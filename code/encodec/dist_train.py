import os
import gc
import json
import time
# import librosa
import argparse
import numpy as np
# from matplotlib import pyplot as plt
from collections import OrderedDict
import re
import sys
import wandb

import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist
from torchaudio import transforms
import torch.multiprocessing as mp
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

# from azureml.core.run import Run

from .dataset import EnCodec_data
from .model import EncodecModel
from .msstftd import MultiScaleSTFTDiscriminator as MSDisc
from .balancer import Balancer
import yaml
import time

class ParamDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

def load_config(cfg_path):
    with open(cfg_path) as file:
        try:
            config_dict = yaml.safe_load(file)   
        except yaml.YAMLError as exc:
            print(exc)
    config = ParamDict(config_dict)
    return config, config_dict

def load_model(model, path):
    state_dict = torch.load(path)
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    model.load_state_dict(model_dict)

def cal_sdr(s, s_hat):

    if len(s.shape) == 3:
        B, C, L = s.shape
        s = s.reshape(B*C, L)
    if len(s_hat.shape) == 3:
        B, C, L = s_hat.shape
        s_hat = s_hat.reshape(B*C, L)

     # s, s_hat - (bt, L)
    return torch.mean(
        -10 * torch.log10(
        torch.sum((s - s_hat)**2, -1) / torch.sum(s**2, -1))
    )

def reconstruction2D(s, x):
    '''
    Reconstruction loss on latent representations.
    '''
    B, C, L = s.shape
    s = s.reshape(B, C*L)
    x = x.reshape(B, C*L)
    loss = torch.sum(torch.abs(s-x), dim=-1)
    return loss.mean()

def entropy_rate(emb):
    '''
    Average entropy rate per batch
    Emb size: [B, C, T]
    '''
    batch_rate = 0
    for b in emb:
        code_count = {}
        for t in range(b.shape[-1]):
            c = b[:,t]
            if c not in code_count:
                code_count[c] = 1
            else:
                code_count[c] +=1
            n = sum(code_count.values())
            n = torch.ones(len(code_count))/n
            e = torch.sum(n*torch.log2(n))
            batch_rate += -e 
    batch_rate /= emb.shape[0]
    return batch_rate


def melspec_loss(s, s_hat, gpu_rank, n_freq):
    
    loss = 0
    sl = s.shape[-1]

    for n in n_freq:

        mel_transform = transforms.MelSpectrogram(
            n_fft=2**n, hop_length=(2**n)//4, 
            win_length=2**n, window_fn=torch.hann_window,n_mels=64,
            normalized=True, center=False, pad_mode=None, power=1).cuda(gpu_rank)
        
        mel_s = mel_transform(s)
        mel_s_hat = mel_transform(s_hat)
    
    loss += torch.mean(abs(mel_s - mel_s_hat)) + torch.mean((mel_s - mel_s_hat) ** 2)

    loss /= 8*sl
    
    return loss

def freeze_params(model, param_name):
    for name, param in model.named_parameters():
        if param_name in name:
            param.requires_grad = False

def valid(model, valid_loader, bandwidth, gpu_rank, use_se_loss, debug):

    # ---------- Run model ------------
    t_loss, sdr_tt, val_entropy = 0, 0, 0
    for idx, batch in enumerate(valid_loader):

        # s shape (64, 1, 16000)
        s, x = batch
        s = s.to(torch.float).cuda(gpu_rank)
        x = x.to(torch.float).cuda(gpu_rank)
        if use_se_loss:
            emb = model.encoder(x)
        else:
            emb = model.encoder(s) # [64, 128, 50]
        frame_rate = 16000 // model.encoder.hop_length
        quantizedResult = model.quantizer(emb, sample_rate=frame_rate, bandwidth=bandwidth) # !!! should be frame_rate - 50
            # Resutls contain - quantized, codes, bw, penalty=torch.mean(commit_loss))
        qtz_emb = quantizedResult.quantized
        #batch_entropy = entropy_rate(qtz_emb)
        #val_entropy+=batch_entropy
        s_hat = model.decoder(qtz_emb) #(64, 1, 16000)

        # ------ Reconstruction loss l_t, l_f --------
        l_t = torch.mean(torch.abs(s - s_hat))
        t_loss += l_t.detach().data.cpu()

        sdr = cal_sdr(s, s_hat)
        sdr_tt += sdr.detach().data.cpu()

        if debug:
            break
    losses = {'val_l_t': t_loss/len(valid_loader), 'sdr_tt': sdr_tt/len(valid_loader)}
    return losses

def get_args():
    
    envvars = [
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "NODE_RANK",
    "NODE_COUNT",
    "HOSTNAME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NCCL_SOCKET_IFNAME",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "AZ_BATCHAI_MPI_MASTER_NODE",
    ]
    args = dict(gpus_per_node=torch.cuda.device_count())
    missing = []
    for var in envvars:
        if var in os.environ:
            args[var] = os.environ.get(var)
            try:
                args[var] = int(args[var])
            except ValueError:
                pass
        else:
            missing.append(var)
    # print(f"II Args: {args}")
    # if missing:
    #     print(f"II Environment variables not set: {', '.join(missing)}.")
    return args

def check_config(inp_args):
    all_valid = True
    if inp_args.use_se_loss and inp_args.use_latent_se_loss:
        print(f"use_se_loss and use_latent_se_loss cannot be used together")
        all_valid = False
    if (inp_args.mixture==False) and (inp_args.use_se_loss or inp_args.use_latent_se_loss):
        print("use_se_loss and use_latent_se_loss require mixture=True")
        all_valid = False

    if inp_args.freeze_dec==True and inp_args.freeze_enc==True:
        print("Both encoder and decoder cannot be frozen")
        all_valid =False
    return all_valid

if __name__ == '__main__':
    config = sys.argv[1]
    inp_args, config_dict = load_config(config)
    config_status = check_config(inp_args)
    if config_status==False:
        exit()
    else:
        print("Checked arguments...")

    train_dataset = EnCodec_data(ds_path=inp_args.data_path, csv_path=inp_args.csv_path, task = 'train', mixture=inp_args.mixture, standardize=inp_args.standardize)
    valid_dataset = EnCodec_data(ds_path=inp_args.data_path, csv_path=inp_args.csv_path, task = 'val',  mixture=inp_args.mixture, standardize=inp_args.standardize)

   
    train_loader = DataLoader(train_dataset, batch_size=inp_args.batch_size, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=inp_args.batch_size, pin_memory=True)
    gpu_rank = 0

    # get new model
    model = EncodecModel._get_model(
                   target_bandwidths = [1.5, 3, 6], 
                   sample_rate = 16000,  # 24_000
                   channels  = 1,
                   causal  = True,
                   model_norm  = 'weight_norm',
                   audio_normalize  = False,
                   segment = None, # tp.Optional[float]
                   name = 'unset').cuda(gpu_rank)
    
    model.set_target_bandwidth(inp_args.bandwidth)

    if inp_args.finetune_model:
        print("Loaded model...")
        load_model(model, inp_args.finetune_model)
    
    disc = MSDisc(filters=32).cuda(gpu_rank)

    if inp_args.finetune_disc:
        print("Loaded discriminator...")
        load_model(disc, inp_args.finetune_disc)
    

    optimizer_G = optim.Adam(model.parameters(), lr=float(inp_args.lr_gen), betas=(0.5, 0.9))
    optimizer_D = optim.Adam(disc.parameters(), lr=float(inp_args.lr_disc), betas=(0.5, 0.9))

    if inp_args.freeze_enc or inp_args.freeze_dec:
        which = 'decoder' if inp_args.freeze_dec else 'encoder'
        freeze_params(model, which)
    
    se_str = f'se_lt_weight_{inp_args.lt_weight}' if inp_args.use_se_loss else ''
    disc_str = 'with_disc' if inp_args.finetune_disc else ''
    freq_str = f'disc_freq{inp_args.disc_freq}'
    std_str = 'std' if {inp_args.standardize} else 'no_std'
    from_ls = 'from_ls' if inp_args.from_ls else ''
    latent_se_str = 'proposed' if inp_args.use_latent_se_loss else ''

    run_name = f"{inp_args.exp_name}_lr{inp_args.lr_gen}_bw{inp_args.bandwidth}_{se_str}_{disc_str}_{freq_str}_{std_str}_{from_ls}_{latent_se_str}"

    if inp_args.freeze_dec or inp_args.freeze_enc:
        freeze_str = "freeze_enc" if inp_args.freeze_enc else "freeze_dec"
    else:
        freeze_str = 'all_unfreeze'
    print(f"RUN NAME: {run_name}")

    if not inp_args.debug:
        wandb.init(config=config_dict, project='encodec-se', entity='anakuzne')
        wandb.run.name = run_name

    '''
    START MAIN TRAINING LOOP HERE
    '''

    step = 1
    log_steps = 100
    tot_steps = 1000000
    track_sdr = torch.tensor(float('-inf'))

    g_loss, d_loss, t_loss, f_loss, w_loss, feat_loss, l_lat = 0,0,0,0,0,0,0
    model.train()

    curr_time = round(time.time()*1000)
    if not os.path.exists(f"{inp_args.output_dir}/{curr_time}"):
        os.makedirs(f"{inp_args.output_dir}/{curr_time}")

    for ep in range(2000):
        for idx, batch in enumerate(train_loader):
            s, x = batch
            # s shape (64, 1, 16000)
            s = s.to(torch.float).cuda(gpu_rank)
            x = x.to(torch.float).cuda(gpu_rank)

            if inp_args.use_se_loss:
                emb = model.encoder(x)
            elif inp_args.use_latent_se_loss:
                emb_x = model.encoder(x)
                emb_s = model.encoder(s)
            else:
                emb = model.encoder(s)
        
            # [64, 128, 50]
            frame_rate = 16000 // model.encoder.hop_length
            if inp_args.use_latent_se_loss:
                quantizedResult_x = model.quantizer(emb_x, sample_rate=frame_rate, bandwidth=inp_args.bandwidth)
                qtz_emb_x = quantizedResult_x.quantized
                quantizedResult_s = model.quantizer(emb_s, sample_rate=frame_rate, bandwidth=inp_args.bandwidth)
                qtz_emb_s = quantizedResult_s.quantized
                s_hat = model.decoder(qtz_emb_s)
            else:
                quantizedResult = model.quantizer(emb, sample_rate=frame_rate, bandwidth=inp_args.bandwidth) # !!! should be frame_rate - 50
                    # Resutls contain - quantized, codes, bw, penalty=torch.mean(commit_loss))
                qtz_emb = quantizedResult.quantized
                s_hat = model.decoder(qtz_emb) #(64, 1, 16000)

            # --- Update Generator ---
            optimizer_G.zero_grad()

            # ---- VQ Commitment loss l_w -----
            if inp_args.use_latent_se_loss:
                l_w = quantizedResult_s.penalty
            else:
                l_w = quantizedResult.penalty # commitment loss
            l_t = torch.mean(torch.abs(s - s_hat))
            l_f = melspec_loss(s, s_hat, gpu_rank, range(5,12))
            if inp_args.use_latent_se_loss:
                l_l = reconstruction2D(qtz_emb_s, qtz_emb_x)
                l_lat += l_l.detach().data.cpu()

            t_loss += l_t.detach().data.cpu()
            f_loss += l_f.detach().data.cpu()
            w_loss += l_w.detach().data.cpu()

            # ---- Discriminator l_d, l_g, l_feat -----

            if inp_args.use_disc:
                s_disc_r, fmap_r = disc(s) # list of the outputs from each discriminator
                s_disc_gen, fmap_gen = disc(s_hat)

                # s_disc_r: [3*[batch_size*[1, 309, 65]]]
                # fmap_r: [3*5*torch.Size([64, 32, 59, 513/257/128..])] 5 conv layers, different stride size
                K = len(fmap_gen)
                L = len(fmap_gen[0])

                l_g = 0
                l_feat = 0

                for d_id in range(len(fmap_r)):

                    l_g += 1/K * torch.mean(torch.max(torch.tensor(0), 1-s_disc_gen[d_id])) # Gen loss

                    for l_id in range(len(fmap_r[0])):
                        l_feat += 1/(K*L) * torch.mean(abs(fmap_r[d_id][l_id] - fmap_gen[d_id][l_id]))/torch.mean(abs(fmap_r[d_id][l_id]))

                g_loss += l_g.detach().data.cpu()
                feat_loss += l_feat.detach().data.cpu()

                l_w.backward(retain_graph=True)

                if inp_args.use_latent_se_loss:
                    l_l = float(inp_args.ll_weight) * l_l
                    l_l.backward(retain_graph=True)
                #if inp_args.use_latent_se_loss:
                #    losses = {'l_t': l_t, 'l_f': l_f, 'l_g': l_g, 'l_feat': l_feat, 'l_l':l_l}
                #    balancer_weights = {'l_t': float(inp_args.lt_weight), 'l_f': 1, 'l_g': 3, 'l_feat': 3, 'l_l':float(inp_args.ll_weight)}
                #else:
                losses = {'l_t': l_t, 'l_f': l_f, 'l_g': l_g, 'l_feat': l_feat}
                balancer_weights = {'l_t': float(inp_args.lt_weight), 'l_f': 1, 'l_g': 3, 'l_feat': 3}
                balancer = Balancer(weights=balancer_weights, rescale_grads=True)
                balancer.backward(losses, s_hat)
                optimizer_G.step()

                # Update Discriminator
                if idx % inp_args.disc_freq == 0:
                    optimizer_D.zero_grad()

                    l_d = 0
                    s_disc_r, fmap_r = disc(s) # list of the outputs from each discriminator
                    s_disc_gen, fmap_gen = disc(s_hat.detach())
                
                    for d_id in range(len(fmap_r)):
                        l_d += 1/K * torch.mean(torch.max(torch.tensor(0), 1-s_disc_r[d_id]) + torch.max(torch.tensor(0), 1+s_disc_gen[d_id])) # Disc loss

                    l_d.backward()
                    optimizer_D.step()

                    d_loss += l_d.detach().data.cpu()

            else:
                loss = (float(inp_args.lt_weight) * l_t) + l_f + l_w
                loss.backward()
                optimizer_G.step()
            
            
            if  step%log_steps==0:
                log_losses = {'l_g': g_loss/log_steps, 
                        'l_d': d_loss/log_steps, 
                        'l_t': t_loss/log_steps, 
                        'l_f': f_loss/log_steps, 
                        'l_w': w_loss/log_steps, 
                        'l_feat': feat_loss/log_steps,
                        'l_l': l_lat/log_steps}
                if not inp_args.debug:
                    wandb.log(log_losses)
                print(f"[Step]: {step} | [Train losses]: {log_losses}")
                g_loss, d_loss, t_loss, f_loss, w_loss, feat_loss, l_lat = 0,0,0,0,0,0,0

                model.eval()
                with torch.no_grad():
                    val_losses = valid(model, valid_loader, inp_args.bandwidth, gpu_rank, inp_args.use_se_loss, inp_args.debug)
                    print(f"[Step]: {step} | [Valid losses]: {val_losses}")
                    if not inp_args.debug:
                        wandb.log(val_losses)
                    vall = list(val_losses.values())[-1] # sdr
                    #If SDR is larger than previous, save new checkpoint
                    
                    if vall > track_sdr:
                        print(f'Prev SDR {track_sdr}, Curr SDR {vall}')
                        track_sdr = vall
                        torch.save(model.state_dict(), f'{inp_args.output_dir}/{curr_time}/{run_name}_best.amlt')
                        torch.save(disc.state_dict(), f'{inp_args.output_dir}/{curr_time}/{run_name}_disc_best.amlt')

                    torch.save(model.state_dict(), f'{inp_args.output_dir}/{curr_time}/{run_name}_latest.amlt')
                    torch.save(disc.state_dict(), f'{inp_args.output_dir}/{curr_time}/{run_name}_disc_latest.amlt')

                model.train()

            step+=1

            if step>=tot_steps:
                break
            #if inp_args.debug:
            #    break