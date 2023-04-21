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
from .dist_train import load_config, check_config, load_model, melspec_loss, reconstruction2D, freeze_params



def valid_latent_only(model, valid_loader, bandwidth, gpu_rank, mixture, debug):
    lat_loss = 0
    for idx, batch in enumerate(valid_loader):

        # s shape (64, 1, 16000)
        s, x = batch
        s = s.to(torch.float).cuda(gpu_rank)
        x = x.to(torch.float).cuda(gpu_rank)
       
        emb_x = model.encoder(x)
        emb_s = model.encoder(s) # [64, 128, 50]
        frame_rate = 16000 // model.encoder.hop_length
        quantizedResult_x = model.quantizer(emb_x, sample_rate=frame_rate, bandwidth=bandwidth) # !!! should be frame_rate - 50
        quantizedResult_s = model.quantizer(emb_s, sample_rate=frame_rate, bandwidth=bandwidth)
        qtz_emb_x = quantizedResult_x.quantized
        qtz_emb_s = quantizedResult_s.quantized

        # ------ Reconstruction loss l_t, l_f --------
        l_l = reconstruction2D(qtz_emb_s, qtz_emb_x)
        lat_loss += l_l.detach().data.cpu()
        if debug:
            break
    losses = {'val_l_l': lat_loss/len(valid_loader)}
    return losses


if __name__ == '__main__':
    '''
    Encoder only training.
    Does not include anything but reconstruction loss in the latent space
    - No discriminator training
    - No additional losses, no balancer
    '''

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

    optimizer_G = optim.Adam(model.parameters(), lr=float(inp_args.lr_gen), betas=(0.5, 0.9))

    if inp_args.freeze_enc or inp_args.freeze_dec:
        which = 'decoder' if inp_args.freeze_dec else 'encoder'
        freeze_params(model, which)
    

    ########RUN NAME ############
    std_str = 'std' if {inp_args.standardize} else 'no_std'
    from_ls = 'from_ls' if inp_args.from_ls else ''
    latent_se_str = 'proposed' if inp_args.use_latent_se_loss else ''

    run_name = f"{inp_args.exp_name}_lr{inp_args.lr_gen}_bw{inp_args.bandwidth}_{std_str}_{from_ls}_{latent_se_str}"

    if inp_args.freeze_dec or inp_args.freeze_enc:
        freeze_str = "freeze_enc" if inp_args.freeze_enc else "freeze_dec"
    else:
        freeze_str = 'all_unfreeze'
    print(f"RUN NAME: {run_name}")

    curr_time = round(time.time()*1000)
    if not os.path.exists(f"{inp_args.output_dir}/{curr_time}"):
        os.makedirs(f"{inp_args.output_dir}/{curr_time}")
    
    config_dict['save_prefix'] = f"{inp_args.output_dir}/{curr_time}/{run_name}"

    if not inp_args.debug:
        wandb.init(config=config_dict, project='encodec-se', entity='anakuzne')
        wandb.run.name = run_name


    '''
    START MAIN TRAINING LOOP HERE
    '''

    step = 1
    log_steps = 100
    tot_steps = 1000000
    track_iter_loss = torch.tensor(float('+inf'))

    l_lat = 0
    model.train()

    for ep in range(2000):
        for idx, batch in enumerate(train_loader):
            s, x = batch
            # s shape (64, 1, 16000)
            s = s.to(torch.float).cuda(gpu_rank)
            x = x.to(torch.float).cuda(gpu_rank)

            emb_x = model.encoder(x)
            emb_s = model.encoder(s)
       
            # [64, 128, 50]
            frame_rate = 16000 // model.encoder.hop_length
           
            quantizedResult_x = model.quantizer(emb_x, sample_rate=frame_rate, bandwidth=inp_args.bandwidth)
            qtz_emb_x = quantizedResult_x.quantized
            quantizedResult_s = model.quantizer(emb_s, sample_rate=frame_rate, bandwidth=inp_args.bandwidth)
            qtz_emb_s = quantizedResult_s.quantized


            # --- Update Generator ---
            optimizer_G.zero_grad()

            l_l = reconstruction2D(qtz_emb_s, qtz_emb_x)
            l_lat += l_l.detach().data.cpu()
            #l_l = float(inp_args.ll_weight) * l_l
            l_l.backward(retain_graph=True)
            optimizer_G.step()
            
            if step%log_steps==0:
                log_losses = {"l_l":l_lat/log_steps}
                if not inp_args.debug:
                    wandb.log(log_losses)
                print(f"[Step]: {step} | [Train losses]: {log_losses}")
                l_lat = 0

                model.eval()
                with torch.no_grad():
                    val_losses = valid_latent_only(model, valid_loader, inp_args.bandwidth, gpu_rank, inp_args.mixture, inp_args.debug)
                    print(f"[Step]: {step} | [Valid losses]: {val_losses}")
                    if not inp_args.debug:
                        wandb.log(val_losses)
                    vall = list(val_losses.values())[-1] 
                    
                    if vall < track_iter_loss:
                        track_iter_loss = vall
                        torch.save(model.state_dict(), f'{inp_args.output_dir}/{curr_time}/{run_name}_best.amlt')
                    torch.save(model.state_dict(), f'{inp_args.output_dir}/{curr_time}/{run_name}_latest.amlt')
                model.train()

            step+=1

            if step>=tot_steps:
                break
    
