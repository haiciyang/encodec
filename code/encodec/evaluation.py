import os
import gc
import re
import json
import time
# import librosa
import argparse
import numpy as np
from scipy import signal
from scipy.io import wavfile

from collections import OrderedDict
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist
from torchaudio import transforms
import torch.multiprocessing as mp
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import EnCodec_data
from .model import EncodecModel
from .msstftd import MultiScaleSTFTDiscriminator as MSDisc
from .balancer import Balancer
from .dist_train import entropy_rate, freeze_params, load_config
import sys
from .dist_train import cal_sdr, load_model
from pystoi import stoi
from pesq import pesq
import matplotlib.pyplot as plt


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
    
    loss += torch.sum(abs(mel_s - mel_s_hat)) + torch.sum((mel_s - mel_s_hat) ** 2)

    loss /= 8*sl
    
    return loss


if __name__ == '__main__':

    inp_args, args_dict = load_config(sys.argv[1])

    # args = get_args()

    # synchronizes all the threads to reach this point before moving on
    # dist.barrier() 

    torch.manual_seed(0)

    # train_dataset = EnCodec_data(inp_args.data_path, task = 'train', seq_len_p_sec = 5, sample_rate=16000, multi=inp_args.multi)
    # valid_dataset = EnCodec_data(inp_args.data_path, task = 'eval', seq_len_p_sec = 5, sample_rate=16000, multi=inp_args.multi)
    valid_dataset = EnCodec_data(inp_args.data_path, inp_args.csv_path, task = inp_args.task, mixture=inp_args.mixture, standardize=inp_args.standardize)
    # train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, pin_memory=True)

    # # get pretrained model
    #if inp_args.sr == 24000:
    ##    model = EncodecModel.encodec_model_24khz().cuda()
     #   model.set_target_bandwidth(inp_args.bandwidth)

    # get new model
    #else:
    model = EncodecModel._get_model(
                    target_bandwidths = [1.5, 3, 6], 
                    sample_rate = 16000,  # 24_000
                    channels  = 1,
                    causal  = True,
                    model_norm  = 'weight_norm',
                    audio_normalize  = False,
                    segment = None, # tp.Optional[float]
                    name = 'unset').cuda()
    model.set_target_bandwidth(inp_args.bandwidth)
    load_model(model, inp_args.inf_model_path)

    model.eval()
    idx = 0
    num_inference = None

    tot_sdr = 0
    tot_stoi = 0
    tot_pesq = 0
    skipped = 0

    for idx, batch in enumerate(valid_loader):
        if not num_inference:
            print(f"[Idx] : {idx+1}/{len(valid_loader)}")
        else:
            print(f"[Idx] : {idx+1}/{num_inference}")

        
        s, x = batch
        s = s.to(torch.float).cuda()
        x = x.to(torch.float).cuda()

        if inp_args.mixture:
            emb = model.encoder(x)
        else:
            emb = model.encoder(s)

        frame_rate = 16000 // model.encoder.hop_length
        quantizedResult = model.quantizer(emb, sample_rate=frame_rate, bandwidth=inp_args.bandwidth) 
        qtz_emb = quantizedResult.quantized
        
        #e_rate = entropy_rate(qtz_emb)
        s_hat = model.decoder(qtz_emb) #(64, 1, 16000)
        if inp_args.plot:
            fig = plt.figure(figsize=(10, 10))
            qtz_emb = qtz_emb.squeeze().cpu().numpy()
            plt.imshow(qtz_emb)
            plt.savefig(f"{inp_args.inf_output_path}/sh_{idx}.png", dpi=200)

        sdr = cal_sdr(s, s_hat)
    
        #if inp_args.sr != 16000:
        #    s_hat = signal.resample(s_hat.squeeze().cpu().data.numpy(), 16000*5)
        #else:
        s_hat = s_hat.squeeze().cpu().data.numpy()
        s = s.squeeze().cpu().data.numpy()
        try:
            p = pesq(inp_args.sr, s, s_hat, 'wb')
        except:
            skipped+=1
            continue
        st = stoi(s, s_hat, inp_args.sr, True)

        tot_sdr += sdr.data
        tot_pesq+=p 
        tot_stoi+=st
        wavfile.write(f"{inp_args.inf_output_path}/s_{idx}.wav", 16000, s/max(abs(s)))
        wavfile.write(f"{inp_args.inf_output_path}/sh_{idx}.wav", 16000, s_hat/max(abs(s_hat)))

        if num_inference and idx==num_inference:
            tot_sdr/=(num_inference-skipped)
            tot_pesq/=(num_inference-skipped)
            tot_stoi/=(num_inference-skipped)
            print(f"[SDR]: {tot_sdr:0.4f} [ePESQ]: {tot_pesq:0.4f} [eSTOI]: {tot_stoi:0.4f}")
            print(f"Skipped {skipped}")
            break
    if not num_inference:
        tot_sdr/=(len(valid_loader)-skipped)
        tot_pesq/=(len(valid_loader)-skipped)
        tot_stoi/=(len(valid_loader)-skipped)
        print(f"[SDR]: {tot_sdr:0.4f} [ePESQ]: {tot_pesq:0.4f} [eSTOI]: {tot_stoi:0.4f}")
        print(f"Skipped {skipped}")