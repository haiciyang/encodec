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

from .dataset_stable import EnCodec_data
from .model import EncodecModel
from .msstftd import MultiScaleSTFTDiscriminator as MSDisc
from .balancer import Balancer


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

        
    parser = argparse.ArgumentParser(description="Encodec_baseline")
    parser.add_argument("--data_path", type=str, default='/data/hy17/dns_pth/*')
    parser.add_argument("--model_path", type=str, default='/home/hy17/Projects/encodec/saved_models/multi_encodec_3.amlt')
    parser.add_argument("--note2", type=str, default='')
    parser.add_argument('--multi', dest='multi', action='store_true')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--bandwidth', type=float, default=3.0)

    
    inp_args = parser.parse_args()

    # args = get_args()

    # synchronizes all the threads to reach this point before moving on
    # dist.barrier() 

    # train_dataset = EnCodec_data(inp_args.data_path, task = 'train', seq_len_p_sec = 5, sample_rate=16000, multi=inp_args.multi)
    # valid_dataset = EnCodec_data(inp_args.data_path, task = 'eval', seq_len_p_sec = 5, sample_rate=16000, multi=inp_args.multi)
    valid_dataset = EnCodec_data(inp_args.data_path, task = 'valid', seq_len_p_sec = 5, sample_rate=16000, multi=inp_args.multi, n_spks = 100)
    # train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, pin_memory=True)

    # # get pretrained model
    if inp_args.sr == 24000:
        model = EncodecModel.encodec_model_24khz().cuda()
        model.set_target_bandwidth(inp_args.bandwidth)

    # get new model
    else:
        model = EncodecModel._get_model(
                    target_bandwidths = [1.5, 3, 6], 
                    sample_rate = 16000,  # 24_000
                    channels  = 1,
                    causal  = True,
                    model_norm  = 'weight_norm',
                    audio_normalize  = False,
                    segment = None, # tp.Optional[float]
                    name = 'unset').cuda()

        state_dict = torch.load(inp_args.model_path)
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k,v in state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict = state_dict
        model.load_state_dict(model_dict)
        # fake()

        model.set_target_bandwidth(inp_args.bandwidth)


    model.eval()
    idx = 0
    note1 = 'multi' if inp_args.multi else 'single'
    note2 = inp_args.note2
    # for s in valid_loader:
    for idx, batch in enumerate(valid_loader):
        
        s = batch[0]
        
        # s shape (64, 1, 16000)
        
        s = s.unsqueeze(1).to(torch.float).cuda()

        # s /= torch.max(torch.abs(s))

        # plt.plot(s.squeeze().cpu().data.numpy())
        # plt.savefig('s.png')
        # plt.clf()
        
        emb = model.encoder(s) # [64, 128, 50]
        quantizedResult = model.quantizer(emb, sample_rate=16000) 
            # Resutls contain - quantized, codes, bw, penalty=torch.mean(commit_loss))
        qtz_emb = quantizedResult.quantized
        s_hat = model.decoder(qtz_emb) #(64, 1, 16000)

        # plt.plot(s_hat.squeeze().cpu().data.numpy())
        # plt.savefig('s_hat.png')
        # # plt.show()
        # plt.clf()

        if inp_args.sr != 16000:
            s_hat = signal.resample(s_hat.squeeze().cpu().data.numpy(), 16000*5)
        else:
            s_hat = s_hat.squeeze().cpu().data.numpy()

        s = s.squeeze().cpu().data.numpy()
        wavfile.write(f"eval_wavs/s_{idx}_{note1}.wav", 16000, s/max(abs(s)))
        wavfile.write(f"eval_wavs/sh_{idx}_{note1}_{note2}.wav", 16000, s_hat/max(abs(s_hat)))

        if inp_args.multi:
            break
        if not inp_args.multi and idx == 1:
            break
        
        
