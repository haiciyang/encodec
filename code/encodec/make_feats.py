import os
import gc
import re
import json
import time
import argparse
from scipy.io import wavfile
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist
from torchaudio import transforms
import torch.multiprocessing as mp
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .model import EncodecModel
from .msstftd import MultiScaleSTFTDiscriminator as MSDisc
from .balancer import Balancer


if __name__ == '__main__':

        
    parser = argparse.ArgumentParser(description="Encodec_baseline")
    parser.add_argument("--data_path", type=str, default='/home/v-haiciyang/data/haici/dns_pth/*')
    parser.add_argument("--model_path", type=str, default='/home/v-haiciyang/amlt/really_with_balancer/manual_use_balancer_50/epoch1600_model.amlt')
    parser.add_argument("--note2", type=str, default='manual_balancer_50_1600_6')
    parser.add_argument('--multi', dest='multi', action='store_true')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--bandwidth', type=float, default=6.0)

    
    inp_args = parser.parse_args()

    model = EncodecModel._get_model(
                target_bandwidths = [1.5, 3, 6], 
                sample_rate = 16000,  # 24_000
                channels  = 1,
                causal  = True,
                model_norm  = 'weight_norm',
                audio_normalize  = False,
                segment = None,
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
    model.set_target_bandwidth(inp_args.bandwidth)


    model.eval()
        
    ls_path = '/media/sdb/anakuzne/data/librispeech' 
    for root, d, files in os.walk(ls_path):
        for f in files:
            if ".wav" in f:
                p = os.path.join(root, f)
                wav, fs = torchaudio.load(p)
                emb = model.encoder(wav.cuda()) # [64, 128, 50]
                quantizedResult = model.quantizer(emb, sample_rate=16000) 
                # Resutls contain - quantized, codes, bw, penalty=torch.mean(commit_loss))
                qtz_emb = quantizedResult.quantized
                print(qtz_emb.shape)


