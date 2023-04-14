from pystoi import stoi
from pesq import pesq
import os
from scipy.io import wavfile
import numpy as np
from mir_eval.separation import bss_eval_sources
import argparse
from tqdm import tqdm
import torch


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




def metrics(data_path, csv_path, output_path, task, fs=16000):
    valid_dataset = EnCodec_data(ds_path=idata_path, csv_path=icsv_path, task = task,  mixture=True, standardize=True)

    SDR = 0
    PESQ = 0
    STOI = 0

    skipped = 0
    
        try:
            PESQ += pesq(fs, s, sh, 'wb')
        except:
            skipped+=1
            continue
        STOI += stoi(s, sh, fs, True)
        s = torch.tensor(s).unsqueeze(0)
        sh = torch.tensor(sh).unsqueeze(0)
        #print(s.shape, sh.shape)
        sdr = cal_sdr(s, sh)
        SDR+=sdr.data
    SDR/=(len(pairs)-skipped)
    PESQ/=(len(pairs)-skipped)
    STOI/=(len(pairs)-skipped)
    print(f'Skipped {skipped}/{len(pairs)}')
    print(f"[SDR]: {SDR:0.4f} [ePESQ]: {PESQ:0.4f} [eSTOI]: {STOI:0.4f}")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encodec_baseline")
    parser.add_argument('--data_path', type=str, default="/data/common")
    parser.add_argument("--csv_path", type=str, default="/home/anakuzne/utils/csvs/encodec_ds.csv")
    parser.add_argument("--output_path", type=str, default='/home/anakuzne/projects/encodec_hy/encodec/code/eval_wavs')
    parser.add_argument("--task", type=str, default='val')
    inp_args = parser.parse_args()
    
    metrics(inp_args.csv_path, inp_args.output_path, inp_args.task)

