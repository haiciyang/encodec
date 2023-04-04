from pystoi import stoi
from pesq import pesq
import os
from scipy.io import wavfile
import numpy as np
from mir_eval.separation import bss_eval_sources
import argparse
from tqdm import tqdm

def metrics(input_path, fs=16000):
    files = os.listdir(input_path)
    pairs = []
    
    for i in range(len(files)//2):
        s = f"{input_path}/s_{i}_single.wav"
        sh = f"{input_path}/sh_{i}_single_.wav"
        pairs.append((s, sh))

    SDR = 0
    PESQ = 0
    STOI = 0

    SDR_ref = 0
    PESQ_ref = 0
    STOI_ref = 0

    for p in tqdm(pairs):
        s, sh = p
        _, s = wavfile.read(s)
        _, sh = wavfile.read(sh)
        sdr, _, _, _ = bss_eval_sources(s, sh)
        SDR+=sdr
        STOI += stoi(s, sh, fs, True)
        PESQ += pesq(fs, s, sh, 'wb')

        sdr, _, _, _ = bss_eval_sources(s, s)
        SDR_ref+= sdr
        STOI_ref+= stoi(s, s, fs, True)
        PESQ_ref+= pesq(fs, s, s, 'wb')

    SDR/=len(pairs)
    PESQ/=len(pairs)
    STOI/=len(pairs)

    SDR_ref/=len(pairs)
    STOI_ref/=len(pairs)
    PESQ_ref/=len(pairs)

    print(f"[SDRref]: {SDR_ref} [ePESQref]: {PESQ_ref} [eSTOIref]: {STOI_ref}")
    print(f"[SDR]: {SDR} [ePESQ]: {PESQ} [eSTOI]: {STOI}")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encodec_baseline")
    parser.add_argument("--data_path", type=str, default='/home/anakuzne/projects/encodec_hy/encodec/code/eval_wavs')
    inp_args = parser.parse_args()
    
    metrics(inp_args.data_path)

