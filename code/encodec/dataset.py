import os
import glob
import torch
import random
import numpy as np
from scipy import signal
from scipy.io import wavfile
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

class EnCodec_data(Dataset):

	def __init__(self, ds_path, csv_path, task='train', mixture=False): 
		self.ds_path = ds_path
		self.df = pd.read_csv(csv_path)
		self.df_part = self.df[self.df['part']==task]
		self.mixture = mixture
		self.mixture_data = ['/data/common/musan/noise/free-sound', '/data/common/musan/noise/sound-bible']

	def __len__(self):
		return len(self.df_part)


	def __getitem__(self, idx):
		output = self.get_seq(idx)
		return output

	def sample_noise(self):
		dir_idx = random.randint(0, len(self.mixture_data)-1)
		noise_dir = self.mixture_data[dir_idx]
		noise_files = os.listdir(noise_dir)
		file_idx = random.randint(0, len(noise_files)-1)
		_, noise = wavfile.read(f"{noise_dir}/{noise_files[file_idx]}")
		return noise

	def pad_noise(self, speech, noise):
		'''
		Cuts noise vector if speech vec is shorter
		Adds noise if speech vector is longer
		'''
		noise_len = noise.shape[1]
		speech_len = speech.shape[1]

		if speech_len > noise_len:
			repeat = (speech_len//noise_len) +1
			noise = torch.tile(noise, (1, repeat))
			diff = speech_len - noise.shape[1]
			noise = noise[:, :noise.shape[1]+diff]          
				
		elif speech_len < noise_len:
			noise = noise[:,:speech_len]
		return noise

	def mix_signals(self, speech, noise, desired_snr):   
		#calculate energies
		energy_s = torch.sum(speech**2, dim=-1, keepdim=True)
		energy_n = torch.sum(noise**2, dim=-1, keepdim=True)

		b = torch.sqrt((energy_s / energy_n) * (10 ** (-desired_snr / 10.)))
		return speech + b * noise


	def get_seq(self, idx, seg_len=5):
		row = self.df_part.iloc[idx]
		fname = row['file']
		fp = f'{self.ds_path}/{fname}'
		
		fs, wav = wavfile.read(fp)

		st_idx = row['seg']
		end_idx = st_idx + seg_len*fs
		seg = wav[st_idx:end_idx]

		# Normalize and add random gain
		seg = seg / (np.std(seg) + 1e-20)
		
		gain = np.random.randint(-10, 7, (1,))
		scale = np.power(10, gain/20)
		seg *= scale
		seg = seg.reshape(1, -1)
		if self.mixture:
			noise = self.sample_noise()
			noise = self.pad_mode(seg, noise)
			snr = random.randint(-5, 5)
			x = self.mix_signals(seg, noise, snr)
		else:
			x_seg = seg
		return seg, x_seg