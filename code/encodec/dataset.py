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

	def __init__(self, ds_path, csv_path, task='train', mixture=False, standardize=True): 
		self.ds_path = ds_path
		self.df = pd.read_csv(csv_path)
		self.df_part = self.df[self.df['part']==task]
		self.mixture = mixture
		self.standardize = standardize
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
		noise_files = [i for i in noise_files if '.wav' in i]
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
			noise = np.tile(noise, (1, repeat))
			diff = speech_len - noise.shape[1]
			noise = noise[:, :noise.shape[1]+diff]          
				
		elif speech_len < noise_len:
			noise = noise[:,:speech_len]
		return noise

	def mix_signals(self, speech, noise, desired_snr):   
		#calculate energies
		energy_s = np.sum(speech**2, axis=1,keepdims=True)
		energy_n = np.sum(noise**2, axis=1, keepdims=True)

		b = np.sqrt((energy_s / energy_n) * (10 ** (-desired_snr / 10.)))
		return speech + b * noise

	def standardize_f(self, seg):
		seg = seg / (np.std(seg) + 1e-20)
		gain = np.random.randint(-10, 7, (1,))
		scale = np.power(10, gain/20)
		seg *= scale
		return seg

	def get_seq(self, idx, seg_len=5):
		row = self.df_part.iloc[idx]
		fname = row['file']
		fp = f'{self.ds_path}/{fname}'
		
		fs, wav = wavfile.read(fp)

		st_idx = row['seg']
		end_idx = st_idx + seg_len*fs
		seg = wav[st_idx:end_idx]

		if (self.standardize) and (self.mixture):
			# Normalize and add random gain
			seg = self.standardize_f(seg)
			seg = seg.reshape(1, -1)

			noise = self.sample_noise().reshape(1, -1)
			noise = self.pad_noise(seg, noise)
			noise = self.standardize_f(noise)
			snr = random.randint(-5, 5)
			x_seg = self.mix_signals(seg, noise, snr)

		elif (self.mixture) and (not self.standardize):
			#if no standartization is done, normalize the signals instead
			seg = seg.reshape(1, -1)
			seg = seg/np.linalg.norm(seg)
			noise = self.sample_noise().reshape(1, -1)
			noise = self.pad_noise(seg, noise)
			noise = noise/np.linalg.norm(noise)
			snr = random.randint(-5, 5)
			x_seg = self.mix_signals(seg, noise, snr)
		
		elif (self.standardize) and (not self.mixture):
			seg = self.standardize_f(seg)
			seg = seg.reshape(1, -1)
			x_seg = seg

		else:
			seg = seg.reshape(1, -1)
			x_seg = seg
		return seg, x_seg