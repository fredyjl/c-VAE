import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import librosa
import mir_eval
import pandas as pd

def singername_from_MIR1Kfilename(MIR1Kfilename):
    return MIR1Kfilename.split('/')[-1].split('.')[0].split('_')[0]

class MIR1Kdataset(Dataset):
    def __init__(self, data_path, sr=16000, transform=None):
        """
        Args:
            data_path: the Dictionary containing key 'wav' and 'pitch', 
                       the corresponding names are lists of path.
            sr: sampling rate
            transform: desired preprocessing and feature extraction
        """
        self.data_path = data_path
        self.sr = sr
        self.transform = transform

        assert 'wav' in self.data_path.keys()
        assert 'pitch' in self.data_path.keys()
        
        wav_path = sorted(data_path['wav'])
        pitch_path = sorted(data_path['pitch'])
        
        self.wav_path = wav_path
        self.pitch_path = pitch_path
        assert len(self.wav_path) == len(self.pitch_path)
        assert sum([singername_from_MIR1Kfilename(i) == singername_from_MIR1Kfilename(j) \
                    for i, j in zip(self.wav_path, self.pitch_path)])\
                    == len(self.wav_path)
        
        # singer_name follows the order of self.wav_dir
        singer_name = [singername_from_MIR1Kfilename(i) for i in self.wav_path]
        # build a look-up table for each singer with corresponding singer id
        # it is also treated as singer label 
        set_singer_name = sorted(list(set(singer_name)))
        singer_label = {}
        for i_singerid, i_singername in enumerate(set_singer_name):
            singer_label[i_singername] = i_singerid
        self.singer_label = singer_label
        
    def __len__(self):
        return len(self.wav_path)

    def __getitem__(self, idx):
        # make sure that self.wav_dir and self.pitch_dir follow the same order
        wavpath = self.wav_path[idx]
        pitchpath = self.pitch_path[idx]
        wav_id = wavpath.split('/')[-1].split('.')[0]
        pitch_id = pitchpath.split('/')[-1].split('.')[0]
        assert wav_id == pitch_id # format: singername_clipid
        # load the audio
        x, _ = librosa.load(wavpath, sr=self.sr, mono=False)
        x = x[1]
        # load the pitch label
        y_pitch = pd.read_csv(pitchpath, header=None).as_matrix().T[0]
        # load the singer label
        singername = singername_from_MIR1Kfilename(wavpath)
        y_singer = self.singer_label[singername]

        if self.transform:
            x = self.transform(x)
            if sum(['ContextWindow' in str(trans_comp) for trans_comp in self.transform.transforms]) > 0:
                center_framepos = self.transform.transforms[2].win_position
                y_pitch = y_pitch[center_framepos]
        
        return {'wav': wavpath, 'pitch': pitchpath, 'X':x,\
                'y_pitch': y_pitch, 'y_singer': y_singer}

# --- Define the transformations ---
class Zscore(object):
    def __init__(self, divide_sigma=True):
        self.divide_sigma = divide_sigma
    def __call__(self, x):
        if len(x.shape) == 2:
            x -= x.mean(axis=1)[:, np.newaxis]
            if self.divide_sigma:
                x /= x.std(axis=1)[:, np.newaxis]
        else:
            x -= x.mean()
            if self.divide_sigma:
                x /= x.std()
        return x

class Spectrogram(object):
    def __init__(self, n_fft=1024, hop_size=160, n_band=256, center=True):
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.n_band = n_band
        self.center = center

    def __call__(self, x, sr=16000):
        S = librosa.core.stft(y=x, n_fft=self.n_fft, hop_length=self.hop_size, 
                              center=self.center)
        S = librosa.feature.melspectrogram(S=S, n_mels=self.n_band)
        S = np.abs(S) ** 2 # convert amplitude to power
        S = librosa.core.power_to_db(S)
        return S

class RMSenergy(object):
    def __init__(self, th=0.2):
        self.th = th
    def __call__(self, X):
        spec_rmse = librosa.feature.rmse(S=X)[0]
        spec_rmse = spec_rmse/max(spec_rmse)
        X = np.vstack([spec_rmse >= self.th, X])
        return X

class ConvertToDb(object):
    def __call__(self, X):
        spec = X[1:, :]
        voice_frame = X[0, :]
        S = librosa.core.power_to_db(spec)
        S = np.vstack([voice_frame, S])
        return S

class ContextWindow(object):
    def __init__(self, window_size=21, hop_size=10):
        assert window_size % 2 == 1
        self.window_size = window_size
        self.hop_size = hop_size
    
    def __call__(self, x):
        pad_size = int(self.window_size/2)
        spec_pad = np.pad(x, pad_width=((0,0), (pad_size,pad_size)),\
                          mode='constant', constant_values=0)
        win_position = np.arange(0, x.shape[1], self.hop_size)
        self.win_position = win_position
        list_context = []
        for i in win_position:
            list_context.append(spec_pad[:, i : i + self.window_size])

        return np.expand_dims(np.array(list_context), 1)

class ToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x)