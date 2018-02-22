import os
import argparse
from data_prepare import *
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.optim as optim
from model import *
from loss import *
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, metavar='N')
parser.add_argument('--batch-size', type=int, default=128, metavar='N')
parser.add_argument('--log-interval', type=int, default=10, metavar='N')
parser.add_argument('--spec-hopsize', type=int, default=320, metavar='N')
parser.add_argument('--spec-winsize', type=int, default=1024, metavar='N')
parser.add_argument('--spec-nband', type=int, default=256, metavar='N')
parser.add_argument('--contextwin-hopsize', type=int, default=10, metavar='N')
parser.add_argument('--contextwin-winsize', type=int, default=21, metavar='N')
parser.add_argument('--code-size', type=int, default=128, metavar='N')
args = parser.parse_args()

result_save_path = '/home/yjluo/projects/c-VAE/result_encoder/'

n_epochs = args.epochs
batch_size = args.batch_size
log_interval = args.log_interval
contextwin_winsize = args.contextwin_winsize
contextwin_hopsize = args.contextwin_hopsize
spec_winsize = args.spec_winsize
spec_hopsize = args.spec_hopsize
spec_nband = args.spec_nband
code_size = args.code_size
print("=" * 20)
print("saving results to: %s" % result_save_path)
print("epochs: %d,\tbatch size: %d,\tlog interval: %d" % (n_epochs, batch_size, log_interval))
print("spec band: %d,\tspec window: %d,\tspec hop: %d,\t" % (spec_nband, spec_winsize, spec_hopsize))
print("contextwin window: %d,\tcontextwin hop: %d" % (contextwin_winsize, contextwin_hopsize))
print("code size: %d" % code_size)
print("=" * 20)

data_dir = {'wav': '/home/yjluo/dataset/MIR-1K/UndividedWavfile/', 
            'pitch': '/home/yjluo/dataset/MIR-1K/UndividedPitchLabel_resample/'}
wav_path = np.array(sorted([os.path.join(data_dir['wav'], i) for i in os.listdir(data_dir['wav']) if '.wav' in i]))
pitch_path = np.array(sorted([os.path.join(data_dir['pitch'], i) for i in os.listdir(data_dir['pitch']) if '.csv' in i]))
singer_id = [singername_from_MIR1Kfilename(i) for i in wav_path]
valid_singerid = ['tammy', 'stool', 'yifen']
train_ind = np.array([i for i in range(len(singer_id)) if singer_id[i] not in valid_singerid])
valid_ind = np.array([i for i in range(len(singer_id)) if singer_id[i] in valid_singerid])

train_wavpath, valid_wavpath = wav_path[train_ind], wav_path[valid_ind]
train_pitchpath, valid_pitchpath = pitch_path[train_ind], pitch_path[valid_ind]
train_path = {'wav': train_wavpath, 'pitch': train_pitchpath}
valid_path = {'wav': valid_wavpath, 'pitch': valid_pitchpath}

mono_train_dataset = MIR1Kdataset(train_path, poly=False, transform=Compose([
    Zscore(divide_sigma=True),
    Spectrogram(n_fft=spec_winsize, hop_size=spec_hopsize, n_band=spec_nband, center=True),
    ContextWindow(hop_size=contextwin_hopsize, window_size=contextwin_winsize), # this setting is for memory save
    ToTensor()]))

mono_valid_dataset = MIR1Kdataset(valid_path, poly=False, transform=Compose([
    Zscore(divide_sigma=True),
    Spectrogram(n_fft=spec_winsize, hop_size=spec_hopsize, n_band=spec_nband, center=True),
    ContextWindow(hop_size=contextwin_hopsize, window_size=contextwin_winsize), # this setting is for memory save
    ToTensor()]))

poly_train_dataset = MIR1Kdataset(train_path, poly=True, transform=Compose([
    Zscore(divide_sigma=True),
    Spectrogram(n_fft=spec_winsize, hop_size=spec_hopsize, n_band=spec_nband, center=True),
    ContextWindow(hop_size=contextwin_hopsize, window_size=contextwin_winsize), # this setting is for memory save
    ToTensor()]))

poly_valid_dataset = MIR1Kdataset(valid_path, poly=True, transform=Compose([
    Zscore(divide_sigma=True),
    Spectrogram(n_fft=spec_winsize, hop_size=spec_hopsize, n_band=spec_nband, center=True),
    ContextWindow(hop_size=contextwin_hopsize, window_size=contextwin_winsize), # this setting is for memory save
    ToTensor()]))

class Logger(object):
    def __init__(self):
        self.list_train_log = []
        self.list_valid_log = []
        self.train_or_valid = 'train'

    def reset(self):
        assert self.train_or_valid in ['train', 'valid']
        if self.train_or_valid == 'train':
            self.list_train_log = []
        else:
            self.list_valid_log = []

    def update(self, val, n_frame=1, n_band=1, contextwin_winsize=1):
        normalized_val = val/n_frame/n_band/contextwin_winsize
        assert self.train_or_valid in ['train', 'valid']
        if self.train_or_valid == 'train':
            self.list_train_log.append(normalized_val)
        else:
            self.list_valid_log.append(normalized_val)

    def return_list(self):
        assert self.train_or_valid in ['train', 'valid']
        if self.train_or_valid == 'train':
            return self.list_train_log
        else:
            return self.list_valid_log

def best_model(path_to_models):
    
#===========================================
mono_model = best_model(path_to_VAE_models)
mono_model.encode_only = True
mono_model.eval()
#===========================================
poly_model = Cnn_VAE(encode_only=True).cuda()
poly_model.double()
#===========================================
optimizer = optim.Adam(poly_model.parameters(), lr=1e-3)
epoch_level_logger = Logger()
song_level_logger = Logger()
batch_level_logger = Logger()

for epoch in range(1, n_epochs+1):
    poly_model.train()
    epoch_level_logger.train_or_valid = 'train'
    song_level_logger.train_or_valid = 'train'
    batch_level_logger.train_or_valid = 'train'
    # input per song clip
    for mono_song, poly_song in zip(mono_train_dataset, poly_train_dataset):
        y, x = mono_song['X'], poly_song['X']
        mono_loader = DataLoader(dataset=y, batch_size=batch_size, shuffle=False,\
                                 pin_memory=True, num_workers=1)
        poly_loader = DataLoader(dataset=x, batch_size=batch_size, shuffle=False,\
                                 pin_memory=True, num_workers=1)
        
        # train per batch (of frames in each song clip)
        for Y, X in zip(mono_loader, poly_loader):
            Y, X = Variable(Y).cuda(), Variable(X).cuda()
            target_mu = mono_model(Y)
            output_mu = poly_model(X)
            loss = loss_mse(output_mu, target_mu)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_level_logger.update(loss.data.cpu().numpy()[0])

        assert len(mono_loader) == len(poly_loader)
        train_loss_per_tf = sum(batch_level_logger.return_list())/len(mono_loader)
        batch_level_logger.reset()
        song_level_logger.update(train_loss_per_tf)
    
    assert len(mono_train_dataset) == len(mono_train_dataset)
    train_loss_epoch = sum(song_level_logger.return_list())/len(mono_train_dataset)
    song_level_logger.reset()
    epoch_level_logger.update(train_loss_epoch)

    poly_model.eval()
    epoch_level_logger.train_or_valid = 'valid'
    song_level_logger.train_or_valid = 'valid'
    batch_level_logger.train_or_valid = 'valid'
    for mono_song, poly_song in zip(mono_valid_dataset, poly_valid_dataset):
        y, x = mono_song['X'], poly_song['X']
        mono_loader = DataLoader(dataset=y, batch_size=batch_size, shuffle=False,\
                                 pin_memory=True, num_workers=1)
        poly_loader = DataLoader(dataset=x, batch_size=batch_size, shuffle=False,\
                                 pin_memory=True, num_workers=1)
        
        # train per batch (of frames in each song clip)
        for Y, X in zip(mono_loader, poly_loader):
            Y, X = Variable(Y).cuda(), Variable(X).cuda()
            target_mu = mono_model(Y)
            output_mu = poly_model(X)
            loss = loss_mse(output_mu, target_mu)
            
            batch_level_logger.update(loss.data.cpu().numpy()[0])

        assert len(mono_loader) == len(poly_loader)
        valid_loss_per_tf = sum(batch_level_logger.return_list())/len(mono_loader)
        batch_level_logger.reset()
        song_level_logger.update(valid_loss_per_tf)
    
    assert len(mono_valid_dataset) == len(mono_valid_dataset)
    valid_loss_epoch = sum(song_level_logger.return_list())/len(mono_valid_dataset)
    song_level_logger.reset()
    epoch_level_logger.update(valid_loss_epoch)    
        
    print("Epoch: %s/%s | Training loss: %s | Evaluating loss: %s" % (epoch, n_epochs, train_loss_epoch, valid_loss_epoch))
    
    epoch_level_logger.train_or_valid = 'train'
    pklSave(epoch_level_logger.return_list(), result_save_path+'train_loss.pkl')
    epoch_level_logger.train_or_valid = 'valid'
    pklSave(epoch_level_logger.return_list(), result_save_path+'valid_loss.pkl')

    if epoch % log_interval == 0:
        print(epoch_level_logger.return_list()[-1] >= epoch_level_logger.return_list()[-log_interval]) 
        if epoch_level_logger.return_list()[-1] >= epoch_level_logger.return_list()[-log_interval]:
            break
        elif epoch_level_logger.return_list()[-log_interval] - epoch_level_logger.return_list()[-1] <= 1e-4:
            break
        else:
            modelname = "epoch-%s.pth.tar" % epoch
            torch.save(poly_model, result_save_path + modelname)
