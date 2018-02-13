import os
import argparse
from data_prepare import *
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.optim as optim
from model import *
from util import *


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, metavar='N')
parser.add_argument('--batch-size', type=int, default=512,metavar='N')
parser.add_argument('--log-interval', type=int, default=10, metavar='N')
parser.add_argument('--spec-hopsize', type=int, default=320, metavar='N')
parser.add_argument('--spec-winsize', type=int, default=1024, metavar='N')
parser.add_argument('--spec-nband', type=int, default=256, metavar='N')
parser.add_argument('--contextwin-hopsize', type=int, default=10, metavar='N')
parser.add_argument('--contextwin-winsize', type=int, default=21, metavar='N')
parser.add_argument('--code-size', type=int, default=128, metavar='N')
args = parser.parse_args()

result_save_path = '/home/yjluo/projects/c-VAE/result_2/'

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
#data_path = {'wav': wav_path, 'pitch': pitch_path}
train_dataset = MIR1Kdataset(train_path, transform=Compose([
    Zscore(divide_sigma=True),
    Spectrogram(n_fft=spec_winsize, hop_size=spec_hopsize, n_band=spec_nband, center=True),
    ContextWindow(hop_size=contextwin_hopsize, window_size=contextwin_winsize), # this setting is for memory save
    ToTensor()]))

valid_dataset = MIR1Kdataset(valid_path, transform=Compose([
    Zscore(divide_sigma=True),
    Spectrogram(n_fft=spec_winsize, hop_size=spec_hopsize, n_band=spec_nband, center=True),
    ContextWindow(hop_size=contextwin_hopsize, window_size=contextwin_winsize), # this setting is for memory save
    ToTensor()]))

class Logger(object):
    def __init__(self):
        self.list_log = []

    def reset(self):
        self.list_log = []

    def update(self, val, n_frame=1, n_band=1, contextwin_winsize=1):
        normalized_val = val/n_frame/n_band/contextwin_winsize
        self.list_log.append(normalized_val)

model = Cnn_VAE().cuda()
model.double()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epoch_level_logger = Logger()
song_level_logger = Logger()
batch_level_logger = Logger()
for epoch in range(1, n_epochs+1):
    model.train()

    for i in range(len(train_dataset)):
        data = train_dataset[i]
        x, y_singer = data['X'], data['y_singer']
        loader = DataLoader(dataset=x, batch_size=batch_size, shuffle=False,\
                            pin_memory=True, num_workers=1)

        for X in loader:
            X = Variable(X).cuda()
            X_recon, mu, var = model(X)
            loss_recon = F.mse_loss(X_recon, X)
            optimizer.zero_grad()
            loss_recon.backward()
            optimizer.step()


    model.eval()
    for j in range(len(valid_dataset)):
        data = valid_dataset[j]
        x, y_singer = data['X'], data['y_singer']
        loader = DataLoader(dataset=x, batch_size=batch_size, shuffle=False,\
                            pin_memory=True, num_workers=1)
        for X in loader:
            X = Variable(X).cuda()
            X_recon, mu, var = model(X)
            loss_recon = F.mse_loss(X_recon, X)
            optimizer.zero_grad()
            loss_recon.backward()
            optimizer.step()

            batch_level_logger.update(loss_recon.data.cpu().numpy()[0],\
                          batch_size, spec_nband, contextwin_winsize)

        loss_recon_per_tf = sum(batch_level_logger.list_log)/len(loader)
        batch_level_logger.reset()
        song_level_logger.update(loss_recon_per_tf)
    
    loss_recon_epoch = sum(song_level_logger.list_log)/len(valid_dataset)
    song_level_logger.reset()
    epoch_level_logger.update(loss_recon_epoch)

    print("Epoch: %s/%s | Evaluating loss: %s" % (epoch, n_epochs, loss_recon_epoch))
    
    if epoch % log_interval == 0 or epoch == 1:
        pklSave(epoch_level_logger.list_log, result_save_path+'evaluating_loss.pkl')
        modelname = "epoch-%s.pth.tar" % epoch
        torch.save(model, result_save_path + modelname)

