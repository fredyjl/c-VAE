import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

class Cnn_VAE(nn.Module):
    def __init__(self, input_size=(1, 256, 21), code_size=128):
        super(Cnn_VAE, self).__init__()
        self.input_size = input_size
        self.code_size = code_size
        self.n_band = self.input_size[1]
        self.n_contextwin = self.input_size[2]
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, (self.n_band, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 128, (1, 3), (1, 2)),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 256, (1, 2), (1, 2)),
            nn.BatchNorm2d(256),
            nn.Tanh()
        )
        # infer flatten size
        self.flat_size = self.infer_flat_size()
        # a fc layer prior to code layer
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.BatchNorm1d(512),
            nn.Tanh()
        )
        # code layer, use linear output without activation
        self.mu_fc = nn.Linear(512, self.code_size)
        self.var_fc = nn.Linear(512, self.code_size)
        # a fc layer prior to decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.code_size, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, self.flat_size),
            nn.BatchNorm1d(self.flat_size),
            nn.Tanh()
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (1, 2), (1, 2)),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 64, (1, 3), (1, 2)),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 1, (self.n_band, 1), (1, 1))
        )

    def infer_flat_size(self):
        encoder_output = self.encoder(Variable(torch.ones(1, *self.input_size)))

        return int(np.prod(encoder_output.size()[1:]))

    def encode(self, x):
        encode_output = self.encoder(x)
        self.encode_output_size = encode_output.size()
        fc_output = self.encoder_fc(encode_output.view(-1, self.flat_size))

        return self.mu_fc(fc_output), self.var_fc(fc_output)
    
    def decode(self, x):
        fc_output = self.decoder_fc(x)
        y = self.decoder(fc_output.view(self.encode_output_size))

        return y
    
    def reparam_trick(self, mu, var):
        # training mode
        if self.training:
            std = var.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, x):
        mu, var = self.encode(x)
        z = self.reparam_trick(mu, var)
        x_recon = self.decode(z)
        
        return x_recon, mu, var

class SelectiveSequential(nn.Module):
    def __init__(self, to_select, modules_dict):
        super(SelectiveSequential, self).__init__()
        for key, module in modules_dict.items():
            self.add_module(key, module)
        self._to_select = to_select
    
    def forward(self, x):
        list = []
        for name, module in self._modules.items():
            x = module(x)
            if name in self._to_select:
                list.append(x)
        return x, list

class Cnn_clfr(nn.Module):
    def __init__(self, n_class, input_size=(1, 256, 21)):
        super(Cnn_clfr, self).__init__()
        self.input_size = input_size
        self.n_class = n_class
        self.n_band = self.input_size[1]
        self.features = SelectiveSequential(
            ['conv1', 'conv2', 'conv3'],
            {'conv1': nn.Conv2d(1, 64, (self.n_band, 1), (1, 1)),
             'batchnorm1': nn.BatchNorm2d(64),
             'tanh1': nn.Tanh(),
             'conv2': nn.Conv2d(64, 128, (1, 3), (1, 2)),
             'batchnorm2': nn.BatchNorm2d(128),
             'tanh2': nn.Tanh(),
             'conv3': nn.Conv2d(128, 256, (1, 2), (1, 2)),
             'batchnorm3': nn.BatchNorm2d(256),
             'tanh3': nn.Tanh()
            }
        )
        # infer flatten size
        self.flat_size = self.infer_flat_size()
        # a fc layer prior to
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.BatchNorm1d(512),
            nn.Tanh()
        )
        # code layer, use linear output without activation
        self.mu_fc = nn.Linear(512, self.n_class)
        
    def infer_flat_size(self):
        x, _ = self.features(Variable(torch.ones(1, *self.input_size)))

        return int(np.prod(x.size()[1:]))

    def forward(self, x):
        x, list_feature = self.features(x)
        x = self.encoder_fc(x.view(-1, self.flat_size))
        x = self.mu_fc(x)
        return x, list_feature

class Conditional_Cnn_VAE(nn.Module):
    def __init__(self, list_condition, z_condition, input_size=(1, 256, 21), code_size=128):
        super(Conditional_Cnn_VAE, self).__init__()
        self.list_condition = list_condition
        self.z_condition = z_condition
        self.input_size = input_size
        self.code_size = code_size
        self.n_band = self.input_size[1]
        self.n_contextwin = self.input_size[2]
        # encoder
        self.conv1 = nn.Conv2d(1, 64, (self.n_band, 1), (1, 1))
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, (1, 3), (1, 2))
        self.batchnorm2 = nn.BatchNorm2d(256+128)
        self.conv3 = nn.Conv2d(256+128, (256+128)*2, (1, 2), (1, 2))
        self.batchnorm3 = nn.BatchNorm2d((256+128)*2 + 256)
        self.tanh = nn.Tanh()
        # infer flatten size
        self.flat_size = self.infer_flat_size()
        # a fc layer prior to code layer
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.BatchNorm1d(512),
            nn.Tanh()
        )
        # code layer, use linear output without activation
        self.mu_fc = nn.Linear(512, self.code_size)
        self.var_fc = nn.Linear(512, self.code_size)
        # a fc layer prior to decoder
        """
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.code_size, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, self.flat_size),
            nn.BatchNorm1d(self.flat_size),
            nn.Tanh()
        )
        """
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.code_size + self.z_condition.size()[1], 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, 256 * 5),
            nn.BatchNorm1d(256 * 5),
            nn.Tanh()
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (1, 2), (1, 2)),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 64, (1, 3), (1, 2)),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 1, (self.n_band, 1), (1, 1))
        )

    def infer_flat_size(self):
        x_cond1, x_cond2, x_cond3 = self.list_condition[0], self.list_condition[1], self.list_condition[2]
        x = self.conv1(Variable(torch.ones(1, *self.input_size)))
        x = torch.cat((x, x_cond1), 1)
        x = self.conv2(x)
        x = torch.cat((x, x_cond2), 1)    
        x = self.conv3(x)
        encoder_output = torch.cat((x, x_cond3), 1)
        
        return int(np.prod(encoder_output.size()[1:]))

    def encode(self, x, list_condition):
        x_cond1, x_cond2, x_cond3 = list_condition[0], list_condition[1], list_condition[2]
        #print(x_cond1.size(), x_cond2.size(), x_cond3.size())
        x = self.conv1(x)
        x = torch.cat((x, x_cond1), 1)
        #print(x.size())
        x = self.batchnorm1(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = torch.cat((x, x_cond2), 1)
        #print(x.size())
        x = self.batchnorm2(x)
        x = self.tanh(x)
        x = self.conv3(x)
        x = torch.cat((x, x_cond3), 1)
        #print(x.size())
        x = self.batchnorm3(x)
        x = self.tanh(x)
        self.encode_output_size = list_condition[2].size()
        fc_output = self.encoder_fc(x.view(-1, self.flat_size))

        return self.mu_fc(fc_output), self.var_fc(fc_output)
    
    def decode(self, x):
        fc_output = self.decoder_fc(x)
        y = self.decoder(fc_output.view(self.encode_output_size))

        return y
    
    def reparam_trick(self, mu, var):
        # training mode
        if self.training:
            std = var.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, x, list_condition, z_condition):
        mu, var = self.encode(x, list_condition)
        z = self.reparam_trick(mu, var)
        z = torch.cat((z, z_condition), 1)
        x_recon = self.decode(z)
        
        return x_recon, mu, var