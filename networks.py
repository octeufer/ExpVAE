import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict
from torch.autograd import Variable
from torch.distributions import bernoulli, normal, categorical
import sys

# class naming convention: p_A_BC -> p(A|B,C)

####### Generative model / Decoder / Model network #######

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, output_dim):
        super(ConvEncoder, self).__init__()
        self.in_channels = in_channels
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(self.in_channels, 32, 1, 1, 0)  # 32 x 32  # 112 x 112
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)  # 16 x 16 # 56 x 56
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)  # 8 x 8 # 28 x 28
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)  # 4 x 4 # 14 x 14
        self.bn4 = nn.BatchNorm2d(256)
        # self.conv5 = nn.Conv2d(256, 512, 3, 2, 1) # 7 x 7
        # self.bn5 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(256, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, self.output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # h = x.view(-1, 1, 64, 64)
        h = self.leakyrelu(self.bn1(self.conv1(x)))
        h = self.leakyrelu(self.bn2(self.conv2(h)))
        h = self.leakyrelu(self.bn3(self.conv3(h)))
        h = self.leakyrelu(self.bn4(self.conv4(h)))
        h = self.leakyrelu(self.bn5(self.conv5(h)))
        # h = self.leakyrelu(self.bn6(self.conv6(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z

class ConvDecoder_y0(nn.Module):
    def __init__(self, input_dim, out_channels):
        super(ConvDecoder_y0, self).__init__()
        self.input_dim = input_dim
        self.out_channels = out_channels

        self.conv1 = nn.ConvTranspose2d(self.input_dim, 512, 4, 1, 0)  # 4 x 4
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)  # 8 x 8
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 16 x 16
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, self.out_channels, 4, 2, 1)  # 32 x 32
        # self.bn4 = nn.BatchNorm2d(64)
        # self.conv5 = nn.ConvTranspose2d(64, 32, 4, 2, 1) # 112 x 112
        # self.bn5 = nn.BatchNorm2d(32)
        # self.conv6 = nn.ConvTranspose2d(32, self.out_channels, 3, 2, 1, 1) # 224 x 224
        # self.bn6 = nn.BatchNorm2d(16)
        # self.conv5 = nn.ConvTranspose2d(32, self.out_channels, 4, 2, 1)  # 32 x 32
        # self.bn5 = nn.BatchNorm2d(32)
        # self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.leakyrelu(self.bn1(self.conv1(h)))
        h = self.leakyrelu(self.bn2(self.conv2(h)))
        h = self.leakyrelu(self.bn3(self.conv3(h)))
        # h = self.leakyrelu(self.bn4(self.conv4(h)))
        # h = self.leakyrelu(self.bn5(self.conv5(h)))
        # h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv4(h)
        # mu_img = self.conv_final(h)
        return self.sigmoid(mu_img)

class ConvDecoder_y1(nn.Module):
    def __init__(self, input_dim, out_channels):
        super(ConvDecoder_y1, self).__init__()
        self.input_dim = input_dim
        self.out_channels = out_channels

        self.conv1 = nn.ConvTranspose2d(self.input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, self.out_channels, 4, 2, 1)  # 32 x 32
        # self.bn5 = nn.BatchNorm2d(32)
        # self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.leakyrelu(self.act(self.bn1(self.conv1(h))))
        h = self.leakyrelu(self.act(self.bn2(self.conv2(h))))
        h = self.leakyrelu(self.act(self.bn3(self.conv3(h))))
        h = self.leakyrelu(self.act(self.bn4(self.conv4(h))))
        # h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv5(h)
        # mu_img = self.conv_final(h)
        return mu_img

class p_x_zy(nn.Module):

    def __init__(self, input_dim, out_channels):
        super().__init__()

        self.input_dim = input_dim
        self.out_channels = out_channels

        self.conv1 = nn.ConvTranspose2d(self.input_dim, 256, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(256, 128, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 16, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.ConvTranspose2d(16, self.out_channels, 4, 2, 1)  # 32 x 32
        # self.bn4 = nn.BatchNorm2d(64)
        # self.conv5 = nn.ConvTranspose2d(64, 32, 4, 2, 1) # 112 x 112
        # self.bn5 = nn.BatchNorm2d(32)
        # self.conv6 = nn.ConvTranspose2d(32, self.out_channels, 3, 2, 1, 1) # 224 x 224
        # self.bn6 = nn.BatchNorm2d(16)
        # self.conv5 = nn.ConvTranspose2d(32, self.out_channels, 4, 2, 1)  # 32 x 32
        # self.bn5 = nn.BatchNorm2d(32)
        # self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # self.decoder_y0 = ConvDecoder_y0(input_dim, out_channels)
        # self.decoder_y1 = ConvDecoder_y1(input_dim, out_channels)

    def forward(self, z, y):
        # print(z)
        # mu_y1 = self.decoder_y1.forward(z)
        # mu_y0 = self.decoder_y0.forward(z)
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.leakyrelu(self.bn1(self.conv1(h)))
        h = self.leakyrelu(self.bn2(self.conv2(h)))
        h = self.leakyrelu(self.bn3(self.conv3(h)))
        h = self.leakyrelu(self.bn4(self.conv4(h)))
        # h = self.leakyrelu(self.bn4(self.conv4(h)))
        # h = self.leakyrelu(self.bn5(self.conv5(h)))
        # h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv5(h)
        # mu_img = self.conv_final(h)
        return self.sigmoid(mu_img)

class p_y_z(nn.Module):

    def __init__(self, dim_in=512, dim_h=256, dim_out=1):
        super().__init__()
        # save required vars
        # self.nh = nh
        # self.dim_out = dim_out
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out

        # dim_in is dim of latent space z
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        # self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)
        self.act = nn.ReLU(inplace=True)
        # self.final = torch.nn.Softmax(dim=1)
        self.final = torch.nn.Sigmoid()

    def forward(self, z):
        h = self.act(self.input(z))
        # for i in range(self.nh):
        #     x = nn.ReLU(self.hidden[i](x))
        # for binary outputs:
        out_p = self.final(self.output(h))
        y = bernoulli.Bernoulli(out_p)
        # print(self.output(h).size())
        # out_p = self.final(self.output(h))
        # y = categorical.Categorical(out_p)
        return y


class p_c_z(nn.Module):

    def __init__(self, dim_in=512, dim_h=256, dim_out=1):
        super().__init__()
        # save required vars
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out

        # Separated forwards for different t values, TAR

        # self.input_t0 = nn.Linear(dim_in, dim_h)
        # # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        # self.hidden_t0 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        # self.mu_t0 = nn.Linear(dim_h, dim_out)

        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        # self.hidden_t1 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)

        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = self.act(self.input(z))
        # for i in range(self.nh):
        #     x = nn.ReLU(self.hidden[i](x))
        # for binary outputs:
        out_p = torch.sigmoid(self.output(h))

        c = bernoulli.Bernoulli(out_p)
        return c


####### Inference model / Encoder #######

class q_y_x(nn.Module):

    def __init__(self, num_class=2):
        super().__init__()
        # save required vars
        # self.nh = nh
        # self.dim_out = dim_out
        self.model = timm.create_model('resnet50', pretrained=True, num_classes=num_class)
        # dim_in is dim of data x
        # self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        # self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        # self.output = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        # x = F.elu(self.input(x))
        # for i in range(self.nh):
        #     x = F.elu(self.hidden[i](x))
        # for binary outputs:
        # out_p = torch.sigmoid(self.output(x))
        # out = bernoulli.Bernoulli(out_p)
        y = self.model(x)
        return y


class q_y_xt(nn.Module):

    def __init__(self, dim_in=2048, nh=3, dim_h=2048, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of data x
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        # separate outputs for different values of t
        # self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x, t):
        # Unlike model network, shared parameters with separated heads
        t = self.act(self.input(t))
        # for i in range(self.nh):
        #     t = F.elu(self.hidden[i](t))
        # only output weights separated
        # mu_t0 = self.mu_t0(x)
        # mu_t1 = self.mu_t1(x)
        # set mu according to t, sigma set to 1
        # y = normal.Normal((1-t)*mu_t0 + t * mu_t1, 1)

        # x_t1 = nn.ReLU(self.input_t1(z))
        # for i in range(self.nh):
        #     x_t1 = nn.ReLU(self.hidden_t1[i](x_t1))
        mu_t1 = self.mu_t1(t).view(t.size(0))
        # set mu according to t value
        # y = normal.Normal((1-t)*mu_t0 + t * mu_t1, 1)
        y = bernoulli.Bernoulli(torch.sigmoid(mu_t1))
        return y


class q_z_cyx(nn.Module):

    def __init__(self, in_channels=3, nh=3, output_dim=512):
        super().__init__()
        self.in_channels = in_channels
        self.output_dim = output_dim

        # self.conv1 = nn.Conv2d(self.in_channels, 32, 1, 1, 0)  # 32 x 32
        # self.bn1 = nn.BatchNorm2d(32)
        # self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        # self.bn3 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        # self.bn4 = nn.BatchNorm2d(64)
        # self.conv5 = nn.Conv2d(64, 512, 4)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.conv_z = nn.Conv2d(512, self.output_dim, 1)
        # # setup the non-linearity
        # self.act = nn.ReLU(inplace=True)
        self.encoder = ConvEncoder(self.in_channels, self.output_dim)

        self.fc_mu = nn.Linear(512+4, output_dim)
        self.fc_var = nn.Linear(512+4, output_dim)

    def forward(self, x, y, c):
        z = self.encoder(x)

        y = F.one_hot(y)
        c = F.one_hot(c)
        z = torch.cat((z, y, c), 1)
        # print(z.size())
        # print(t.size())
        # print(y.size())
        # mu_t0 = self.mu_t0(x)
        mu = self.fc_mu(z)
        # sigma_t0 = self.softplus(self.sigma_t0(x))
        log_var = self.fc_var(z)

        # Set mu and sigma according to t
        # z = normal.Normal(mu_t1, sigma_t1)
        return mu, log_var

class VAE(nn.Module):
    num_iter = 0 #Global step counter
    def __init__(self, cfg, use_cuda=True, z_dim=512, t_dim=2048, gamma=1000):
        super().__init__()

        self.gamma = gamma
        self.z_dim = cfg['z_dim']
        self.x_zy = p_x_zy(cfg['z_dim'], cfg['x_out_channel'])
        self.y_z = p_y_z(dim_in=cfg['z_dim'], dim_h=cfg['h_dim'], dim_out=1)
        self.c_z = p_c_z(dim_in=cfg['z_dim'], dim_h=cfg['h_dim'], dim_out=1)
        self.y_x = q_y_x(num_class=2)
        # self.y_xt = q_y_xt(dim_in=cfg['t_out_dim'], nh=3, dim_h=cfg['h_dim'], dim_out=1)
        self.q_xyc = q_z_cyx(in_channels=3, nh=3, output_dim=512)

        self.params = list(self.x_zy.parameters()) + \
                      list(self.q_xyc.parameters()) + \
                      list(self.y_z.parameters()) + \
                      list(self.c_z.parameters())
                      
                    #   list(self.y_x.parameters()) + \
                      
                      #  list(q_t_x_dist.parameters()) + \
        
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
    
    def encode(self, x, y, c):
        mu, logvar = self.q_xyc.forward(x, y, c)
        return mu, logvar
    
    def decode_x(self, z, y):
        x_recon = self.x_zy.forward(z, y)
        return x_recon
    
    def decode_y(self, z):
        y_prob = self.y_z.forward(z)
        return y_prob

    def decode_c(self, z):
        c_prob = self.c_z.forward(z)
        return c_prob
    
    def predict(self, x):
        y_prob = self.y_x.forward(x)
        return y_prob
    
    # def approximate_y(self, x, t):
    #     y = self.y_xt.forward(x, t)
    #     return y
    
    # def approximate_t(self, x):
    #     t = self.t_x.forward(x)
    #     return t
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def sample_prior(self, num_samples, device):
        z = torch.randn(num_samples, self.z_dim)
        z = z.to(device)
        samples = self.decode_x(z)
        return samples
        
    def loss_function(self, x_recon, c_recon, y_recon, x_input, y_input, c_input, y_pred, mu, logvar, cfg):
        self.num_iter += 1
        kld_weight = cfg['M_N']

        loss = defaultdict(list)
        
        pred_loss = nn.CrossEntropyLoss()
        pred_loss_y = pred_loss(y_pred, y_input)
        loss['pred_y'] = pred_loss_y
        
        log_sigma = ((x_input - x_recon) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()
        log_sigma = softclip(log_sigma, -6)
        recon_loss_x = gaussian_nll(x_recon, log_sigma, x_input).sum()
        # recon_loss = nn.MSELoss()
        # # recon_loss = nn.BCELoss()
        # recon_loss_x = recon_loss(x_recon, x_input)
        loss['recon_x'] = recon_loss_x
        
        c_input = c_input.float()
        # print(c_input)
        recon_loss_c = -torch.mean(c_recon.log_prob(c_input).squeeze())
        loss['recon_c'] = recon_loss_c
        
        y_input = y_input.float()
        # print(y_input.size())
        # print(y_recon)
        recon_loss_y = -torch.mean(y_recon.log_prob(y_input).squeeze())
        # print(recon_loss_y.size())
        loss['recon_y'] = recon_loss_y
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1),dim=0)
        loss['kld'] = kld_loss
        
        loss_total = recon_loss_x + recon_loss_y + recon_loss_c +\
            kld_weight * kld_loss + pred_loss_y
        
        # loss_total = recon_loss_x + kld_weight * kld_loss
        loss['total'] = loss_total
        # print(kld_loss)
        # print(loss_total)
        return loss
