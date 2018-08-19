import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
# a special module that converts [batch, channel, w, h] to [batch, units]


cuda = torch.cuda.is_available()


class CuriosityNetwork(nn.Module):
    def __init__(self, n_actions, feat_dim, in_channels, imsize):

        super(CuriosityNetwork, self).__init__()
        self.n_actions = n_actions
        self.feat_dim = feat_dim

        self.forwardDynamics = ForwardDynamics(n_actions, feat_dim)
        self.inverseDynamics = InverseDynamics(n_actions, feat_dim)
        self.perception = Encoder(encode_dim=feat_dim, in_channels=in_channels, imsize=imsize, model_idx=1)

    def get_all_params(self):
        p = list(self.forwardDynamics.parameters()) \
          + list(self.inverseDynamics.parameters()) \
          + list(self.perception.parameters())

        return p
class ForwardDynamics(nn.Module):
    # take features and actions to predict next features

    def __init__(self, n_actions, feat_dim):
        super(ForwardDynamics, self).__init__()

        self.linear1 = nn.Linear(n_actions + feat_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, feat_dim)

    def forward(self, feat, act_t):
        x = torch.cat((feat, act_t), 1)

        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = self.linear3(x)

        return x

class InverseDynamics(nn.Module):
    # take two features and predict the actions 

    def __init__(self, n_actions, feat_dim):
        super(InverseDynamics, self).__init__()

        self.linear1 = nn.Linear(2 * feat_dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, n_actions)

    def forward(self, feat1, feat2):
        x = torch.cat((feat1, feat2), 1)

        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = F.softmax(self.linear3(x))

        return x





class DecisionNetwork(nn.Module):
    def __init__(self, n_actions, memory_dim):

        super(DecisionNetwork, self).__init__()

        self.memory_dim = memory_dim
        self.n_actions = n_actions

        self.linear1 = nn.Linear(memory_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.logits = nn.Linear(128, n_actions)
        self.state_value = nn.Linear(128, 1)

        self.rnn = nn.LSTMCell(128, 128)

    def forward(self, encoded_memory):

        x = F.relu(self.linear1(encoded_memory))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))

        logits = self.logits(x)
        state_value = self.state_value(x)

        return (logits, state_value)

    # def forward(self, prev_state, encoded_memory):

    #     hx, cx = prev_state

    #     x = F.relu(self.linear1(encoded_memory))
    #     x = F.relu(self.linear2(x))
    #     # x = F.relu(self.linear3(x))

    #     hx, cx = self.rnn(x, (hx, cx))
    #     new_state = (hx, cx)

    #     x = hx


    #     logits = self.logits(x)
    #     state_value = self.state_value(x)

    #     return new_state, (logits, state_value)


class LongMemoryLSTM(nn.Module):
    def __init__(self, memory_dim=128):

        super(LongMemoryLSTM, self).__init__()

        self.memory_dim = memory_dim
        self.lstm = nn.LSTMCell(memory_dim, memory_dim)

    def forward(self, prev_state, x):

        hx, cx = prev_state
        hx, cx = self.lstm(x, (hx, cx))
        new_state = (hx, cx)

        return new_state

    def get_initial_state(self, batch_size):
        if not cuda:
            return (Variable(torch.zeros((batch_size, self.memory_dim))),
                    Variable(torch.zeros((batch_size, self.memory_dim))))
        else:
            return (Variable(torch.zeros((batch_size, self.memory_dim)).cuda()),
                    Variable(torch.zeros((batch_size, self.memory_dim)).cuda()))

class MemoryLSTM(nn.Module):
    def __init__(self, memory_dim=128):

        super(MemoryLSTM, self).__init__()

        self.memory_dim = memory_dim
        self.lstm = nn.LSTM(memory_dim, memory_dim, 1)

    def forward(self, prev_state, x):

        hx, cx = prev_state
        hx, cx = self.lstm(x, (hx, cx))
        new_state = (hx, cx)

        return new_state

    # def get_initial_state(self, batch_size):
        # if not cuda:
        #     return (Variable(torch.zeros((batch_size, self.memory_dim))),
        #             Variable(torch.zeros((batch_size, self.memory_dim))))
        # else:
        #     return (Variable(torch.zeros((batch_size, self.memory_dim)).cuda()),
        #             Variable(torch.zeros((batch_size, self.memory_dim)).cuda()))

    def get_initial_state(self, batch_size):
        if not cuda:
            return (Variable(torch.zeros((1, batch_size, self.memory_dim))),
                    Variable(torch.zeros((1, batch_size, self.memory_dim))))
        else:
            return (Variable(torch.zeros((1, batch_size, self.memory_dim)).cuda()),
                    Variable(torch.zeros((1, batch_size, self.memory_dim)).cuda()))
class Encoder(nn.Module):
    # for perception unit

    def __init__(self, encode_dim=128, in_channels=3, imsize=42, nfeat=64, extra_layer=0, batchNorm=True, activation='ReLU', model_idx=0):

        super(Encoder, self).__init__()
        self.initial = ConvSet(in_channels, nfeat, 4, 2, 1, False, 
                               batchNorm=batchNorm, activation=activation, name='initial')

        self.model_idx = model_idx
        if model_idx == 0:
            encoder_list = []
            name_list = []
            c_imsize, c_feat = imsize / 2, nfeat
            
            ind = 0       
            while c_imsize >= 4:
                in_feat = c_feat
                out_feat = c_feat * 2
                
                ind += 1
                layer_name = 'pyramid_' + str(ind)

                convnet = ConvSet(in_feat, out_feat, 4, 2, 1, bias=False, 
                                                batchNorm=batchNorm, activation=activation, name=layer_name)
                
                entries, names = convnet.getEntriesAndNames()
                encoder_list.extend(entries)
                name_list.extend(names)
            
                c_feat *= 2
                c_imsize = c_imsize / 4
            # Tensor[None, 256, 8, 8]
     
            # final convolutional layer, out_feat=20 is to be changed
            final_conv = ConvSet(c_feat, 20, 4, 2, 1, bias=False, 
                                                batchNorm=batchNorm, activation=activation, name='final_conv')
            entries, names = final_conv.getEntriesAndNames()
            encoder_list.extend(entries)
            name_list.extend(names)
            
            encoder_dict = OrderedDict(zip(name_list, encoder_list))
            self.encoder =  nn.Sequential(encoder_dict)
            # Tensor[None, 20, 4, 4] -> 320
            
            self.sigma = nn.Sequential()
            self.sigma.add_module('sigma',
                                        nn.Linear(320, encode_dim))
            self.logvar = nn.Sequential()
            self.logvar.add_module('logvar',
                                        nn.Linear(320, encode_dim))
        
    # def forward(self, x):
    #     # for Variational autoencoder
    #     x = self.initial(x)
    #     x = self.encoder(x)
    #     # print(x)
    #     sigma = self.sigma(x.view(x.shape[0], 320))
    #     logvar = self.logvar(x.view(x.shape[0], 320))
    #     return sigma, logvar
        elif model_idx == 1:
        # small model
            self.conv0 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=(2,2))
            self.conv1 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(2,2))
            self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(2,2))
            self.flatten = Flatten()
            self.linear = nn.Linear(512, encode_dim)

        else:
            NotImplementedError()


    def forward(self, x):
        # for conventional autoencoder
        if self.model_idx == 0:
            x = self.initial(x)
            x = self.encoder(x)
            sigma = self.sigma(x.view(x.shape[0], 320))
        elif self.model_idx == 1:
            x = F.elu(self.conv0(x))
            x = F.elu(self.conv1(x))
            x = F.elu(self.conv2(x))
            sigma = self.linear(self.flatten(x))
        # logvar = self.logvar(x.view(x.shape[0], 320))
        return sigma
    
    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

class Decoder(nn.Module):
    # for recoveryUnit
    def __init__(self, encode_dim=128, out_channels=3, imsize=64, nfeat=64, extra_layer=0, batchNorm=True, activation='ReLU'):

        super(Decoder, self).__init__()

        self.encode_dim = encode_dim
        
        cngf, tisize = nfeat//2, 4
        while tisize != imsize:
            cngf = cngf * 2
            tisize = tisize * 2
            
        nz = encode_dim
        
        # initial input
        self.initial = TConvSet(nz, cngf, 4, 1, 0, False, 
                               batchNorm=batchNorm, activation=activation, name='initial')
        
        # pyramid structure
        decoder_list = []
        name_list = []        
        c_imsize, c_feat = 4, cngf
        
        ind = 0
        while c_imsize < imsize // 2:
            ind += 1
            layer_name = 'pyramid_' + str(ind)
            tconvnet = TConvSet(c_feat, c_feat//2, 4, 2, 1, bias=False,
                               batchNorm=batchNorm, activation=activation, name=layer_name)
            
            entries, names = tconvnet.getEntriesAndNames()
            decoder_list.extend(entries)
            name_list.extend(names)
            
            c_feat = c_feat // 2
            c_imsize = c_imsize * 2
        
        # extra layer
        ind = 0
        for i in range(extra_layer):
            ind += 1
            layer_name = 'extra_layer_' + str(ind)
            extra_tconvnet = ConvSet(c_feat, c_feat, 3, 1, 1, bias=False, name=layer_name)
            
            entries, names = extra_tconvnet.getEntriesAndNames()
            decoder_list.extend(entries)
            name_list.extend(names)       
        
        # final output layer
        final_tconvnet = TConvSet(c_feat, out_channels, 4, 2, 1, bias=False,
                   batchNorm=batchNorm, activation=activation, name='final_layer')

        entries, names = final_tconvnet.getEntriesAndNames()
        decoder_list.extend(entries)
        name_list.extend(names)
        
        decoder_dict = OrderedDict(zip(name_list, decoder_list))
        self.decoder =  nn.Sequential(decoder_dict)   
        
        
    def forward(self, *args):

        x = args[0]
        assert x.shape[1] == self.encode_dim, "Dimension of encoded vector does not match"
        
        x = x.view((x.shape[0], x.shape[1], 1, 1))
        x = self.initial(x)        
        x = self.decoder(x)
        return x
    
    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

class ConvSet(nn.Module):
    def __init__(self, in_channels, out_channels, kernersize, stride, padding, bias=False, 
                 batchNorm=True, activation='ReLU', name='Default_net'):
        
        super(ConvSet, self).__init__()
        
        assert activation in ['ReLU', 'LeakyReLU'], "Activation methods not implemented!!!"
        self.convSet = nn.Sequential()
        self.nameList = []
        
        conv_name = name + '-{}-{}-conv'.format(in_channels, out_channels)
        activ_name = name + '-{}-' + activation
        activ_name = activ_name.format(out_channels)
        if batchNorm:
            batch_name = name + '-{}-batchnorm'.format(out_channels)

        self.convSet.add_module(conv_name,
                               nn.Conv2d(in_channels, out_channels, kernersize, stride, padding, bias=bias))
        self.nameList.append(conv_name)
        
        if batchNorm:
            self.convSet.add_module(batch_name,
                                   nn.BatchNorm2d(out_channels))
            self.nameList.append(batch_name)
        
        if activation == 'ReLU':
            self.convSet.add_module(activ_name,
                                   nn.ReLU(inplace=True))
        elif activation == 'LeakyReLU':
            self.convSet.add_module(activ_name,
                                   nn.LeakyReLU(0.2, inplace=True))
        self.nameList.append(activ_name)

    def forward(self, x):
        x = self.convSet(x)
        return x
    
    def getEntriesAndNames(self):
        moduleList = list(self.convSet)
        return moduleList, self.nameList

class TConvSet(nn.Module):
    def __init__(self, in_channels, out_channels, kernersize, stride, padding, bias=False, 
                 batchNorm=True, activation='ReLU', name='Default_net'):
        
        super(TConvSet, self).__init__()
        
        assert activation in ['ReLU', 'LeakyReLU'], "Activation methods not implemented!!!"
        self.convSet = nn.Sequential()
        self.nameList = []
        
        conv_name = name + '-{}-{}-transconv'.format(in_channels, out_channels)
        activ_name = name + '-{}-' + activation
        activ_name = activ_name.format(out_channels)
        if batchNorm:
            batch_name = name + '-{}-batchnorm'.format(out_channels)
            
        self.convSet.add_module(conv_name,
                               nn.ConvTranspose2d(in_channels, out_channels, kernersize, stride, padding, bias=bias))
        self.nameList.append(conv_name)
        
        if batchNorm:
            self.convSet.add_module(batch_name,
                                   nn.BatchNorm2d(out_channels))
            self.nameList.append(batch_name)
        
        if activation == 'ReLU':
            self.convSet.add_module(activ_name,
                                   nn.ReLU(inplace=True))
        elif activation == 'LeakyReLU':
            self.convSet.add_module(activ_name,
                                   nn.LeakyReLU(0.2, inplace=True))
        self.nameList.append(activ_name)

    def forward(self, x):
        x = self.convSet(x)
        return x
    
    def getEntriesAndNames(self):
        moduleList = list(self.convSet)
        return moduleList, self.nameList

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)