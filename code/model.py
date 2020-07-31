# -*- coding: utf-8 -*-

"""
####################

# Models definition

# Defines basic models and how wrappers should behave

# author    : Philippe Esling
             <esling@ircam.fr>

####################
"""

import torch
import torch.nn as nn
import numpy as np
import mir_eval

from models.vae.ae import AE
from models.vae.vae import VAE
from models.vae.vae_flow import VAEFlow
from models.vae.wae import WAE
from utils import plot_batch_images, plot_batch_compare

"""
###################

Abstract lottery model operations:
    - Test / train functions
    
###################
"""
class LotteryModel(nn.Module):
    
    def __init__(self, args):
        super(LotteryModel, self).__init__()
        self.pruning = args.pruning
    
    # Function for one full epoch of training
    def train_epoch(model, train_loader, optimizer, criterion, args):
        train_loss = 0
        """ You will need to explicitly call self.pruning.train_callback() """
        return train_loss;
    
    # Function for one full epoch of testing
    def test_epoch(model, test_loader, criterion, args):
        test_loss = 0
        return test_loss;
    
    # Function for regular evaluation
    def eval_epoch(model, test_loaader, criterion, args):
        pass
    
"""
###################

Trainer for classification models

###################
"""
class LotteryClassification(nn.Module):
    
    def __init__(self, args):
        super(LotteryClassification, self).__init__(args)
        self.pruning = args.pruning
    
    def train_epoch(self, train_loader, optimizer, criterion, iteration, args):
        self.train()
        full_loss = torch.zeros(1).to(args.device, non_blocking=True)
        n_ex = torch.zeros(1).to(args.device, non_blocking=True)
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Send the data to device
            data, targets = data.to(args.device, non_blocking=True), targets.to(args.device, non_blocking=True)
            # Set gradients to zero
            optimizer.zero_grad()
            # Compute output of the model
            output = self(data)
            # Compute criterion
            train_loss = criterion(output, targets)
            # Run backward
            train_loss.backward()
            """
            # Call the pruning strategy prior to optimization
            """
            self.pruning.train_callback(self, iteration)
            # Take optimizer step
            optimizer.step()
            # Compute max probability index
            pred = output.data.max(1, keepdim=True)[1]
            n_ex += pred.shape[0]
            # Compute max accuracy
            full_loss += pred.eq(targets.data.view_as(pred)).sum().item()
        full_loss = 1. - (full_loss / n_ex)
        return full_loss.item()

    # Function for Testing
    def test_epoch(self, test_loader, criterion, iteration, args):
        self.eval()
        accuracy = torch.zeros(1).to(args.device, non_blocking=True)
        n_ex = torch.zeros(1).to(args.device, non_blocking=True)
        with torch.no_grad():
            for data, target in test_loader:
                # Send data to device
                data, target = data.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)
                # Forward pass the model
                output = self(data)
                # Compute max probability index
                pred = output.data.max(1, keepdim=True)[1]
                n_ex += pred.shape[0]
                # Compute max accuracy
                accuracy += pred.eq(target.data.view_as(pred)).sum().item()
            loss = 1. - (accuracy / n_ex)
        return loss.item()

    # Function for cumulative gradients
    def cumulative_epoch(self, test_loader, optimizer, criterion, limit, args):
        self.train()
        # Set gradients to zero
        optimizer.zero_grad()
        nb_examples = 0
        full_targets = []
        # Put the whole dataset through
        for data, target in test_loader:
            # Send data to device
            data, target = data.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)
            # Forward pass the model
            output = self(data)
            # Compute criterion
            train_loss = criterion(output, target)
            # Run backward
            train_loss.backward()
            # Add targets
            full_targets.append(target.unsqueeze(1))
            nb_examples += len(target)
            if (nb_examples > limit):
                break
        self.outputs = full_targets
                
    def evaluate_epoch(self, test_loader, criterion, iteration, args):
        self.eval()
        with torch.no_grad():
            f = open(args.model_save + '/it_' + str(args.prune_it) + '_' + str(iteration) + '.txt', 'w')
            f.write('Dummy')
            #img_name = args.model_save + '/it_' + str(args.prune_it) + '_' + str(iteration)
            # Fixed batch to work with
            #fixed_data  = next(iter(test_loader))
            #fixed_data = fixed_data.to(args.device, non_blocking=True)            

    
"""
###################

Trainer for transcription models

###################
"""
class LotteryTranscription(nn.Module):
    
    def __init__(self, args):
        super(LotteryTranscription, self).__init__(args)
        self.pruning = args.pruning
    
    def train_epoch(self, train_loader, optimizer, criterion, iteration, args):
        self.train()
        full_loss = torch.zeros(1).to(args.device, non_blocking=True)
        n_ex = torch.zeros(1).to(args.device, non_blocking=True)
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Send the data to device
            data, targets = data.to(args.device, non_blocking=True), targets.to(args.device, non_blocking=True)
            # Set gradients to zero
            optimizer.zero_grad()
            # Compute output of the model
            output = self(data)
            # Compute criterion
            train_loss = criterion(output, targets)
            # Run backward
            train_loss.backward()
            """
            # Call the pruning strategy prior to optimization
            """
            self.pruning.train_callback(self, iteration)
            # Take optimizer step
            optimizer.step()
            # Compute max accuracy
            full_loss += train_loss
            n_ex += output.shape[0]
        full_loss /= n_ex
        return full_loss.item()

    # Function for Testing
    def test_epoch(self, test_loader, criterion, iteration, args):
        self.eval()
        test_loss = torch.zeros(1).to(args.device, non_blocking=True)
        n_ex = torch.zeros(1).to(args.device, non_blocking=True)
        f_score = torch.zeros(1).to(args.device, non_blocking=True)
        with torch.no_grad():
            for data, target in test_loader:
                # Send data to device
                data, target = data.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)
                # Forward pass the model
                output = self(data)
                # Compute test loss
                test_loss += criterion(output, target)
                target = target.cpu()
                output_vals = torch.max(output, dim=1)[1].cpu()
                x_vals = np.linspace(0, output.shape[-1] - 1, output.shape[-1])
                for b in range(output.shape[0]):
                    for i in range(output.shape[2]):
                        cur_score = mir_eval.onset.f_measure(x_vals[output_vals[b, i, :] > 0], x_vals[target[b, i, :] > 0])[0]
                        f_score += cur_score
                n_ex += output.shape[0]
            test_loss = (test_loss / n_ex)
            f_score /= (n_ex * output.shape[2])
        return 1. - f_score.item()

    # Function for cumulative gradients
    def cumulative_epoch(self, test_loader, optimizer, criterion, limit, args):
        self.train()
        # Set gradients to zero
        optimizer.zero_grad()
        nb_examples = 0
        full_targets = []
        # Put the whole dataset through
        for data, target in test_loader:
            # Send data to device
            data, target = data.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)
            # Forward pass the model
            output = self(data)
            # Compute criterion
            train_loss = criterion(output, target)
            # Run backward
            train_loss.backward()
            # Add targets
            full_targets.append(target.unsqueeze(1))
            nb_examples += len(target)
            if (nb_examples > limit):
                break
        self.outputs = full_targets
                
    def evaluate_epoch(self, test_loader, criterion, iteration, args):
        self.eval()
        with torch.no_grad():
            f = open(args.model_save + '/it_' + str(args.prune_it) + '_' + str(iteration) + '.txt', 'w')
            f.write('Dummy')
            #img_name = args.model_save + '/it_' + str(args.prune_it) + '_' + str(iteration)
            # Fixed batch to work with
            #fixed_data  = next(iter(test_loader))
            #fixed_data = fixed_data.to(args.device, non_blocking=True)   
    
"""
###################

Trainer for autoencoder models

###################
"""
class LotteryAE(LotteryModel, AE):
    
    def __init__(self, args):
        AE.__init__(self, args)
        self.pruning = args.pruning
        self.encoder_dims = args.encoder_dims
        self.latent_dims = args.latent_dims
    
    def train_epoch(self, train_loader, optimizer, criterion, iteration, args):
        self.train()
        full_loss = 0
        for x, y in train_loader:
            # Send to device
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            # Perform backward
            optimizer.zero_grad()
            # Auto-encode
            x_tilde, z_tilde, z_loss = self(x)
            # Reconstruction loss
            rec_loss = criterion(x_tilde, x)
            # Final loss
            b_loss = (rec_loss + (args.beta * z_loss)).mean(dim=0)
            # Run backward
            b_loss.backward()
            """
            # Call the pruning strategy prior to optimization
            """
            self.pruning.train_callback(self, iteration)
            # Take optimizer step
            optimizer.step()
            full_loss += b_loss
        full_loss /= len(train_loader)
        return full_loss
    
    def test_epoch(self, test_loader, criterion, iteration, args):
        self.eval()
        full_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
                # Auto-encode
                x_tilde, z_tilde, z_loss = self(x)
                # Regression loss
                rec_loss = criterion(x_tilde, x)
                full_loss += rec_loss
            full_loss /= len(test_loader)
        return full_loss
    
    def evaluate_epoch(self, test_loader, criterion, iteration, args):
        self.eval()
        with torch.no_grad():
            img_name = args.model_save + '/it_' + str(args.prune_it) + '_' + str(iteration)
            # Fixed batch to work with
            fixed_data, fixed_target = next(iter(test_loader))
            fixed_data, fixed_target = fixed_data.to(args.device, non_blocking=True), fixed_target.to(args.device, non_blocking=True)
            # Reconstruct the batch
            x_tilde, z_tilde, z_loss = self(fixed_data)
            # Plot the comparison
            plot_batch_compare(fixed_data.cpu(), x_tilde.cpu(), name = img_name + '_reconstruction')
            # Perform random sampling
            z_rand = torch.randn_like(z_tilde)
            x_rand = self.decode(z_rand)
            plot_batch_images(x_rand.cpu(), name = img_name + '_sampling')
            # Perform a linear interpolation between two random z
            idx = torch.randint(0, 15, [2])
            z_interp = torch.randn_like(z_tilde)
            for i in range(16):
                cur_r = float(i / 15)
                z_interp[i] = cur_r * z_tilde[idx[0]] + ((1 - cur_r) * z_tilde[idx[1]])
            interp_data = self.decode(z_interp)
            plot_batch_images(interp_data.cpu(), name = img_name + '_interpolation')

"""
###################
VAE - Duck typing to lottery model
###################
"""
class LotteryVAE(LotteryAE, VAE):
    """
    Definition of the Variational version of the Lottery AE
    seen as a variational auto-encoding architecture. 
    """

    def __init__(self, args):
        super(LotteryVAE, self).__init__(args)
        LotteryAE.__init__(self, args)
        VAE.__init__(self, args)
        self.pruning = args.pruning
        self.mu.unprunable = True
        self.log_var.unprunable = True

"""
###################
VAE - Duck typing to lottery model
###################
"""
class LotteryVAEFlow(LotteryAE, VAEFlow):
    """
    Definition of the Variational version of the Lottery AE
    seen as a variational auto-encoding architecture. 
    """

    def __init__(self, args):
        super(LotteryVAEFlow, self).__init__(args)
        LotteryAE.__init__(self, args)
        VAEFlow.__init__(self, args)
        self.pruning = args.pruning
        self.mu.unprunable = True
        self.log_var.unprunable = True

"""
###################
WAE - Duck typing to lottery model
###################
"""
class LotteryWAE(LotteryAE, WAE):
    """
    Definition of the Variational version of the Lottery AE
    seen as a variational auto-encoding architecture. 
    """

    def __init__(self, args):
        super(LotteryWAE, self).__init__(args)
        LotteryAE.__init__(self, args)
        WAE.__init__(self, args)
        self.pruning = args.pruning
        self.mu.unprunable = True
        self.log_var.unprunable = True

def construct_encoder_decoder(args):
    """ Construct encoder and decoder layers for AE models """
    in_size = args.input_size
    out_size = args.output_size
    # MLP layers
    if (args.type_mod in ['mlp', 'gated_mlp']):
        args.output_size = args.encoder_dims
        args.encoder = GatedMLP(args)
        args.input_size = args.latent_dims
        args.output_size = in_size
        args.decoder = DecodeMLP(args)
    elif (args.type_mod in ['cnn', 'gated_cnn', 'res_cnn']):
        args.output_size = args.encoder_dims
        args.encoder = GatedCNN(args)
        args.input_size = args.latent_dims
        args.output_size = in_size
        args.cnn_size = args.encoder.cnn_size
        args.decoder = DecodeCNN(args)
    args.input_size = in_size
    args.output_size = out_size
    return args
    
"""
###################

Basic layers definitions

###################
"""

class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=torch.relu):
        super(GatedDense, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation( self.h( x ) )
        g = self.sigmoid( self.g( x ) )
        return h * g

class GatedConv2d(nn.Module):
    
    def __init__(self, in_c, out_c, kernel, stride, pad, dilation=1, act=torch.relu):
        super(GatedConv2d, self).__init__()
        self.activation = act
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Conv2d(in_c, out_c, kernel, stride, pad, dilation)
        self.g = nn.Conv2d(in_c, out_c, kernel, stride, pad, dilation)

    def forward(self, x):
        h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))
        return h * g

class ResConv2d(nn.Module):
    
    def __init__(self, in_c, out_c, kernel, stride, pad, dilation=1, act=torch.relu):
        super(ResConv2d, self).__init__()
        self.activation = act
        self.h = nn.Conv2d(in_c, out_c, kernel, stride, pad, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_c)
        self.g = nn.Conv2d(in_c, out_c, kernel, stride, pad, dilation)
        for i in range(self.g.weight.shape[0]):
            for j in range(self.g.weight.shape[1]):
                nn.init.eye_(self.g.weight.data[i, j])
        nn.init.constant_(self.g.bias.data, 0)
        self.g.weight.requires_grad = False
        self.g.bias.requires_grad = False

    def forward(self, x):
        h = self.activation(self.bn(self.h(x)))
        g = self.g(x)
        return h + g

class ResConvTranspose2d(nn.Module):
    
    def __init__(self, in_c, out_c, kernel, stride, pad, output_padding=0, dilation=1, act=torch.relu):
        super(ResConvTranspose2d, self).__init__()
        self.activation = act
        self.h = nn.ConvTranspose2d(in_c, out_c, kernel, stride, pad, output_padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_c)
        self.g = nn.ConvTranspose2d(in_c, out_c, kernel, stride, pad, output_padding, dilation=dilation)
        for i in range(self.g.weight.shape[0]):
            for j in range(self.g.weight.shape[1]):
                nn.init.eye_(self.g.weight.data[i, j])
        nn.init.constant_(self.g.bias.data, 0)
        self.g.weight.requires_grad = False
        self.g.bias.requires_grad = False

    def forward(self, x):
        h = self.activation(self.bn(self.h(x)))
        g = self.g(x)
        return h + g
    
class GatedConvTranspose2d(nn.Module):
    
    def __init__(self, in_c, out_c, kernel, stride, pad, output_padding=0, dilation=1, act=torch.relu):
        super(GatedConvTranspose2d, self).__init__()
        self.activation = act
        self.sigmoid = nn.Sigmoid()
        self.h = nn.ConvTranspose2d(in_c, out_c, kernel, stride, pad, output_padding, dilation=dilation)
        self.g = nn.ConvTranspose2d(in_c, out_c, kernel, stride, pad, output_padding, dilation=dilation)

    def forward(self, x):
        h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))
        return h * g

"""
###################

Basic MLP definition

###################
"""

model_layer = {'mlp':'normal', 'gated_mlp':'gated', 'cnn':'normal', 'gated_cnn':'gated', 'res_cnn':'residual'}


class GatedMLP(LotteryClassification):
    
    def __init__(self, args):
        super(GatedMLP, self).__init__(args)
        in_size = np.prod(args.input_size)
        out_size = np.prod(args.output_size)
        hidden_size = args.n_hidden
        n_layers = args.n_layers
        type_mod = model_layer[args.type_mod]
        dense_module = (type_mod == 'gated') and GatedDense or nn.Linear
        # Create modules
        modules = nn.Sequential()
        for l in range(n_layers):
            in_s = (l==0) and in_size or hidden_size
            out_s = (l == n_layers - 1) and out_size or hidden_size
            cur_dense = dense_module(in_s, out_s)
            modules.add_module('l%i'%l, cur_dense)
            if (l < n_layers - 1):
                modules.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                modules.add_module('a%i'%l, nn.ReLU())
                modules.add_module('d%i'%l, nn.Dropout(p=.2))
            else:
                cur_dense.unprunable = True
                modules.add_module('a%i'%l, nn.ReLU())
        self.net = modules
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
        
    def forward(self, inputs):
        # Flatten the input
        out = inputs.view(inputs.shape[0], -1)
        for m in range(len(self.net)):
            out = self.net[m](out)
        return out

   
"""
###################

Basic CNN definitions

###################
"""

class GatedCNN(LotteryClassification):
    
    def __init__(self, args):
        super(GatedCNN, self).__init__(args)
        type_mod = model_layer[args.type_mod]
        conv_module = (type_mod == 'gated') and GatedConv2d or nn.Conv2d
        conv_module = (type_mod == 'residual') and ResConv2d or conv_module
        dense_module = (type_mod == 'gated') and GatedDense or nn.Linear
        # Retrieve arguments
        in_size = args.input_size
        out_size = args.output_size
        hidden_size = args.n_hidden
        n_layers = args.n_layers
        channels = args.channels
        n_mlp = args.n_layers
        # Create modules
        modules = nn.Sequential()
        size = [in_size[-2], in_size[-1]]
        in_channel = 1 if len(in_size)<3 else in_size[0] #in_size is (C,H,W) or (H,W)
        kernel = args.kernel
        stride = 2
        """ First do a CNN """
        for l in range(n_layers):
            dil = ((args.dilation == 3) and (2 ** l) or args.dilation)
            pad = 3 * (dil + 1)
            in_s = (l==0) and in_channel or channels
            out_s = (l == n_layers - 1) and 1 or channels
            cur_conv = conv_module(in_s, out_s, kernel, stride, pad, dilation = dil)
            modules.add_module('c2%i'%l, cur_conv)
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('d2%i'%l, nn.Dropout2d(p=.25))
            else:
                cur_conv.unprunable = True
            size[0] = int((size[0]+2*pad-(dil*(kernel-1)+1))/stride+1)
            size[1] = int((size[1]+2*pad-(dil*(kernel-1)+1))/stride+1)
        self.net = modules
        self.mlp = nn.Sequential()
        """ Then go through MLP """
        for l in range(n_mlp):
            in_s = (l==0) and (size[0] * size[1]) or hidden_size
            out_s = (l == n_mlp - 1) and out_size or hidden_size
            cur_dense = dense_module(in_s, out_s)
            self.mlp.add_module('h%i'%l, cur_dense)
            if (l < n_layers - 1):
                self.mlp.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i'%l, nn.ReLU())
                self.mlp.add_module('d%i'%l, nn.Dropout(p=.25))
            else:
                cur_dense.unprunable = True
        self.cnn_size = size
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
        
    def forward(self, inputs):
        out = inputs.unsqueeze(1) if len(inputs.shape) < 4 else inputs # force to (batch, C, H, W)
        for m in range(len(self.net)):
            out = self.net[m](out)
        out = out.view(inputs.shape[0], -1)
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        return out
    
"""
###################

Decoding version of MLP definitions

###################
"""
class DecodeMLP(GatedMLP):
    
    def __init__(self, args):
        super(DecodeMLP, self).__init__(args)
        # Record final size
        self.out_size = args.output_size
        
    def forward(self, inputs):
        # Use super function
        out = GatedMLP.forward(self, inputs)
        # Reshape output
        if (len(self.out_size) > 1):
            out = out.view(inputs.shape[0], *self.out_size)
        return out
    
"""
###################

Decoding version of CNN definitions

###################
"""    
class DecodeCNN(LotteryClassification):
    
    def __init__(self, args):
        super(DecodeCNN, self).__init__(args)
        type_mod = model_layer[args.type_mod]
        conv_module = (type_mod == 'gated') and GatedConvTranspose2d or nn.ConvTranspose2d
        conv_module = (type_mod == 'residual') and ResConvTranspose2d or conv_module
        dense_module = (type_mod == 'gated') and GatedDense or nn.Linear
        # Retrieve arguments
        in_size = args.input_size
        out_size = args.output_size
        hidden_size = args.n_hidden
        n_layers = args.n_layers
        channels = args.channels
        n_mlp = args.n_layers
        # Create modules
        self.cnn_size = [args.cnn_size[0], args.cnn_size[1]]
        size = self.cnn_size.copy()
        kernel = args.kernel
        stride = 2
        self.mlp = nn.Sequential()
        """ First go through MLP """
        for l in range(n_mlp):
            in_s = (l==0) and (in_size) or hidden_size
            out_s = (l == n_mlp - 1) and np.prod(self.cnn_size) or hidden_size
            cur_dense = dense_module(in_s, out_s)
            self.mlp.add_module('h%i'%l, cur_dense)
            if (l < n_layers - 1):
                self.mlp.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i'%l, nn.ReLU())
                self.mlp.add_module('d%i'%l, nn.Dropout(p=.25))
            else:
                cur_dense.unprunable = True
        """ Protect layer from pruning """
        #self.mlp._modules['h%i'%(n_layers-1)].unprunable = True
        # Pass to CNN
        modules = nn.Sequential()
        """ Then do a CNN """
        for l in range(n_layers):
            dil = ((args.dilation == 3) and (2 ** ((n_layers - 1) - l)) or args.dilation)
            pad = 3 * (dil + 1)
            if (args.dilation == 1):
                pad = 2
            out_pad = (pad % 2)
            in_s = (l==0) and 1 or channels
            out_s = (l == n_layers - 1) and out_size[0] or channels
            cur_conv = conv_module(in_s, out_s, kernel, stride, pad, output_padding=out_pad, dilation = dil)
            modules.add_module('c2%i'%l, cur_conv)
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('d2%i'%l, nn.Dropout2d(p=.2))
            else:
                modules.add_module('a%i'%l, nn.Sigmoid())
                cur_conv.unprunable = True
            size[0] = int((size[0] - 1) * stride - (2 * pad) + dil * (kernel - 1) + out_pad + 1)
            size[1] = int((size[1] - 1) * stride - (2 * pad) + dil * (kernel - 1) + out_pad + 1)
        self.net = modules
        self.out_size = out_size #(H,W) or (C,H,W)
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
        
    def forward(self, inputs):
        out = inputs
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        out = out.unsqueeze(1).view(-1, 1, self.cnn_size[0], self.cnn_size[1])
        for m in range(len(self.net)):
            out = self.net[m](out)
        if len(self.out_size) < 3:
            out = out[:, :, :self.out_size[0], :self.out_size[1]].squeeze(1)
        else:
            out = out[:, :, :self.out_size[1], :self.out_size[2]]
        return out

"""
###################

Simplest RNN possible definitions

###################
"""    
# Recurrent neural network (many-to-one)
class SimpleRNN(LotteryClassification):
    
    def __init__(self, args):#input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__(args)
        self.in_size = args.input_size[1]
        self.seq_size = args.input_size[2]
        self.hidden_size = args.n_hidden
        self.num_layers = args.n_layers
        self.device = args.device
        self.model = args.model
        if (args.model == 'lstm'):
            self.recurrent = nn.LSTM(self.in_size, self.hidden_size, self.num_layers, batch_first=True)
        elif (args.model == 'gru'):
            self.recurrent = nn.GRU(self.in_size, self.hidden_size, self.num_layers, batch_first=True)
        elif (args.model == 'rnn'):
            self.recurrent = nn.RNN(self.in_size, self.hidden_size, self.num_layers, batch_first=True)
        self.norm = nn.BatchNorm1d(self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, args.output_size)
        self.fc.unprunable = True
    
    def forward(self, x):
        x = x.view(-1, self.seq_size, self.in_size)
        # Set initial hidden and cell states 
        self.hidden_size = self.recurrent.weight_hh_l0.shape[1]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # Forward propagate LSTM
        if (self.model == 'lstm'):
            out = self.recurrent(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        else:
            out = self.recurrent(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        if (type(out) == tuple):
            out, _ = out
        out = self.norm(out[:, -1, :])
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out
    
"""
###################

Simplest CNN - 1 dimensional

###################
"""    
# Recurrent neural network (many-to-one)
class SimpleCNN1D(LotteryClassification):
    
    def __init__(self, args):#input_size, hidden_size, num_layers, num_classes):
        super(SimpleCNN1D, self).__init__(args)
        # Retrieve arguments
        in_size = np.prod(args.input_size)
        out_size = np.prod(args.output_size)
        hidden_size = args.n_hidden
        n_layers = args.n_layers
        channels = args.channels
        n_mlp = args.n_layers
        # Create modules
        modules = nn.Sequential()
        self.in_size = in_size
        size = in_size
        in_channel = 1 #if len(in_size)<3 else in_size #in_size is (C,H,W) or (H,W)
        kernel = args.kernel
        stride = 2
        """ First do a CNN """
        for l in range(n_layers):
            dil = ((args.dilation == 3) and (2 ** l) or args.dilation)
            pad = 3 * (dil + 1)
            in_s = (l==0) and in_channel or channels
            out_s = (l == n_layers - 1) and 1 or channels
            modules.add_module('c2%i'%l, nn.Conv1d(in_s, out_s, kernel, stride, pad, dilation = dil))
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm1d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('d2%i'%l, nn.Dropout2d(p=.25))
            size = int((size+2*pad-(dil*(kernel-1)+1))/stride+1)
        self.net = modules
        self.mlp = nn.Sequential()
        """ Then go through MLP """
        for l in range(n_mlp):
            in_s = (l==0) and (size) or hidden_size
            out_s = (l == n_mlp - 1) and out_size or hidden_size
            self.mlp.add_module('h%i'%l, nn.Linear(in_s, out_s))
            if (l < n_layers - 1):
                self.mlp.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i'%l, nn.ReLU())
                self.mlp.add_module('d%i'%l, nn.Dropout(p=.25))
        self.cnn_size = size
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
        
    def forward(self, x):
        x = x.view(-1, 1, self.in_size)
        out = x
        for m in range(len(self.net)):
            out = self.net[m](out)
        out = out.view(x.shape[0], -1)
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        return out
    
    