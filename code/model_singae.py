import torch
from torch.nn import functional as F

from models.sing_ae.sing_ae import ConvolutionalAE
from model import LotteryModel
from utils import plot_batch_wav, plot_batch_compare_wav, write_batch_wav, write_batch_compare_wav

"""
###################

Trainer for sing-autoencoder models

###################
"""
class LotterySINGAE(LotteryModel, ConvolutionalAE):
    def __init__(self,args):
        #ConvolutionalAE.__init__(self, channels=4096, stride=256, dimension=args.encoder_dims, kernel_size=1024, context_size=9, rewrite_layers=2, window_name="hann", squared_window=True)
        ConvolutionalAE.__init__(self, channels=2048, stride=256, dimension=args.encoder_dims, kernel_size=1024, context_size=6, rewrite_layers=2, window_name="hann", squared_window=True)
        self.pruning = args.pruning
        self.encoder_dims = args.encoder_dims

    # Function for one full epoch of training
    def train_epoch(self, train_loader, optimizer, criterion, iteration, args):
        self.train()
        full_loss = 0
        """ You will need to explicitly call self.pruning.train_callback() """
        for inputs in train_loader:
            # Perform backward
            optimizer.zero_grad()
            # Auto-encode
            rebuilt, z, target = self._get_rebuilt_target(inputs,args)
            # Reconstruction loss       
            b_loss = criterion(rebuilt, target)
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

    # Function for one full epoch of testing
    def test_epoch(self, test_loader, criterion, iteration, args):
        self.eval()
        full_loss = 0
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(args.device, non_blocking=True)
                # Auto-encode
                rebuilt, z, target = self._get_rebuilt_target(inputs,args)
                # Regression loss 
                b_loss = criterion(rebuilt, target)
                full_loss += b_loss
            full_loss /= len(test_loader)
        return full_loss
    
    # Function for regular evaluation
    def evaluate_epoch(self, test_loader, criterion, iteration, args):
        self.eval()
        with torch.no_grad():
            img_name = args.model_save + '/it_' + str(args.prune_it) + '_' + str(iteration)
            # Fixed batch to work with
            fixed_data  = next(iter(test_loader))
            fixed_data = fixed_data.to(args.device, non_blocking=True)
            # Reconstruct the batch
            rebuilt, z, target = self._get_rebuilt_target(fixed_data,args)
            # Plot the comparison
            plot_batch_compare_wav(target.cpu(), rebuilt.cpu(), name = img_name + '_reconstruction')
            write_batch_compare_wav(target.cpu(), rebuilt.cpu(), args.sample_rate, name = img_name + '_reconstruction')
            # Perform random sampling
            z_rand = torch.randn_like(z)
            x_rand = self.decode(z_rand)
            plot_batch_wav(x_rand.cpu(), name = img_name + '_sampling')
            write_batch_wav(x_rand.cpu(), args.sample_rate, name = img_name + '_sampling')

    # Function for one full epoch of training
    def cumulative_epoch(self, train_loader, optimizer, criterion, limit, args):
        self.train()
        # Set gradients to zero
        optimizer.zero_grad()
        nb_examples = 0
        full_targets = []
        """ You will need to explicitly call self.pruning.train_callback() """
        for inputs in train_loader:
            # Auto-encode
            rebuilt, z, target = self._get_rebuilt_target(inputs,args)
            # Reconstruction loss       
            b_loss = criterion(rebuilt, target)
            # Run backward
            b_loss.backward()
            # Add targets
            full_targets.append(target.unsqueeze(1))
            nb_examples += inputs.shape[0]
            if (nb_examples > limit):
                break
        self.outputs = full_targets


    def _get_rebuilt_target(self, wav, args):
        wav = wav.to(args.device, non_blocking = True)
        wav = F.pad(wav, (args.pad, args.pad))
        target = self.unpad1d(wav, self.decoder.strip)
        rebuilt, z = self(wav.unsqueeze(1))
        return rebuilt, z, target
    
    def unpad1d(self, tensor, pad):
        if pad > 0:
            return tensor[..., pad:-pad]
        return tensor
