from os import path

import torch
import librosa as li
import soundfile as sf
import numpy as np

from udls import SimpleDataset

from model import LotteryModel
from models.ddsp import NeuralSynth

#from crepe import predict
from tqdm import tqdm


class hparams:
    # ENCODER PARAMS
    LATENT_HIDDEN_DIM = 128
    LATENT_SIZE = 16
    KERNEL_SIZE = 7
    DILATION = 2
    STRIDE = [4, 2, 4, 5]

    # DECODER PARAMS
    HIDDEN_SIZE = 512
    N_PARTIAL = 100
    FILTER_SIZE = 160
    SAMPLERATE = 16000
    BLOCK_SIZE = 160
    SEQUENCE_SIZE = 200
    FFT_SCALES = [2048, 1024, 512, 256, 128, 64]


@torch.no_grad()
def preprocess(name):
    try:
        x = li.load(name, hparams.SAMPLERATE)[0]
    except KeyboardInterrupt:
        exit()
    except:
        return None

    n_signal = hparams.SEQUENCE_SIZE * hparams.BLOCK_SIZE
    border = len(x) % n_signal
    if border:
        if len(x) < n_signal:
            x = np.pad(x, (0, n_signal - len(x)))
        else:
            x = x[:-border]

    f0 = predict(
        x,
        hparams.SAMPLERATE,
        model_capacity='full',
        viterbi=True,
        center=True,
        step_size=10,
        verbose=0,
    )[1]

    lo = li.feature.rms(
        x,
        frame_length=2 * hparams.BLOCK_SIZE,
        hop_length=hparams.BLOCK_SIZE,
    )

    f0 = f0[:x.shape[-1] // hparams.BLOCK_SIZE]
    lo = lo[..., :x.shape[-1] // hparams.BLOCK_SIZE]

    f0_interp = np.interp(
        np.linspace(
            0,
            1,
            len(f0) * hparams.BLOCK_SIZE,
        ),
        np.linspace(0, 1, len(f0)),
        f0,
    )

    phi = np.zeros(len(f0_interp))
    for i in np.arange(1, len(phi)):
        phi[i] = 2 * np.pi * f0_interp[i] / hparams.SAMPLERATE + phi[i - 1]
        phi[i] = phi[i] % (2 * np.pi)

    x = x.reshape(-1, n_signal).astype(np.float32)
    phi = phi.reshape(-1, n_signal).astype(np.float32)
    f0 = f0.reshape(-1, hparams.SEQUENCE_SIZE).astype(np.float32)
    lo = lo.reshape(-1, hparams.SEQUENCE_SIZE).astype(np.float32)

    return zip(x, phi, f0, lo)


class LotteryDDSP(LotteryModel, NeuralSynth):
    def __init__(self, args):
        NeuralSynth.__init__(
            self,
            latent_hidden_dim=hparams.LATENT_HIDDEN_DIM,
            latent_size=hparams.LATENT_SIZE,
            kernel_size=hparams.KERNEL_SIZE,
            dilation=hparams.DILATION,
            stride=hparams.STRIDE,
            hidden_size=hparams.HIDDEN_SIZE,
            n_partial=hparams.N_PARTIAL,
            filter_size=hparams.FILTER_SIZE,
            block_size=hparams.BLOCK_SIZE,
            samplerate=hparams.SAMPLERATE,
            sequence_size=hparams.SEQUENCE_SIZE,
            fft_scales=hparams.FFT_SCALES,
        )

        self.pruning = args.pruning
        self.running_loss = None

    def train_epoch(self, train_loader, optimizer, criterion, iteration, args):
        self.train()
        full_loss = torch.zeros(1).to(args.device, non_blocking=True)
        step = iteration * len(train_loader)

        #train_loader = tqdm(train_loader)

        if iteration == 0:
            self.running_loss = None

        for batch in train_loader:
            x = batch[0].to(args.device, non_blocking=True)
            phi = batch[1].to(args.device, non_blocking=True)
            f0 = batch[2].to(args.device, non_blocking=True)
            lo = batch[3].to(args.device, non_blocking=True)

            inputs = torch.stack([f0, lo], -1)

            noise_pass = True if step > 2000 else False
            conv_pass = True  #if step > 500 else False

            y, amp, _, _, kl_loss = self(
                x=x,
                cdt=inputs,
                noise_pass=noise_pass,
                conv_pass=conv_pass,
                phi=phi,
                only_y=False,
            )

            stfts = self.multiScaleFFT(x)
            stfts_rec = self.multiScaleFFT(y)

            lin_loss = sum([
                torch.mean(abs(stfts[i] - stfts_rec[i]))
                for i in range(len(stfts_rec))
            ])

            log_loss = sum([
                torch.mean(
                    abs(
                        torch.log(stfts[i] + 1e-4) -
                        torch.log(stfts_rec[i] + 1e-4)))
                for i in range(len(stfts_rec))
            ])

            amp_loss = torch.mean(-torch.log(amp + 1e-10))

            loss = lin_loss + log_loss + 1e-4 * kl_loss

            # Used to avoid a silence collapse during early stage of training
            if step < 1000:
                loss += .1 * amp_loss

            optimizer.zero_grad()
            loss.backward()
            self.pruning.train_callback(self, iteration)

            optimizer.step()

            step += 1

            full_loss += loss

            if self.running_loss is not None:
                self.running_loss = loss.item() * .01 + self.running_loss * .99
            else:
                self.running_loss = loss.item()

            #train_loader.set_description("epoch %d, runningloss %.3f" %
            #                             (iteration, self.running_loss))

        full_loss /= len(train_loader)
        return full_loss.item()

    @torch.no_grad()
    def test_epoch(self, test_loader, criterion, iteration, args):
        self.eval()
        full_loss = torch.zeros(1).to(args.device, non_blocking=True)

        step = iteration * len(test_loader)

        for batch in test_loader:
            x = batch[0].to(args.device, non_blocking=True)
            phi = batch[1].to(args.device, non_blocking=True)
            f0 = batch[2].to(args.device, non_blocking=True)
            lo = batch[3].to(args.device, non_blocking=True)

            inputs = torch.stack([f0, lo], -1)

            noise_pass = True if step > 2000 else False
            conv_pass = True  #if step > 500 else False

            y, amp, _, _, kl_loss = self(
                x=x,
                cdt=inputs,
                noise_pass=noise_pass,
                conv_pass=conv_pass,
                phi=phi,
                only_y=False,
            )

            stfts = self.multiScaleFFT(x)
            stfts_rec = self.multiScaleFFT(y)

            lin_loss = sum([
                torch.mean(abs(stfts[i] - stfts_rec[i]))
                for i in range(len(stfts_rec))
            ])

            log_loss = sum([
                torch.mean(
                    abs(
                        torch.log(stfts[i] + 1e-4) -
                        torch.log(stfts_rec[i] + 1e-4)))
                for i in range(len(stfts_rec))
            ])

            amp_loss = torch.mean(-torch.log(amp + 1e-10))

            loss = lin_loss + log_loss + 1e-3 * kl_loss

            full_loss += loss
        
        full_loss /= len(test_loader)
        return full_loss.item()

    def cumulative_epoch(self, train_loader, optimizer, criterion, limit,
                         args):
        self.train()
        # Set gradients to zero
        optimizer.zero_grad()
        nb_examples = 0
        full_targets = []
        #train_loader = tqdm(train_loader)
        step = 0
        for batch in train_loader:
            x = batch[0].to(args.device)
            phi = batch[1].to(args.device)
            f0 = batch[2].to(args.device)
            lo = batch[3].to(args.device)

            inputs = torch.stack([f0, lo], -1)

            noise_pass = True if step > 2000 else False
            conv_pass = True  #if step > 500 else False

            y, amp, _, _, kl_loss = self(
                x=x,
                cdt=inputs,
                noise_pass=noise_pass,
                conv_pass=conv_pass,
                phi=phi,
                only_y=False,
            )

            stfts = self.multiScaleFFT(x)
            stfts_rec = self.multiScaleFFT(y)

            lin_loss = sum([
                torch.mean(abs(stfts[i] - stfts_rec[i]))
                for i in range(len(stfts_rec))
            ])

            log_loss = sum([
                torch.mean(
                    abs(
                        torch.log(stfts[i] + 1e-4) -
                        torch.log(stfts_rec[i] + 1e-4)))
                for i in range(len(stfts_rec))
            ])

            amp_loss = torch.mean(-torch.log(amp + 1e-10))

            loss = lin_loss + log_loss + 1e-3 * kl_loss

            # Run backward
            loss.backward()
            # Add targets
            full_targets.append(x.unsqueeze(1))
            nb_examples += x.shape[0]
            step += 1

            if (nb_examples > limit):
                break
        self.outputs = full_targets

    @torch.no_grad()
    def evaluate_epoch(self, test_loader, criterion, iteration, args):
        self.eval()
        outputs = []
        for batch in test_loader:
            x = batch[0].to(args.device)
            phi = batch[1].to(args.device)
            f0 = batch[2].to(args.device)
            lo = batch[3].to(args.device)

            inputs = torch.stack([f0, lo], -1)

            y = self(
                x=x,
                cdt=inputs,
                noise_pass=True,
                conv_pass=True,
                phi=phi,
            )

            outputs.append(y[:16].reshape(-1))
            break
        outputs = torch.cat(outputs, -1).cpu().numpy()
        filename = path.join(args.model_save, "eval_%d_%d_out.wav" % (args.prune_it, iteration))
        sf.write(filename, outputs, hparams.SAMPLERATE)
        filename = path.join(args.model_save, "eval_%d_%d_in.wav" % (args.prune_it, iteration))
        sf.write(filename, x[:16, :].reshape(-1).cpu().numpy(), hparams.SAMPLERATE)
