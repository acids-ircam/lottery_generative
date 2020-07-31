from os import path

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import librosa as li
import soundfile as sf

import math

from model import LotteryModel
from models.wavenet import Wavenet, MelEncoder

import matplotlib.pyplot as plt

from tqdm import tqdm

writer = SummaryWriter("debug", flush_secs=20)


class hparams:
    CYCLE_NB = 2
    N_LAYER = 20
    IN_SIZE = 256
    RES_SIZE = 128
    SKP_SIZE = 256
    CDT_SIZE = 80
    OUT_SIZE = 256
    DIM_REDUCTION = 512
    SAMPRATE = 16000
    N_SIGNAL = 8192


melencoder = MelEncoder(hparams.SAMPRATE, hparams.DIM_REDUCTION,
                        hparams.CDT_SIZE, True)


@torch.no_grad()
def preprocess(name):
    x = li.load(name, hparams.SAMPRATE)[0]
    x = torch.from_numpy(x).float()

    border = len(x) % hparams.N_SIGNAL
    x = x[:-border] if border else x
    mel = melencoder(x.unsqueeze(0))  # 1 x CDT x T / DR

    x = x.reshape(-1, hparams.N_SIGNAL)
    mel = mel.permute(0, 2, 1)
    mel = mel.reshape(-1, hparams.N_SIGNAL // hparams.DIM_REDUCTION,
                      hparams.CDT_SIZE)
    mel = mel.permute(0, 2, 1)

    # MULAW

    x = torch.sign(x) * torch.log(1 + 255 * abs(x)) / math.log(256)

    x += 1
    x /= 2
    x = torch.clamp(x, 0, 1)
    x *= 255
    x = x.long()

    return zip(x, mel)


class LotteryWavenet(LotteryModel, Wavenet):
    def __init__(self, args):
        Wavenet.__init__(self, hparams.CYCLE_NB, hparams.N_LAYER,
                         hparams.IN_SIZE, hparams.RES_SIZE, hparams.SKP_SIZE,
                         hparams.CDT_SIZE, hparams.OUT_SIZE,
                         hparams.DIM_REDUCTION)

        self.pruning = args.pruning
        self.step = 0

    def train_epoch(self, train_loader, optimizer, criterion, iteration, args):
        self.train()
        full_loss = 0
        criterion = nn.CrossEntropyLoss()
        for sample, condition in tqdm(train_loader, desc="train_epoch"):
            # TRANSFORM INPUT
            sample = sample.to(args.device)
            condition = condition.to(args.device)
            sample_oh = torch.nn.functional.one_hot(sample, hparams.IN_SIZE)
            sample_oh = sample_oh.permute(0, 2, 1).float()

            # FORWARD PASS
            prediction = self(sample_oh, condition)

            if self.step % 100 == 0:
                image = torch.softmax(prediction[0], 0).cpu().detach().numpy()

                plt.figure(figsize=(20, 4))
                plt.imshow(
                    image,
                    origin="lower",
                    aspect="auto",
                )

                writer.add_figure("pred", plt.gcf(), self.step)
                plt.close()

            # LOSS COMPUTATION
            prediction_cropped = prediction[..., self.receptive_field:-1]
            target_cropped = sample[..., self.receptive_field + 1:]

            prediction_cropped = prediction_cropped.permute(0, 2, 1).reshape(
                -1,
                self.in_size,
            )

            loss = criterion(prediction_cropped, target_cropped.reshape(-1))

            writer.add_scalar("loss", loss.item(), self.step)
            self.step += 1

            # PRUNING AND OPTIMIZATION
            optimizer.zero_grad()
            loss.backward()
            self.pruning.train_callback(self, iteration)
            optimizer.step()

            full_loss += loss.item()

        full_loss /= len(train_loader)
        return full_loss

    @torch.no_grad()
    def test_epoch(self, test_loader, criterion, iteration, args):
        self.eval()
        full_loss = 0
        for sample, condition in tqdm(test_loader, desc="test_epoch"):
            # TRANSFORM INPUT
            sample = sample.to(args.device)
            condition = condition.to(args.device)
            sample_oh = torch.nn.functional.one_hot(sample, hparams.IN_SIZE)
            sample_oh = sample_oh.permute(0, 2, 1).float()

            # FORWARD PASS
            prediction = self(sample_oh, condition)

            # LOSS COMPUTATION
            prediction_cropped = prediction[..., self.receptive_field:-1]
            target_cropped = sample[..., self.receptive_field + 1:]

            prediction_cropped = prediction_cropped.permute(0, 2, 1).reshape(
                -1,
                self.in_size,
            )

            loss = criterion(prediction_cropped, target_cropped.reshape(-1))

            full_loss += loss.item()

        full_loss /= len(test_loader)
        return full_loss

    def cumulative_epoch(self, train_loader, optimizer, criterion, limit,
                         args):
        self.train()
        # Set gradients to zero
        optimizer.zero_grad()
        nb_examples = 0
        full_targets = []
        for sample, condition in tqdm(train_loader, desc="cumulative_epoch"):
            # TRANSFORM INPUT
            sample = sample.to(args.device)
            condition = condition.to(args.device)
            sample_oh = torch.nn.functional.one_hot(sample, hparams.IN_SIZE)
            sample_oh = sample_oh.permute(0, 2, 1).float()

            # FORWARD PASS
            prediction = self(sample_oh, condition)

            # LOSS COMPUTATION
            prediction_cropped = prediction[..., self.receptive_field:-1]
            target_cropped = sample[..., self.receptive_field + 1:]

            prediction_cropped = prediction_cropped.permute(0, 2, 1).reshape(
                -1,
                self.in_size,
            )

            loss = criterion(prediction_cropped, target_cropped.reshape(-1))

            # PRUNING AND OPTIMIZATION
            loss.backward()
            # Add targets
            full_targets.append(target_cropped.unsqueeze(1))
            nb_examples += sample.shape[0]
            if (nb_examples > limit):
                break
        self.outputs = full_targets

    @torch.no_grad()
    def evaluate_epoch(self, test_loader, criterion, iteration, args):
        self.eval()
        save_path = path.join(args.model_save,
                              "it_%d_%d" % (args.prune_it, iteration))
        for i, (sample, condition) in enumerate(test_loader):
            condition = condition.to(args.device)

            sample = self.generate_fast(condition)
            sample = sample.cpu().numpy().reshape(-1)

            # inverse mu law
            sample = np.sign(sample) * (1 / 255) * ((256)**abs(sample) - 1)

            sf.write(
                "%s_%d.wav" % (save_path, iteration),
                sample,
                hparams.SAMPRATE,
            )
            break
