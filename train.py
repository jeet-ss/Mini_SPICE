import torch
import numpy as np
# import pandas as pd
import argparse
import sys
import os

from utils.model import Spice_model
from utils.training_script import Trainer
from optims.loss import Huber_loss, Recons_loss, Conf_loss


def train():
    parser = argparse.ArgumentParser(description='Training of SPICE model for F0 estimation')
    # parser.add_argument()
    args = parser.parse_args()

    # define Hyperparams
    learning_rate = 0.1         # original 
    epochs_num = 10000
    loss_threshold = 0.01
    batch_size = 64             # original 64
    tau = 0.1                   # for huber loss

    # Architecture params
    channel_enc_list = [1, 64, 128, 256, 512, 512, 512] 
    channel_dec_list = [512, 256, 256, 256, 128, 64, 32]
    unPooling_list = [True, False, False, False, False, False]

    # Load Data

    # Split into batches and train-val set

    # set up model 
    spice = Spice_model(channel_enc_list, channel_dec_list, unPooling_list)
    # set up loss funcitons
    pitch_loss = Huber_loss(tau=tau)
    recons_loss = Recons_loss()
    conf_loss = Conf_loss()
    # set up optimizers
    adam_optim = torch.optim.Adam(spice.parameters(), lr=learning_rate)
    # set up Trainer object
    trainer = Trainer(model=spice, loss_pitch=pitch_loss, loss_recons=recons_loss, 
                        loss_conf=conf_loss,
                        optim=adam_optim, train_ds=None, val_test_ds= None,
                        w_pitch=None, w_recon=None)
    # run training
    loss_train = trainer.fit_model(epochs=epochs_num)

    # plot data


if __name__ == '__main__':
    train()
