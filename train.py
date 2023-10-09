import torch
import numpy as np
import pandas as pd
import argparse
#import sys
#import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.model import Spice_model
from utils.training_script import Trainer
from optims.loss import Huber_loss, Recons_loss, Conf_loss
from data_files.dataset import CQT_Dataset
from utils.decoders import Spice_model_Sterne, Spice_model_1Unpool, Spice_model_Mirror

#
import cProfile, pstats, io
from pstats import SortKey
import re

def scaling_factor(Q, fmax, fmin):
    # take care of negative inside log
    # for Mir1k values are 666.6649397699019, 66.9456331525636
    # since we are not using interpolated labels for training we use original 
    # dataset values for fmin and fmax    
    return 1 / (Q * np.log2(fmax / fmin))
    #return 1/2


def train(args):
    
    # define Hyperparams
    learning_rate = 1e-4       # original 1e-4
    epochs_st = 0
    epochs_end = 20
    batch_size = 64             # original 64
    CQT_bins_per_octave = 24
    wpitch = np.power(10, 4)
    wrecon = 1
    retrain_epoch_num = 9
    name_variant='withCOnf'

    ### Architecture params
    # for 1 Unpool
    channel_enc_list = [1, 64, 128, 256, 512, 512, 512] 
    channel_dec_list = [512, 256, 256, 256, 128, 64, 32]
    unPooling_list = [True, False, False, False, False, False]
    # mirror of encoder
    channel_dec_list_rev = [512, 512, 512, 256, 128, 64, 1]
    unPooling_list_rev = [True, True, True, True, True, True]

    # Load Data
    data_pd = pd.read_pickle(args.fs) 
    print("total data shape: ",data_pd.shape)
    # Decide the fmin and fmax of the dataset
    fmax, fmin = 666.664939769901, 66.9456331525636256      # MIR1k
    #fmax, fmin = 1210.180328, 46.259328        # MDBSynth

    # get scaling factor sigma
    sigma_ = scaling_factor(Q=CQT_bins_per_octave, fmax=fmax, fmin=fmin)
    # set tau for huber loss
    tau = 0.25*sigma_
    print(f'Data shape:{data_pd.shape}, fmax,fmin:{fmax,fmin}, sigma:{sigma_}, tau:{tau}')

    # Split into batches and Dataloader 
    train, val = train_test_split(data_pd[:50], train_size=0.8, test_size=0.2, random_state=1)
    print(f'trainData:{train.shape}, valData:{val.shape}')
    train_batches = DataLoader(CQT_Dataset(data=train, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_batches = DataLoader(CQT_Dataset(data=val, mode='val'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"train batches:{len(train_batches)},val batches:{len(val_batches)}")

    # set up model 
    spice_u = Spice_model(channel_enc_list, channel_dec_list, unPooling_list)
    spice_r = Spice_model(channel_enc_list, channel_dec_list_rev, unPooling_list_rev)
    spice_s = Spice_model_Sterne()
    #sp_r = Spice_model_Mirror()        # serialized model
    #sp_u = Spice_model_1Unpool()       # serialized model
    
    # set up loss funcitons
    #pitch_loss = torch.nn.MSELoss()        # L2 loss function
    pitch_loss = torch.nn.HuberLoss(delta=tau)
    recons_loss = Recons_loss()
    conf_loss = Conf_loss()
    # set up optimizers
    adam_optim = torch.optim.Adam(spice_r.parameters(), lr=learning_rate)
    # set up Trainer object
    trainer = Trainer(model=spice_r, loss_pitch=pitch_loss, loss_recons=recons_loss, 
                        loss_conf=conf_loss,
                        optim=adam_optim, train_ds=train_batches, val_test_ds= val_batches,
                        w_pitch=wpitch, w_recon=wrecon, sigma = sigma_, name_variant=name_variant)
    # run training
    #trainer.restore_checkpoint(retrain_epoch)
    loss_data = trainer.fit(epochs_st, epochs_end)
    loss_data.update({
        'condts' : 'Shuffle->True, Augmentation->False',
        'params' : 'Mir1k data, 1k epoch,',
        'name' : name_variant,
    })

    # save loss data
    np.save(f'{name_variant}_{epochs_st}-{epochs_end}_data.npy', loss_data)

    # get encoder output
    #y_hat = trainer.val_test_epoch(batch_data=val_batches,  mode='encoder_out')
    #np.save(name_variant+'y_hat.npy', y_hat)
    #print(yh1.shape)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of SPICE model for F0 estimation')
    parser.add_argument('--fp', '-filepath', type=str, default="./CQT_data/MIR1k.pkl", help='file path of data')
    args = parser.parse_args()

    train(args)