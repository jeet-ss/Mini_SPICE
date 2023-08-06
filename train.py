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
from utils.decoders import Spice_model_Sterne

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
    epochs_num = 1
    batch_size = 64             # original 64
    #tau = 0.1                  # for huber loss
    CQT_bins_per_octave = 24
    wpitch = 3*np.power(10, 4)
    wrecon = 1

    ### Architecture params
    # for 1 Unpool
    channel_enc_list = [1, 64, 128, 256, 512, 512, 512] 
    channel_dec_list = [512, 256, 256, 256, 128, 64, 32]
    unPooling_list = [True, False, False, False, False, False]
    # mirror of encoder
    channel_dec_list_rev = [512, 512, 512, 256, 128, 64, 1]
    unPooling_list_rev = [True, True, True, True, True, True]

    # Load Data
    #data_np = np.load('./CQT_data/MedleyDB.npy')                              # load nd.array from file
    #data_pd = pd.DataFrame(data=data_np)  
    data_pd = pd.read_pickle("./CQT_data/MIR1k.pkl") 
    print("total data shape: ",data_pd.shape)
    # remove rows of cqt where label (last) column is zero
    #data_pd.drop(data_pd.loc[data_pd.iloc[:, -1]<=0].index, inplace=True) 
    # Shuffle data
    #data_pd = data_pd.sample(frac=1, random_state=10).reset_index()
    #data_pd.drop(columns=['index'],axis=1, inplace=True)
    # 
    # Mir1kdf
    fmax, fmin = 666.664939769901, 66.9456331525636256
    # MDBSynth
    #fmax, fmin = 1210.180328, 46.259328
    # get scaling factor sigma
    sigma_ = scaling_factor(Q=CQT_bins_per_octave, fmax=fmax, fmin=fmin)
    # set tau for huber loss
    tau = 0.25*sigma_
    #tau = 0.1

    # Split into batches and Dataloader 
    train, val = train_test_split(data_pd, train_size=0.8, test_size=0.2, random_state=1)
    train_batches = DataLoader(CQT_Dataset(data=train, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_batches = DataLoader(CQT_Dataset(data=val, mode='val'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print("no of batches: ", len(train_batches)," sigma: ", sigma_)

    # set up model 
    spice_o = Spice_model(channel_enc_list, channel_dec_list, unPooling_list)
    spiceC = torch.compile(Spice_model(channel_enc_list, channel_dec_list, unPooling_list))
    spice_r = Spice_model(channel_enc_list, channel_dec_list_rev, unPooling_list_rev)
    spice_rC = torch.compile(Spice_model(channel_enc_list, channel_dec_list_rev, unPooling_list_rev))
    spice_s = Spice_model_Sterne()
    # set up loss funcitons
    #pitch_loss = Huber_loss(tau=tau)
    pitch_loss = torch.nn.HuberLoss(delta=tau)
    recons_loss = Recons_loss()
    conf_loss = Conf_loss()
    # set up optimizers
    adam_optim = torch.optim.Adam(spice_o.parameters(), lr=learning_rate)
    # set up Trainer object
    trainer = Trainer(model=spice_o, loss_pitch=pitch_loss, loss_recons=recons_loss, 
                        loss_conf=conf_loss,
                        optim=adam_optim, train_ds=train_batches, val_test_ds= val_batches,
                        w_pitch=wpitch, w_recon=wrecon, sigma = sigma_)
    # run training
    loss_train = trainer.fit_model(epochs=epochs_num)
    #trainer.save_model_onnx("check2.onnx")


    # plot data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of SPICE model for F0 estimation')
    parser.add_argument('--fp', '-filepath', type=str, default="./CQT_data/MIR1k.pkl", help='file path of data')
    args = parser.parse_args()

    # pr = cProfile.Profile()
    # pr.enable()
    train(args)
    #cProfile.run('re.compile("train(args)")', 'profileOut')
    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.strip_dirs()
    # print("s. value: " , s.getvalue())
    # ps.dump_stats('pD_bF.crof')
