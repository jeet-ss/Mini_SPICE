import numpy as np
import librosa
import argparse
import scipy
import os
from data_files.dataloader import MedleyDBLoader, MDBMelodySynthLoader, MIR1KLoader
# extra imports
from data_files.dataset import CQT_Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def dataset_select(indx: int, fs: int):
    match indx:
        case 1 :
            return MedleyDBLoader(fs), "MedleyDB.npy"
        case 2 :
            return MDBMelodySynthLoader(fs), "MDBSynth.npy"
        case 3:
            return MIR1KLoader(fs), "MIR1k.npy"



def generate_data(args):
    

    dataset, file_name  = dataset_select(args.dataset, args.fs)
    id_list = dataset.get_ids()

    # load audio
    songs = []
    f0_list = []
    for i, s in enumerate(id_list[:1]):
        song, f0 = dataset.load_data(s)
        print(song.shape, f0.shape)
        # convert stereo to mono
        songs.append(librosa.to_mono(song))
        f0_list.append(f0)

    # Convert to CQT array and concat
    Cqtt = np.zeros((1, 190))
    F0_interp = np.zeros(1)
    for s, f in zip(songs, f0_list):
        C = np.abs(librosa.cqt(s, sr=args.fs, hop_length=512, 
                    #window=librosa.filters.get_window('hann', Nx=1024, fftbins=False), 
                    fmin= librosa.note_to_hz('C1'),
                    n_bins=190, bins_per_octave=24))
        #print("CQT shape: ", C.shape)
        Cqtt = np.vstack((Cqtt, C.T))
        # interpolate f0 for labels 
        interpolator = scipy.interpolate.interp1d(x=f[0], y=f[2], axis=0)
        f0_new = interpolator(C[0])
        F0_interp = np.concatenate((F0_interp, f0_new))
        #print("F0 interpolated shape: ", f0_new)data_pd = pd.DataFrame(data=data_np) 
    print("CQT & F0 Shape: ", Cqtt.shape, F0_interp.shape)

    # remove zero rows
    Cqtt = Cqtt[1:, :]
    F0_interp = F0_interp[1:]
    #elevate f0
    F0_interp = F0_interp.reshape(-1, 1)
    #print("CQT Shape: ", Cqtt.shape)
    #print("F0 shape: ", F0_interp.shape)

    # make the last column as f0s
    data_np = np.hstack((Cqtt, F0_interp))
    #print('final data: ', data_np.shape)

    # save CQT to file
    # get root directory and file path
    root_path = os.path.join(os.path.abspath(os.getcwd()), args.data_dir)
    # is directory not present already
    if os.path.isdir(root_path) != True:
        os.makedirs(root_path)
    # save file
    file_path = os.path.join(root_path, file_name)
    np.save(file=file_path, arr=data_np)


    ################################################################################
    ##  Extra part form Train.py
    ## for testing
    data_pd = pd.DataFrame(data=data_np) 
    train, val = train_test_split(data_pd, train_size=0.8, test_size=0.2, random_state=1)
    print("train shape: ", train.shape)
    train_batches = DataLoader(CQT_Dataset(data=train, mode='train'), batch_size=64, shuffle=True)
    print("train_batch shape: ", len(train_batches))
    diff, slice1, slice2, f0 = next(iter(train_batches))
    print(f"diff batch shape: {diff.size()}")
    print(f"slice1 batch shape: {slice1.size()}")
    print(f"slice2 batch shape: {f0.size()}")
    ##
    ###############################################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of SPICE model for F0 estimation')
    parser.add_argument('--fs', type=int, default=16000, help='Sampling rate of Dataset')
    parser.add_argument('-ds', '--dataset', type=int, default=1, help='Dataset to Load')
    parser.add_argument('-dir', '--data_dir', type=str, default='CQT_data', help='Directory to store data')
    args = parser.parse_args()


    generate_data(args)