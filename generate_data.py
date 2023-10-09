import numpy as np
import librosa
import argparse
import scipy
import os
from data_files.dataloader import MedleyDBLoader, MDBMelodySynthLoader, MIR1KLoader
# extra imports
from data_files.dataset import CQT_Dataset
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.model import Spice_model


def dataset_select(indx: int, fs: int):
    extension = ".pkl"
    match indx:
        case 1 :
            return MedleyDBLoader(fs), "MedleyDB" + extension
        case 2 :
            return MDBMelodySynthLoader(fs), "MDBSynth" + extension
        case 3:
            return MIR1KLoader(fs), "MIR1k" + extension
    

def generate_data(args):
    """ generates CQT data and add label values after interpolation """
    # constants
    fs = args.fs
    hop_len = 512
    #
    dataset, file_name  = dataset_select(args.dataset, args.fs)
    id_list = dataset.get_ids()
    dataset_name = file_name.split('.')[0]
    # load audio
    songs = []
    f0_list = []
    foo = []
    for i, s in enumerate(id_list[:3]):
        song, f0 = dataset.load_data(s)
        #print(song.shape, f0.shape)
        # convert stereo to mono
        songs.append(librosa.to_mono(song))
        f0_list.append(f0)
        foo = np.concatenate((foo, f0[2]), axis=0)
    # remove zeros
    foo = foo[foo!=0]
    #np.save('f0_mir1k', foo)
    print(f'Dataset Name: {dataset_name}, Songs Shape:{songs[0].shape}, Max_Freq:{ np.max(foo)}, Min_Freq:{np.min(foo)} ')
        
    # Convert to CQT array and concat
    Cqtt = np.zeros((1, 190))
    F0_interp = np.zeros(1)
    for s, f in zip(songs, f0_list):
        C = np.abs(librosa.cqt(s, sr=fs, hop_length=hop_len, 
                    fmin= librosa.note_to_hz('C1'),
                    n_bins=190, bins_per_octave=24))
        Cqtt = np.vstack((Cqtt, C.T))
        # interpolate f0 for labels 
        interpolator = scipy.interpolate.interp1d(x=f[0], y=f[2], axis=0, fill_value = 'extrapolate')
        interp_time = np.arange(0, C.shape[1], 1)*hop_len/fs
        f0_new = interpolator(interp_time)
        F0_interp = np.concatenate((F0_interp, f0_new))
        print(C.shape, f0_new.shape)
    print("CQT & F0 Shape: ", Cqtt.shape, F0_interp.shape)

    # remove empty rows at the start
    Cqtt = Cqtt[1:, :]
    F0_interp = F0_interp[1:]
    F0_interp = F0_interp.reshape(-1, 1)

    # make the last column of CQT array as f0 labels
    data_np = np.hstack((Cqtt, F0_interp))
    print('final data shape: ', data_np.shape)
    
    # load into pandas
    df = pd.DataFrame(data=data_np)

    # save dataframe to file
    # get root directory and file path
    root_path = os.path.join(os.path.abspath(os.getcwd()), args.data_dir)
    # is directory not present already
    if os.path.isdir(root_path) != True:
        os.makedirs(root_path)
    # save file
    file_path = os.path.join(root_path, file_name)
    print(file_path)
    np.save(file=file_path, arr=data_np)
    df.to_pickle(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of SPICE model for F0 estimation')
    parser.add_argument('--fs', type=int, default=16000, help='Sampling rate of Dataset')
    parser.add_argument('-ds', '--dataset', type=int, default=3, help='Dataset to Load')
    parser.add_argument('-dir', '--data_dir', type=str, default='CQT_data', help='Directory to store data')
    args = parser.parse_args()
    print(args)

    generate_data(args)
