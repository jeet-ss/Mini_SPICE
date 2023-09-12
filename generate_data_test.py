import numpy as np
import librosa
import argparse
import scipy
import os
import sys
global_workspace_path = "/home/hpc/iwal/iwal132h/Mini_SPICE/"
sys.path.append(global_workspace_path)
from data_files.dataloader import MedleyDBLoader, MDBMelodySynthLoader, MIR1KLoader
# # extra imports
# from data_files.dataset import CQT_Dataset
import pandas as pd
# import torch
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader
# from utils.model import Spice_model


def dataset_select(indx: int, fs: int):
    extension = ".pkl"
    match indx:
        case 1 :
            return MedleyDBLoader(fs), "MedleyDB" + extension
        case 2 :
            return MDBMelodySynthLoader(fs), "MDBSynth" + extension
        case 3:
            return MIR1KLoader(fs), "MIR1k" + extension
        
def step_func(n, threshold=0.5):
    if n < threshold:
        return 0.
    if n >= threshold:
        return 1.    

def generate_data(args):
#     """ generates CQT data and add label values after interpolation """
    # constants
    fs = args.fs
    hop_len = 512
    #
    dataset, file_name  = dataset_select(args.dataset, args.fs)
    print("dataset: ", file_name)
    id_list = dataset.get_ids()

#     # load audio
    songs = []
    f0_list = []
    foo = []
    for i, s in enumerate(id_list):
        try:
            song, f0 = dataset.load_data(s)
            print(song.shape, f0.shape)
            # convert stereo to mono
            songs.append(librosa.to_mono(song))
            f0_list.append(f0)
            foo = np.concatenate((foo, f0[2]), axis=0)
        except AssertionError:
            print("{} song has problem".format(i))
    # remove zeros
    foo = foo[foo!=0]
    print(f'Songs Shape:{songs[0].shape}, {len(songs)}')
    #print("fo", foo.shape, np.max(foo), np.min(foo)   )
    # Convert to CQT array and concat
    Cqtt = np.zeros((1, 190))
    voicing_interp = np.zeros(1)
    F0_interp = np.zeros(1)
    time_arr = np.zeros(1)
    for s, f in zip(songs, f0_list):
        C = np.abs(librosa.cqt(s, sr=fs, hop_length=hop_len, 
                    #window=librosa.filters.get_window('hann', Nx=1024, fftbins=False), 
                    fmin= librosa.note_to_hz('C1'),
                    n_bins=190, bins_per_octave=24))
        #print("CQT shape: ", C.shape)
        Cqtt = np.vstack((Cqtt, C.T))
        # interpolate f0 for labels 
        interpolator = scipy.interpolate.interp1d(x=f[0], y=f[2], axis=0, fill_value = 'extrapolate')
        interp_time = np.arange(0, C.shape[1], 1)*hop_len/fs
        f0_new = interpolator(interp_time)
        F0_interp = np.concatenate((F0_interp, f0_new))
        # interpolate voicing for labels 

        interpolator_voice = scipy.interpolate.interp1d(x=f[0], y=f[1], axis=0, fill_value = 'extrapolate')
        uv_new = interpolator_voice(interp_time)
        voicing_interp = np.concatenate((voicing_interp, uv_new))

        time_arr=np.concatenate((time_arr, interp_time))
        #print("F0 interpolated shape: ", f0_new)data_pd = pd.DataFrame(data=data_np) 
    print("Time, CQT, Voiicing & F0 Shape: ",time_arr.shape, Cqtt.shape, voicing_interp.shape, F0_interp.shape)
    apply_step = np.vectorize(step_func)
    voicing_interp = apply_step(voicing_interp)
#     # remove empty rows at the start
    time_arr = time_arr[1:]
    time_arr = time_arr.reshape(-1, 1)
    Cqtt = Cqtt[1:, :]
    F0_interp = F0_interp[1:]
    #elevate f0
    F0_interp = F0_interp.reshape(-1, 1)
    voicing_interp = voicing_interp[1:]
    #elevate voicing
    voicing_interp = voicing_interp.reshape(-1, 1)
    print("Time Shape: ", time_arr.shape)
    print("CQT Shape: ", Cqtt.shape)
    print("F0 shape: ", F0_interp.shape)
    print("Voicing shape: ", voicing_interp.shape)

#     # make the last column as f0s
    data_np = np.hstack((time_arr, Cqtt, voicing_interp, F0_interp))
    print('final data: ', data_np.shape)
    
    # load into pandas
    df = pd.DataFrame(data=data_np)
    print('Before', df.shape)
    # remove rows of cqt where label (last) column is zero
    # df.drop(df.loc[df.iloc[:, -1]<=20].index, inplace=True) 
    print('after', df.shape)
    
#     # save dataframe to file
#     # get root directory and file path
    
    root_path = os.path.join(global_workspace_path, args.data_dir)
    # is directory not present already
    if os.path.isdir(root_path) != True:
        os.makedirs(root_path)
#     # save file
    file_path = os.path.join(root_path, file_name)
    print(file_path)
    np.save(file=file_path, arr=data_np)
    df.to_pickle(file_path)


#     ################################################################################
#     ##  Extra part form Train.py
#     ## for testing
#     # data_pd = pd.DataFrame(data=data_np) 
#     # train, val = train_test_split(data_pd, train_size=0.8, test_size=0.2, random_state=1)
#     # print("train shape: ", train.shape)
#     # train_batches = DataLoader(CQT_Dataset(data=train, mode='train'), batch_size=64, shuffle=True)
#     # print("train_batch shape: ", len(train_batches))
#     # diff, slice1, slice2, f0 = next(iter(train_batches))
#     # print(f"diff batch shape: {diff.size()}")
#     # print(f"slice1 batch shape: {slice1.size()}")
#     # print(f"slice2 batch shape: {f0.size()}")

#     # spice = Spice_model()


#     # for b in train_batches:
#     #     pitch_diff, x_1, x_2, f0 = b
#     #     x_1 = x_1.type(torch.FloatTensor)
#     #     #print(x_1.shape, x_1.type())
#     #     a, x, y = spice(x_1)
#     #     print(a.size(), x.size(), y.size())
#     #
#     ###############################################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of SPICE model for F0 estimation')
    parser.add_argument('--fs', type=int, default=16000, help='Sampling rate of Dataset')
    parser.add_argument('-ds', '--dataset', type=int, default=3, help='Dataset to Load')
    parser.add_argument('-dir', '--data_dir', type=str, default='CQT_data', help='Directory to store data')
    args = parser.parse_args()
    print(args)

    generate_data(args)
