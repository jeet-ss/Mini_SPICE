import numpy as np
import librosa
import argparse
import scipy
from dataloader import MedleyDBLoader, MDBMelodySynthLoader, MIR1KLoader


def dataset_select(indx: int, fs: int):
    match indx:
        case 1 :
            return MedleyDBLoader(fs)
        case 2 :
            return MDBMelodySynthLoader(fs)
        case 3:
            return MIR1KLoader(fs)



def generate_data():
    parser = argparse.ArgumentParser(description='Training of SPICE model for F0 estimation')
    parser.add_argument('--fs', type=int, default=16000, help='Sampling rate of Dataset')
    parser.add_argument('-ds', '--dataset', type=int, default=1, help='Dataset to Load')
    parser.add_argument('-dir', '--data_dir', type=str, default='MedleyDB', help='name of file')
    args = parser.parse_args()

    dataset = dataset_select(args.dataset, args.fs)
    id_list = dataset.get_ids()

    # load audio
    songs = []
    f0_list = []
    for i, s in enumerate(id_list[:5]):
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
        #print("F0 interpolated shape: ", f0_new)
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


    # save CQT to file
    np.save()




if __name__ == "__main__":
    generate_data()