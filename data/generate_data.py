import numpy as np
import librosa
import argparse
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
    for i, s in enumerate(id_list[:5]):
        song, _ = dataset.load_data(s)
        print(song.shape, s)
        # convert stereo to mono
        songs.append(librosa.to_mono(song))

    # Convert to CQT array and concat
    Cqtt = np.zeros((190, 1))
    for i, s in enumerate(songs):
        C = np.abs(librosa.cqt(s, sr=args.fs, hop_length=512, 
                    #window=librosa.filters.get_window('hann', Nx=1024, fftbins=False), 
                    fmin= librosa.note_to_hz('C1'),
                    n_bins=190, bins_per_octave=24))
        print(C.shape)
        Cqtt = np.hstack((Cqtt, C))
    print("CQT Shape: ", Cqtt.shape)

    # save CQT to file
    np.save()




if __name__ == "__main__":
    generate_data()