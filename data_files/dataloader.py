"""
Load F0 annotations and audio files from the MedleyDB dataset in a unified format

Author: Simon Schwär <simon.schwaer@audiolabs-erlangen.de>
Author: Franca Bittner <franca.bittner@tu-ilmenau.de>
Date: 01.06.2023
"""

import numpy as np
import soundfile as sf
import resampy
import os
import pathlib
import scipy



class DataLoader:
    fs = None # must be set by derived classes
    tol = 0.001 # tolerance for equal timestamps (in seconds)

    def get_num_elements(self):
        """ Return the number of elements in the dataset
        """
        raise NotImplementedError

    def get_ids(self):
        """ Get a list of the IDs of all elements in this dataset
        """
        raise NotImplementedError

    def load_data(self, file_id : str):
        """ Load the (possibly resampled) audio data and F0 annotations for a given ID

        Returns
        =======
            x : np.ndarray
                A shape (C, L) numpy array for time domain audio with C channels and L samples
            f0 : np.ndarray
                A shape (3, N) numpy array for N frames. First row is time in seconds,
                second row is whether the frame is voiced (1) or unvoiced (0),
                third row is the F0 value in Hz (0 if unvoiced)
        """
        raise NotImplementedError


    def _load_audio(self, path):
        """Load an audio file and potentially resample it to the global target sampling rate

        Parameters
        ==========
        path : string
            absolute or relative path to the audio file

        Returns
        =======
        y : numpy array
            2D numpy array with shape (L, C), where C is the number of channels in the audio file
            and L is the number of samples
        """
        assert self.fs is not None

        x, _fs = sf.read(path)
        if _fs != self.fs:
            #print("Resampling \"%s\" from %i Hz to %i Hz." % (path, _fs, self.fs))
            y = resampy.core.resample(x, _fs, self.fs, filter="kaiser_best", axis=0)
        else:
            y = x

        if len(y.shape) == 1: # add channel dimension if it is missing
            y = np.expand_dims(y, axis=1)

        return y.T # soundfile has channel dimension last but we want it first

    def _isin(self, a, b):
        """Check whether each element of a is in b, within a tolerance tol

        Adapted from https://stackoverflow.com/questions/51744613/numpy-setdiff1d-with-tolerance-comparing-a-numpy-array-to-another-and-saving-o/
        """
        idx = np.searchsorted(b, a)

        mask_l = (idx == len(b))
        idx[mask_l] = len(b) - 1
        lval = b[idx] - a
        lval[mask_l] *=-1

        mask_r = (idx == 0)
        idx1 = idx - 1
        idx1[mask_r] = 0
        rval = a - b[idx1]
        rval[mask_r] *=-1
        return np.minimum(lval, rval) <= self.tol

class MedleyDBLoader(DataLoader):
    def __init__(self, fs=44100, audio_loc="../datasets/MedleyDB/Audio", f0_loc="../datasets/MedleyDB/Pitch_Annotations", annot_hop=256, orig_fs=44100):
        """ Initialize the DataLoader for MedleyDB
        
        Parameters
        ==========
        fs : int
            the desired sampling rate. If it is different from the sampling rate of the audio files,
            they will be resampled upon loading
        audio_loc : string
            path to the audio files relative to this `dataloader.py`, probably doesn't have to be
            changed
        f0_loc : string
            path to the F0 files relative to this `dataloader.py`, probably doesn't have to be
            changed
        annot_hop : int
            hop size of the F0 annotations, probably doesn't have to be changed
        orig_fs : int
            original sampling rate of the data, probably doesn't have to be changed
        """

        self.fs = fs
        self.annot_dt = annot_hop / orig_fs

        db_path = pathlib.Path(__file__).parent.resolve()
        # self.audio_path = os.path.join(db_path, audio_loc)
        # self.f0_path = os.path.join(db_path, f0_loc)
        self.audio_path = audio_loc
        self.f0_path = f0_loc
        assert os.path.exists(self.audio_path), "Audio location not found."
        assert os.path.exists(self.f0_path), "F0 location not found."

        self.ids = []

        for f in os.listdir(self.audio_path):
            audio_file = os.path.join(self.audio_path, f)
            if os.path.isfile(audio_file) and f[-4:] == ".wav":
                fid = f[:-4]
                self.ids.append(fid)

    def get_num_elements(self):
        """ Return the number of elements in the dataset
        """
        return len(self.ids)

    def get_ids(self):
        """ Get a list of the IDs of all elements in this dataset
        """
        return self.ids

    def load_data(self, file_id : str):
        """ Load the (possibly resampled) audio data and F0 annotations for a given ID

        Returns
        =======
            x : np.ndarray
                A shape (C, L) numpy array for time domain audio with C channels and L samples
            f0 : np.ndarray
                A shape (3, N) numpy array for N frames. First row is time in seconds,
                second row is whether the frame is voiced (1) or unvoiced (0),
                third row is the F0 value in Hz (0 if unvoiced)
        """
        assert file_id in self.ids, "Unknown file_id."
        ap = os.path.join(self.audio_path, file_id + ".wav")
        fp = os.path.join(self.f0_path, file_id + ".csv")

        # load audio
        x = self._load_audio(ap)
        N = np.ceil(x.shape[1] / self.fs / self.annot_dt).astype(int)

        # load annotations
        # specifying cols because some annotation files contain mysterious values in a third column
        raw_data = np.loadtxt(fp, delimiter=",", usecols=[0,1]).T

        # reformat annotations
        f0 = np.zeros((3, N))
        f0[0,:] = np.arange(N) * self.annot_dt
        idx = np.where(self._isin(f0[0,:], raw_data[0,:]))
        f0[1,idx] = 1
        assert len(idx[0]) == len(raw_data[1,:]), "The item cannot be loaded due to an annotation length mismatch."
        f0[2,idx] = raw_data[1,:]

        return x, f0


class MDBMelodySynthLoader(DataLoader):
    def __init__(self, fs=44100, audio_loc="../datasets/MDB-melody-synth/audio_melody", f0_loc="../datasets/MDB-melody-synth/annotation_melody", annot_hop=128, orig_fs=44100):
        """ Initialize the DataLoader for MDB-melody-synth
        
        Parameters
        ==========
        fs : int
            the desired sampling rate. If it is different from the sampling rate of the audio files,
            they will be resampled upon loading
        audio_loc : string
            path to the audio files relative to this `dataloader.py`, probably doesn't have to be
            changed
        f0_loc : string
            path to the F0 files relative to this `dataloader.py`, probably doesn't have to be
            changed
        annot_hop : int
            hop size of the F0 annotations, probably doesn't have to be changed
        orig_fs : int
            original sampling rate of the data, probably doesn't have to be changed
        """

        self.fs = fs
        self.annot_dt = annot_hop / orig_fs

        db_path = pathlib.Path(__file__).parent.resolve()
        self.audio_path = os.path.join(db_path, audio_loc)
        self.f0_path = os.path.join(db_path, f0_loc)

        assert os.path.exists(self.audio_path), "Audio location not found."
        assert os.path.exists(self.f0_path), "F0 location not found."

        self.ids = []

        for f in os.listdir(self.audio_path):
            audio_file = os.path.join(self.audio_path, f)
            if os.path.isfile(audio_file) and f[-10:] == ".RESYN.wav":
                fid = f[:-10]
                self.ids.append(fid)

    def get_num_elements(self):
        """ Return the number of elements in the dataset
        """
        return len(self.ids)

    def get_ids(self):
        """ Get a list of the IDs of all elements in this dataset
        """
        return self.ids

    def load_data(self, file_id : str):
        """ Load the (possibly resampled) audio data and F0 annotations for a given ID

        Returns
        =======
            x : np.ndarray
                A shape (C, L) numpy array for time domain audio with C channels and L samples
            f0 : np.ndarray
                A shape (3, N) numpy array for N frames. First row is time in seconds,
                second row is whether the frame is voiced (1) or unvoiced (0),
                third row is the F0 value in Hz (0 if unvoiced)
        """
        assert file_id in self.ids, "Unknown file_id."
        ap = os.path.join(self.audio_path, file_id + ".RESYN.wav")
        fp = os.path.join(self.f0_path, file_id + ".RESYN.csv")

        # load audio
        x = self._load_audio(ap)

        # load annotations
        # specifying cols because some annotation files contain mysterious values in a third column
        raw_data = np.loadtxt(fp, delimiter=",", usecols=[0,1]).T
        N = raw_data.shape[1]

        # reformat annotations
        f0 = np.zeros((3, N))
        f0[0,:] = np.arange(N) * self.annot_dt
        idx = np.where(raw_data[1,:] > 1) # np.where(self._isin(f0[0,:], raw_data[0,:]))
        f0[1,idx] = 1
        f0[2,:] = raw_data[1,:]

        return x, f0


class MIR1KLoader(DataLoader):
    def __init__(self, fs=44100, audio_loc="../datasets/MIR-1K/Wavfile", f0_loc="../datasets/MIR-1K/PitchLabel",
                 voiced_loc="../datasets/MIR-1K/vocal-nonvocalLabel", annot_hop=320, orig_fs=16000):
        """ Initialize the DataLoader for MIR-1K
        
        Parameters
        ==========
        fs : int
            the desired sampling rate. If it is different from the sampling rate of the audio files,
            they will be resampled upon loading
        audio_loc : string
            path to the audio files relative to this `dataloader.py`, probably doesn't have to be
            changed
        f0_loc : string
            path to the F0 files relative to this `dataloader.py`, probably doesn't have to be
            changed
        f0_loc : string
            path to the voiced/unvoiced annotation files relative to this `dataloader.py`, probably 
            doesn't have to be changed
        annot_hop : int
            hop size of the F0 annotations, probably doesn't have to be changed
        orig_fs : int
            original sampling rate of the data, probably doesn't have to be changed
        """

        self.fs = fs
        self.annot_dt = annot_hop / orig_fs

        db_path = pathlib.Path(__file__).parent.resolve()
        # self.audio_path = os.path.join(db_path, audio_loc)
        # self.f0_path = os.path.join(db_path, f0_loc)
        # self.voiced_path = os.path.join(db_path, voiced_loc)
        self.audio_path = audio_loc
        self.f0_path = f0_loc
        self.voiced_path = voiced_loc
        assert os.path.exists(self.audio_path), "Audio location not found."
        assert os.path.exists(self.f0_path), "F0 location not found."
        assert os.path.exists(self.voiced_path), "Voiced/Unvoiced location not found."

        self.ids = []

        for f in os.listdir(self.audio_path):
            audio_file = os.path.join(self.audio_path, f)
            if os.path.isfile(audio_file) and f[-4:] == ".wav":
                fid = f[:-4]
                self.ids.append(fid)

    def get_num_elements(self):
        """ Return the number of elements in the dataset
        """
        return len(self.ids)

    def get_ids(self):
        """ Get a list of the IDs of all elements in this dataset
        """
        return self.ids

    def load_data(self, file_id : str):
        """ Load the (possibly resampled) audio data and F0 annotations for a given ID

        Returns
        =======
            x : np.ndarray
                A shape (C, L) numpy array for time domain audio with C channels and L samples
            f0 : np.ndarray
                A shape (3, N) numpy array for N frames. First row is time in seconds,
                second row is whether the frame is voiced (1) or unvoiced (0),
                third row is the F0 value in Hz (0 if unvoiced)
        """
        assert file_id in self.ids, "Unknown file_id."
        ap = os.path.join(self.audio_path, file_id + ".wav")
        fp = os.path.join(self.f0_path, file_id + ".pv")
        vp = os.path.join(self.voiced_path, file_id + ".vocal")

        # load audio
        x = self._load_audio(ap)
        x = x[1:,:] # only use right channel, left channel is backing track
        N = np.floor(x.shape[1] / self.fs / self.annot_dt).astype(int) - 1

        # load annotations
        raw_f0 = np.loadtxt(fp, usecols=[0])
        raw_f0_unvoiced = np.where(raw_f0 == 0) # store original pitch values of 0 (possibly different from voicing labels)
        raw_f0 = 440 * np.power(2, (raw_f0 - 69)/12) # convert from MIDI pitch to Hz
        raw_f0[raw_f0_unvoiced] = 0 # reset unvoiced frames to 0Hz
        raw_voiced = np.loadtxt(vp, usecols=[0])

        # reformat annotations
        f0 = np.zeros((3, N))
        f0[0,:] = np.arange(1, N+1) * self.annot_dt
        f0[1,:] = raw_voiced
        f0[2,:] = raw_f0

        return x, f0

    
    
class DatagenLoader(DataLoader):
    def __init__(self, fs=44100, audio_loc="../datasets/pre_post/wavfile", f0_loc="../datasets/pre_post/labels", annot_hop=320, orig_fs=16000):
        """ Initialize the DataLoader for MIR-1K
        
        Parameters
        ==========
        fs : int
            the desired sampling rate. If it is different from the sampling rate of the audio files,
            they will be resampled upon loading
        audio_loc : string
            path to the audio files relative to this `dataloader.py`, probably doesn't have to be
            changed
        f0_loc : string
            path to the F0 files relative to this `dataloader.py`, probably doesn't have to be
            changed
        annot_hop : int
            hop size of the F0 annotations, probably doesn't have to be changed
        orig_fs : int
            original sampling rate of the data, probably doesn't have to be changed
        """

        self.fs = fs
        self.annot_dt = annot_hop / orig_fs

        db_path = pathlib.Path(__file__).parent.resolve()
        # self.audio_path = os.path.join(db_path, audio_loc)
        # self.f0_path = os.path.join(db_path, f0_loc)
        # self.voiced_path = os.path.join(db_path, voiced_loc)
        self.audio_path = audio_loc
        self.f0_path = f0_loc
        # self.voiced_path = voiced_loc
        assert os.path.exists(self.audio_path), "Audio location not found."
        assert os.path.exists(self.f0_path), "F0 location not found."
        # assert os.path.exists(self.voiced_path), "Voiced/Unvoiced location not found."

        self.ids = []

        for f in os.listdir(self.audio_path):
            audio_file = os.path.join(self.audio_path, f)
            if os.path.isfile(audio_file) and f[-4:] == ".wav":
                fid = f[:-4]
                self.ids.append(fid)

    def get_num_elements(self):
        """ Return the number of elements in the dataset
        """
        return len(self.ids)

    def get_ids(self):
        """ Get a list of the IDs of all elements in this dataset
        """
        return self.ids

    def load_data(self, file_id : str):
        """ Load the (possibly resampled) audio data and F0 annotations for a given ID

        Returns
        =======
            x : np.ndarray
                A shape (C, L) numpy array for time domain audio with C channels and L samples
            f0 : np.ndarray
                A shape (3, N) numpy array for N frames. First row is time in seconds,
                second row is whether the frame is voiced (1) or unvoiced (0),
                third row is the F0 value in Hz (0 if unvoiced)
        """
        assert file_id in self.ids, "Unknown file_id."
        ap = os.path.join(self.audio_path, file_id + ".wav")
        fp = os.path.join(self.f0_path, file_id + ".csv")
        # vp = os.path.join(self.voiced_path, file_id + ".vocal")

        # load audio
        x = self._load_audio(ap)
        N = np.ceil(x.shape[1] / self.fs / self.annot_dt).astype(int)
        raw_data = np.loadtxt(fp, delimiter=",", usecols=[0,2]).T

        # reformat annotations
        f0 = np.zeros((3, N))
        print(self.annot_dt)
        f0[0,:] = np.arange(N) * self.annot_dt
        idx = np.where(self._isin(f0[0,:], raw_data[0,:]))
        f0[1,idx] = 1
        # interpolator = scipy.interpolate.interp1d(x=raw_data[0,:], y=raw_data[1,:], axis=0, fill_value = 'extrapolate')
        # interp_time = np.arange(0, len(idx[0]), 1)*self.annot_dt
        # f0_new = interpolator(interp_time)
        print(len(idx[0]),len(raw_data[1,:]))
        assert len(idx[0]) == len(raw_data[1,:]), "The item cannot be loaded due to an annotation length mismatch."
        f0[2,idx] = raw_data[1,:]

        return x, f0