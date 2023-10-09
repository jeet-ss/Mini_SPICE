import math
import random
import numpy as np
import pandas as pd
import torch
import sys
import os 
import librosa
from tqdm import tqdm
from utils.model import Spice_model
import os
import tensorflow as tf
import tensorflow_hub as hub

MAX_ABS_INT16 = 32768.0
os.environ['http_proxy']="http://proxy:80"
os.environ['https_proxy']="http://proxy:80"

class Calibrator_SPICE():
    def __init__(self, model,CQT=True, M=5,low=110,high=440):
        self.model = model
        self.CQT = CQT
        self.M = M
        self.low = low
        self.high = high
        print(self.CQT)
        # if isinstance(self.model, torch.nn.Module):
        #     print("******* Model Loaded for Calibration *******\n")
        # else:
        #     sys.exit("Fatal : Model is not supported for calibration")

    def _pick_semitone_freq(self):
        semitone_ratio = 2 ** (1/12)
        semitone_freq_list = []
        
        current_freq = self.low
        while current_freq <= self.high:
            semitone_freq_list.append(current_freq)
            current_freq *= semitone_ratio
        
        return random.choice(semitone_freq_list)
    
    def _generate_harmonic_wave(self,fundamental_freq, sampling_rate=16000, N=11, H=512):
        amplitude_f0 = np.random.normal(0, 1)
        t = np.linspace(0, (N*H) / sampling_rate, (N*H), endpoint=False)
        fundamental_signal = amplitude_f0*np.sin(2 * np.pi * fundamental_freq * t)
        harmonic_signal = fundamental_signal.copy()
        

        for i in range(2, 4):  
            frequency = fundamental_freq * i
            random_phase = random.uniform(0, 2 * math.pi)
            harmonic_signal += random.uniform(0, 1) * np.sin(2 * np.pi * frequency * t+random_phase)

        return harmonic_signal
    
    def _generate_samples(self):
        print("******* Generating Audio Samples *******\n")
        samples = []
        labels = []
        for i in range(self.M):
            f0 = self._pick_semitone_freq()
            wave = self._generate_harmonic_wave(f0)
            samples.append(wave)
            labels.append(f0)
        
        self.samples, self.labels = np.array(samples), np.array(labels)
        print("******* Generating Audio Samples Completed *******\n")
        
    def _get_cqt(self):
        print("******* Generating CQT Samples *******\n")
        Cqtt = np.zeros((1, 190))
        F0 = np.zeros(1)
        for s, f in zip(self.samples, self.labels):
            C = np.abs(librosa.cqt(s, sr=16000, hop_length=512, 
                        fmin= librosa.note_to_hz('C1'),
                        n_bins=190, bins_per_octave=24))
            Cqtt = np.vstack((Cqtt, C.T))
            f0 = np.repeat(f, C.shape[1])
            F0 = np.concatenate((F0, f0))
        Cqtt = Cqtt[1:, :]
        F0 = F0[1:].reshape(-1, 1)
        self.cqt_data = np.hstack((Cqtt, F0))
        print("******* Generating CQT Samples Completed *******\n")
        
    def _get_eqn(self):
        print("******* Generating Linear Equations *******\n")
        A = []
        B = []
        if self.CQT:
            for row in tqdm(self.cqt_data):
                pitch_h1,conf_h1,x_hat1 = self.model(torch.from_numpy(row[0:128].reshape(1,128)).float())
                coeff_x = [1,pitch_h1.detach().numpy()[0][0]]
                y = 12*math.log(row[-1]/10,2)
                A.append(coeff_x)
                B.append(y)
            self.A = np.array(A[6::12])
            self.B = np.array(B[6::12])
        else:
            print("Here")
            for s, f in zip(self.samples, self.labels):
                model_output = self.model.signatures["serving_default"](tf.constant(list(s), tf.float32))
                pitch_outputs = model_output["pitch"]
                pitch_outputs = [ float(x) for x in pitch_outputs]
                y = 12*math.log(f/10,2)
                A.append(pitch_outputs)
                B.append(y)

            A = [item for sublist in A for item in sublist]    
            self.A = np.hstack((np.ones((self.M, 1)),np.array(A[6::12]).reshape(self.M,1)))
            self.B = np.array(B)
            print(self.A)
        print("******* Calibration Equations *******\n")
        for x,y in zip(self.A,self.B):
            eq = "b + s*{} = {}\n".format(x[1], y) 
            print(eq)
        
    def get_values(self):
        self._generate_samples()
        if self.CQT:
            self._get_cqt()
        self._get_eqn()
        print("******* Solving Equations *******\n")
        x, residuals, rank, singular_values = np.linalg.lstsq(self.A,self.B, rcond=None)
        b,s = x[0],x[1]
        print("******* Equations Solved *******\n")
        print("Value of PT_OFFSET :{}, Value of PT_SLOPE: {}\n".format(b,s))
        return b,s
    
    def get_data(self):
        return self.A, self.B
    