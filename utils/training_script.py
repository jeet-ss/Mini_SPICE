import torch
import numpy as np
#from logger import Logger

# Global variables
LOG_EVERY_N_STEPS = 100

# CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

# logging
#logger = Logger('./logs')

# helper functions
def to_np(x):
    return x.data.cpu().numpy() 

def scaling_factor():
    # 1 / (Q * torch.log2(fmax / fmin))
    return 1/2

class Trainer:
    
    def __init__(self, 
                encoder,        # model to train
                decoder,        
                loss_pitch,     # Loss function
                loss_recons,    
                optim,          # Optimizer
                train_ds,       # Train dataset
                val_test_ds,    # Validation dataset
                w_pitch,        # weight for pitch loss
                w_recon,        # weight for recons loss
                ):
        self._encoder = encoder.type(dtype)
        self._decoder = decoder.type(dtype)
        self._lossPitch = loss_pitch.type(dtype)
        self._lossRecons = loss_recons.type(dtype)
        self._optim = optim
        self._trainDs = train_ds.dtype(dtype)
        self._valDs = val_test_ds.dtype(dtype)
        self.w_pitch = w_pitch.dtype(dtype)
        self.w_recon = w_recon.dtype(dtype)
        #
        self.epoch_counter = 0

    def save_checkpoint(self, epoch):
        pass

    def restore_checkpoint(Self, epoch):
        pass

    def save_model(self, filePath):
        pass

    def train_step(self, x_batch):
        """
            x_batch : each batch element is a three-tuple of two slices of 128 dim and (kt,1 - kt,2)
        """
        # reset grad to zero
        self._optim.zero_grad()
        #
        x_1 = x_batch[:, 0]
        x_2 = x_batch[:, 1]
        # Encoder 
        pitch_H_1, conf_H_1, indx_mat_list_1 = self._encoder(x_1)
        pitch_H_2, conf_H_2, indx_mat_list_2 = self._encoder(x_2)
        # calculate loss
        pitch_error = torch.abs((pitch_H_1 - pitch_H_2) - scaling_factor()*x_batch[:, 2])
        lossPitch = self._lossPitch(pitch_error)  
        # Decoder 
        hat_x_1 = self._decoder(pitch_H_1, indx_mat_list_1)
        hat_x_2 = self._decoder(pitch_H_2, indx_mat_list_2)
        # take care of reshape
        if x_1.size() != hat_x_1.size():
            pass
        # Decoder Loss
        lossRecons = self._lossRecons(x_1, x_2, hat_x_1, hat_x_2)
        lossTotal = self.w_pitch*lossPitch + self.w_recon*lossRecons
        # Backprop
        lossTotal.backward()
        # Update weights
        self._optim.step()
        # return 
        return lossTotal

    def val_step(self, x_batch):
        pass

    def train_epoch(self):
        # set training mode
        self._model.training = True
        # iterate through the training set
        loss = 0
        for x in self._trainDs:
            # x is One batch of data
            loss += self.train_step(x)
        # calculate avg batch loss for logging
        avg_loss = loss/self._trainDs.__len__()
        return avg_loss



    def val_test_epoch(self):
        pass

    def fit_model(self, epochs = -1):
        assert epochs > 0
        #
        loss_train = np.array([])
        loss_val = np.array([])
        epoch_counter = 0
        min_loss = np.Inf
        #
        while True:
            # stop by epoch number
            if epoch_counter >= epochs:
                break
            # increment Counter
            epoch_counter += 1
            self.epoch_counter = epoch_counter
            # train for an epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            loss_train = np.append(loss_train, to_np(train_loss))
            #
            if train_loss < min_loss:
                min_loss = train_loss
                self.save_checkpoint(epoch_counter)
            
        return loss_train