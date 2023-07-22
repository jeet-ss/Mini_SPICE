import torch
import os
import numpy as np
from utils.logger import Logger

# Global variables
LOG_EVERY_N_STEPS = 100

# CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.DoubleTensor if USE_CUDA else torch.DoubleTensor
dlongtype = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
device = 'cuda' if USE_CUDA else 'cpu'

# logging
logger = Logger('./logs')

# helper functions
def to_np(x):
    return x.data.cpu().numpy() 



class Trainer:
    
    def __init__(self, 
                model,        # model to train
                loss_pitch,     # Loss function
                loss_recons,    
                loss_conf,
                optim,          # Optimizer
                train_ds,       # Train dataset
                val_test_ds,    # Validation dataset
                w_pitch,        # weight for pitch loss
                w_recon,        # weight for recons loss
                sigma,        # pitch difference scaling factor
                ):
        self._model = model.type(dtype)
        self._lossPitch = loss_pitch.type(dtype)
        self._lossRecons = loss_recons.type(dtype)
        self._lossConf = loss_conf.type(dtype)
        self._optim = optim
        self._trainDs = train_ds#.type(dtype)
        self._valDs = val_test_ds#.type(dtype)
        self.w_pitch = w_pitch#.type(dtype)
        self.w_recon = w_recon#.dtype(dtype)
        self.sigma = sigma#.dtype(dtype)
        #
        self.epoch_counter = 0
        # path for saving and loading models
        self.model_path = os.path.join(os.path.abspath(os.getcwd()), "M_checkpoints")

    def save_checkpoint(self, epoch):
        # create path
        
        if os.path.isdir(self.model_path) != True:
            os.makedirs(self.model_path)
        torch.save({'state_dict': self._model.state_dict()}, os.path.join(self.model_path, 'checkpoint_{:03d}.ckp'.format(epoch)))

    def restore_checkpoint(self, epoch):
        checkpoint = torch.load(os.path.join(self.model_path, 'checkpoint_{:03d}.ckp'.format(epoch)), 'cuda' if USE_CUDA else None)
        self._model.load_state_dict(checkpoint['state_dict'])

    def save_model_onnx(self, filePath):
        '''raise NotImplementedError'''
        m = self._model.cpu()
        m.eval()
        x = torch.randn(1, 128, requires_grad=True)
        p, c, h = self._model(x)
        torch.onnx.export(m,                                    # model being run
              x,                                                # model input (or a tuple for multiple inputs)
              filePath,                                         # where to save the model (can be a file or file-like object)
              export_params=True,                               # store the trained parameter weights inside the model file
              opset_version=10,                                 # the ONNX version to export the model to
              do_constant_folding=True,                         # whether to execute constant folding for optimization
              input_names = ['input'],                          # the model's input names
              output_names = ['output'],                        # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},       # variable length axes
                            'output' : {0 : 'batch_size'}})
        

    def train_step(self, x_batch):
        """
            x_batch : each batch element is a three-tuple of two slices of 128 dim and (kt,1 - kt,2)
        """
        # reset grad to zero
        if USE_CUDA:
            torch.cuda.empty_cache()
        self._optim.zero_grad()
        #
        print("before ", len(x_batch) )
        pitch_diff, x_1, x_2, f0 = x_batch
        pitch_diff = pitch_diff.type(dtype)
        x_1 = x_1.type(dtype)
        x_2 = x_2.type(dtype)
        
        # model 
        pitch_H_1, conf_H_1, hat_x_1 = self._model(x_1)
        pitch_H_2, conf_H_2, hat_x_2 = self._model(x_2)
        
        # calculate loss
        print('in train', pitch_H_1.size(), pitch_diff.size(), pitch_H_2.size(), self.sigma)
        pitch_error = torch.abs((pitch_H_1.squeeze() - pitch_H_2.squeeze()) - self.sigma*pitch_diff)
        print('train 2 ', pitch_error.size())
        lossPitch = self._lossPitch(pitch_error)  
        # conf head loss
        lossConf = self._lossConf(conf_H_1, conf_H_2, pitch_error, self.sigma)
        # take care of reshape
        if x_1.size() != hat_x_1.size():
            hat_x_1 = torch.reshape(hat_x_1, (hat_x_1.size()[0], -1))
            hat_x_2 = torch.reshape(hat_x_2, (hat_x_1.size()[0], -1))
        # Decoder Loss
        lossRecons = self._lossRecons(x_1, x_2, hat_x_1, hat_x_2)
        lossTotal = self.w_pitch*lossPitch + self.w_recon*lossRecons
        #
        pitch_diff.detach()
        x_1.detach()
        x_2.detach()
        pitch_H_1.detach()
        pitch_H_2.detach()
        conf_H_1.detach()
        conf_H_2.detach()
        hat_x_1.detach()
        hat_x_2.detach()

        #
        ''' should I train the conf head while training the pitch head '''
        # freeze conf head
        self._model.enc_block.conf_head.weight.requires_grad = False
        # Backprop
        # for n, param in self._model.named_parameters():
        #     if param.requires_grad == False:
        #         print ('111111',n)
        lossTotal.backward(retain_graph=True)
        ''' Do I need to pass gradient for this algebraic loss func also?? '''
        # Update weights
        #self._optim.step()
        # freeze network for conf head
        for param in self._model.parameters():
            param.requires_grad = False
        # unfreeze conf head
        self._model.enc_block.conf_head.weight.requires_grad = True
        # update conf head weights
        # for n, param in self._model.named_parameters():
        #     if param.requires_grad:
        #         print ('222222',n, param.data)
        lossConf.backward()
        # update weights
        self._optim.step()
        # unfreeze model
        for param in self._model.parameters():
            param.requires_grad = True
        # return 
        return lossTotal.detach().item()

    def val_step(self, x_batch):
        pitch_diff, x_1, x_2, f0 = x_batch
        # predict
        pitch_H_1, conf_H_1, hat_x_1 = self._model(x_1)
        pitch_H_2, conf_H_2, hat_x_2 = self._model(x_2)
        # calculate frequency from pitch 
        # some function
        freq_0 = ()
        # calc difference to f0
        diff = np.abs(freq_0 - f0)
        return diff


    def train_epoch(self):
        # set training mode
        self._model.training = True
        # iterate through the training set
        loss = 0
        for b in self._trainDs:
            # if USE_CUDA:
            #         b = b.cuda()
            # x is One batch of data
            
            loss += self.train_step(b)
            
        # calculate avg batch loss for logging
        avg_loss = loss/self._trainDs.__len__()
        return avg_loss


    def val_test_epoch(self):
        #
        #self._model.eval()
        #self._valDs = self._valDs.cuda()
        print("val ds size", self._valDs.__len__())
        # itr through val set
        with torch.no_grad():
            for b in self._valDs:
                # if USE_CUDA:
                #     b = b.cuda()

                loss = self.val_step(b)


    def fit_model(self, epochs = -1):
        assert epochs > 0, 'Epochs > 0'
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
            train_loss = self.val_test_epoch()
            loss_train = np.append(loss_train, train_loss)
            print("loss", epoch_counter , train_loss)
            logger.scalar_summary("loss", train_loss, epoch_counter)
            #
            if train_loss < min_loss:
                
                min_loss = train_loss
                self.save_checkpoint(epoch_counter)
            
        return loss_train