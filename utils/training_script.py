import torch
import os
import numpy as np
from utils.logger import Logger
import tqdm

# Global variables
LOG_EVERY_N_STEPS = 100

# CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.DoubleTensor
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
                name_variant,   # string to create diff logs
                ):
        self._model = model.type(dtype)
        self._lossPitch = loss_pitch.type(dtype)
        self._lossRecons = loss_recons.type(dtype)
        self._lossConf = loss_conf.type(dtype)
        self._optim = optim
        self._trainDs = train_ds
        self._valDs = val_test_ds
        self.w_pitch = w_pitch
        self.w_recon = w_recon
        self.sigma = sigma
        #
        self.epoch_counter = 0
        logger.update_dir('l_'+name_variant)
        # path for saving and loading models
        self.model_path = os.path.join(os.path.abspath(os.getcwd()), name_variant+"checkpoints")

    def save_checkpoint(self, epoch):
        # create path
        if os.path.isdir(self.model_path) != True:
            os.makedirs(self.model_path)
        torch.save({'state_dict': self._model.state_dict()}, os.path.join(self.model_path, 'checkpoint_{:03d}.ckp'.format(epoch)))

    def restore_checkpoint(self, epoch):
        checkpoint = torch.load(os.path.join(self.model_path, 'checkpoint_{:03d}.ckp'.format(epoch)), 'cuda' if USE_CUDA else None)
        self._model.load_state_dict(checkpoint['state_dict'])

    def train_step(self, x_batch, batch_counter,):
        """
            This is one batch of training 
            x_batch : Input mini-batch data, each element is a three-tuple of two slices of 128 dim and (kt,1 - kt,2)
        """
        # reset grad to zero
        if USE_CUDA:
            torch.cuda.empty_cache()
        self._optim.zero_grad()
        #
        pitch_diff, x_1, x_2, f0 = x_batch
        pitch_diff = pitch_diff.type(dtype)
        x_1 = x_1.type(dtype)
        x_2 = x_2.type(dtype)
        
        # pass through the model 
        pitch_H_1, conf_H_1, hat_x_1 = self._model(x_1)
        pitch_H_2, conf_H_2, hat_x_2 = self._model(x_2)
        
        # calculate loss
        # Huber Loss
        pitch_hat_diff = torch.subtract(pitch_H_1.squeeze(), pitch_H_2.squeeze())
        pitch_diff = self.sigma*pitch_diff
        lossPitch = self._lossPitch(pitch_hat_diff, pitch_diff)

        # conf head loss
        pitch_error = torch.abs((pitch_H_1.squeeze() - pitch_H_2.squeeze()) - pitch_diff)
        lossConf = self._lossConf(conf_H_1, conf_H_2, pitch_error, self.sigma)

        # Decoder Loss
        # take care of reshape
        if x_1.size() != hat_x_1.size():
            hat_x_1 = torch.reshape(hat_x_1, (hat_x_1.size()[0], -1))
            hat_x_2 = torch.reshape(hat_x_2, (hat_x_1.size()[0], -1))
        lossRecons = self._lossRecons(x_1, x_2, hat_x_1, hat_x_2)

        # Total Loss
        lossTotal = self.w_pitch*lossPitch + self.w_recon*lossRecons

        # detach everything except loss
        pitch_diff.detach()
        x_1.detach()
        x_2 = x_2.detach()
        pitch_H_1 = pitch_H_1.cpu().detach()
        pitch_H_2 = pitch_H_2.cpu().detach()
        conf_H_1 = conf_H_1.cpu().detach()
        conf_H_2 = conf_H_2.cpu().detach()
        hat_x_1 = hat_x_1.cpu().detach()
        hat_x_2 = hat_x_2.cpu().detach()

        # freeze conf head
        self._model.enc_block.conf_head.weight.requires_grad = False
        lossTotal.backward(retain_graph=True)
        # freeze network for conf head
        for param in self._model.parameters():
            param.requires_grad = False
        # unfreeze conf head
        self._model.enc_block.conf_head.weight.requires_grad = True
        lossConf.backward()

        ### for excluding conf loss
        ###lossTotal.backward()

        # update weights
        self._optim.step()
        # unfreeze model
        for param in self._model.parameters():
            param.requires_grad = True
        # return 
        return lossTotal.detach().item(), lossConf.detach().item(), lossRecons.detach().item(), lossPitch.detach().item()

    def val_step(self, x_batch):
        '''
            Performs validation for one batch 
            As the method is Self-supervised, we perform validation by calculating the loss on the unseen data
                and note if they converge similar to training data
            x_batch : Input mini-batch data, each element is a three-tuple of two slices of 128 dim and (kt,1 - kt,2)
        '''
        pitch_diff, x_1, x_2, f0 = x_batch
        pitch_diff = pitch_diff.type(dtype)
        x_1 = x_1.type(dtype)
        x_2 = x_2.type(dtype)
        # predict
        pitch_H_1, conf_H_1, hat_x_1 = self._model(x_1)
        pitch_H_2, conf_H_2, hat_x_2 = self._model(x_2)

        # calculate error
        
        # pitch loss
        pitch_hat_diff = torch.subtract(pitch_H_1.squeeze(), pitch_H_2.squeeze())
        pitch_diff = self.sigma*pitch_diff
        lossPitch = self._lossPitch(pitch_hat_diff, pitch_diff)

        # Conf loss
        pitch_error = torch.abs((pitch_H_1.squeeze() - pitch_H_2.squeeze()) - pitch_diff)
        lossConf = self._lossConf(conf_H_1, conf_H_2, pitch_error, self.sigma)

        # Decoder loss
        # take care of reshape
        if x_1.size() != hat_x_1.size():
            hat_x_1 = torch.reshape(hat_x_1, (hat_x_1.size()[0], -1))
            hat_x_2 = torch.reshape(hat_x_2, (hat_x_1.size()[0], -1))
        lossRecons = self._lossRecons(x_1, x_2, hat_x_1, hat_x_2)

        # Total loss
        lossTotal = self.w_pitch*lossPitch + self.w_recon*lossRecons

        # detach everything 
        pitch_H_1 = pitch_H_1.cpu().detach().numpy().squeeze()
        pitch_H_2 = pitch_H_2.cpu().detach()
        conf_H_1 = conf_H_1.cpu().detach()
        conf_H_2 = conf_H_2.cpu().detach()
        hat_x_1 = hat_x_1.cpu().detach()
        hat_x_2 = hat_x_2.cpu().detach()
        
        return lossTotal.detach().item(), pitch_error.cpu().detach().numpy().mean(), lossConf.detach().item(), lossRecons.detach().item(), lossPitch.detach().item(), np.mean(pitch_H_1[pitch_H_1>1])
        
    def train_epoch(self):
        '''
            Trains the model for one epoch
            returns the total loss, conf los, recons loss, and pitch loss
        '''
        # set training mode
        self._model.train()
        # iterate through the training set
        loss_Total = 0
        loss_Conf = 0
        loss_Pitch = 0
        loss_Recons = 0
        batch_counter = 0
        for b in self._trainDs:
            t_loss, c_loss, r_loss, p_loss = self.train_step(b, batch_counter)
            loss_Total += t_loss
            loss_Conf += c_loss
            loss_Recons += r_loss
            loss_Pitch += p_loss
            batch_counter += 1
            
        # calculate avg batch loss for logging
        avg_loss_t = loss_Total/self._trainDs.__len__()
        avg_loss_c = loss_Conf/self._trainDs.__len__()
        avg_loss_r = loss_Recons/self._trainDs.__len__()
        avg_loss_p = loss_Pitch/self._trainDs.__len__()

        return avg_loss_t, avg_loss_c, avg_loss_r, avg_loss_p


    def val_test_epoch(self, batch_data, mode='val'):
        '''
            Validates the model for one epoch
            returns the total loss, conf los, recons loss, and pitch loss

            If mode is 'encoder out' , gives only the encoder outputs for evaluation
        '''

        self._model.eval()
        loss_Total = 0
        loss_Conf = 0
        loss_Pitch = 0
        loss_Recons = 0
        p_error = 0
        pHead = 0
        y_hat1 = np.zeros((64,1))
        y_hat2 = np.zeros((64,1))
        # itr through val set
        with torch.no_grad():
            for b in batch_data:
                # if USE_CUDA:
                #     b = b.cuda()
                if mode == 'val':
                    t_loss, pitch_error, c_loss, r_loss, p_loss, phead  = self.val_step(b)
                    loss_Total += t_loss
                    loss_Conf += c_loss
                    loss_Recons += r_loss
                    loss_Pitch += p_loss
                    p_error += pitch_error
                    pHead += phead
                if mode == 'test':
                    pass
                if mode == 'encoder_out':
                    p1, p2 = self.test_step(b)
                    y_hat1 = np.vstack((y_hat1, p1))
                    y_hat2 = np.vstack((y_hat2, p2))

        if mode == 'val':
            avg_loss_t = loss_Total/batch_data.__len__()
            avg_pitchError = p_error/batch_data.__len__()
            avg_loss_r = loss_Recons/batch_data.__len__()
            avg_loss_c = loss_Conf/batch_data.__len__()
            avg_loss_p = loss_Pitch/batch_data.__len__()
            avgPhead = pHead/batch_data.__len__()

            return avg_loss_t, avg_pitchError, avg_loss_c, avg_loss_r, avg_loss_p, avgPhead
        if mode == 'encoder_out':
            return {'yhat1': y_hat1[64:],
                    'yhat2': y_hat2[64:]}

    

    def fit(self, epochs_start=1, epochs_end=0):
        '''
            Fits the data to the model
            Returns all loss values in a dictionary format
        '''
        epochs = epochs_end - epochs_start
        assert epochs > 0, 'Epochs > 0'
        #
        loss_train_total = np.array([])
        loss_train_conf = np.array([])
        loss_train_recons = np.array([])
        loss_train_pitch = np.array([])
        loss_val_total = np.array([])
        loss_val_conf = np.array([])
        loss_val_recons = np.array([])
        loss_val_pitch = np.array([])
        
        min_loss = np.Inf
        #
        for i in range(epochs_start, epochs_end):
            # increment Counter
            self.epoch_counter = i
            # train for an epoch and then calculate the loss and metrics on the validation set
            train_loss_t, train_loss_c, train_loss_r, train_loss_p = self.train_epoch()
            # save
            loss_train_total = np.append(loss_train_total, train_loss_t)
            loss_train_conf = np.append(loss_train_conf, train_loss_c)
            loss_train_recons = np.append(loss_train_recons, train_loss_r)
            loss_train_pitch = np.append(loss_train_pitch, train_loss_p)
            # log
            logger.scalar_summary("train_Total_loss", train_loss_t, i)
            logger.scalar_summary("train_Conf_loss", train_loss_c, i)
            logger.scalar_summary("train_Recons_loss", train_loss_r, i)
            logger.scalar_summary("train_Pitch_loss", train_loss_p, i)
            # validation
            val_loss_t, p_error, val_loss_c, val_loss_r, val_loss_p, pHead = self.val_test_epoch(self._valDs, mode='val')
            # save
            loss_val_total = np.append(loss_val_total, val_loss_t)
            loss_val_conf = np.append(loss_val_conf, val_loss_c)
            loss_val_recons = np.append(loss_val_recons, val_loss_r)
            loss_val_pitch = np.append(loss_val_pitch, val_loss_p)
            # log
            logger.scalar_summary("val_Total_loss", val_loss_t, i)
            logger.scalar_summary("pitch_Error", p_error, i)
            logger.scalar_summary("val_Conf_loss", val_loss_c, i)
            logger.scalar_summary("val_Recons_loss", val_loss_r, i)
            logger.scalar_summary("val_Pitch_loss", val_loss_p, i)
            logger.scalar_summary("val_Pitch_HEad", pHead, i)
            
            # print after 20 epochs
            if i%2==0:
                print(f"Epoch:{i}, trainL:{train_loss_t}, valL:{val_loss_t}")

            # save checkpoint if better
            if train_loss_t < min_loss:
                
                min_loss = train_loss_t
                self.save_checkpoint(i)
            
        return {
            'min_Loss':min_loss,
            'train_loss_total':loss_train_total,
            'train_loss_conf': loss_train_conf,
            'train_loss_recons':loss_train_recons,
            'train_loss_pitch':loss_train_pitch,
            'val_loss_total':loss_val_total,
            'val_loss_conf':loss_val_conf,
            'val_loss_recons':loss_val_recons,
            'val_loss_pitch':loss_val_pitch
            }