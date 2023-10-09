import torch
from torch import nn
from utils.model import Spice_Encoder

    
class Deconv_block(nn.Module):
    def __init__(self, in_channels=64, out_channels=32, filter_size=3, stride=1, 
                padding=1, batch_norm=False):
        super().__init__()
        """
            This deconv block does not do MaxUnpooling
        """
        #
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=filter_size, stride=stride, padding=padding, bias=False)
        self.batchNorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        # self.seq_deconv = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels, out_channels, kernel_size=filter_size, stride=stride, padding=padding, bias=False),
        #     nn.BatchNorm1d(out_channels),
        #     nn.ReLU()
        # )

    def forward(self, input_1D):
        
        # Transpose conv
        input_1D = self.deconv(input_1D)
        # relu 
        input_1D = self.relu(input_1D)
        # batch norm
        input_1D = self.batchNorm(input_1D)
        
        # return
        return input_1D


class Spice_Decoder_1Unpool(nn.Module):
    def __init__(self, channel_list = [512, 256, 256, 256, 128, 64, 32], ):
        super().__init__()
        """
            This is a serialized version of the 1UnPool decoder for optimization
            This version of Decoder uses the model with 1 unpool layer
            channel_list : is a list with channels for all conv blocks, first entry is input channels
                default is set to original values from paper

            unPooling_list : is a list with bool values for deciding of Unpooling in each deconv layer
        """
        #
        self.input_channels = channel_list[0]
        #
        self.deconv_block1 = Deconv_block(channel_list[0], channel_list[1])
        self.deconv_block2 = Deconv_block(channel_list[1], channel_list[2])
        self.deconv_block3 = Deconv_block(channel_list[2], channel_list[3])
        self.deconv_block4 = Deconv_block(channel_list[3], channel_list[4])
        self.deconv_block5 = Deconv_block(channel_list[4], channel_list[5])
        self.deconv_block6 = Deconv_block(channel_list[5], channel_list[6])
        # 
        self.fc1 = nn.Linear(1, 48)
        self.fc2 = nn.Linear(48, 1024)
        self.unpool = nn.MaxUnpool1d(kernel_size=3, stride=2, padding=1)
        


        
    def forward(self, input_1D, unpool_mat=None):
        ''' just do unsqueeze to make batch size one '''
        input_1D = input_1D.unsqueeze(dim=1)
        batch_size = input_1D.size()[0]
        #
        input_1D = self.fc2(self.fc1(input_1D))
        # reshape
        input_1D = input_1D.reshape(batch_size, self.input_channels, -1)
        ###
        # do one unpooling
        input_1D = self.unpool(input_1D, unpool_mat[5], output_size=(input_1D.size()[0], input_1D.size()[1], input_1D.size()[2]*2))   

        # do the deconv layer     
        input_1D = self.deconv_block1(input_1D)
        
        input_1D = self.deconv_block2(input_1D)
        
        input_1D = self.deconv_block3(input_1D)
        
        input_1D = self.deconv_block4(input_1D)
        
        input_1D = self.deconv_block5(input_1D)
        
        input_1D = self.deconv_block6(input_1D)

        return input_1D
    
class Spice_model_1Unpool(nn.Module):
    def __init__(self, channel_enc_list = [1, 64, 128, 256, 512, 512, 512], 
                channel_dec_list = [512, 256, 256, 256, 128, 64, 32],):
        super().__init__()
        """
        Unified SPICE model
            1Unpool version 
        """
        self.enc_block = Spice_Encoder(channel_list=channel_enc_list)
        self.dec_block = Spice_Decoder_1Unpool(channel_list=channel_dec_list)

    def forward(self, input_1D):
        # inout is [64, 128]
        # pass through encoder
        pitch_H, conf_H, mat_list = self.enc_block(input_1D)
        # some reshaping of p_head

        # decoder
        hat_x = self.dec_block(pitch_H, mat_list)

        return pitch_H, conf_H, hat_x
    
class Spice_Decoder_Mirror(nn.Module):
    def __init__(self, channel_list = [512, 512, 512, 256, 128, 64, 1], ):
        super().__init__()
        """
            This is a serialized version of the Mirror decoder for optimization
            This version of Decoder uses the model with 1 unpool layer
            channel_list : is a list with channels for all conv blocks, first entry is input channels
                default is set to original values from paper

            unPooling_list : is a list with bool values for deciding of Unpooling in each deconv layer
        """
        #
        self.input_channels = channel_list[0]
        #
        self.deconv_block1 = Deconv_block(channel_list[0], channel_list[1])
        self.deconv_block2 = Deconv_block(channel_list[1], channel_list[2])
        self.deconv_block3 = Deconv_block(channel_list[2], channel_list[3])
        self.deconv_block4 = Deconv_block(channel_list[3], channel_list[4])
        self.deconv_block5 = Deconv_block(channel_list[4], channel_list[5])
        self.deconv_block6 = Deconv_block(channel_list[5], channel_list[6])
        # 
        self.fc1 = nn.Linear(1, 48)
        self.fc2 = nn.Linear(48, 1024)
        self.unpool = nn.MaxUnpool1d(kernel_size=3, stride=2, padding=1)

        
    def forward(self, input_1D, unpool_mat_list=None):
        ''' just do unsqueeze to make batch size one '''
        input_1D = input_1D.unsqueeze(dim=1)
        batch_size = input_1D.size()[0]
        #
        input_1D = self.fc2(self.fc1(input_1D))
        # reshape
        input_1D = input_1D.reshape(batch_size, self.input_channels, -1)
        ###
        # do one unpooling
        input_1D = self.unpool(input_1D, unpool_mat_list[5], output_size=(input_1D.size()[0], input_1D.size()[1], input_1D.size()[2]*2))   
        # do the deconv layer     
        input_1D = self.deconv_block1(input_1D)
        # repeat for each layer
        input_1D = self.unpool(input_1D, unpool_mat_list[4], output_size=(input_1D.size()[0], input_1D.size()[1], input_1D.size()[2]*2))
        input_1D = self.deconv_block2(input_1D)
        
        input_1D = self.unpool(input_1D, unpool_mat_list[3], output_size=(input_1D.size()[0], input_1D.size()[1], input_1D.size()[2]*2))
        input_1D = self.deconv_block3(input_1D)
        
        input_1D = self.unpool(input_1D, unpool_mat_list[2], output_size=(input_1D.size()[0], input_1D.size()[1], input_1D.size()[2]*2))
        input_1D = self.deconv_block4(input_1D)
        
        input_1D = self.unpool(input_1D, unpool_mat_list[1], output_size=(input_1D.size()[0], input_1D.size()[1], input_1D.size()[2]*2))
        input_1D = self.deconv_block5(input_1D)
        
        input_1D = self.unpool(input_1D, unpool_mat_list[0], output_size=(input_1D.size()[0], input_1D.size()[1], input_1D.size()[2]*2))
        input_1D = self.deconv_block6(input_1D)

        return input_1D
    
class Spice_model_Mirror(nn.Module):
    def __init__(self, channel_enc_list = [1, 64, 128, 256, 512, 512, 512], 
                channel_dec_list = [512, 512, 512, 256, 128, 64, 1],):
        super().__init__()
        """
        Unified SPICE model
            Mirrored Decoder
        """
        self.enc_block = Spice_Encoder(channel_list=channel_enc_list)
        self.dec_block = Spice_Decoder_Mirror(channel_list=channel_dec_list)

    def forward(self, input_1D):
        # inout is [64, 128]
        # pass through encoder
        pitch_H, conf_H, mat_list = self.enc_block(input_1D)
        # some reshaping of p_head

        # decoder
        hat_x = self.dec_block(pitch_H, mat_list)

        return pitch_H, conf_H, hat_x

class Spice_Decoder_sterne(nn.Module):
    def __init__(self, channel_list = [1, 256, 256, 256, 128, 64, 64], ):
        super().__init__()
        """
            This version of Decoder uses the model from sterme implementation
            this decoder has maxpool instead of maxunpooling and 1fc layer
            channel_list : is a list with channels for all conv blocks, first entry is input channels
                default is set to original values from paper

            unPooling_list : is a list with bool values for deciding of Unpooling in each deconv layer
        """
        #
        self.input_channels = channel_list[0]
        #
        self.deconv_block1 = Deconv_block(channel_list[0], channel_list[1])
        self.deconv_block2 = Deconv_block(channel_list[1], channel_list[2])
        self.deconv_block3 = Deconv_block(channel_list[2], channel_list[3])
        self.deconv_block4 = Deconv_block(channel_list[3], channel_list[4])
        self.deconv_block5 = Deconv_block(channel_list[4], channel_list[5])
        self.deconv_block6 = Deconv_block(channel_list[5], channel_list[6])
        # 
        self.fc1 = nn.Linear(1, 48)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, return_indices=False )
        


        
    def forward(self, input_1D):
        ''' just do unsqueeze to make batch size one '''
        input_1D = input_1D.unsqueeze(dim=1)
        batch_size = input_1D.size()[0]
        #
        input_1D = self.fc1(input_1D)
        # reshape
        input_1D = input_1D.reshape(batch_size, self.input_channels, -1)
        ###
        # do the deconv layer        
        input_1D = self.deconv_block1(input_1D)
        
        input_1D = self.deconv_block2(input_1D)
        input_1D = self.maxpool(input_1D)
        
        input_1D = self.deconv_block3(input_1D)
        input_1D = self.maxpool(input_1D)
        
        input_1D = self.deconv_block4(input_1D)
        input_1D = self.maxpool(input_1D)
        
        input_1D = self.deconv_block5(input_1D)
        input_1D = self.maxpool(input_1D)
        
        input_1D = self.deconv_block6(input_1D)
        input_1D = self.maxpool(input_1D)

        return input_1D
    
class Spice_model_Sterne(nn.Module):
    def __init__(self, channel_enc_list = [1, 64, 128, 256, 512, 512, 512], 
                channel_dec_list = [1, 256, 256, 256, 128, 64, 64] ):
        super().__init__()
        """
        Unified SPICE model
            sterme version of the decoder
        """
        self.enc_block = Spice_Encoder(channel_list=channel_enc_list)
        self.dec_block = Spice_Decoder_sterne(channel_list=channel_dec_list)

    def forward(self, input_1D):
        # inout is [64, 128]
        # pass through encoder
        pitch_H, conf_H, mat_list = self.enc_block(input_1D)
        # some reshaping of p_head

        # decoder
        hat_x = self.dec_block(pitch_H)

        return pitch_H, conf_H, hat_x