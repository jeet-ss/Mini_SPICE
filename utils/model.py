import torch
from torch import nn

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1, 
                padding=1, pool_filter_size=3, pool_stride=2, pool_padding=1):
        '''
            Convolution block of the Spice Encoder
            all default values set according to paper
        '''
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=filter_size, stride=stride, padding=padding, bias=False)
        self.batch = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool1d(kernel_size=pool_filter_size, stride=pool_stride, padding=pool_padding, return_indices=True)

    def forward(self, input_1D):
        input_1D = self.conv(input_1D)
        input_1D = self.relu(self.batch(input_1D))
        input_1D, indx_mat = self.maxPool(input_1D)

        return input_1D, indx_mat

class Spice_Encoder(nn.Module):
    def __init__(self, channel_list=[1, 64, 128, 256, 512, 512, 512]):
        ''' 
            channel_list : is a list with channels for all conv blocks, first entry is input channels
                default is set to original values from paper
        '''
        super().__init__()
        # Encoder is fixed 
        self.conv_block1 = Conv_block(channel_list[0], channel_list[1],  )
        self.conv_block2 = Conv_block(channel_list[1], channel_list[2],  )
        self.conv_block3 = Conv_block(channel_list[2], channel_list[3],  )
        self.conv_block4 = Conv_block(channel_list[3], channel_list[4],  )
        self.conv_block5 = Conv_block(channel_list[4], channel_list[5],  )
        self.conv_block6 = Conv_block(channel_list[5], channel_list[6],  )
        # 
        # Pitch estimantion Head
        self.fc1 = nn.Linear(1024, 48)
        self.fc2 = nn.Linear(48, 1)
        # Conf Head
        self.conf_head = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1D):
        # channel and batch size
        input_1D = input_1D.unsqueeze(dim=1)
        batch_size = input_1D.size()[0]
        # conv blocks
        input_1D, mat_1 = self.conv_block1(input_1D)
        input_1D, mat_2 = self.conv_block2(input_1D)
        input_1D, mat_3 = self.conv_block3(input_1D)
        input_1D, mat_4 = self.conv_block4(input_1D)
        input_1D, mat_5 = self.conv_block5(input_1D)
        input_1D, mat_6 = self.conv_block6(input_1D)
        # flatten by batch size
        input_1D = input_1D.reshape(batch_size, -1)
         # Pitch head 
        pitch_head = self.fc2(self.fc1(input_1D))
        # Conf Head 
        conf_head = self.conf_head(input_1D)

        # return both heads and pooling matrices
        return pitch_head, conf_head, [mat_1, mat_2, mat_3, mat_4, mat_5, mat_6]

class Deconv_block(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1, 
                padding=1, unPool_filter_size=3, unPool_stride=2, unPool_padding=1, unPooling=False, batch_norm=False):
        super().__init__()
        """
            The default values are as per the general understanding of the model in the paper.
            We use these values for the direct implementation without passing them agian.
        """
        #
        self.unPooling = unPooling
        self.batch_norm = batch_norm  # bool to select batch norm
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=filter_size, stride=stride, padding=padding, bias=False)
        self.batchNorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.unpool = nn.MaxUnpool1d(kernel_size=unPool_filter_size, stride=unPool_stride, padding=unPool_padding)

    def forward(self, input_1D, unpool_mat = None, output_size = None):
        
        # Unpool
        if self.unPooling and unpool_mat is not None:
            # check if indices and input are of same size to prevent error
            if unpool_mat.size() == input_1D.size():
                input_1D = self.unpool(input_1D, unpool_mat, output_size=output_size)
            else:
                print("Unpool matrix dont match size", input_1D.size(), unpool_mat.size())
        # else:
        #     print("no unpooling", self.unPooling, unpool_mat is None)
        # Transpose conv
        input_1D = self.deconv(input_1D)
        # batch norm
        #if self.batch_norm:
        input_1D = self.batchNorm(input_1D)
        # relu 
        input_1D = self.relu(input_1D)
        return input_1D
        
class Spice_Decoder(nn.Module):
    def __init__(self, channel_list = [512, 256, 256, 256, 128, 64, 32], 
                unPooling_list = [True, False, False, False, False, False], ):
        super().__init__()
        """
            This version of Decoder uses only one Unpool layer after the first deconv layer.
            It follows similar architecture to that of the paper.
            We use the Fc layers and one Unpooling as default setting
            Since, One layer of Unpooling is necessary to match the dimension of input to Encoder.

            channel_list : is a list with channels for all conv blocks, first entry is input channels
                default is set to original values from paper

            unPooling_list : is a list with bool values for deciding of Unpooling in each deconv layer
        """
        #
        self.input_channels = channel_list[0]
        self.unPooling_list = unPooling_list
        #
        self.deconv_block1 = Deconv_block(channel_list[0], channel_list[1], unPooling = unPooling_list[0])
        self.deconv_block2 = Deconv_block(channel_list[1], channel_list[2], unPooling = unPooling_list[1])
        self.deconv_block3 = Deconv_block(channel_list[2], channel_list[3], unPooling = unPooling_list[2])
        self.deconv_block4 = Deconv_block(channel_list[3], channel_list[4], unPooling = unPooling_list[3])
        self.deconv_block5 = Deconv_block(channel_list[4], channel_list[5], unPooling = unPooling_list[4])
        self.deconv_block6 = Deconv_block(channel_list[5], channel_list[6], unPooling = unPooling_list[5])
        # 
        self.fc1 = nn.Linear(1, 48)
        self.fc2 = nn.Linear(48, 1024)


        
    def forward(self, input_1D, unpool_mat_list = None):
        # make batch size 1
        input_1D = input_1D.unsqueeze(dim=1)
        batch_size = input_1D.size()[0]
        #
        input_1D = self.fc2(self.fc1(input_1D))
        # reshape
        input_1D = input_1D.reshape(batch_size, self.input_channels, -1)
        ###
        # do the deconv layer with or without Unpooling separately for each layer
        ###
        if self.unPooling_list[0] and len(unpool_mat_list)>5:
            input_1D = self.deconv_block1(input_1D, unpool_mat_list[5], output_size=(input_1D.size()[0], input_1D.size()[1], input_1D.size()[2]*2))
        else:
            input_1D = self.deconv_block1(input_1D)
        
        if self.unPooling_list[1] and len(unpool_mat_list)>4:
            input_1D = self.deconv_block2(input_1D, unpool_mat_list[4], output_size=(input_1D.size()[0], input_1D.size()[1], input_1D.size()[2]*2))
        else:
            input_1D = self.deconv_block2(input_1D)
        #
        if self.unPooling_list[2] and len(unpool_mat_list)>3:
            input_1D = self.deconv_block3(input_1D, unpool_mat_list[3], output_size=(input_1D.size()[0], input_1D.size()[1], input_1D.size()[2]*2))
        else:
            input_1D = self.deconv_block3(input_1D)
        #
        if self.unPooling_list[3] and len(unpool_mat_list)>2:
            input_1D = self.deconv_block4(input_1D, unpool_mat_list[2], output_size=(input_1D.size()[0], input_1D.size()[1], input_1D.size()[2]*2))
        else:
            input_1D = self.deconv_block4(input_1D)
        #
        if self.unPooling_list[4] and len(unpool_mat_list)>1:
            input_1D = self.deconv_block5(input_1D, unpool_mat_list[1], output_size=(input_1D.size()[0], input_1D.size()[1], input_1D.size()[2]*2))
        else:
            input_1D = self.deconv_block5(input_1D)
        #
        if self.unPooling_list[5] and len(unpool_mat_list)>0:
            input_1D = self.deconv_block6(input_1D, unpool_mat_list[0], output_size=(input_1D.size()[0], input_1D.size()[1], input_1D.size()[2]*2))
        else:
            input_1D = self.deconv_block6(input_1D)

        return input_1D
    
class Spice_model(nn.Module):
    def __init__(self, channel_enc_list = [1, 64, 128, 256, 512, 512, 512], 
                channel_dec_list = [512, 256, 256, 256, 128, 64, 32],
                unPooling_list = [True, False, False, False, False, False], ):
        super().__init__()
        """
        Unified SPICE model
            default version 
        """
        self.enc_block = Spice_Encoder(channel_list=channel_enc_list)
        self.dec_block = Spice_Decoder(channel_list=channel_dec_list, unPooling_list=unPooling_list)

    def forward(self, input_1D):
        # inout is [64, 128]
        # pass through encoder
        pitch_H, conf_H, mat_list = self.enc_block(input_1D)
        # some reshaping of p_head

        # decoder
        hat_x = self.dec_block(pitch_H, mat_list)

        return pitch_H, conf_H, hat_x