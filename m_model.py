import torch.nn as nn
import torch
import random

class model_lstm(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,dropout_rate,tf=0.5) -> None:
        super(model_lstm,self).__init__()
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim
        self.__dropout_rate = dropout_rate
        self.__tf = tf

        self.__dropout = nn.Dropout(self.__dropout_rate)
        self.__encoder = Encoder(self.__embedding_dim,self.__hidden_dim,self.__dropout_rate)
        self.__decoder = Decoder(self.__embedding_dim,self.__hidden_dim,self.__dropout_rate)

    def forward(self,encoder_x,decoder_x):  
        # [batch,mod,7]     [batch,mod,7]
        bs,mod,_ = decoder_x.shape

        hidden,cell = self.__encoder(encoder_x)
        outputs = torch.zeros(mod,bs,7).cuda()
        t = encoder_x[:,-1,:]
        # return the next feature
        for i in range(mod):
            # [bs,7]
            out,hidden,cell = self.__decoder(t,hidden,cell)
            outputs[i] = out
            tf = random.random() < self.__tf
            if (i+1) != mod:
                t = decoder_x[:,i+1,:] if tf else out 
        outputs = outputs.transpose(0,1)
        return outputs  # [bs,mod,7]


class Encoder(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,dropout_rate) -> None:
        super(Encoder,self).__init__()
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim
        self.__dropout_rate = dropout_rate

        self.__dropout = nn.Dropout(self.__dropout_rate)
        self.__encoder_layer = nn.LSTM(
            input_size=self.__embedding_dim,hidden_size=self.__hidden_dim,
            batch_first=True,bidirectional=False,
            dropout=self.__dropout_rate,num_layers=1)

    def forward(self,encoder_x):
        # encoder_x :[batch,mod,7]
        _,(hidden_out,cell_out) = self.__encoder_layer(encoder_x) 
        hidden_out = self.__dropout(hidden_out)
        cell_out = self.__dropout(cell_out)
        # [1,bs,hidden_dim]
        return hidden_out,cell_out


class Decoder(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,dropout_rate) -> None:
        super(Decoder,self).__init__()
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim
        self.__dropout_rate = dropout_rate

        self.__decoder_layer = nn.LSTM(
            input_size = self.__embedding_dim,hidden_size=self.__hidden_dim,
            batch_first = False,bidirectional = False,
            dropout = self.__dropout_rate,num_layers = 1
        )

    def forward(self,decoder_x,en_hidden,en_cell):
        # en_hidden [1,bs,hidden_dim]
        # decoder_x [bs,7]
        decoder_x = decoder_x.unsqueeze(0) # [1,bs,7]
        out,(de_hidden,de_cell) = self.__decoder_layer(decoder_x,(en_hidden,en_cell))
        # [1,bs,7]
        return out.squeeze(0),de_hidden,de_cell







