import torch.nn as nn
import torch
import random

class BeLSTM(nn.Module):
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
        hidden,cell = torch.randn(1,bs,self.__hidden_dim).cuda(),torch.randn(1,bs,self.__hidden_dim).cuda()

        en_out = self.__encoder(encoder_x) # [bs,mod,7]
        outputs = torch.zeros(mod,bs,7).cuda()
        t = en_out[:,-1,:]
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


class EncoderLayer(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,dropout_rate):
        super().__init__()
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim
        self.__dropout_rate = dropout_rate
        self.mh = MultiHead(self.__embedding_dim,self.__hidden_dim,self.__dropout_rate)

    def forward(self, x):
        # [bs,mod,7]
        score = self.mh(x, x, x)

        return score


class Encoder(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,dropout_rate):
        super().__init__()
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim
        self.__dropout_rate = dropout_rate
        self.layer_1 = EncoderLayer(self.__embedding_dim,self.__hidden_dim,self.__dropout_rate)
        self.layer_2 = EncoderLayer(self.__embedding_dim,self.__hidden_dim,self.__dropout_rate)
        self.layer_3 = EncoderLayer(self.__embedding_dim,self.__hidden_dim,self.__dropout_rate)

    def forward(self, x):
        # [bs,mod,7]
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x



class MultiHead(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,dropout_rate):
        super().__init__()
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim
        self.__dropout_rate = dropout_rate
        self.__head_count = 1

        self.fc_Q = torch.nn.Linear(self.__embedding_dim, self.__hidden_dim)
        self.fc_K = torch.nn.Linear(self.__embedding_dim, self.__hidden_dim)
        self.fc_V = torch.nn.Linear(self.__embedding_dim, self.__hidden_dim)
        self.out_fc = torch.nn.Linear(self.__hidden_dim, self.__embedding_dim)
        self.__dropout = nn.Dropout(self.__dropout_rate)
        self.__LN = nn.LayerNorm(normalized_shape=embedding_dim, elementwise_affine=True)


    def forward(self, Q, K, V):
        b,mod,_ = Q.shape
        clone_Q = Q.clone()
        K = self.fc_K(self.__LN(K))
        V = self.fc_V(self.__LN(V))
        Q = self.fc_Q(self.__LN(Q))
        Q = Q.reshape(b, mod, self.__head_count, self.__embedding_dim).permute(0, 2, 1, 3)
        K = K.reshape(b, mod, self.__head_count, self.__embedding_dim).permute(0, 2, 1, 3)
        V = V.reshape(b, mod, self.__head_count, self.__embedding_dim).permute(0, 2, 1, 3)
        score = self.attention(Q, K, V,mod)
        score = self.__dropout(self.out_fc(score))
        score = clone_Q + score
        return score


    def attention(self,Q, K, V,mod):
        score = torch.matmul(Q, K.permute(0, 1, 3, 2))
        score /=  (self.__embedding_dim/self.__head_count) ** 0.5
        score = torch.softmax(score, dim=-1)
        score = torch.matmul(score, V)
        score = score.permute(0, 2, 1, 3).reshape(-1, mod, self.__embedding_dim)

        return score

