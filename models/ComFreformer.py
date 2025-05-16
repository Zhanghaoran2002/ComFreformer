import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.ComplexTransformer import *
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.revin = configs.revin
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_deep = configs.d_deep
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.attention_type = configs.attention_type
        self.d_model = configs.d_model
        self.input_dim = self.seq_len//2+self.d_deep//2+2 if self.d_deep>0 else self.seq_len//2+1
        self.length_ratio = (self.seq_len + self.pred_len)/self.seq_len
        self.out_freq = (self.seq_len+self.pred_len)//2+1
        self.use_modrelu = configs.use_modrelu
        if self.use_modrelu:
            self.activation = ModReLU(self.channels,configs.modrelub,np.sqrt(1e-5/configs.learning_rate))  
        self.real_dropout = nn.Dropout(configs.dropout)
        self.complex_dropout = ComplexDropout(configs.dropout)
        self.layernorm = ComplexLayerNorm(embed_dim=self.d_model) if configs.use_layernorm else None

        if self.individual:
            self.time_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.time_upsampler.append(nn.Linear(self.seq_len, self.d_deep))
        else:
            self.time_upsampler = nn.Linear(self.seq_len, self.d_deep)
        # Embedding
        self.enc_embedding = nn.Linear(self.input_dim,self.d_model).to(torch.cfloat)
        self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            ComplexAttention(attention_type=configs.attention_type, attention_dropout=configs.dropout,
                                          output_attention=False), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        use_layernorm=configs.use_layernorm,
                        dropout=configs.dropout,
                        activation=configs.activation
                    )
                      for l in range(configs.e_layers)
                ]
                ,norm_layer=self.layernorm
            )
        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.d_model, self.out_freq).to(torch.cfloat))
        else:
            self.freq_upsampler = nn.Linear(self.d_model, self.out_freq).to(torch.cfloat)
            
    
    def forecast(self, x_enc):
        _, _, N = x_enc.shape # B L N

        x = self.enc_embedding(x_enc.permute(0,2,1))
        x = self.complex_dropout(x)
        
        enc_out, attns = self.encoder(x, attn_mask=None)
        # enc_out = self.layernorm(enc_out) if self.layernorm else enc_out

        if self.individual:
            weights = torch.stack([layer.weight for layer in self.freq_upsampler], dim=0)  # [D, d_out, d_in]
            biases = torch.stack([layer.bias for layer in self.freq_upsampler], dim=0)    #[D,d_out]
            freq = torch.einsum('coi,bci->bco', weights, enc_out) + biases 
            freq = freq.permute(0,2,1)
        else:
            freq = self.freq_upsampler(enc_out).permute(0,2,1)
        return freq


    def forward(self, x, mask=None):
        if self.revin:
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x = x - x_mean
            x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
            x = x / torch.sqrt(x_var)
        if self.individual:
            weights = torch.stack([layer.weight for layer in self.time_upsampler], dim=0)  # [D, d_out, d_in]
            biases = torch.stack([layer.bias for layer in self.time_upsampler], dim=0)    #[D,d_out]
            z = torch.einsum('coi,bic->bco', weights, x) + biases 
            z = z.permute(0,2,1)
        else:
            z = self.time_upsampler(x.permute(0,2,1)).permute(0,2,1)
        specxx = torch.fft.rfft(x, dim=1)
        if self.d_deep>0:
            specxz = torch.fft.rfft(z, dim=1)  
            specx = torch.cat((specxx, specxz), dim=1)
        else:
            specx = specxx   
        if self.use_modrelu:
            masks = torch.abs(specx)>1e-3
            specx = specx*masks
            specx = self.activation(specx)
        dec_out = self.forecast(specx)
        low_specxy = torch.zeros([dec_out.size(0),(self.seq_len+self.pred_len)//2+1,dec_out.size(2)],dtype=dec_out.dtype).to(dec_out.device)
        low_specxy[:,0:dec_out.size(1),:]=dec_out 
        low_xy=torch.fft.irfft(low_specxy, dim=1)
        low_xy=low_xy * self.length_ratio
        xy=(low_xy) * torch.sqrt(x_var) + x_mean if self.revin else low_xy
        return xy