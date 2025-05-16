import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, x):
        real = F.dropout(x.real, self.p, self.training)
        imag = F.dropout(x.imag, self.p, self.training)
        return torch.complex(real, imag)
    
class ComplexSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(ComplexSoftmax, self).__init__()
        self.dim = dim
    def forward(self, x):
        real = x.real
        imag = x.imag
        softmax_real = F.softmax(real, dim=self.dim)
        softmax_imag = F.softmax(imag, dim=self.dim)
        return torch.complex(softmax_real, softmax_imag)


class MagnitudeSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(MagnitudeSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        magnitude = torch.abs(x)
        magnitude_sum = torch.sum(magnitude, dim=self.dim, keepdim=True)
        normalized_magnitude = magnitude / magnitude_sum
        
        real = x.real * normalized_magnitude
        imag = x.imag * normalized_magnitude
        
        return torch.complex(real, imag)
    
class ComplexLayerNorm(nn.Module):

    def __init__(self, embed_dim=None, eps=1e-05, elementwise_affine=True, device='cuda'):
        super().__init__()
        assert not(elementwise_affine and embed_dim is None), 'Give dimensions of learnable parameters or disable them'
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.embed_dim = embed_dim
            self.register_parameter(name='weights', param=torch.nn.Parameter(torch.empty([2, 2], dtype=torch.complex64)))
            self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(embed_dim, dtype=torch.complex64)))
            self.weights = torch.nn.Parameter(torch.eye(2))
            self.weights = torch.nn.Parameter((torch.Tensor([1, 1, 0]).repeat([embed_dim, 1])).unsqueeze(-1))
            self.bias = torch.nn.Parameter(torch.zeros([1, 1, embed_dim], dtype=torch.complex64))
        self.eps = eps

    def forward(self, input):

        ev = torch.unsqueeze(torch.mean(input, dim=-1), dim=-1)
        var_real = torch.unsqueeze(torch.unsqueeze(torch.var(input.real, dim=-1)+self.eps, dim=-1), dim=-1)
        var_imag = torch.unsqueeze(torch.unsqueeze(torch.var(input.imag, dim=-1)+self.eps, dim=-1), dim=-1)

        input = input - ev
        cov = torch.unsqueeze(torch.unsqueeze(torch.mean(input.real * input.imag, dim=-1), dim=-1), dim=-1)
        cov_m_0 = torch.cat((var_real, cov), dim=-1)
        cov_m_1 = torch.cat((cov, var_imag), dim=-1)
        cov_m = torch.unsqueeze(torch.cat((cov_m_0, cov_m_1), dim=-2), dim=-3)
        in_concat = torch.unsqueeze(torch.cat((torch.unsqueeze(input.real, dim=-1), torch.unsqueeze(input.imag, dim=-1)), dim=-1), dim=-1)

        cov_sqr = self.sqrt_2x2(cov_m)

        # out = self.inv_2x2(cov_sqr).matmul(in_concat)  # [..., 0]
        if self.elementwise_affine:
            real_var_weight = (self.weights[:, 0, :] ** 2).unsqueeze(-1).unsqueeze(0)
            imag_var_weight = (self.weights[:, 1, :] ** 2).unsqueeze(-1).unsqueeze(0)
            cov_weight = (torch.sigmoid(self.weights[:, 2, :].unsqueeze(-1).unsqueeze(0)) - 0.5) * 2 * torch.sqrt(real_var_weight * imag_var_weight)
            weights_mult = torch.cat([torch.cat([real_var_weight, cov_weight], dim=-1), torch.cat([cov_weight, imag_var_weight], dim=-1)], dim=-2).unsqueeze(0)
            mult_mat = self.sqrt_2x2(weights_mult).matmul(self.inv_2x2(cov_sqr))
            out = mult_mat.matmul(in_concat)  # makes new cov_m = self.weights
        else:
            out = self.inv_2x2(cov_sqr).matmul(in_concat)  # [..., 0]
        out = out[..., 0, 0] + 1j * out[..., 1, 0]  # torch.complex(out[..., 0], out[..., 1]) not used because of memory requirements
        if self.elementwise_affine:
            return out + self.bias
        return out

    def inv_2x2(self, input):
        a = torch.unsqueeze(torch.unsqueeze(input[..., 0, 0], dim=-1), dim=-1)
        b = torch.unsqueeze(torch.unsqueeze(input[..., 0, 1], dim=-1), dim=-1)
        c = torch.unsqueeze(torch.unsqueeze(input[..., 1, 0], dim=-1), dim=-1)
        d = torch.unsqueeze(torch.unsqueeze(input[..., 1, 1], dim=-1), dim=-1)
        divisor = a * d - b * c
        mat_1 = torch.cat((d, -b), dim=-2)
        mat_2 = torch.cat((-c, a), dim=-2)
        mat = torch.cat((mat_1, mat_2), dim=-1)
        return mat / divisor

    def sqrt_2x2(self, input):
        a = torch.unsqueeze(torch.unsqueeze(input[..., 0, 0], dim=-1), dim=-1)
        b = torch.unsqueeze(torch.unsqueeze(input[..., 0, 1], dim=-1), dim=-1)
        c = torch.unsqueeze(torch.unsqueeze(input[..., 1, 0], dim=-1), dim=-1)
        d = torch.unsqueeze(torch.unsqueeze(input[..., 1, 1], dim=-1), dim=-1)

        s = torch.sqrt(a * d - b * c)  # sqrt(det)
        t = torch.sqrt(a + d + 2 * s)  # sqrt(trace + 2 * sqrt(det))
        # maybe use 1/t * (M + sI) later, see Wikipedia

        return torch.cat((torch.cat((a + s, b), dim=-2), torch.cat((c, d + s), dim=-2)), dim=-1) / t
    
class ComplexReLU(nn.Module):
    
    def __init__(self):
        super(ComplexReLU, self).__init__()

    def forward(self, x):
        real = x.real
        imag = x.imag

        relu_real = F.relu(real)
        relu_imag = F.relu(imag)

        return torch.complex(relu_real, relu_imag)
    
class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.05):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        real = F.leaky_relu(x.real, negative_slope=self.negative_slope)
        imag = F.leaky_relu(x.imag, negative_slope=self.negative_slope)
        return torch.complex(real, imag)
    
class ComplexGELU(nn.Module):
    def __init__(self):
        super(ComplexGELU, self).__init__()

    def forward(self, x):
        real = x.real
        imag = x.imag

        gelu_real = F.gelu(real)
        gelu_imag = F.gelu(imag)

        return torch.complex(gelu_real, gelu_imag)
    
class ModReLU(nn.Module):
    def __init__(self, num_channels=1, init_bias=0.0, m=1):
        super(ModReLU, self).__init__()
        self.num_channels = num_channels
        self.m=m
        if num_channels==1:
            self.b = nn.Parameter(torch.tensor(init_bias))
        else:
            self.b = nn.Parameter(torch.full((num_channels,), init_bias))  

    def forward(self, z):
        if self.num_channels > 1:
            b = self.b.view(1, -1)
        magnitude = torch.abs(z)  
        phase = z / (magnitude + 1e-5)  
        activated_magnitude = torch.relu(magnitude + self.m*self.b) 
        output = activated_magnitude * phase
        output[magnitude == 0] = 0
        return output


class ComplexAttention(nn.Module):
    def __init__(self, attention_type=complex, mask_flag=False, factor=5, scale=None, 
                 attention_dropout=0.1, output_attention=False):
        super(ComplexAttention, self).__init__()  
        self.scale = scale
        self.mask_flag = mask_flag
        self.attention_type = attention_type
        self.output_attention = output_attention
        if self.attention_type == 'complex':
            self.dropout = ComplexDropout(attention_dropout)
            self.softmax = ComplexSoftmax(dim=-1)
        else:
            self.dropout = nn.Dropout(attention_dropout)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape    
        _, S, _, D = values.shape   
        scale = self.scale or 1. / sqrt(E)

        if self.attention_type == 'irfft':
            queries1 = torch.fft.irfft(queries, dim=-1)
            keys1 = torch.fft.irfft(keys, dim=-1)
            scores = torch.einsum("blhe,bshe->bhls", queries1, keys1)
        else:
            scores = torch.einsum("blhe,bshe->bhls", queries, keys.conj())  
        
        
        if self.attention_type == 'real':
            scores = scores.real
        A = self.dropout(self.softmax(scale * scores))
       
        if self.attention_type == 'complex':
            V = torch.einsum("bhls,bshd->blhd", A, values).to(queries.dtype)
        else:
            V = torch.einsum("bhls,bshd->blhd", A.type_as(values), values).to(queries.dtype)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads).to(torch.cfloat)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads).to(torch.cfloat)
        self.value_projection = nn.Linear(d_model, d_values * n_heads).to(torch.cfloat)
        self.out_projection = nn.Linear(d_values * n_heads, d_model).to(torch.cfloat)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
# without normlayer
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, use_layernorm=False, dropout=0.1, activation="gelu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Linear(d_model,d_ff).to(torch.cfloat)
        self.conv2 = nn.Linear(d_ff,d_model).to(torch.cfloat)
        if use_layernorm:
            self.norm1 = ComplexLayerNorm(elementwise_affine=False)
            self.norm2 = ComplexLayerNorm(elementwise_affine=False)
        else:
            self.norm1 =None
        self.dropout = ComplexDropout(dropout)
        self.activation = ComplexGELU() 

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        if self.norm1 is not None:
            y = x = self.norm1(x)
            y = self.dropout(self.activation(self.conv1(y)))
            y = self.dropout(self.conv2(y))
            return self.norm2(x + y), attn
        else:
            y = x
            y = self.dropout(self.activation(self.conv1(y)))
            y = self.dropout(self.conv2(y))

        return x + y, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns