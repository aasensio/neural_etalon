import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as pl


def init_kaiming(m):
    if type(m) == nn.Linear:
        init.kaiming_uniform_(m.weight, nonlinearity='relu')

class MLPConditioning(nn.Module):
    def __init__(self, n_input, n_output, dim_hidden=1, n_hidden=1, activation=nn.ReLU(), bias=True, final_activation=nn.Identity()):
        """Simple fully connected network, potentially including FiLM conditioning

        Parameters
        ----------
        n_input : int
            Number of input neurons
        n_output : int
            Number of output neurons
        n_hidden : int, optional
            number of neurons per hidden layers, by default 1
        n_hidden_layers : int, optional
            Number of hidden layers, by default 1        
        activation : _type_, optional
            Activation function to be used at each layer, by default nn.Tanh()
        bias : bool, optional
            Include bias or not, by default True
        final_activation : _type_, optional
            Final activation function at the last layer, by default nn.Identity()
        """
        super().__init__()


        self.activation = activation
        self.final_activation = final_activation

        self.layers = nn.ModuleList([])        
        
        self.layers.append(nn.Linear(n_input, dim_hidden, bias=bias))
        
        for i in range(n_hidden):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden, bias=bias))
            self.layers.append(self.activation)
        
        self.gamma = nn.Linear(dim_hidden, n_output)
        self.beta = nn.Linear(dim_hidden, n_output)
        

        self.layers.apply(init_kaiming)
        self.gamma.apply(init_kaiming)
        self.beta.apply(init_kaiming)
        
    def forward(self, x):

        # Apply all layers
        for layer in self.layers:
            x = layer(x)
        
        gamma = self.gamma(x)
        beta = self.beta(x)
        
        return gamma, beta


class MLP(nn.Module):
    def __init__(self, n_input, n_output, dim_hidden=1, n_hidden=1, activation=nn.ReLU(), bias=True, final_activation=nn.Identity()):
        """Simple fully connected network, potentially including FiLM conditioning

        Parameters
        ----------
        n_input : int
            Number of input neurons
        n_output : int
            Number of output neurons
        n_hidden : int, optional
            number of neurons per hidden layers, by default 1
        n_hidden_layers : int, optional
            Number of hidden layers, by default 1        
        activation : _type_, optional
            Activation function to be used at each layer, by default nn.Tanh()
        bias : bool, optional
            Include bias or not, by default True
        final_activation : _type_, optional
            Final activation function at the last layer, by default nn.Identity()
        """
        super().__init__()


        self.activation = activation
        self.final_activation = final_activation

        self.layers = nn.ModuleList([])        
        
        self.layers.append(nn.Linear(n_input, dim_hidden, bias=bias))
        
        for i in range(n_hidden):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden, bias=bias))            
        
        self.last_layer = nn.Linear(dim_hidden, n_output)

        self.layers.apply(init_kaiming)
        self.last_layer.apply(init_kaiming)
        
    def forward(self, x, gamma=None, beta=None):

        # Apply all layers
        for layer in self.layers:

            # Apply conditioning if present
            if (gamma is not None):
                x = layer(x) * gamma
            else:
                x = layer(x)

            if (beta is not None):
                x += beta

            x = self.activation(x)
        
        x = self.last_layer(x)
        x = self.final_activation(x)
        
        return x

    def weights_init(self, type='xavier', nonlinearity='relu'):
        for module in self.modules():
            if (type == 'xavier'):
                xavier_init(module)
            if (type == 'kaiming'):
                kaiming_init(module, nonlinearity=nonlinearity)

class PositionalEncoding(nn.Module):
    """Positional encoding for the input vector

    gamma(v) = [..., cos(2 * pi * sigma ** (j / m) * v), sin(2 * pi * sigma ** (j / m) * v), ...]

    Parameters
    ----------
    nn : _type_
        _description_
    """
    def __init__(self, sigma, n_freqs, n_input):
        super().__init__()
        self.sigma = sigma
        self.n_freqs = n_freqs
        self.dim_encoding = (2 * n_freqs + 1) * n_input

    def forward(self, v, alpha=None):

        n_batch, n_input = v.shape

        if (alpha is None):
            alpha = self.n_freqs
        
        k = torch.arange(self.n_freqs, device=v.device)

        weight = 0.5 * (1.0 - torch.cos((alpha - k) * np.pi))
        weight[alpha < k] = 0.0
        weight[alpha - k >= 1] = 1.0
        weight = weight[None, None, :]

        coeffs = 2 * np.pi * self.sigma ** (1.0 * k / self.n_freqs)
        vp = coeffs * torch.unsqueeze(v, -1)        
        vp_cat = torch.cat((weight * torch.cos(vp), weight * torch.sin(vp)), dim=-1)

        out = vp_cat.flatten(-2, -1)

        out = torch.cat((v, out), dim=-1)

        return out
        
    
class GaussianEncoding(nn.Module):
    def __init__(self, input_size, encoding_size, sigma=None):
        self.sigma = sigma
        self.input_size = input_size
        self.encoding_size = encoding_size

        # Fourier matrix        
        B = self.sigma * torch.randn((self.encoding_size, self.input_size))
        
        self.register_buffer("B", B)

    def forward(self, v):
        vp = 2.0 * np.pi * v @ b.T
    
        return torch.cat([torch.cos(vp), torch.sin(vp)], dim=-1)

if (__name__ == '__main__'):

    dim_in = 1
    dim_encoding = 128
    dim_out = 1
    dim_hidden = 128
    n_hidden = 3
    n_freqs = 3
    sigma = 0.1

    encoding = PositionalEncoding(sigma=sigma, n_freqs=n_freqs, n_input=dim_in)
    mlp = MLP(n_input=encoding.dim_encoding, n_output=dim_out, dim_hidden=dim_hidden, n_hidden=n_hidden, activation=nn.ReLU())
    

    v = np.linspace(-1, 1, 1000)
    v1 = torch.tensor(v[:, None].astype('float32'))
    v2 = torch.tensor(v[:, None].astype('float32'))
    v = torch.cat((v1, v2), dim=-1)

    mlp2 = MLP(n_input=dim_in, n_output=dim_out, dim_hidden=dim_hidden, n_hidden=n_hidden, activation=nn.ReLU())
    
    out = encoding(v1, alpha=3)
    out1 = mlp(out)
    # out2 = mlp2(v)

    
    
    # dim_in = 2
    # dim_hidden = 128
    # dim_out = 1
    # num_layers = 15
    
    # tmp = MLPMultiFourier(n_input=dim_in, n_output=dim_out, n_hidden=dim_hidden, n_hidden_layers=num_layers, sigma=[0.03, 1], activation=nn.ReLU()) #, 0.1, 1.0])
    # tmp.weights_init(type='kaiming')

    # print(f'N. parameters : {sum(x.numel() for x in tmp.parameters())}')

    # x = np.linspace(-1, 1, 128)
    # y = np.linspace(-1, 1, 128)
    # X, Y = np.meshgrid(x, y)

    # xin = torch.tensor(np.vstack([X.flatten(), Y.flatten()]).T.astype('float32'))
    
    # out = tmp(xin).squeeze().reshape((128, 128)).detach().numpy()

    # pl.imshow(out)