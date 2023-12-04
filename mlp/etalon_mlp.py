import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as pl
import mlp
import normalization
import time
import PSF
from matplotlib import colors

Et = {
      'R' : 0.892,
      'n' : 2.3268,
      'd' : 281e-6,
      'fnum' : 60
      }
    
class Etalon(object):
    def __init__(self, gpu=0, verbose=False):
        """
        Object to evaluate the etalon neural model

        Parameters
        ----------
        gpu : int, optional
            What GPU to use, by default 0. If gpu=-1, then force CPU
        verbose : bool, optional
            Verbosity level, by default False
        """

        checkpoint = '../train/weights/2023-10-05-10:27:40.best.pth'

        if (verbose):
            print(f"Loading model {checkpoint}")
        chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)

        self.hyperparameters = chk['hyperparameters']

        self.cuda = torch.cuda.is_available()
        self.gpu = gpu        

        if (self.gpu == -1):
            self.device = torch.device("cpu")    
        else:
            self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        if (verbose):
            print(f"Computing in {self.device}")
        
                       
        # Model
        self.encoding = mlp.PositionalEncoding(n_input=1, 
                                               sigma=self.hyperparameters['sigma'], 
                                               n_freqs=self.hyperparameters['n_freqs']).to(self.device)
        
        self.model = mlp.MLP(n_input=self.encoding.dim_encoding,
                                n_output=1,
                                dim_hidden=self.hyperparameters['n_hidden_mlp'],                                 
                                n_hidden=self.hyperparameters['num_layers_mlp'],
                                activation=nn.LeakyReLU(),
                                final_activation=nn.Sigmoid()).to(self.device)
                
        self.conditioning = mlp.MLPConditioning(n_input=5,
                                                n_output=self.hyperparameters['n_hidden_mlp'],
                                                  dim_hidden=self.hyperparameters['n_hidden_conditioning'],                                                   
                                                  n_hidden=self.hyperparameters['num_layers_conditioning'],
                                                  activation=nn.LeakyReLU()).to(self.device)
        
        if (verbose):
            print('N. total parameters MLP : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
            print('N. total parameters CONDITIONING : {0}'.format(sum(p.numel() for p in self.conditioning.parameters() if p.requires_grad)))

        if (verbose):
            print("Setting weights of the model...")
        self.conditioning.load_state_dict(chk['conditioning_dict'])
        self.model.load_state_dict(chk['siren_dict'])

        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.conditioning.eval()

        self.range_min = [0, 0, -200e-6, -200e-6, 0.999999]
        self.range_max = [0.4, 0.4, 200e-6, 200e-6, 1.000001]

        # First normalization trying to put all profiles with amplitude close to 1
        x = np.array([1.3614228004339918e-05, 6.311172221422113e-05, 0.0001021886913272852, 0.00013866052916614507, 0.00017513236700500488, 0.00021030163920676258, 0.0002506811739569288])
        y = np.array([-0.35578027835803994, -3.997183316251814, -5.5880875561083165, -6.61333695512695, -7.355758933726651, -7.956767202116886, -8.522422042954753])
        out = np.polyfit(x, y, 3)
        self.polyn = np.poly1d(out)

    def evaluate(self, angle1, angle2, xi, eta, Da, wavelength):
        """
        Evaluate the frunction for the parameters and wavelengths given

        """
        
        n_models = len(angle1)
        n_wavelengths = len(wavelength)

        pars = np.hstack([angle1[:,None], angle2[:,None], xi[:,None], eta[:,None], Da[:,None]])
        wavelengths = np.repeat(wavelength[None, :], n_models, axis=0)
        
        # Compute the normalization for the peak amplitude
        rr = np.sqrt(pars[:, 2]**2 + pars[:, 3]**2)
        factor = np.exp(self.polyn(rr))[:, None]

        wvl = normalization.normalize_input(wavelengths, 6172.5, 6174.5)
                
        for i in range(5):
            pars[:, i] = normalization.normalize_input(pars[:, i], self.range_min[i], self.range_max[i])

        wavelengths = torch.tensor(wvl.astype('float32')).to(self.device)
        pars = torch.tensor(pars.astype('float32')).to(self.device)
        
        pars = pars[:, None, :].expand(-1, n_wavelengths, -1)

        wavelengths = wavelengths.reshape(-1, 1)        
        pars = pars.reshape(-1, 5)                    
        
        
        with torch.no_grad():
                                    
            # FiLM conditioning
            beta, gamma = self.conditioning(pars)

            # MLP
            wvl_encoded = self.encoding(wavelengths)
            out = self.model(wvl_encoded, beta=beta, gamma=gamma)
                                    
        out = out.cpu().numpy()
        
        # De-normalize
        # out = normalization.denormalize_output(out, 0.0, 1.0)               
        out = out.reshape(n_models, n_wavelengths)
        
        # Now undo the normalization for the peak amplitude        
        out *= factor

        return out
             
if (__name__ == '__main__'):
    
    n_models = 32*32
    n_wavelengths = 400
    wavelength = np.linspace(6173.5 - 0.2, 6173.5 + 0.2, n_wavelengths)
    
    angle1 = np.random.uniform(low=0.0, high=0.4, size=n_models)
    angle2 = np.random.uniform(low=0.0, high=0.4, size=n_models)
    xi = np.random.uniform(low=-100e-6, high=100e-6, size=n_models)
    eta = np.random.uniform(low=-100e-6, high=100e-6, size=n_models)
    Da = np.random.uniform(low=0.999999, high=1.000001, size=n_models)
        
    etalon = Etalon(gpu=0, verbose=True)

    start = time.time()
    out = etalon.evaluate(angle1, angle2, xi, eta, Da, wavelength)
    end = time.time()
    print(f'Time for computing {n_models} profiles : {end-start} s (i.e. {n_models/(end-start)} profiles/s or {1e3*(end-start)/n_models} ms/profiles)')
    
    start2 = time.time()
    out_theory = PSF.PSF(wavelength * 1e-10, angle1[0], angle2[0], xi[0], eta[0], Da[0], Et)
    end2 = time.time()
    print(f'Time for theory : {end2-start2} s - {(end2-start2)/((end-start)/n_models)} times slower than the model')

    fig, ax = pl.subplots()
    ax.plot(wavelength, out_theory, label='theory')
    ax.plot(wavelength, out[0, :], label='model')
    ax.legend()
    ax.set_xlabel('Wavelength [A]')


    n_models = 500
    angle1 = np.ones(n_models) * 0.25
    angle2 = np.ones(n_models) * 0.0
    xi = np.linspace(-200e-6, 200e-6, n_models)
    eta = np.ones(n_models) * 0.0
    Da = np.ones(n_models) * 1.0

    out = etalon.evaluate(angle1, angle2, xi, eta, Da, wavelength)

    fig, ax = pl.subplots()
    im = ax.imshow(out, norm=colors.LogNorm())
    pl.colorbar(im)