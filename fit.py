import etalon
import numpy as np
import torch
import matplotlib.pyplot as pl
import transform


if (__name__ == '__main__'):
    
    lower = [0.0, 0.0, -200e-6, -200e-6, 0.999999]
    upper = [0.4, 0.4, 200e-6, 200e-6, 1.000001]

    n_iter = 50
    lr = 5.0

    pl.close('all')
    emulator = etalon.Etalon(checkpoint='weights.pth', gpu=0, verbose=True)

    # Generate a fake observation
    n_wavelengths = 400
    wavelength = np.linspace(6173.5 - 0.2, 6173.5 + 0.2, n_wavelengths)

    # Parameters of the fake model
    n_models = 1
    angle1 = np.ones(n_models) * 0.25
    angle2 = np.ones(n_models) * 0.0
    xi = np.ones(n_models) * (-200e-6)
    eta = np.ones(n_models) * 0.0
    Da = np.ones(n_models) * 1.0

    # Evaluate the model and transform to PyTorch tensors
    obs = emulator.evaluate(angle1, angle2, xi, eta, Da, wavelength)
    obs = torch.tensor(obs.astype('float32'), device=emulator.device)

    #-----------------------------------
    # Now fit the model
    #-----------------------------------

    # Given that the parameters are tightly constrained, we can use a simple
    # transformation from the physical space to an uncosntrained space    
    angle1 = np.ones(n_models) * 0.35
    angle2 = np.ones(n_models) * 0.0
    xi = np.ones(n_models) * (-200e-6)
    eta = np.ones(n_models) * 0.0
    Da = np.ones(n_models) * 1.0

    angle1 = transform.physical_to_transformed(angle1, lower[0], upper[0])    
    Da = transform.physical_to_transformed(Da, lower[4], upper[4])

    # Transform to PyTorch tensors and set the angle1 as a trainable parameter
    angle1 = torch.tensor(angle1.astype('float32'), requires_grad=True, device=emulator.device)
    angle2 = torch.tensor(angle2.astype('float32'), device=emulator.device)
    xi = torch.tensor(xi.astype('float32'), device=emulator.device)
    eta = torch.tensor(eta.astype('float32'), device=emulator.device)
    Da = torch.tensor(Da.astype('float32'), requires_grad=True, device=emulator.device)
    wavelength = torch.tensor(wavelength.astype('float32'), device=emulator.device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam([angle1, Da], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=0.1*lr)
    
    ext_loss = []
    ext_model = []

    for i in range(n_iter):        
        # Zero gradients
        optimizer.zero_grad()

        # Transform back to physical space
        angle1_ = transform.transformed_to_physical(angle1, lower[0], upper[0])
        Da_ = transform.transformed_to_physical(Da, lower[4], upper[4])
            
        # Evaluate model and compute loss. We use the PyTorch version of the evaluation
        out = emulator.evaluate_pt(angle1_, angle2, xi, eta, Da_, wavelength, grad=True)
        loss = torch.mean((out - obs)**2)
        loss.backward()
            
        # Optimizer step
        optimizer.step()
        scheduler.step()
                
        print(f'It: {i:04d} - angle1={angle1_[0].item():9.5f} - Da={Da_[0].item():9.5f} - loss={loss.item():9.5e} - lr={scheduler.get_last_lr()[0]:9.5f}')

    fig, ax = pl.subplots()
    ax.plot(obs[0, :].cpu().numpy())
    ax.plot(out[0, :].detach().cpu().numpy())