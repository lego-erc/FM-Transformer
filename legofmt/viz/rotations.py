import torch

class Rotations:
    def __init__(self):
        pass

    def phi_rot_arr(self, circ_frac, steps=30, phi_0=0.):
        phi_arr = torch.linspace(phi_0, 2 * torch.pi * circ_frac + phi_0, steps)
        return phi_arr
    
    def phi_rot_sph_norm(self, circ_frac, steps=30, phi_0=0.):
        phi_pos_arr = self.phi_rot_arr(circ_frac, steps, phi_0)
        phi_mom_arr = phi_pos_arr - torch.pi
        theta_arr = torch.pi / 2 * torch.ones_like(phi_pos_arr)
        return torch.stack((theta_arr, phi_mom_arr, theta_arr, phi_pos_arr), dim=-1)
    
    def phi_rot_sph_mom(self, circ_frac, steps=30, phi_0=0.):
        phi_mom_arr = self.phi_rot_arr(circ_frac, steps, torch.pi)
        phi_pos_arr = torch.pi / 2 * torch.ones_like(phi_mom_arr)
        theta_arr = torch.pi / 2 * torch.ones_like(phi_mom_arr)
        return torch.stack((theta_arr, phi_mom_arr, theta_arr, phi_pos_arr), dim=-1)
