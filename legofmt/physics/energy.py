import torch
from scipy.constants import physical_constants


class EnergyTrafos:
    def __init__(self):
        self.mass_pdgid = torch.empty(2212 + 1, dtype=torch.float32)
        self.mass_pdgid[11] = torch.tensor(
            [physical_constants["electron mass energy equivalent in MeV"][0]]
        )
        self.mass_pdgid[2112] = torch.tensor(
            [physical_constants["neutron mass energy equivalent in MeV"][0]]
        )
        self.mass_pdgid[22] = 0.

    def p_to_e(self, cc, pdgids, e_kin = True):
        if cc.shape[-1] != 3:
            p, _ = cc.split(3, -1)
        else:
            p = cc
        masses = self.mass_pdgid[pdgids.nan_to_num(22).long().abs()]
        e = torch.sqrt((p**2).sum(dim=-1) + masses**2)
        if e_kin:
            return e - masses
        return e
