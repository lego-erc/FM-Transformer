import torch
from scipy.constants import physical_constants


# pdgid → rest-mass energy [MeV]. Anti-particles use |pdgid|, so positrons share
# the electron entry. Unknown pdgids resolve to mass = 0 in `p_to_e` (so they
# behave like photons — the same default as `nan_to_num(22)` already implies,
# and avoids the uninitialized-memory bug that the previous 2213-slot tensor
# silently produced).
_MASSES_MEV: dict[int, float] = {
    11:   physical_constants["electron mass energy equivalent in MeV"][0],
    22:   0.0,
    2112: physical_constants["neutron mass energy equivalent in MeV"][0],
}


class EnergyTrafos:
    def __init__(self) -> None:
        # Parallel small tensors for vectorised broadcast-match lookup.
        self._pdgids = torch.tensor(list(_MASSES_MEV.keys()), dtype=torch.long)
        self._masses_mev = torch.tensor(list(_MASSES_MEV.values()), dtype=torch.float32)

    def p_to_e(self, cc: torch.Tensor, pdgids: torch.Tensor, e_kin: bool = True) -> torch.Tensor:
        p = cc if cc.shape[-1] == 3 else cc.split(3, -1)[0]
        p_abs = pdgids.nan_to_num(22).long().abs()
        # Lookup table is tiny (len ~3), so a (..., N_known) broadcast-equal
        # is cheaper and clearer than a global-shape indexed gather.
        known = self._pdgids.to(p_abs.device)
        masses_mev = self._masses_mev.to(p.device)
        match = (p_abs.unsqueeze(-1) == known).to(masses_mev.dtype)
        masses = (match * masses_mev).sum(-1)
        e = torch.sqrt((p**2).sum(dim=-1) + masses**2)
        return e - masses if e_kin else e
