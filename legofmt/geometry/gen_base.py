import torch
import torch.nn.functional as F

from legofmt.geometry.geom_trafos import GeomTrafos


class GenerateBase:
    """Base-distribution sampler for the flow.

    Produces the noise prior the flow starts from, matched to the model's
    manifold and energy scale. :meth:`poles` builds the conditioning-aware
    prior (von Mises-Fisher directions and positions around the incoming
    particle's ray); :meth:`iso` is the role-agnostic isotropic prior used
    for arbitrary masks.
    """

    def __init__(self, config: dict):
        """Reads base-distribution settings from ``config["base_conf"]``.

        Args:
            config: full model config. The ``base_conf`` block selects the
                prior (only ``"poles"`` is currently supported) and its
                concentration, energy-scale, and beam-spread parameters.

        Raises:
            ValueError: if a base distribution other than ``"poles"`` is
                requested.
        """
        self.geom_trafos = GeomTrafos()
        self.cutoff_mev = config["dl_conf"]["lds_args"].get("cutoff_mev", 10.0)
        base_conf = config.get("base_conf")
        base_dist = base_conf.get("base_dist", "poles")
        self.tanh_theta = base_conf.get("tanh_theta", False)
        self.kappa = base_conf.get("kappa", torch.tensor(10.0))
        self.e_dep_max = base_conf.get("e_dep_max", 1.)
        self.bs_frac = base_conf.get("bs_frac", 0.0)
        self.scale_dist = base_conf.get("scale_dist", "trunc_norm")

        if base_dist != "poles":
            raise ValueError("base_dist's other than poles are currently deprecated")

    def __call__(self, shape, incoming_rt):
        """Samples the configured prior; see :meth:`poles`."""
        return self.poles(shape, incoming_rt=incoming_rt)

    @torch.no_grad()
    def rd_scale(self, shape, e_in):
        """Draws a radial energy scale and applies it to ``e_in``.

        The multiplier follows ``scale_dist`` (truncated-normal, uniform,
        or a softened half-normal), keeping sampled energies at or below
        the incoming energy.

        Args:
            shape: leading shape of the sampled tensor.
            e_in: per-event incoming energy.

        Returns:
            Scaled energy ``(*shape, 1)``.

        Raises:
            ValueError: if ``scale_dist`` is unknown.
        """
        if self.scale_dist == "trunc_norm":
            u = torch.nn.init.trunc_normal_(
                e_in.new_empty((*shape, 1)), std=1.0, a=-1.0, b=0.0,
            ) + 1.0
        elif self.scale_dist == "uniform":
            u = torch.rand((*shape, 1), device=e_in.device)
        elif self.scale_dist == "sm_norm":
            u = 1 - torch.tanh(torch.randn((*shape, 1), device=e_in.device).abs() / 2)
        else:
            raise ValueError("Unknown scale_dist")
        return e_in.view(-1, 1, 1) * u

    @torch.no_grad()
    def poles(self, shape, incoming_rt, **kwargs):
        """Conditioning-aware prior centred on the incoming particle's ray.

        Samples scaled energies and von Mises-Fisher directions and
        positions concentrated (by ``kappa``) around the incoming
        momentum and entry point, then prepends the unchanged conditioning
        rows.

        Args:
            shape: leading shape of the generated slots.
            incoming_rt: incoming-particle row ``[energy, mom(3), pos(3)]``.

        Returns:
            Base features with the conditioning rows prepended.
        """
        e_in = incoming_rt[..., 0:1]
        p_cc = F.normalize(incoming_rt[..., 1:4], dim=-1)
        loc_cc = incoming_rt[..., -3:]
        e_sc = self.rd_scale(shape, torch.ones_like(e_in))
        x = self.geom_trafos.sample(shape, loc_cc, self.kappa, self.bs_frac, self.tanh_theta)
        p_ = self.geom_trafos.sample(shape, p_cc, self.kappa, 0.0, self.tanh_theta)
        base = torch.cat((e_sc, p_, x), dim=-1)
        base = torch.cat((incoming_rt, base), dim=1)
        return base

    @torch.no_grad()
    def iso(self, shape, device):
        """Role-agnostic isotropic prior for arbitrary masks.

        Uniform energy in ``[0, 1]`` with isotropic directions and
        positions on the sphere, independent of any conditioning.

        Args:
            shape: leading shape of the sampled tensor.
            device: device to sample on.

        Returns:
            Base features ``(*shape, 7)`` as ``[energy, dir(3), pos(3)]``.
        """
        e = torch.rand((*shape, 1), device=device)
        p = self.geom_trafos.sample_iso(shape, 1, device=device)
        x = self.geom_trafos.sample_iso(shape, 1, device=device)
        return torch.cat((e, p, x), dim=-1)

    @torch.no_grad()
    def insert_add(self, base):
        """Seeds the energy-deposition slot with random noise in place.

        Args:
            base: base features whose slot ``1`` holds the edep value.

        Returns:
            The same tensor with its edep slot overwritten.
        """
        base[:, 1, 0] = self.e_dep_max * torch.sigmoid(torch.randn_like(base[:, 1, 0]))
        return base
