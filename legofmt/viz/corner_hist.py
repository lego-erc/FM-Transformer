"""``CornerHist``: corner plots and per-step animations of model output.

Calls the LightningModule directly, so ``odeint_conf.return_timesteps`` decides
whether the animation has intermediate frames to draw.
"""

import shutil

import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from flow_matching.utils.manifolds import Euclidean, Sphere
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from legofmt.data.struct import _F
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.geometry.product_manifold import ProductManifold
from legofmt.main.modules import LEGOLtng
from legofmt.viz.plot_geom import PlotGeom

# on top of plot_geom's style (imported above)
plt.rcParams.update({"text.usetex": all(shutil.which(b) for b in ("latex", "dvipng", "gs")), "axes.labelsize": 20})


class CornerHist:
    def __init__(
        self,
        config: dict,
        figsize=(8, 6),
        plot_vars="default",
        title=None,
        anim_save_path=None,
        cube=False,
        corner=True,
        plot_en=False,
        plot_edep=False,
        anim_intermediates=False,
        cutoff_en=10.0,
        **kwargs,
    ):
        if config is not None:
            config["config"].setdefault("odeint_conf", {})["return_timesteps"] = anim_intermediates
            self.model = LEGOLtng(config).to(kwargs.get("device", torch.get_default_device()))
        self.anim_intermediates = anim_intermediates
        self.geom_trafos = GeomTrafos()
        self.disp_man = ProductManifold([Euclidean(), Sphere(), Sphere()], (1, 3, 3))
        self.figsize = figsize
        self.plot_vars = plot_vars
        self.title = title
        self.anim_save_path = anim_save_path
        self.cube = cube
        self.corner_ = corner
        self.plot_en = plot_en
        self.plot_edep = plot_edep
        self.create_handle = lambda col: Line2D([0], [0], linestyle="None", marker="s", markersize=10,
                                                markerfacecolor=col)

    def __call__(self, batch: tuple, truth=None):
        self.fig_sup, self.fig = self.make_fig(self.title, cube=self.cube, corner_=self.corner_)

        def is_t_tup(x):
            return isinstance(x, tuple) and all(isinstance(t, torch.Tensor) for t in x)

        if is_t_tup(batch) and truth is not None and is_t_tup(truth):
            return self._make_anim(lambda i: self.plot_tensors(batch[i], truth[i]), len(batch))

        if is_t_tup(batch) and not self.anim_intermediates:
            if batch[0].ndim > 3:  # sequence of batches

                def draw(i):
                    prepped = self.prep(batch[i])
                    self.fig_sup.suptitle(
                        rf"$\mathrm{{Density:\;}}{self.sols_density.item():.2f}"
                        rf"\mathrm{{,\;Deposited\;Energy\;Mean:\;}}"
                        rf"{self.sols_e_dep.item():.3f}$",
                        fontsize=20,
                    )
                    return prepped

                return self._make_anim(draw, len(batch))
            return self.prep(batch)

        if isinstance(batch, torch.Tensor) and not self.anim_intermediates:
            return self.plot_tensors(batch, truth)

        if self.anim_intermediates:
            sols, _, _ = self.model(batch)
            return self._make_anim(lambda i: self.prep(batch, sols=sols[i]), len(sols))

    def _make_anim(self, draw, frames):
        axes_fig = self.fig[1] if self.cube else self.fig

        def frame(i):
            for axis in axes_fig.get_axes():  # blank the previous frame; draw() builds a fresh axes grid
                axis.clear()
            return draw(i)

        anim = FuncAnimation(self.fig_sup, frame, frames=frames, interval=100, repeat_delay=2000, blit=False)
        if self.anim_save_path:
            anim.save(self.anim_save_path, writer="pillow")
        return anim

    # sols / truth are (B, N, 7) model blocks or (B, N, 8) with the pdgid column; [..., :7] is the block for both

    def plot_tensors(self, sols, truth=None):
        truth_e_dep = truth_cc = None
        if truth is not None:
            sols = sols[:, :truth.shape[1]]
            truth_e_dep, truth_cc = truth[:, 1, 0], truth[..., :7]
        sols_cc = sols[..., :7]
        return self.arrange_plots_(self.fig_sup, self.fig, sols_cc, sols_true=truth_cc,
                                   incoming=_F(sols_cc).in_p if self.cube else None,
                                   data_add=(sols[:, 1, 0], truth_e_dep))

    def prep(self, batch, sols=None):
        if sols is None:
            sols, _, _ = self.model(batch)
        e_dep = sols[:, 1, 0]
        self.sols_density, self.sols_e_dep = sols[:, 0, 0].mean(), e_dep.mean()

        truth = _F(batch[0])
        sols_true = batch[0][..., :7]
        sols_cc = sols[..., :7][:, :sols_true.shape[1]]
        sols_true = torch.where(torch.isnan(sols_cc), torch.nan, sols_true)
        return self.arrange_plots_(self.fig_sup, self.fig, sols_cc, sols_true,
                                   incoming=truth.in_cc if self.cube else None, data_add=(e_dep, truth.edep))

    def make_fig(self, title=None, cube=False, corner_=True):
        title = title or r"$\mathrm{LEGO\;Fixed\;Gun\;with\;Random\;Training,\;Isotropic\;Base\;Noise,\;}t=1.0$"
        fig_w, fig_h = self.figsize
        fig_sup = plt.figure(figsize=(fig_w, fig_h * 2 if cube and corner_ else fig_h))
        _, fig = fig_sup.subfigures(2, 1, height_ratios=[0.1, 0.9])
        fig.legend(handles=[self.create_handle("#FF9D00"), self.create_handle("maroon")],
                   labels=[r"$\mathrm{Model}$", r"$\mathrm{Truth}$"], loc="upper right")
        fig_sup.suptitle(title, fontsize=20)
        if not cube:
            return fig_sup, fig

        fig_t = fig_b = fig
        if corner_:
            fig_t, fig_b = fig.subfigures(2, 1, height_ratios=[0.5, 0.4])
        ax_l, ax_r = fig_b.add_subplot(121, projection="3d"), fig_b.add_subplot(122, projection="3d")
        return fig_sup, (fig, fig_t, PlotGeom(figure=fig_b, ax=ax_l), PlotGeom(figure=fig_b, ax=ax_r))

    @torch.no_grad()
    def make_corner(self, data_cc, fig, color="#FF9D00", data_add=None):
        proj = self.disp_man.projx(_F(data_cc).out_p.reshape(-1, 7))
        data = self.geom_trafos.to_sph(proj[:, 1:7]).cpu().numpy()
        labels = [r"$\theta_\mathrm{mom}$", r"$\phi_\mathrm{mom}$", r"$\theta_\mathrm{pos}$", r"$\phi_\mathrm{pos}$"]
        range_ = [(0.0, np.pi), (-np.pi, np.pi)] * 2
        if self.plot_en:
            labels.append(r"$\hat{E}$")
            range_.append((-0.2, 1.2))
            data = np.concatenate([data, proj[:, 0:1].clamp(0.0, 1.0).cpu().numpy()], axis=-1)
        if self.plot_edep:
            labels.append(r"$E_\mathrm{dep}$")
            range_.append((-0.2, 1.2))
            data_add = data_add.repeat_interleave(data.shape[0] // data_add.shape[0]).unsqueeze(-1).clamp_min(0.0)
            data = np.concatenate([data, data_add.cpu().numpy()], axis=-1)
        return corner.corner(data, bins=99, fig=fig, labelpad=0.01, labels=labels, color=color, max_n_ticks=4,
                             range=range_)

    @torch.no_grad()
    def make_cube(self, data, cube, incoming, color="#FF9D00"):
        if incoming is not None:
            incoming = self.geom_trafos.to_cube(self.disp_man.projx(incoming)[..., 1:7])
        return cube.plot_cube_with_points(self.geom_trafos.to_cube(data[..., 1:7]), incoming=incoming, arr_c=color,
                                          arr_lr=0.0, arr_l=1.0, arr_lw=1.0)

    @torch.no_grad()
    def arrange_plots_(self, fig_sup, fig, sols, sols_true=None, incoming=None, data_add=None):
        if self.cube:
            _, _, pc_s, pc_t = fig
            incoming = incoming[:2**10] if incoming is not None else None
            self.make_cube(sols[:2**10], pc_s, incoming)
            if sols_true is not None:
                self.make_cube(sols_true[:2**10], pc_t, incoming, color="maroon")

        if self.corner_:
            fig_inner = fig[1] if self.cube else fig
            self.make_corner(sols, fig_inner, data_add=data_add[0] if data_add else None)
            if sols_true is not None:
                self.make_corner(sols_true, fig_inner, color="maroon", data_add=data_add[1] if data_add else None)
        return fig_sup
