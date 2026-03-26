import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from flow_matching.utils.manifolds import Sphere
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from ..main.modules import LEGOLtng
from ..geometry.path_sample_mult import ProductManifold
from ..geometry.vmf_sampling import VMF
from .plot_geom import PlotGeom

plt.rcParams.update(
    {
        "axes.labelpad": 8,
        "text.usetex": True,
        "font.serif": "Computer Modern",
        "axes.labelsize": 20,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


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
        return_base=False,
        anim_intermediates=False,
        cutoff_en=10.0,
        **kwargs,
    ):
        device = kwargs.get("device", torch.get_default_device())
        if config is not None:
            config["config"]["odeint_conf"] = config["config"].get("odeint_conf", {})
            config["config"]["odeint_conf"].update(
                {
                    "return_base": return_base,
                    "return_timesteps": anim_intermediates,
                }
            )
            self.model = LEGOLtng(config).to(device)
            self.cutoff_en = config["config"]["dl_conf"]["lds_args"].get("cutoff_mev", cutoff_en)
        else:
            self.cutoff_en = cutoff_en
        self.anim_intermediates = anim_intermediates
        self.vmf_utils = VMF()
        self.disp_man = ProductManifold([Sphere(), Sphere()], (3, 3))
        self.figsize = figsize
        self.plot_vars = plot_vars
        self.title = title
        self.anim_save_path = anim_save_path
        self.cube = cube
        self.corner_ = corner
        self.plot_en = plot_en
        self.plot_edep = plot_edep

        self.create_handle = lambda col: Line2D(
            [0], [0], linestyle="None", marker="s", markersize=10, markerfacecolor=col
        )

    def __call__(self, batch: tuple, truth=None):
        self.fig_sup, self.fig = self.make_fig(
            self.title, cube=self.cube, corner_=self.corner_
        )

        def is_t(x):
            return isinstance(x, torch.Tensor)

        def is_t_tup(x):
            return isinstance(x, tuple) and all(is_t(t) for t in x)

        if is_t_tup(batch) and truth is not None and is_t_tup(truth):

            def anim_wrapper_(i):
                for axis in (self.fig[1] if self.cube else self.fig).get_axes():
                    axis.clear()
                return self.plot_tensors(batch[i], truth[i])

            return self._make_anim(anim_wrapper_, len(batch))

        if is_t_tup(batch) and not self.anim_intermediates:
            if batch[0].ndim > 3:  # sequence of batches

                def anim_wrapper_(i):
                    for axis in (self.fig[1] if self.cube else self.fig).get_axes():
                        axis.clear()
                    prepped = self.prep(batch[i])
                    self.fig_sup.suptitle(
                        rf"$\mathrm{{Density:\;}}{str(self.sols_density.item())[:4]}\mathrm{{,\;Deposited\;Energy\;Mean:\;}}{str(self.sols_e_dep.item())[:4]}$",
                        fontsize=20,
                    )
                    return prepped

                return self._make_anim(anim_wrapper_, len(batch))
            return self.prep(batch)

        if is_t(batch) and not self.anim_intermediates:
            return self.plot_tensors(batch, truth)

        if self.anim_intermediates:
            sols = self.model(batch)

            def anim_wrapper_(i):
                for axis in (self.fig[1] if self.cube else self.fig).get_axes():
                    axis.clear()
                return self.prep(batch, sols=sols[i])

            return self._make_anim(anim_wrapper_, len(sols))

    def _make_anim(self, func, frames):
        anim = FuncAnimation(
            self.fig_sup,
            func,
            frames=frames,
            interval=100,
            repeat_delay=2000,
            blit=False,
        )
        if self.anim_save_path:
            anim.save(self.anim_save_path, writer="pillow")
        return anim

    def plot_tensors(self, sols, truth=None):
        if truth is not None:
            truth_e_dep = truth[:, 1, 0]
            sols = sols[:, : truth.shape[1]]
            truth = truth[:, 3:].contiguous().view(-1, 6)
        else:
            truth_e_dep = None

        truth_e_dep_tuple = (sols[:, 1, 0], truth_e_dep)
        en_in = sols[:, 2, :3].norm(dim=-1, keepdim=True)
        en_in = en_in.expand_as(sols[:, 3:, 0]).reshape(-1, 1)
        sols = sols[:, 3:].reshape(-1, 6)
        incoming = sols[:, 2:3] if self.cube else None

        norm_fac = torch.log(self.cutoff_en / en_in.clamp_min(1e-3)).abs()
        sols = sols / norm_fac
        if truth is not None:
            truth = truth / norm_fac

        return self.arrange_plots_(
            self.fig_sup,
            self.fig,
            sols,
            sols_true=truth,
            incoming=incoming,
            data_add=truth_e_dep_tuple,
        )

    def prep(self, batch, sols=None):
        if sols is None:
            sols = self.model(batch)
        e_dep = sols[:, 1, 0]

        sols_true = (
            batch[0][:, 3:] if batch[0].shape[-1] == 6 else batch[0][:, 3:, -7:-1]
        )

        en_in = sols[:, 2:3, :3].norm(dim=-1, keepdim=True)
        sols = sols[:, 3:]
        sols[..., :3] -= sols[..., :3] / sols[..., :3].norm(dim=-1, keepdim=True)
        sols_true[..., :3] -= sols_true[..., :3] / sols_true[..., :3].norm(dim=-1, keepdim=True)
        sols = sols[:, : sols_true.shape[1]]
        en_in = en_in.expand_as(sols[..., :1])
        sols_true = torch.where(torch.isnan(sols), torch.nan, sols_true)

        norm_fac = torch.log(en_in.clamp_min(1e-3) / self.cutoff_en)
        sols = sols / norm_fac
        sols_true = sols_true / norm_fac

        return self.arrange_plots_(
            self.fig_sup,
            self.fig,
            sols.reshape(-1, 6),
            sols_true.reshape(-1, 6),
            incoming=(batch[0][:, 2:3, -7:-1] if self.cube else None),
            data_add=(e_dep, batch[0][:, 1, 1]),
        )

    def make_fig(self, title=None, cube=False, corner_=True):
        title = title or (
            r"$\mathrm{LEGO\;Fixed\;Gun\;with\;Random\;Training,\;"
            + r"Isotropic\;Base\;Noise,\;}t=1.0$"
        )

        fig_w, fig_h = self.figsize
        fig_sup = plt.figure(figsize=(fig_w, fig_h * 2 if cube and corner_ else fig_h))
        _, fig = fig_sup.subfigures(2, 1, height_ratios=[0.1, 0.9])

        fig.legend(
            handles=[self.create_handle("#FF9D00"), self.create_handle("maroon")],
            labels=[r"$\mathrm{Model}$", r"$\mathrm{Truth}$"],
            loc="upper right",
        )
        fig_sup.suptitle(title, fontsize=20)

        if cube:
            if corner_:
                fig_t, fig_b = fig.subfigures(2, 1, height_ratios=[0.5, 0.4])
                ax_l = fig_b.add_subplot(121, projection="3d")
                ax_r = fig_b.add_subplot(122, projection="3d")
                return fig_sup, (
                    fig,
                    fig_t,
                    PlotGeom(figure=fig_b, ax=ax_l),
                    PlotGeom(figure=fig_b, ax=ax_r),
                )
            else:
                ax_l = fig.add_subplot(121, projection="3d")
                ax_r = fig.add_subplot(122, projection="3d")
                return fig_sup, (
                    fig,
                    fig,
                    PlotGeom(figure=fig, ax=ax_l),
                    PlotGeom(figure=fig, ax=ax_r),
                )

        return fig_sup, fig

    @torch.no_grad()
    def make_corner(self, data_cc, fig, color="#FF9D00", data_add=None):
        data = self.vmf_utils.to_sph(self.disp_man.projx(data_cc)).cpu().numpy()
        labels = [
            r"$\theta_\mathrm{mom}$",
            r"$\phi_\mathrm{mom}$",
            r"$\theta_\mathrm{pos}$",
            r"$\phi_\mathrm{pos}$",
        ]
        range_ = [(0.0, np.pi), (-np.pi, np.pi)] * 2
        
        if self.plot_en is not False:
            labels += [
                r"$-\log \frac{\| \vec{p} \|_2}{\| \vec{p}_\mathrm{incoming} \|_2}$"
            ]
            data_en = data_cc[..., :3].norm(dim=-1, keepdim=True)
            range_ += [(-0.2, 1.2)]
            data = np.concatenate([data, data_en.cpu().numpy()], axis=-1)
        if self.plot_edep is not False:
            labels += [r"$E_\mathrm{dep}$"]
            data_add = data_add.repeat((data.shape[0] // data_add.shape[0])).unsqueeze(
                -1
            )
            range_ += [(-0.2, 1.2)]
            data = np.concatenate([data, data_add.cpu().numpy()], axis=-1)

        return corner.corner(
            data,
            bins=2**6,
            fig=fig,
            labelpad=0.01,
            labels=labels,
            color=color,
            max_n_ticks=4,
            range=range_,
        )

    @torch.no_grad()
    def make_cube(self, data, cube, incoming, color="#FF9D00"):
        if incoming is not None:
            incoming = self.vmf_utils.to_cube(self.disp_man.projx(incoming))
        return cube.plot_cube_with_points(
            self.vmf_utils.to_cube(data),
            incoming=incoming,
            arr_c=color,
            arr_lr=0.0,
            arr_l=1.0,
            arr_lw=1.0,
        )

    @torch.no_grad()
    def arrange_plots_(
        self, fig_sup, fig, sols, sols_true=None, incoming=None, data_add=None,
    ):
        if self.cube:
            _, fig_inner, pc_s, pc_t = fig
            self.make_cube(
                sols[: 2**10], pc_s, incoming[: 2**10] if incoming is not None else None
            )
            if sols_true is not None:
                self.make_cube(
                    sols_true[: 2**10],
                    pc_t,
                    incoming[: 2**10] if incoming is not None else None,
                    color="maroon",
                )

        if self.corner_:
            fig_inner = fig[1] if self.cube else fig
            self.make_corner(
                sols, fig_inner, data_add=data_add[0] if data_add else None
            )
            if sols_true is not None:
                self.make_corner(
                    sols_true,
                    fig_inner,
                    color="maroon",
                    data_add=data_add[1] if data_add else None,
                )

        return fig_sup
