import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from flow_matching.utils.manifolds import Euclidean, Sphere
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from ..main.modules import LEGOLtng
from ..geometry.energy_proj import EnergyProjections
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
        full_gen=False,
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
        **kwargs,
    ):
        device = kwargs.get("device", torch.get_default_device())
        self.proj_en_out = config["config"]["model_conf"].get("proj_en_out", False)
        self.proj_en = config["config"]["model_conf"].get("proj_en", False)
        config["config"]["odeint_conf"] = config["config"].get("odeint_conf", {})
        config["config"]["odeint_conf"].update({
            "return_base": return_base,
            "return_timesteps": anim_intermediates,
        })
        self.anim_intermediates = anim_intermediates
        self.vmf_utils = VMF()
        if isinstance(full_gen, str):
            from ..main.generate import GenerateOut

            self.model = GenerateOut(config, full_gen, device=device)
        else:
            self.model = LEGOLtng(config).to(device)
        self.disp_man = ProductManifold([Sphere(), Sphere()], (3, 3))
        self.en_norm = EnergyProjections(self.proj_en)
        if self.proj_en_out is not False:
            self.en_norm_out = EnergyProjections(self.proj_en_out)
        self.figsize = figsize
        self.plot_vars = plot_vars
        self.title = title
        self.anim_save_path = anim_save_path
        self.cube = cube
        self.corner_ = corner
        self.plot_en = plot_en
        self.plot_edep = plot_edep

        self.create_handle = lambda col: Line2D(
            [0],
            [0],
            linestyle="None",
            marker="s",
            markersize=10,
            markerfacecolor=col,
        )

    def __call__(
        self,
        batch: tuple,
    ):
        self.fig_sup, self.fig = self.make_fig(
            self.title, cube=self.cube, corner_=self.corner_
        )

        if (
            isinstance(batch[0], torch.Tensor)
            and batch[0].ndim <= 3
            and not self.anim_intermediates
        ):
            return self.prep(batch)

        if self.anim_intermediates:
            sols = self.model(batch)

        def anim_wrapper_(i):
            for axis in (self.fig[1] if self.cube else self.fig).get_axes():
                axis.clear()
            if self.anim_intermediates:
                return self.prep(batch, sols=sols[i])
            prepped = self.prep(batch[i])
            self.fig_sup.suptitle(r"$\mathrm{Density:\;}" + str(self.sols_density.item())[:4] +
                                  r"\mathrm{,\;Deposited\;Energy\;Mean:\;}" + str(self.sols_e_dep.item())[:4] + r"$", fontsize=20)
            return prepped

        anim = FuncAnimation(
            self.fig_sup,
            anim_wrapper_,
            frames=torch.arange(len(sols if self.anim_intermediates else batch)),
            interval=100,
            repeat_delay=2000,
            blit=False,
        )

        if self.anim_save_path is not None:
            anim.save(self.anim_save_path, writer="pillow")

        return anim

    def prep(self, batch):
        sols = self.model(batch)
        data_add_f = sols[:, 1, 0]
        self.sols_density = sols[0, 0, 0]
        self.sols_e_dep = sols[:, 1, 0].mean()
        sols = sols[:, 3:]
        sols_true = batch[0][:, 3:] if batch[0].shape[-1] == 6 else batch[0][:, 3:, -7:-1]
        sols = sols[:, :sols_true.shape[1]]
        sols_true = torch.where(torch.isnan(sols), torch.nan, sols_true)
        if self.proj_en_out is not False:
            sols = self.en_norm_out(sols)
            sols_true = self.en_norm_out(sols_true)
        return self.arrange_plots_(
            self.fig_sup,
            self.fig,
            sols.contiguous().view(-1, 6),
            sols_true.contiguous().view(-1, 6),
            incoming=(batch[0][:, 2:3, -7:-1] if self.cube else None),
            data_add=(data_add_f, batch[0][:, 1, 1]),
        )

    def make_fig(self, title=None, cube=False, corner_=True):
        if title is None:
            title = (
                r"$\mathrm{LEGO\;Fixed\;Gun\;with\;Random\;Training,\;"
                + r"Isotropic\;Base\;Noise,\;}t=1.0$"
            )

        fig_w, fig_h = self.figsize
        fig_sup = plt.figure(figsize=(fig_w, fig_h * 2 if cube and corner_ else fig_h))
        _, fig = fig_sup.subfigures(2, 1, height_ratios=[0.1, 0.9])

        orange_handle = self.create_handle("#FF9D00")
        maroon_handle = self.create_handle("maroon")

        fig.legend(
            handles=[orange_handle, maroon_handle],
            labels=[r"$\mathrm{Model}$", r"$\mathrm{Truth}$"],
            loc="upper right",
        )
        fig_sup.suptitle(title, fontsize=20)

        if cube and corner_:
            fig_t, fig_b = fig.subfigures(2, 1, height_ratios=[0.5, 0.4])
            ax_l = fig_b.add_subplot(121, projection="3d")
            ax_r = fig_b.add_subplot(122, projection="3d")

            pc_s = PlotGeom(figure=fig_b, ax=ax_l)
            pc_t = PlotGeom(figure=fig_b, ax=ax_r)

            return fig_sup, (fig, fig_t, pc_s, pc_t)

        if cube and not corner_:
            ax_l = fig.add_subplot(121, projection="3d")
            ax_r = fig.add_subplot(122, projection="3d")

            pc_s = PlotGeom(figure=fig, ax=ax_l)
            pc_t = PlotGeom(figure=fig, ax=ax_r)

            return fig_sup, (fig, fig, pc_s, pc_t)

        return fig_sup, fig

    @torch.no_grad()
    def make_corner(self, data_cc, fig, color="#FF9D00", data_add=None):
        if self.plot_vars != "raw_full":
            data_cc_norm = self.disp_man.projx(data_cc)
            data = self.vmf_utils.to_sph(data_cc_norm).cpu().numpy()
            labels = [
                r"$\theta_\mathrm{mom}$",
                r"$\phi_\mathrm{mom}$",
                r"$\theta_\mathrm{pos}$",
                r"$\phi_\mathrm{pos}$",
            ]
            range_ = [(0.0, np.pi), (-np.pi, np.pi)] * 2
        elif self.plot_vars == "raw_full":
            data = self.vmf_utils.to_cube(data_cc).cpu().numpy()
            labels = [r"$p_x$", r"$p_y$", r"$p_z$", r"$x$", r"$y$", r"$z$"]
            range_ = [(-3.0, 3.0)] * 3 + [(-1.1, 1.1)] * 3
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
            _, fig, pc_s, pc_t = fig
            self.make_cube(sols[:2**10], pc_s, incoming[:2**10])
            if sols_true is not None:
                self.make_cube(sols_true[:2**10], pc_t, incoming[:2**10], color="maroon")
        if self.corner_:
            self.make_corner(sols, fig, data_add=data_add[0])
            if sols_true is not None:
                self.make_corner(sols_true, fig, color="maroon", data_add=data_add[1])

        return fig_sup
