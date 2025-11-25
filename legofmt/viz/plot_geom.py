import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from ..geometry.vmf_sampling import VMF

plt.rcParams.update(
    {
        "axes.labelpad": 8,
        "text.usetex": True,
        "font.serif": "Computer Modern",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


class PlotGeom:
    def __init__(self, figure=None, ax=None):
        self.figure = figure
        self.ax = ax
        self.vmf_utils = VMF()

    def do_ax(self):
        if self.figure is None:
            self.figure = plt.figure(figsize=(8, 8))
            self.ax = self.figure.add_subplot(projection="3d")
        else:
            self.ax.cla()
        return self.ax

    def plot_points(
        self,
        ax: plt.Axes,
        coord: torch.Tensor,
        incoming: torch.Tensor | None = None,
        arr_lw: float = 1.0,
        arr_l: float = 1.0 / 4,
        arr_lr: float = 0.04,
        arr_c: str = "y",
        lims: float = 1.4,
    ) -> None:
        try:
            vec, pts = coord.movedim(-1, 0).split(3, 0)
            ax.quiver(
                *pts.cpu().numpy(),
                *vec.cpu().numpy(),
                length=arr_l,
                normalize=True,
                arrow_length_ratio=arr_lr,
                linewidths=arr_lw,
                color=arr_c,
                zorder=3,
                alpha=0.2,
            )
        except ValueError:
            pts = coord.movedim(-1, 0)

        ax.scatter(*pts.cpu().numpy(), color=arr_c, depthshade=True, zorder=3)

        if incoming is not None:
            vec_in, pts_in = incoming.split(3, -1)
            ax.scatter(
                *pts_in.movedim(-1, 0).cpu().numpy(),
                color="r",
                depthshade=True,
                zorder=3,
            )
            ax.quiver(
                *pts_in.movedim(-1, 0).cpu().numpy(),
                *-vec_in.movedim(-1, 0).cpu().numpy(),
                length=2 * arr_l,
                normalize=True,
                arrow_length_ratio=0.0,
                linewidths=2.0,
                color="r",
                zorder=3,
            )

        ax.set_xlim(-lims, lims)
        ax.set_ylim(-lims, lims)
        ax.set_zlim(-lims, lims)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
        ax.set_box_aspect((1, 1, 1))

        return ax

    def plot_cube_with_points(
        self,
        coords: torch.Tensor,
        incoming: torch.Tensor | None = None,
        arr_lw: float = 1.0,
        arr_l: float = 1.0 / 4,
        arr_lr: float = 0.1,
        arr_c: str = "y",
    ) -> None:
        vertices = torch.unique(
            torch.combinations(
                torch.tensor([-1, 1, -1, 1]), r=3, with_replacement=True
            ),
            dim=0,
        )
        cc_edges = vertices[torch.combinations(torch.arange(8), r=2)]
        cc_edges = cc_edges[cc_edges.diff(dim=1).squeeze(1).abs().sum(dim=-1) == 2.0]

        ax = self.do_ax()
        ax.add_collection(
            Line3DCollection(
                cc_edges.cpu().numpy(), linestyles="--", colors="k", linewidths=1
            )
        )

        return self.plot_points(ax, coords, incoming, arr_lw, arr_l, arr_lr, arr_c)

    def plot_sphere_with_points(
        self,
        coords: torch.Tensor,
        incoming: torch.Tensor | None = None,
        arr_lw: float = 1.0,
        arr_l: float = 1.0 / 4,
        arr_lr: float = 0.1,
        arr_c: str = "b",
    ) -> None:
        pi_range = torch.linspace(0, torch.pi, 37)
        grid_sph = torch.stack(
            torch.meshgrid(pi_range, 2 * pi_range, indexing="xy"), dim=-1
        )
        cc = self.vmf_utils.to_cc(grid_sph)

        ax = self.do_ax()
        for i in range(0, 36, 2):
            ax.plot(
                *cc[i].movedim(-1, 0).cpu().numpy(),
                linestyle="--",
                color="k",
                linewidth=0.5,
                alpha=0.5,
            )
            ax.plot(
                *cc[:, i].movedim(-1, 0).cpu().numpy(),
                linestyle="--",
                color="k",
                linewidth=0.5,
                alpha=0.5,
            )

        return self.plot_points(ax, coords, incoming, arr_lw, arr_l, arr_lr, arr_c, 1.0)
