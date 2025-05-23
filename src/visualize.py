from random import sample
import sqlite3
from typing import Any
from matplotlib import cm
from matplotlib import colormaps
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, shapiro
from scipy.spatial import Delaunay
from scipy.interpolate import CubicSpline, griddata
from sklearn.preprocessing import StandardScaler
from src.instance_builder import build_instances
from src.scent_marks import ScentMarks
from src.simulation import Simulation
from src.wolf import Wolf


"""
This file contains methods used to generate visualizations and graphs
"""


db_path: str = "data/simulations.sqlite"

fontsize_label: int = 18
fontsize_title: int = 18
fontsize_legend: int = 14
fig_height: int = 7
ax_width: int = 9
ax_width_wide: int = 15


def calculate_MSD_in_time(sim_ids: list[int], db_name: str = db_path) -> pd.DataFrame:
    # calculates mean square displacement of the border location in time for all passed ids
    with sqlite3.connect(db_name) as conn:
        data: pd.DataFrame = pd.read_sql_query(
            f"""SELECT * FROM simulations
            WHERE id IN ({", ".join(["?"] * len(sim_ids))})
            """, 
            params=sim_ids,
            con=conn)


    return data[["id", "tick", "border"]].groupby(by=["id", "tick"], as_index=False).var(ddof=0)


def MSD_for_periods(period_lengths: list[float] = [0.187, 0.437]) -> None:
    # plots for each period length a plot of log mean square displacement 
    # with regard to L (x axis) and d (color)
    # interpolates points with cubic spline for every diffusion coefficient for clarity

    with sqlite3.connect(db_path) as conn:
        data: pd.DataFrame = pd.read_sql_query(
                "SELECT * FROM statistics s JOIN runs r ON s.id = r.id ", 
                con=conn
                )
        
    # get colors for diffusion coefficient values
    cmap = colormaps["viridis_r"]
    num_colors: int = len(set(data["d"]))
    colors = [cmap(i / (num_colors - 1 )) for i in range(num_colors)] 

    fig, axes = plt.subplots(
        1, len(period_lengths), 
        figsize=(ax_width*len(period_lengths), fig_height), 
        constrained_layout=True)

    # plot for each period length
    for i, T in enumerate(period_lengths):
        # plot every diffusion coefficient as separate color
        for d, color in zip(sorted(set(data["d"])), colors):
            # get only needed data
            data_tmp = data[(data["d"] == d) & (data["T"] == T)].sort_values("L")

            # cubic spline
            x = data_tmp["L"]
            y = np.log(data_tmp["border_msd"])
            cs = CubicSpline(x, y)
            x_new = np.linspace(x.min(), x.max(), 1000)
            y_new = cs(x_new)

            # plot simulated values and spline            
            axes[i].scatter(x, y, color=color, marker="o", s=15)  
            axes[i].plot(x_new, y_new, color=color)  
        
    # set labels and axes
    ylim_min = min([ax.get_ylim()[0] for ax in axes])
    ylim_max = max([ax.get_ylim()[1] for ax in axes])
    axes[0].set_ylabel(r"$\operatorname{log}(\operatorname{border\_msd})$", fontsize=20)    
    for ax, T in zip(axes, period_lengths):
        ax.set_title(f"T = {T}", fontsize=fontsize_title)
        ax.set_xlabel(r"$L$", fontsize=fontsize_label)
        ax.set_ylim(ylim_min, ylim_max)

    # create legend
    custom_legend = [
        Line2D([0], [0], color="black", marker="o", markersize=4, linestyle="None", label="simulated values"),
        Line2D([0], [0], color="black", label="cubic spline")
    ]
    axes[-1].legend(handles=custom_legend, fontsize=fontsize_legend)
    
    # add colorbar
    norm: mcolors.Normalize = mcolors.Normalize(vmin=data["d"].min(), vmax=data["d"].max())  
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="vertical", fraction=0.05, pad=0.05)
    cbar.set_label(r"$d$", fontsize=fontsize_label)
    
    # plt.savefig("figs/msd_for_periods.png")
    plt.show()


def MSD_dimensionless_groups() -> None:
    # plots for each period length a plot of log mean square displacement 
    # with regard to beta_1 and beta_2
    # colored by 1/T (gamma), adds best fit 5th degree polynomial for clarity

    with sqlite3.connect(db_path) as conn:
        data: pd.DataFrame = pd.read_sql_query(
                "SELECT * FROM statistics s JOIN runs r ON s.id = r.id ", con=conn
                )
        
    poly_degree: int = 5
    
    # create transformations needed
    data["gamma"] = (1 / data["T"])
    data["beta_1"] = data["d"] / (1 * (data["L"] ** 2))     # decay = 1 in dimensionless form
    data["beta_2"] = (data["T"] * data["d"]) / (data["L"] ** 2)
    data["border_msd"] = np.log(data["border_msd"])
    
    # colors for each value of 1/T (gamma)
    cmap = colormaps["viridis_r"]
    colors: list[tuple[float, float, float, float]] = [
        cmap(i / (len(set(data["gamma"])) - 1)) 
        for i in range(len(set(data["gamma"])))
        ] 
    
    fig, axes = plt.subplots(1, 2, figsize=(2*ax_width, fig_height), constrained_layout=True)

    # for each dimensionless group (beta)
    for i, beta in enumerate(["beta_1", "beta_2"]): 
        # for each value of gamma
        for gamma, color in zip(sorted(set(data["gamma"])), colors):
            # extract needed data
            data_tmp = data[data["gamma"] == gamma]

            # fit polynomial of 5-th degree
            coefficients = np.polyfit(data_tmp[beta], data_tmp["border_msd"], poly_degree)
            polynomial = np.poly1d(coefficients)
            x_fit = np.linspace(min(data_tmp[beta]), max(data_tmp[beta]), 100)
            y_fit = polynomial(x_fit)

            # plot data and polynomial fit
            axes[i].plot(x_fit, y_fit, color=color, lw=1)
            axes[i].scatter(data_tmp[beta], data_tmp["border_msd"], marker="o", s=15, color=color) 
    
            axes[i].set_xlabel(rf"$\{beta}$", fontsize=fontsize_label)

    # ylabel and legend
    axes[0].set_ylabel(r"$\operatorname{log}(\operatorname{border\_msd})$", fontsize=18)
    
    custom_legend = [
        Line2D([0], [0], color="black", marker="o", markersize=4, linestyle="None", label="simulated values"),
        Line2D([0], [0], color="black", label="best fit $P_5(k_1)$")
    ]
    axes[1].legend(handles=custom_legend, fontsize=fontsize_legend)
    
    # colorbar
    norm = mcolors.Normalize(vmin=data["gamma"].min(), vmax=data["gamma"].max())  
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="vertical", fraction=0.05, pad=0.05)
    cbar.set_label(r"$\gamma$", fontsize=18)
        
    # plt.savefig("figs/msd_beta.png")
    plt.show()
       

def buffer_width_vector_field() -> None:
    # plot of 
    # (L, T, d) -> (
    #       ( bwz_end(L + ΔL, T, d), 
    #         bwz_end(L, T + ΔT, d), 
    #         bwz_end(L, T, d + Δd)
    #       ) - bwz_end(L, T, d)
    #   ) / bwz_end(L, T, d)
    db: str = "data/simulations.sqlite"
        
    with sqlite3.connect(db) as conn:
        data: pd.DataFrame = pd.read_sql_query(
                "SELECT * FROM statistics s JOIN runs r ON s.id = r.id ", con=conn
                )
        

    def closest_greater_value(value: float, column: pd.Series):
        # return the closest greater value if available, else None
        sorted_values: np.ndarray = np.sort(column)        
        greater_values: np.ndarray = sorted_values[sorted_values > value]
        return greater_values[0] if len(greater_values) > 0 else None
        
    
    def buffer_widths_after_increment(L: float, T: float, d: float, data: pd.DataFrame) -> tuple[float, float, float]:
        if L == data["L"].max() or T == data["T"].max() or d == data["d"].max():
            return None, None, None     # no increment can be done
        
        # compute buffer zone widths after one parameter is increment
        bwz_new: tuple[float, float, float] = (
            data[(data["L"] == closest_greater_value(L, data["L"])) & (data["T"] == T) & (data["d"] == d)]["buffer_width"].iloc[0],
            data[(data["L"] == L) & (data["T"] == closest_greater_value(T, data["T"])) & (data["d"] == d)]["buffer_width"].iloc[0],
            data[(data["L"] == L) & (data["T"] == T) & (data["d"] == closest_greater_value(d, data["d"]))]["buffer_width"].iloc[0]
        )

        # relative change
        bwz_old: float = data[(data["L"] == L) & (data["T"] == T) & (data["d"] == d)]["buffer_width"].iloc[0]
        print(bwz_new)
        return ((bwz - bwz_old) / bwz_old for bwz in bwz_new) 
    

    data[["L", "T", "d"]] = data[["L", "T", "d"]].round(2)

    # compute relative change in buffer zone width for increments of each parameter 
    # for each configuration
    increments_L = []
    increments_T = []
    increments_d = []
    for i, row in data[["L", "T", "d"]].iterrows():  
        inc_L, inc_T, inc_d = buffer_widths_after_increment(row["L"], row["T"], row["d"], data)
        increments_L.append(inc_L)
        increments_T.append(inc_T)
        increments_d.append(inc_d)
    
    # arrange into grid
    grid = pd.DataFrame(data={
        "x": data["L"],
        "y": data["T"],
        "z": data["d"],
        "u": pd.to_numeric(np.array(increments_L), errors="coerce"),
        "v": pd.to_numeric(np.array(increments_T), errors="coerce"),
        "w": pd.to_numeric(np.array(increments_d), errors="coerce")
        }).dropna()
        
    # visualize
    fig = plt.figure(figsize=(ax_width*1.5, ax_width*1.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(grid["x"], grid["y"], grid["z"], grid["u"], grid["v"], grid["w"], length=0.2, linewidths=1)
    ax.set_xlabel(r"$L$", fontsize=fontsize_label)
    ax.set_ylabel(r"$T$", fontsize=fontsize_label)
    ax.set_zlabel(r"$d$", fontsize=fontsize_label)
    
    
    # plt.savefig("figs/buffer_width_relative_changes.png")
    plt.show()

    
def plot_3D_functions() -> None:
    def mesh_3d(data: pd.DataFrame, x_lab: str, y_lab: str, z_lab: str, 
                log: bool = False, savetitle: str = "3D_fig") -> None:
        # creates surface with mesh
        x = data[x_lab].to_numpy()
        y = data[y_lab].to_numpy()
        z = data[z_lab].to_numpy()
        
        # triangulation
        points2D = np.vstack([x, y]).T
        tri = Delaunay(points2D)

        # 3D surface
        surface = go.Mesh3d(
            x=x, y=y, z=z,
            i=tri.simplices[:, 0], 
            j=tri.simplices[:, 1], 
            k=tri.simplices[:, 2],
            intensity=np.log10(z) if log else z,        # color by the z value for clarity
            colorscale="viridis_r",
            showscale=False
        )

        # mesh
        mesh_x, mesh_y, mesh_z = [], [], []
        for triangle in tri.simplices:
            for i in range(3):
                i1, i2 = triangle[i], triangle[(i+1) % 3]
                mesh_x.extend([x[i1], x[i2], None])  # None breaks the line
                mesh_y.extend([y[i1], y[i2], None])
                mesh_z.extend([z[i1], z[i2], None])

        mesh = go.Scatter3d(
            x=mesh_x, y=mesh_y, z=mesh_z,
            mode="lines",
            line=dict(color="black", width=1),
            name="Mesh Lines"
        )
               

        # combine surface and mesh
        fig = go.Figure(data=[surface, mesh])
        fig.update_layout(
            scene_camera=dict(eye=dict(x=1.7, y=-1.7, z=0.3)),
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis_title=x_lab,
                yaxis_title=y_lab,
                zaxis_title=f"log({z_lab})" if log else z_lab,
                zaxis=dict(type="log", range=[-5.7, -2]) if log else dict(range=[0, 0.35]),
                aspectmode="cube"
                )
        )
        
        fig.show()
        exit()
        fig.write_image(f"figs/3D_plots/{savetitle}.png", width=800, height=800, scale=2)      
     
    
    with sqlite3.connect(db_path) as conn:
        statistics: pd.DataFrame = pd.read_sql_query(
            "SELECT * FROM statistics",
            con=conn)
    
    # 3D plots for mean square displacement
    for d in sorted(set(statistics["d"])):
        sims_filtered: pd.DataFrame = statistics[statistics["d"] == d]
        mesh_3d(sims_filtered, "T", "L", "border_msd", log=True, savetitle=f"MSD_{f"{d:.2f}".replace(".", ",")}")
          
    # 3d plots for buffer zone width
    for d in sorted(set(statistics["d"])):
        sims_filtered: pd.DataFrame = statistics[statistics["d"] == d]
        mesh_3d(sims_filtered, "T", "L", "buffer_width", savetitle=f"bw_{f"{d:.2f}".replace(".", ",")}")


def visualize_simulation() -> None:    
    s: Simulation = Simulation(*build_instances(T_a=0.312, T_b=0.312, diffusion_coefficient=0.25))
    time_steps: int = 10_000
    s.simulate(time_steps, track_border=True, track_locations=True, track_scent_marks=True)
    res: dict[str, Any] = s.get_results(
        time_steps, round(s.wolf_a.T / s.sm.dt),
        True, True, True
        )
    
    # configure fig
    fig = plt.figure(figsize=(ax_width_wide, fig_height))  
    gs = gridspec.GridSpec(1, 2, width_ratios=[9, 2]) 
    
    # plot of locations, dens, border and buffer zone
    ax1 = fig.add_subplot(gs[0])
    times: list[float] = [i * s.sm.dt for i in range(time_steps)]
    
    # buffer zone
    intervals: list[float] = [tick * s.sm.dt for tick in res["tick"]]
    buffer_a: list[float] = np.array(res["r_95_a"])
    buffer_b: list[float] = np.array(res["r_95_b"])
    mask = buffer_a > buffer_b

    ax1.step(intervals, np.where(mask, np.nan, buffer_a), where="pre", color="gray", alpha=0.7)
    ax1.step(intervals, np.where(mask, np.nan, buffer_b), where="pre", color="gray", alpha=0.7)
    ax1.fill_between(intervals, buffer_a, buffer_b, step="pre", label="buffer zone",
                     interpolate=True, alpha=0.5, color="lightgray")    
    
    # border
    ax1.plot(times, s.borders, color="tab:green", label="border")

    # locations
    ax1.plot(times, [s.wolf_a.den] * time_steps, color="tab:blue", linestyle="--")
    ax1.plot(times, [s.wolf_b.den] * time_steps, color="tab:orange", linestyle="--")
    ax1.plot(times, s.locs_a, label=r"$X(t)$", color="tab:blue")
    ax1.plot(times, s.locs_b, label=r"$Y(t)$", color="tab:orange")
    # ax1.plot([], [], color="black", linestyle="--", label="dens")

    # legend
    handles, labels = plt.gca().get_legend_handles_labels()
    new_order = [2, 3, 0, 1]  
    ax1.legend(
        [handles[i] for i in new_order], [labels[i] for i in new_order],
        fontsize=fontsize_legend, loc="upper left"
        )
    

    # axes
    ax1.set_xlabel(r"$t$", fontsize=fontsize_label)
    ax1.set_ylabel(r"$x$", fontsize=fontsize_label)
    ax1.set_xlim((0, times[-1]))
    ax1.set_ylim((0, 1))
    ax1.tick_params(right=False)

    # plot of scent marking function at the end, border location and buffer zone
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    # scent marking functions
    ax2.plot(s.sm.get_scent_field(s.wolf_a), np.linspace(0, 1, s.sm.M + 1), label=r"$f(x, t_{\text{end}})$")
    ax2.plot(s.sm.get_scent_field(s.wolf_b), np.linspace(0, 1, s.sm.M + 1), label=r"$g(x, t_{\text{end}})$")

    # border and buffer zone
    ymax = ax2.get_xlim()[1]
    ax2.hlines(y=s.borders[-1], xmin=0, xmax=ymax, color="tab:green", linestyle="-")
    ax2.hlines(y=res["r_95_a"][-1], xmin=0, xmax=ymax, color="gray", linestyle="-", alpha=0.7)
    ax2.hlines(y=res["r_95_b"][-1], xmin=0, xmax=ymax, color="gray", linestyle="-", alpha=0.7)
    ax2.fill_between(np.linspace(0, ymax, 100), res["r_95_a"][-1], res["r_95_b"][-1], color="lightgray", alpha=0.5)
    
    # axes and legend
    ax2.legend(fontsize=fontsize_legend, loc="upper right")
    ax2.set_xlabel("scent concentration", fontsize=fontsize_label)
    ax2.set_xlim((0, ax2.get_xlim()[1]))
    ax2.set_ylim((0, 1))
    ax2.yaxis.set_visible(False)    
    ax2.tick_params(left=False)
    
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    gap = -0.003  # tiny overlap to hide any white line
    ax1.set_position([pos1.x0, pos1.y0, pos1.width + gap, pos1.height])
    ax2.set_position([pos2.x0 - gap, pos2.y0, pos2.width + gap, pos2.height])

    plt.tight_layout()
    # plt.savefig("figs/sample_simulation_run.png")
    plt.show()


    # scent marking function with buffer zone and arrows
    """fig, ax2 = plt.subplots(figsize=(10, 6))
    
    ax2.plot(np.linspace(0, 1, s.sm.M + 1), s.sm.get_scent_field(s.wolf_a), label=r"$f^{(A)}(x, t_{\text{end}})$")
    ax2.plot(np.linspace(0, 1, s.sm.M + 1), s.sm.get_scent_field(s.wolf_b), label=r"$f^{(B)}(x, t_{\text{end}})$")
    ymax = ax2.get_ylim()[1]
    
    ax2.vlines(x=s.borders[-1], ymin=0, ymax=ymax, color="green", linestyle="-", label="border")
    ax2.vlines(x=res["r_95_a"][-1], ymin=0, ymax=ymax, color="gray", linestyle="-")
    ax2.vlines(x=res["r_95_b"][-1], ymin=0, ymax=ymax, color="gray", linestyle="-")
    
    ax2.fill_between(np.linspace(res["r_95_a"][-1], res["r_95_b"][-1], 100), 0, ymax, color='gray', alpha=0.3, label="buffer zone")
    ax2.annotate(
        '', xy=(res["r_95_a"][-1], 2), xytext=(res["r_95_a"][-1]-0.1, 2),
        arrowprops=dict(
        arrowstyle='->',
        color='black',
        linewidth=2.5,
        mutation_scale=20  # increase for bigger arrowheads
    )
    )
    ax2.annotate(
        '', xy=(res["r_95_a"][-1], 2), xytext=(res["r_95_b"][-1]+0.1, 2),
        arrowprops=dict(
        arrowstyle='->',
        color='black',
        linewidth=2.5,
        mutation_scale=20  # increase for bigger arrowheads
    )
    )
    
    ax2.set_ylabel("scent value", fontsize=20)
    ax2.set_xlabel("location", fontsize=20)
    ax2.legend(fontsize=16, loc="upper right")
    ax2.set_xlim((0, 1))
    plt.tight_layout()
    plt.show()"""


def compare_parameters() -> None:
    # configure fig
    fig, ax = plt.subplots(figsize=(ax_width_wide, fig_height))  

    L1, L2 = 0.2, 0.8
    T1, T2 = 0.312, 0.312
    d1, d2 = 0.25, 0.25
    for i, color in zip(range(2), ["tab:blue", "tab:orange"]):
        if i == 0:
            s: Simulation = Simulation(*build_instances(
                den_a=round(0.5 - (L1/2), 3), den_b=round(0.5 + (L1/2), 3),
                T_a=T1, T_b=T1, 
                diffusion_coefficient=d1)
                )


            if T1 != T2:
                label = rf"$T$ = {T1}"
            elif L1 != L2:
                label = rf"$L$ = {L1}"
            elif d1 != d2:
                label = rf"$d$ = {d1}"
            else:
                label = None    # define suitable label
        elif i == 1:
            s: Simulation = Simulation(*build_instances(
                den_a=round(0.5 - (L2/2), 3), den_b=round(0.5 + (L2/2), 3),
                T_a=T2, T_b=T2, 
                diffusion_coefficient=d2)
                )
            
            if T1 != T2:
                label = rf"$T$ = {T2}"
            elif L1 != L2:
                label = rf"$L$ = {L2}"
            elif d1 != d2:
                label= rf"$d$ = {d2}"
            else:
                label = None    # define suitable label

        time_steps: int = 5_000
        s.simulate(time_steps, track_border=True, track_locations=True, track_scent_marks=True)
        
        # plot of locations, dens, border and buffer zone
        times: list[float] = [i * s.sm.dt for i in range(time_steps)]
        
        # locations
        ax.plot(times, [s.wolf_a.den] * time_steps, color=color if L1 != L2 else "gray", linestyle="--")
        ax.plot(times, [s.wolf_b.den] * time_steps, color=color if L1 != L2 else "gray", linestyle="--")
        ax.plot(times, s.locs_a, color=color)
        ax.plot(times, s.locs_b, color=color)
        ax.plot([], [], color=color, linestyle="-", label=label)

    
    # ax.plot([], [], color="gray", linestyle="--", label="dens")

    # legend
    ax.legend(
        fontsize=fontsize_legend, loc="upper left"
        )
    

    # axes
    ax.set_xlabel(r"$t$", fontsize=fontsize_label)
    ax.set_ylabel(r"$x$", fontsize=fontsize_label)
    ax.set_xlim((0, times[-1]))
    ax.set_ylim((0, 1))
    ax.tick_params(right=False)

    plt.tight_layout()
    # plt.savefig("figs/sample_simulation_run.png")
    plt.show()


def main() -> None:
    visualize_simulation()
        
    
if __name__ == "__main__":
    main()
