import sys

sys.path.append("../")
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets.widgets as widgets
from ipywidgets import interact, interactive, fixed


def read_field_datasets(
    s2_lai_fname="data/Ghana_avg_archetype_lai.csv",
    ground_yield_fname="data/Yield_Maize_Biomass_V2.csv",
):
    df_lai = pd.read_csv(s2_lai_fname, parse_dates=["time"])
    colum_names = [
        "quadrant_code",
        "yield_quad",
        "Yield",
        "n_plants_quad",
        "avg_nplants",
        "plant_dens",
        "avg_plant_dens",
        "ncobs_1m",
        "avg_ncobs_1m",
        "wet_weight",
        "avg_wet_weight",
        "dry_weight",
        "avg_dry_weight",
        "biomass",
        "total_biomass",
    ]
    df_yield = pd.read_csv(ground_yield_fname, names=colum_names, skiprows=1)
    df_yield["field_code"] = df_yield.quadrant_code.str[:-2]
    avg_yield = (
        df_yield.groupby("field_code")[["yield_quad"]]
        .agg(["mean", "std"])
        .reset_index(level=[0])
    )
    avg_yield.columns = ["field_code", "yield_mean", "yield_std"]
    return df_lai, df_yield, avg_yield


def process_field_data(
    s2_lai_fname="data/Ghana_avg_archetype_lai.csv",
    ground_yield_fname="data/Yield_Maize_Biomass_V2.csv",
):
    (df_lai, df_yield, avg_yield) = read_field_datasets(
        s2_lai_fname=s2_lai_fname, ground_yield_fname=ground_yield_fname
    )
    # Only keep fields with LAI observations
    avg_yield = avg_yield[avg_yield.field_code.isin(df_lai.field_code)]
    avg_yield = avg_yield.sort_values("yield_mean")
    # Remove weird field
    fields = avg_yield.field_code.tolist()
    fields = [f for f in fields if (f != "7069ZIN")]
    avg_yield = avg_yield[avg_yield.field_code.isin(fields)]
    df_lai = df_lai[df_lai.field_code.isin(fields)]
    return avg_yield, df_lai, fields


def read_data(fname=f"data/wofost_sims_dvs125.npz"):
    f = np.load(fname, allow_pickle=True)
    parameters = f.f.parameters
    t_axis = f.f.t_axis
    samples = f.f.samples
    lais = f.f.lais
    yields = f.f.yields
    DVS = f.f.DVS
    print("Read in simulations")
    avg_yield, df_lai, fields = process_field_data()
    return parameters, t_axis, samples, lais, yields, DVS, avg_yield, df_lai, fields


def slider_plots_func(
    sel_field,
    sel_dos,
    sel_beta_early,
    sel_beta_late,
    df_lai,
    avg_yield,
    t_axis,
    lais,
    yields,
    parameters,
):

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    axs = axs.flatten()
    time_lai = df_lai[df_lai.field_code == sel_field][["time"]].values
    retval = df_lai[df_lai.field_code == sel_field][["lai_mean", "lai_std"]].values
    (obs_lai_mean, obs_lai_std) = retval[:, 0], retval[:, 1]
    axs[0].plot(time_lai, obs_lai_mean, "o", markerfacecolor="none", label="Observed")
    axs[0].vlines(
        time_lai, obs_lai_mean - obs_lai_std, obs_lai_mean + obs_lai_std, color="0.8"
    )

    diff = np.abs(parameters - np.array([sel_dos, sel_beta_early, sel_beta_late]))
    ilocs = np.abs(diff).sum(axis=1).argmin()
    axs[0].plot(t_axis, lais[ilocs], "o", markerfacecolor="none", label="Modelled")
    axs[0].legend(loc="best", frameon=False)

    yield_mean = avg_yield[avg_yield.field_code == sel_field].yield_mean.values
    yield_std = avg_yield[avg_yield.field_code == sel_field].yield_std.values
    axs[1].axvspan(yield_mean - yield_std, yield_mean + yield_std, color="0.9")
    axs[1].plot(yield_mean, yields[ilocs], "o")
    axs[1].plot([0, 5000], [0, 5000], "k--")
    axs[1].set_xlim(0, 5000)
    axs[1].set_ylim(0, 5000)
    axs[0].set_ylabel("Leaf Area Index [m2/m2]", fontsize=9)
    axs[1].set_ylabel("Modelled yield [m2/m2]", fontsize=9)
    axs[0].set_xlabel("Time", fontsize=9)
    axs[1].set_xlabel("Observed yield [m2/m2]", fontsize=9)


def slider_plots():
    (
        parameters,
        t_axis,
        samples,
        lais,
        yields,
        DVS,
        avg_yield,
        df_lai,
        fields,
    ) = read_data()
    interact(
        slider_plots_func,
        sel_field=widgets.Dropdown(options=fields),
        sel_dos=widgets.IntSlider(min=181, max=224, value=200),
        sel_beta_early=widgets.FloatSlider(min=0.05, max=0.55, step=0.001, value=0.35),
        sel_beta_late=widgets.FloatSlider(min=0.05, max=0.55, step=0.001, value=0.35),
        df_lai=widgets.fixed(df_lai),
        avg_yield=widgets.fixed(avg_yield),
        t_axis=widgets.fixed(t_axis),
        lais=widgets.fixed(lais),
        yields=widgets.fixed(yields),
        parameters=widgets.fixed(parameters),
    )
