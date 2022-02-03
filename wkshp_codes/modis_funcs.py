#!/usr/bin/env python
"""Functions to play around with MODIS data"""

import json

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

gdal.UseExceptions()

try:
    from tqdm.autonotebook import tqdm
except ImportError:
    pass


import ipywidgets as widgets
from IPython.display import display

districts_list = json.load(open("../data/carto/Ghana_districts.geojson", "r"))
district_names = [feat["properties"]["NAME"] for feat in districts_list["features"]]

landcover_types = [
    ("Water", 0),
    ("Evergreen needleleaf forest", 1),
    ("Evergreen broadleaf forest", 2),
    ("Deciduous needleleaf forest", 3),
    ("Deciduous broadleaf forest ", 4),
    ("Mixed forest", 5),
    ("Closed shrublands", 6),
    ("Open shrublands", 7),
    ("Woody savannas", 8),
    ("Savannas", 9),
    ("Grasslands", 10),
    ("Permanent wetlands", 11),
    ("Croplands", 12),
    ("Urban and built-up", 13),
    ("Cropland/natural vegetation mosaic", 14),
    ("Snow and ice", 15),
    ("Barren or sparsely vegetated", 16),
]


def extract_avg_par(
    district,
    par="LAI",
    lc_class=14,
    lp_min=10,
    years=(2015, 2017),
    mcd15_url="http://gws-access.ceda.ac.uk/public/odanceo/MCD15",
    mcd12_url="http://gws-access.ceda.ac.uk/public/odanceo/MCD12",
    cutline_ds="../data/carto/Map_of_Districts_216.shp",
    field_name="NAME",
):
    start_year = years[0]
    end_year = years[1]

    if par == "LAI":
        lai_files = [
            f"/vsicurl/{mcd15_url:s}/Lai_500m_{year:d}.tif"
            for year in range(start_year, end_year)
        ]
    elif par == "fAPAR":
        lai_files = [
            f"/vsicurl/{mcd15_url:s}/Fpar_500m_{year:d}.tif"
            for year in range(start_year, end_year)
        ]
    lai_qa_files = [
        f"/vsicurl/{mcd15_url:s}/FparLai_QC_{year:d}.tif"
        for year in range(start_year, end_year)
    ]
    lp_files = [
        f"/vsicurl/{mcd12_url:s}/LC_Prop1.A{year:d}001.tif"
        for year in range(start_year, end_year)
    ]
    lc_files = [
        f"/vsicurl/{mcd12_url:s}/LC_Type1.A{year:d}001.tif"
        for year in range(start_year, end_year)
    ]

    aggregator = lambda fname: gdal.Warp(
        "",
        fname,
        format="MEM",
        cutlineDSName=cutline_ds,
        cutlineWhere=f"{field_name:s}='{district:s}'",
        cropToCutline=True,
    ).ReadAsArray()
    mask57 = 0b11100000  # Select bits 5, 6 and 7
    get_sfc_qc = lambda fname: (
        np.right_shift(np.bitwise_and(aggregator(fname), mask57), 5) <= 2
    )

    with ThreadPoolExecutor(max_workers=10) as exec:
        lai = list(tqdm(exec.map(aggregator, lai_files), total=len(lai_files)))
    with ThreadPoolExecutor(max_workers=10) as exec:
        lai_qa = list(tqdm(exec.map(get_sfc_qc, lai_qa_files), total=len(lai_qa_files)))
    with ThreadPoolExecutor(max_workers=10) as exec:
        lp = list(tqdm(exec.map(aggregator, lp_files), total=len(lp_files)))
    with ThreadPoolExecutor(max_workers=10) as exec:
        lc = list(tqdm(exec.map(aggregator, lc_files), total=len(lc_files)))

    sel_lai = []
    lc_name = [x for x, y in landcover_types if y == lc_class][0]
    for i, year in enumerate(range(start_year, end_year)):
        if par == "LAI":
            L = lai[i] * 0.1
        elif par == "fAPAR":
            L = lai[i] * 0.01
        qa = lai_qa[i]
        L[~qa] = np.nan
        mask = np.logical_and(lc[i] == lc_class, lp[i] >= lp_min)

        print(f"Year {year} => {mask.sum():d} usable pixels in class {lc_name}")
        for i in range(L.shape[0]):
            L[i][mask] = np.nan
        # L[:,mask] = np.nan
        sel_lai.append(L)
    df = get_par_df(sel_lai, list(range(start_year, end_year)))
    plt.figure(figsize=(15, 5))

    df["mean"].plot(title=f"Mean {district} {lc_name} {par}")  # Change title!
    plt.fill_between(df["date"], df["q05"].values, df["q95"].values, color="0.5")
    plt.fill_between(df["date"], df["q25"].values, df["q75"].values, color="0.8")
    plt.savefig(f"{district}_{lc_name}_{par}.png", dpi=140)  # Change filename!
    print(f"Saved plot as {district}_{lc_name}_{par}.png")
    plt.savefig(f"{district}_{lc_name}_{par}.pdf", dpi=140)  # Change filename!
    print(f"Saved plot as {district}_{lc_name}_{par}.pdf")

    df.to_csv(f"{district}_{lc_name}_{par}.csv", encoding="utf-8", index=False)
    print(f"Saved data as {district}_{lc_name}_{par}.csv")
    return df


def get_par_df(retval, years):
    frame = {
        "date": [],
        "mean": [],
        "std": [],
        "q05": [],
        "q25": [],
        "q50": [],
        "q75": [],
        "q95": [],
    }

    for ii, year in enumerate(years):
        ar = retval[ii]
        t0 = dt.datetime(year, 1, 1)
        t_axs = [t0 + j * dt.timedelta(days=8) for j in range(46)]
        mean = np.nanmean(ar, axis=(1, 2))
        std = np.nanstd(ar, axis=(1, 2))
        q05, q25, q50, q75, q95 = np.nanpercentile(ar, [5, 25, 50, 75, 95], axis=(1, 2))
        frame["date"] += t_axs
        frame["mean"] += mean.tolist()
        frame["std"] += std.tolist()
        frame["q05"] += q05.tolist()
        frame["q25"] += q25.tolist()
        frame["q50"] += q50.tolist()
        frame["q75"] += q75.tolist()
        frame["q95"] += q95.tolist()

    df = pd.DataFrame(frame)
    df = df.set_index(pd.DatetimeIndex(df["date"]))

    return df


def select_region_modis_lai():
    w = widgets.interactive(
        extract_avg_par,
        {"manual": True},
        par=widgets.RadioButtons(
            options=["LAI", "fAPAR"],
            value="LAI",
            description="Select LAI or fAPAR aggreation:",
            disabled=False,
        ),
        lp_min=widgets.IntSlider(
            value=10,
            min=0,
            max=100,
            step=10,
            description="Mininum Class Pcntge:",
            orientation="horizontal",
            readout=True,
            readout_format="d",
        ),
        lc_class=widgets.Dropdown(
            options=landcover_types, value=12, description="Land Cover Class"
        ),
        years=widgets.IntRangeSlider(
            value=[2010, 2017],
            min=2002,
            max=2018,
            step=1,
            description="Years:",
            orientation="horizontal",
            readout=True,
            readout_format="d",
        ),
        district=widgets.Dropdown(options=district_names, value="Garu Tempane"),
        mcd15_url=widgets.fixed("http://gws-access.ceda.ac.uk/public/odanceo/MCD15"),
        mcd12_url=widgets.fixed("http://gws-access.ceda.ac.uk/public/odanceo/MCD12"),
        cutline_ds=widgets.fixed("../data/carto/Map_of_Districts_216.shp"),
        field_name=widgets.fixed("NAME"),
    )
    return w


def accum_lai(df, start_date, end_date, field="q50"):
    plt.figure(figsize=(15, 7))
    years = np.unique([x.year for x in df["date"]])
    for year in years:
        s0 = dt.datetime(year, start_date.month, start_date.day)
        s1 = dt.datetime(year, end_date.month, end_date.day)
        passer = np.logical_and(df["date"] >= s0, df["date"] <= s1)
        y = np.cumsum(df["q50"][passer].values)
        x = [int(z.strftime("%j")) for z in df["date"][passer]]
        if year % 2 == 0:
            plt.plot(x, y, "-", label=year, lw=2)
        else:
            plt.plot(x, y, "--", label=year, lw=2)
    plt.legend(loc="best", frameon=False)
    plt.xlabel("Day of year [d]")
    plt.ylabel("Accumulated LAI [-]")


def cummulative_lai_plots(df):
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2010, 12, 31)

    dates = pd.date_range(start_date, end_date, freq="D")

    options = [(date.strftime(" %d %b %Y "), date) for date in dates]
    index = (0, len(options) - 1)

    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description="Sowing & Harvest",
        orientation="horizontal",
        layout={"width": "600px"},
    )

    def plot_aggr_meteo(sowing_harvesting):
        sowing, harvesting = sowing_harvesting
        accum_lai(df, sowing, harvesting)

    widgets.interact_manual(plot_aggr_meteo, sowing_harvesting=selection_range_slider)

def get_sfc_qc(qa_data, mask57 = 0b11100000):
    sfc_qa = np.right_shift(np.bitwise_and(qa_data, mask57), 5)
    return sfc_qa

def get_scaling(sfc_qa, golden_ratio=0.61803398875):
    weight = np.zeros_like(sfc_qa, dtype=np.float)
    for qa_val in [0, 1, 2, 3]:
        weight[sfc_qa == qa_val] = np.power(golden_ratio, float(qa_val))
    return weight
