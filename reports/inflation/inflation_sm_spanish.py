# %%
# ----------------------------------
# We load fonts and stylesheet.
# ----------------------------------
import datetime
from re import X
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from highlight_text import ax_text
from dateutil.relativedelta import relativedelta
import matplotlib as mpl
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from urllib import response
import sys
from distutils import core
import os


module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\utils")
    sys.path.append(module_path+"\\assets")

from tukan_helper_functions import get_tukan_api_request

# For creating cool charts :)
plt.style.use(module_path + '\\utils\\tukan_style.mpl')

# Change for local settings to spanish
import locale
locale.setlocale(locale.LC_ALL, 'es_es')
# %%
# ----------------------------------
# We load the product_mapping_inegi csv file to map the products to the INEGI category
# ----------------------------------


def map_tukan_inegi_products(from_d="2019-01-01"):

    # Map to match INEGI classifications
    tukan_inegi_map = pd.read_csv(
        "product_mapping_inegi_es.csv", encoding="utf-8")
    tukan_inegi_map.drop(["weight"], axis=1, inplace=True)

    weight_payload = {
        "type": "data_table",
        "operation": "sum",
        "language": "es",
        "group_by": [
            "product"
        ],
        "categories": {
            "product": "all"
        },
        "request": [
            {
                "table": "mex_inegi_inpc_product_weights",
                "variables": "all"
            }
        ]
    }

    cpi_payload = {
        "type": "data_table",
        "operation": "sum",
        "language": "es",
        "group_by": [
            "product"
        ],
        "categories": {
            "product": "all"
        },
        "request": [
            {
                "table": "mex_inegi_inpc_product_monthly",
                "variables": [
                    "c572db59b8cd109"
                ]
            }
        ],
        "from": from_d
    }

    response_weight = get_tukan_api_request(weight_payload)
    cpi_response = get_tukan_api_request(cpi_payload)

    weight_data = response_weight["data"]
    cpi_data = cpi_response["data"]

    weight_data.drop("date", axis=1, inplace=True)

    final_data = pd.merge(cpi_data, weight_data, how="left", on=[
                          "product", "product__ref"])
    final_data = pd.merge(final_data, tukan_inegi_map,
                          how="left", on="product__ref")

    return final_data

# ------------------------------------------------------------------
#
#                       CHARTS FOR REPORT
#
# ------------------------------------------------------------------
# %%

# ------------------------------------------------------------------
#
# CHART 1: YOY CHANGE IN CPI
#
# ------------------------------------------------------------------


def plot_chart_1(from_d="2000-01-01"):

    payload = {
        "type": "data_table",
        "operation": "yoy_growth_rel",
        "language": "es",
        "group_by": [
            "product"
        ],
        "categories": {
            "product": [
                "193b800af2978be"
            ]
        },
        "request": [
            {
                "table": "mex_inegi_inpc_product_monthly",
                "variables": [
                    "c572db59b8cd109"
                ]
            }
        ],
        "from": from_d
    }

    response = get_tukan_api_request(payload)
    data = response["data"]

    # ----
    # The chart
    # ----
    cmap = mpl.cm.get_cmap("GnBu_r", 5)
    fig = plt.figure(figsize=(9, 5), dpi=200)
    ax = plt.subplot(111)

    ax.plot(data["date"], data["c572db59b8cd109"],
            marker="o", ms=6, mec="white", markevery=[-1], color=cmap(0))

    Y_end = data["c572db59b8cd109"].iloc[-1]
    X_max = data["date"].iloc[-1]

    ax_text(x=X_max + relativedelta(months=3), y=Y_end,
            s=f"<{Y_end:.2%}>",
            highlight_textprops=[{"color": cmap(0)}],
            ax=ax, weight="bold", font="Dosis", ha="left", size=9)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))
    ax.set_ylim(0)

    fig.text(
    0.1,
    1,
    "Índice Nacional de Precios al Consumidor - Variación Anual",
    size=14,
    weight = "bold"
    )

    tukan_im = image.imread(module_path + "\\assets\\logo\\logo192.png")
    newax = fig.add_axes([0.68, .88, 0.18, 0.21], anchor="NE", zorder=1)
    newax.imshow(tukan_im)
    newax.axis("off")

    plt.savefig(
        "plots/es_yoy_cpi_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    # ---
    print(
        f"Annual inflation came in at {Y_end:.2%} during {X_max.strftime('%b-%Y')}")

# %%
# ------------------------------------------------------------------
#
# CHART 2: UNDERLYING AND NON-UNDERLYING CONTRIBUTION TO INFLATION
#
# ------------------------------------------------------------------


def plot_chart_2(from_d="2019-01-01"):

    weight_payload = {
        "type": "data_table",
        "operation": "sum",
        "language": "es",
        "group_by": [
            "product"
        ],
        "categories": {
            "product": "all"
        },
        "request": [
            {
                "table": "mex_inegi_inpc_product_weights",
                "variables": "all"
            }
        ]
    }

    cpi_payload = {
        "type": "data_table",
        "operation": "sum",
        "language": "es",
        "group_by": [
            "product"
        ],
        "categories": {
            "product": "all"
        },
        "request": [
            {
                "table": "mex_inegi_inpc_product_monthly",
                "variables": [
                    "c572db59b8cd109"
                ]
            }
        ],
        "from": from_d
    }

    response_weight = get_tukan_api_request(weight_payload)
    cpi_response = get_tukan_api_request(cpi_payload)

    weight_data = response_weight["data"]
    cpi_data = cpi_response["data"]

    # ---
    # Underlying CPI 2dac8b4b2fc8037
    # ---

    # Adjust weights
    underlying_weights = weight_data[[
        "2dac8b4b2fc8037", "5993e5e787e4259", "product__ref"]]
    underlying_weights = underlying_weights[underlying_weights["2dac8b4b2fc8037"] != 0].copy(
    )
    underlying_weights.reset_index(drop=True, inplace=True)

    cpi_data_underlying = pd.merge(
        cpi_data, underlying_weights, how="right", on="product__ref")

    cpi_data_underlying.loc[:, "cpi_under"] = [
        x*y for x, y in zip(cpi_data_underlying["c572db59b8cd109"], cpi_data_underlying["2dac8b4b2fc8037"])]

    cpi_data_underlying = cpi_data_underlying.groupby(
        ["date"]).sum(["cpi_under"]).reset_index()

    cpi_data_underlying["lag_under"] = cpi_data_underlying["cpi_under"].shift(
        12)

    cpi_data_underlying["yoy_under"] = cpi_data_underlying["cpi_under"] / \
        cpi_data_underlying["lag_under"] - 1

    cpi_data_underlying = cpi_data_underlying[[
        "date", "yoy_under", "5993e5e787e4259"]]

    # Non-underlying CPI a5a09b8e3b7cbb5

    non_underlying_weights = weight_data[[
        "a5a09b8e3b7cbb5", "5993e5e787e4259", "product__ref"]]
    non_underlying_weights = non_underlying_weights[non_underlying_weights["a5a09b8e3b7cbb5"] != 0].copy(
    )
    non_underlying_weights.reset_index(drop=True, inplace=True)

    cpi_data_non_underlying = pd.merge(
        cpi_data, non_underlying_weights, how="right", on="product__ref")

    cpi_data_non_underlying.loc[:, "cpi_non_under"] = [
        x*y for x, y in zip(cpi_data_non_underlying["c572db59b8cd109"], cpi_data_non_underlying["a5a09b8e3b7cbb5"])]

    cpi_data_non_underlying = cpi_data_non_underlying.groupby(
        ["date"]).sum(["cpi_non_under"]).reset_index()

    cpi_data_non_underlying["lag_non_under"] = cpi_data_non_underlying["cpi_non_under"].shift(
        12)

    cpi_data_non_underlying["yoy_non_under"] = cpi_data_non_underlying["cpi_non_under"] / \
        cpi_data_non_underlying["lag_non_under"] - 1

    cpi_data_non_underlying = cpi_data_non_underlying[[
        "date", "yoy_non_under", "5993e5e787e4259"]]

    # ---
    # The chart
    # ---

    plot_df = pd.merge(cpi_data_underlying,
                       cpi_data_non_underlying, how="left", on="date")
    plot_df = plot_df[plot_df["date"] >=
                      plot_df["date"].max() - relativedelta(months=12)]

    cmap = mpl.cm.get_cmap("GnBu_r", 5)
    fig = plt.figure(figsize=(9, 5), dpi=200)
    ax = plt.subplot(111)

    index_ticks = np.arange(plot_df["date"].shape[0])
    width = 0.25

    ax.bar(index_ticks, plot_df["yoy_under"], color=cmap(
        0), width=width, zorder=3, label="Subyacente")
    ax.bar(index_ticks + width, plot_df["yoy_non_under"],
           color=cmap(1), width=width, zorder=3, label="No subyacente")

    ax.set_xticks(index_ticks + width / 2,
                  labels=plot_df["date"].dt.strftime("%b-%y"), rotation=90)

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))
    ax.set_ylim(0)

    ax.legend(loc="lower center", bbox_to_anchor=(0.55, -0.45), ncol=2)

    fig.text(
    0.1,
    1,
    "Inflación Subyacente vs. No Subyacente - Variación Anual",
    size=14,
    weight = "bold"
    )

    tukan_im = image.imread(module_path + "\\assets\\logo\\logo192.png")
    newax = fig.add_axes([0.68, .88, 0.18, 0.21], anchor="NE", zorder=1)
    newax.imshow(tukan_im)
    newax.axis("off")

    plt.savefig(
        "plots/es_core_vs_non_core_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )


# %%

# ------------------------------------------------------------------
#
# CHART 3: CORE INFLATION SUB-INDEX
#
# ------------------------------------------------------------------

def plot_chart_3(from_d="2019-01-01"):

    data = map_tukan_inegi_products(from_d)

    core_data = data[data["2dac8b4b2fc8037"] != 0].reset_index(drop=True)
    core_data.dropna(inplace=True)
    core_data.loc[:, "index_weighted"] = core_data["c572db59b8cd109"] * \
        core_data["2dac8b4b2fc8037"]

    aux_weights = core_data.groupby(["date", "primary"])[
        "2dac8b4b2fc8037"].sum().reset_index()
    aux_weights.drop(["date"], axis=1, inplace=True)
    aux_weights.drop_duplicates(inplace=True)
    aux_weights.rename(columns={"2dac8b4b2fc8037": "rel_weight"}, inplace=True)

    core_data = core_data.groupby(["date", "primary"])[
        "index_weighted"].sum().reset_index()

    core_data = pd.merge(core_data, aux_weights, on="primary", how="left")

    core_data.loc[:, "12_m_lag"] = core_data.groupby(
        ["primary", "rel_weight"])["index_weighted"].shift(12)
    core_data.loc[:, "1_m_lag"] = core_data.groupby(
        ["primary", "rel_weight"])["index_weighted"].shift(1)
    core_data.loc[:, "yoy_change"] = core_data["index_weighted"] / \
        core_data["12_m_lag"] - 1
    core_data.loc[:, "mom_change"] = core_data["index_weighted"] / \
        core_data["1_m_lag"] - 1

    # Sort subindex
    sort_index = core_data.sort_values(by="rel_weight", ascending=False)

    cmap = mpl.cm.get_cmap(
        "GnBu_r", sort_index[["primary"]].drop_duplicates().shape[0] + 4)

    yoy_data = core_data.dropna()

    X_min = yoy_data["date"].min()
    X_max = yoy_data["date"].max()

    fig = plt.figure(figsize=(9, 5), dpi=200)

    ax = plt.subplot(111)

    sub_indices = list(sort_index["primary"].unique())

    for index, sub_index in enumerate(sub_indices):
        plot_data_aux = yoy_data[yoy_data["primary"] == sub_index].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["yoy_change"],
                marker="o", markevery=[-1], color=cmap(index), mec="white", ms=5)
        Y_end = plot_data_aux["yoy_change"].iloc[-1]
        weight_end = plot_data_aux["rel_weight"].iloc[-1]
        if sub_index== 'Educativos':
            ax_text(x=X_max + datetime.timedelta(15), y=Y_end + 0.005,
                    s=f"<{sub_index}>",
                    highlight_textprops=[{"color": cmap(index)}],
                    ax=ax, weight="bold", font="Dosis", ha="left", size=7.5)
        else:
            ax_text(x=X_max + datetime.timedelta(15), y=Y_end + 0.001,
                    s=f"<{sub_index}>",
                    highlight_textprops=[{"color": cmap(index)}],
                    ax=ax, weight="bold", font="Dosis", ha="left", size=7.5)

    ax.set_xlim(X_min, X_max + relativedelta(months=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    if yoy_data["yoy_change"].min() < 0:
        ax.hlines(0, X_min, X_max, ls="--", color="black", lw=0.75)
    else:
        ax.set_ylim(0)

    fig.text(
    0.1,
    1,
    "Componentes Inflación Subyacente - Variación Anual",
    size=14,
    weight = "bold"
    )

    tukan_im = image.imread(module_path + "\\assets\\logo\\logo192.png")
    newax = fig.add_axes([0.68, .88, 0.18, 0.21], anchor="NE", zorder=1)
    newax.imshow(tukan_im)
    newax.axis("off")

    plt.savefig(
        "plots/es_core_subindices.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,

    )

# %%

# ------------------------------------------------------------------
#
# CHART 4: NON-CORE INFLATION SUB-INDEX
#
# ------------------------------------------------------------------


def plot_chart_4(from_d="2019-01-01"):

    data = map_tukan_inegi_products(from_d)

    core_data = data[data["a5a09b8e3b7cbb5"] != 0].reset_index(drop=True)
    core_data.dropna(inplace=True)
    core_data.loc[:, "index_weighted"] = core_data["c572db59b8cd109"] * \
        core_data["a5a09b8e3b7cbb5"]

    aux_weights = core_data.groupby(["date", "primary"])[
        "a5a09b8e3b7cbb5"].sum().reset_index()
    aux_weights.drop(["date"], axis=1, inplace=True)
    aux_weights.drop_duplicates(inplace=True)
    aux_weights.rename(columns={"a5a09b8e3b7cbb5": "rel_weight"}, inplace=True)

    core_data = core_data.groupby(["date", "primary"])[
        "index_weighted"].sum().reset_index()

    core_data = pd.merge(core_data, aux_weights, on="primary", how="left")

    core_data.loc[:, "12_m_lag"] = core_data.groupby(
        ["primary", "rel_weight"])["index_weighted"].shift(12)
    core_data.loc[:, "1_m_lag"] = core_data.groupby(
        ["primary", "rel_weight"])["index_weighted"].shift(1)
    core_data.loc[:, "yoy_change"] = core_data["index_weighted"] / \
        core_data["12_m_lag"] - 1
    core_data.loc[:, "mom_change"] = core_data["index_weighted"] / \
        core_data["1_m_lag"] - 1

    # Sort subindex
    sort_index = core_data.sort_values(by="rel_weight", ascending=False)

    cmap = mpl.cm.get_cmap(
        "GnBu_r", sort_index[["primary"]].drop_duplicates().shape[0] + 3)

    yoy_data = core_data.dropna()

    X_min = yoy_data["date"].min()
    X_max = yoy_data["date"].max()

    fig = plt.figure(figsize=(9, 5), dpi=200)

    ax = plt.subplot(111)

    sub_indices = list(sort_index["primary"].unique())

    for index, sub_index in enumerate(sub_indices):
        plot_data_aux = yoy_data[yoy_data["primary"] == sub_index].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["yoy_change"],
                marker="o", markevery=[-1], color=cmap(index), mec="white", ms=5)
        Y_end = plot_data_aux["yoy_change"].iloc[-1]
        weight_end = plot_data_aux["rel_weight"].iloc[-1]
        ax_text(x=X_max + datetime.timedelta(15), y=Y_end + 0.001,
                s=f"<{sub_index}>",
                highlight_textprops=[{"color": cmap(index)}],
                ax=ax, weight="bold", font="Dosis", ha="left", size=7.5)

    ax.set_xlim(X_min, X_max + relativedelta(months=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    if yoy_data["yoy_change"].min() < 0:
        ax.hlines(0, X_min, X_max, ls="--", color="black", lw=0.75)
    else:
        ax.set_ylim(0)

    fig.text(
    0.1,
    1,
    "Componentes Inflación No subyacente - Variación Anual",
    size=14,
    weight = "bold"
    )

    tukan_im = image.imread(module_path + "\\assets\\logo\\logo192.png")
    newax = fig.add_axes([0.68, .88, 0.18, 0.21], anchor="NE", zorder=1)
    newax.imshow(tukan_im)
    newax.axis("off")

    plt.savefig(
        "plots/es_non_core_subindices.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )


# ------------------------------------------------------------------
#
# CHART 5: Top 10 Products with highest contribution to non-core inflation
#
# ------------------------------------------------------------------

def plot_chart_5(from_d="2020-01-01"):

    data = map_tukan_inegi_products(from_d)

    core_data = data[data["a5a09b8e3b7cbb5"] != 0].reset_index(drop=True)
    core_data.dropna(inplace=True)
    core_data.loc[:, "index_weighted"] = core_data["c572db59b8cd109"] * \
        core_data["a5a09b8e3b7cbb5"]

    core_data.loc[:, "12_m_lag"] = core_data.groupby(
        ["product", "primary"])["index_weighted"].shift(12)
    core_data.loc[:, "1_m_lag"] = core_data.groupby(
        ["product", "primary"])["index_weighted"].shift(1)
    core_data.loc[:, "yoy_change"] = core_data["index_weighted"] - \
        core_data["12_m_lag"]
    core_data.loc[:, "mom_change"] = core_data["index_weighted"] - \
        core_data["1_m_lag"]

    core_inflation = core_data.groupby(
        ["date"])[["yoy_change", "mom_change"]].sum()
    core_inflation.reset_index(inplace=True)
    core_inflation.rename(
        columns={"yoy_change": "yoy_core", "mom_change": "mom_core"}, inplace=True)

    core_data = pd.merge(core_data, core_inflation, how="left", on="date")
    core_data.loc[:, "yoy_contribution"] = core_data["yoy_change"] / \
        core_data["yoy_core"]

    core_data = core_data[core_data["date"] == core_data["date"].max()]
    core_data = core_data.sort_values(
        by="yoy_contribution", ascending=False).head(9)

    # True YoY and MoM Change
    core_data.loc[:, "yoy_change"] = core_data["yoy_change"] / \
        core_data["12_m_lag"]
    core_data.loc[:, "mom_change"] = core_data["mom_change"] / \
        core_data["1_m_lag"]

    top_10 = list(core_data["product__ref"])

    core_data.reset_index(drop=True, inplace=True)

    core_data = core_data[["product", "primary",
                           "yoy_change", "mom_change", "yoy_contribution"]]
    core_data.replace({"product": {
                      "Servicios de transmisión de energía eléctrica": "Energía eléctrica"}}, inplace=True)
    core_data.set_index("product", inplace=True)
    core_data.columns = ["Grupo", "YoY %", "MoM %", "Cont. to\nYoY%"]

    fig = plt.figure(figsize=(6.5, 6), dpi=200)

    ax = fig.add_subplot(111)
    ax.set_ylim(-8.25, 1.5)
    ax.set_xlim(-0.75, 9.05)
    ax.set_axis_off()

    for row, x in enumerate(core_data.values):
        ax.annotate(
            core_data.index[row],
            xy=(-0.15, -1*row),
            xycoords="data",
            xytext=(0, 0),
            textcoords="offset points",
            color="black",
            size=8,
            va="center",
            ha="left",
            weight="bold"
        )
        column_aux = 0
        columns_auxes = []
        for column, label in enumerate(x):
            if isinstance(label, str):
                label_x = label
            else:
                label_x = f"{label:.1%}"
            if column == 0:
                column_x = 3
            elif column == 1:
                column_aux = 2
            else:
                column_x = 1.5
            column_aux = column_aux + column_x
            ax.annotate(
                label_x,
                xy=(column_aux, -1*row),
                xycoords="data",
                xytext=(10, 1),
                textcoords="offset points",
                color="black",
                size=7.5,
                va="center",
                ha="center"
            )
            if row == 8:
                color = "black"
                linewidth = 0.85
            else:
                color = "lightgrey"
                linewidth = 0.5
            ax.plot([-0.75, 9.15], [-1*row - .25, -1*row - .25],
                    color=color, linewidth=linewidth, ls="-")
            columns_auxes.append(column_aux)

    for index, column_aux in enumerate(columns_auxes):
        col_label = core_data.columns[index]
        ax.annotate(
            col_label,
            xy=(column_aux, 1.25),
            xycoords="data",
            xytext=(10, 2),
            textcoords="offset points",
            color="white",
            size=8,
            va="top",
            ha="center",
            weight="bold"
        )

    ax.plot([1.9, 8.15], [.75, .75], color="black", linewidth=0.85)

    ax.add_patch(mpl.patches.Rectangle((1.9, .75), width=7.15, height=0.8, linewidth=1,
                                       color='#2B5AA5', fill=True))

    plt.savefig(
        "plots/es_non_core_top_10.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    cpi_payload = {
        "type": "data_table",
        "operation": "yoy_growth_rel",
        "language": "es",
        "group_by": [
            "product"
        ],
        "categories": {
            "product": top_10
        },
        "request": [
            {
                "table": "mex_inegi_inpc_product_monthly",
                "variables": [
                    "c572db59b8cd109"
                ]
            }
        ],
        "from": "2010-01-01"
    }

    cpi_data = get_tukan_api_request(cpi_payload)
    cpi_data = cpi_data["data"]
    cpi_data.replace({"product": {
                     "Servicios de transmisión de energía eléctrica": "Energía eléctrica"}}, inplace=True)

    cmap = mpl.cm.get_cmap(
        "GnBu_r", 5)

    for index, x in enumerate(top_10):
        fig = plt.figure(figsize=(3, 1.5), dpi=200)
        ax = plt.subplot(111)
        aux_aux_df = cpi_data[cpi_data["product__ref"] == x]
        product = aux_aux_df["product"].iloc[0]
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.plot(aux_aux_df["date"], aux_aux_df["c572db59b8cd109"],
                zorder=10, lw=1, color=cmap(0))
        ax.tick_params(axis='both', which='major', labelsize=6)
        Y_max = aux_aux_df["c572db59b8cd109"].max()
        Y_min = aux_aux_df["c572db59b8cd109"].min()
        ax.text(
            aux_aux_df["date"].min() + datetime.timedelta(days=20),
            Y_max + (Y_max - Y_min)/4,
            f"{product}",
            horizontalalignment="left",
            verticalalignment="top",
            size=7,
            weight="bold",
            zorder=3
        )

        if Y_min < 0:
            ax.hlines(0, xmin=aux_aux_df["date"].min(), xmax=aux_aux_df["date"].max(),
                      ls="--", color="black", lw=0.75)
        else:
            ax.set_ylim(0)

        tukan_im = image.imread(module_path + "\\assets\\logo\\logo192.png")
        newax = fig.add_axes([0.68, .88, 0.18, 0.21], anchor="NE", zorder=1)
        newax.imshow(tukan_im)
        newax.axis("off")
    
        plt.savefig(
            f"plots/es_{product}_ts.png",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            transparent=False,
        )

# ------------------------------------------------------------------
#
# CHART 6: Top 10 Products with highest contribution to core inflation
#
# ------------------------------------------------------------------


def plot_chart_6(from_d="2020-01-01"):

    data = map_tukan_inegi_products(from_d)

    core_data = data[data["2dac8b4b2fc8037"] != 0].reset_index(drop=True)
    core_data.dropna(inplace=True)
    core_data.loc[:, "index_weighted"] = core_data["c572db59b8cd109"] * \
        core_data["2dac8b4b2fc8037"]

    core_data.loc[:, "12_m_lag"] = core_data.groupby(
        ["product", "primary"])["index_weighted"].shift(12)
    core_data.loc[:, "1_m_lag"] = core_data.groupby(
        ["product", "primary"])["index_weighted"].shift(1)
    core_data.loc[:, "yoy_change"] = core_data["index_weighted"] - \
        core_data["12_m_lag"]
    core_data.loc[:, "mom_change"] = core_data["index_weighted"] - \
        core_data["1_m_lag"]

    core_inflation = core_data.groupby(
        ["date"])[["yoy_change", "mom_change"]].sum()
    core_inflation.reset_index(inplace=True)
    core_inflation.rename(
        columns={"yoy_change": "yoy_core", "mom_change": "mom_core"}, inplace=True)

    core_data = pd.merge(core_data, core_inflation, how="left", on="date")
    core_data.loc[:, "yoy_contribution"] = core_data["yoy_change"] / \
        core_data["yoy_core"]

    core_data = core_data[core_data["date"] == core_data["date"].max()]
    core_data = core_data.sort_values(
        by="yoy_contribution", ascending=False).head(9)

    # True YoY and MoM Change
    core_data.loc[:, "yoy_change"] = core_data["yoy_change"] / \
        core_data["12_m_lag"]
    core_data.loc[:, "mom_change"] = core_data["mom_change"] / \
        core_data["1_m_lag"]

    top_10 = list(core_data["product__ref"])

    core_data.reset_index(drop=True, inplace=True)

    core_data = core_data[["product", "primary",
                           "yoy_change", "mom_change", "yoy_contribution"]]
    core_data.replace({"product": {"Loncherías, fondas, torterías y taquerías": "Taquerías y similares",
                      "Passenger air transportation": "Air transportation"}}, inplace=True)
    core_data.set_index("product", inplace=True)
    core_data.columns = ["Grupo", "YoY %", "MoM %", "Cont. to\nYoY%"]

    fig = plt.figure(figsize=(6.5, 6), dpi=200)

    ax = fig.add_subplot(111)
    ax.set_ylim(-8.25, 1.5)
    ax.set_xlim(-0.75, 9.05)
    ax.set_axis_off()

    for row, x in enumerate(core_data.values):
        ax.annotate(
            core_data.index[row],
            xy=(-0.15, -1*row),
            xycoords="data",
            xytext=(0, 0),
            textcoords="offset points",
            color="black",
            size=8,
            va="center",
            ha="left",
            weight="bold"
        )
        column_aux = 0
        columns_auxes = []
        for column, label in enumerate(x):
            if isinstance(label, str):
                label_x = label
            else:
                label_x = f"{label:.1%}"
            if column == 0:
                column_x = 3
            elif column == 1:
                column_aux = 2
            else:
                column_x = 1.5
            column_aux = column_aux + column_x
            ax.annotate(
                label_x,
                xy=(column_aux, -1*row),
                xycoords="data",
                xytext=(10, 1),
                textcoords="offset points",
                color="black",
                size=7.5,
                va="center",
                ha="center"
            )
            if row == 8:
                color = "black"
                linewidth = 0.85
            else:
                color = "lightgrey"
                linewidth = 0.5
            ax.plot([-0.75, 9.15], [-1*row - .25, -1*row - .25],
                    color=color, linewidth=linewidth, ls="-")
            columns_auxes.append(column_aux)

    for index, column_aux in enumerate(columns_auxes):
        col_label = core_data.columns[index]
        ax.annotate(
            col_label,
            xy=(column_aux, 1.25),
            xycoords="data",
            xytext=(10, 2),
            textcoords="offset points",
            color="white",
            size=8,
            va="top",
            ha="center",
            weight="bold"
        )

    ax.plot([1.9, 9.15], [.75, .75], color="black", linewidth=0.85)

    ax.add_patch(mpl.patches.Rectangle((1.9, .75), width=7.15, height=0.8, linewidth=1,
                                       color='#2B5AA5', fill=True))

    plt.savefig(
        "plots/es_core_top_10.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    cpi_payload = {
        "type": "data_table",
        "operation": "yoy_growth_rel",
        "language": "es",
        "group_by": [
            "product"
        ],
        "categories": {
            "product": top_10
        },
        "request": [
            {
                "table": "mex_inegi_inpc_product_monthly",
                "variables": [
                    "c572db59b8cd109"
                ]
            }
        ],
        "from": "2010-01-01"
    }

    cpi_data = get_tukan_api_request(cpi_payload)
    cpi_data = cpi_data["data"]
    cpi_data.replace({"product": {"Diners, inns, torterías and taquerías": "Diners, inns & others",
                     "Passenger air transportation": "Air transportation"}}, inplace=True)

    cmap = mpl.cm.get_cmap(
        "GnBu_r", 5)

    for index, x in enumerate(top_10):
        fig = plt.figure(figsize=(3, 1.5), dpi=200)
        ax = plt.subplot(111)
        aux_aux_df = cpi_data[cpi_data["product__ref"] == x]
        product = aux_aux_df["product"].iloc[0]
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.plot(aux_aux_df["date"], aux_aux_df["c572db59b8cd109"],
                zorder=10, lw=1, color=cmap(0))
        ax.tick_params(axis='both', which='major', labelsize=6)
        Y_max = aux_aux_df["c572db59b8cd109"].max()
        Y_min = aux_aux_df["c572db59b8cd109"].min()
        ax.text(
            aux_aux_df["date"].min() + datetime.timedelta(days=20),
            Y_max + (Y_max - Y_min)/4,
            f"{product}",
            horizontalalignment="left",
            verticalalignment="top",
            size=7,
            weight="bold",
            zorder=3
        )

        if Y_min < 0:
            ax.hlines(0, xmin=aux_aux_df["date"].min(), xmax=aux_aux_df["date"].max(),
                      ls="--", color="black", lw=0.75)
        else:
            ax.set_ylim(0)

        plt.savefig(
            f"plots/es_{product}_ts.png",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            transparent=False,
        )

# %%
# ------------------------------------------------------------------
#
# CHART 7: NON-AGRICULTURAL PRODUCTS IN BASIC BASKET
#
# ------------------------------------------------------------------


def plot_chart_7(from_d="2019-01-01"):

    payload = {
        "type": "data_table",
        "operation": "yoy_growth_rel",
        "language": "es",
        "group_by": [
            "product"
        ],
        "categories": {
            "product": [
                "d14323a232586a6",
                "a475f2b84d6a94b",
                "c912248098f41da",
                "aa9fec12e179a94",
                "bfc34ac08cc27f3",
                "40fd62913aab87a",
                "b8560ee27a9dd95",
                "0e92d6a5900b8f1",
                "3ba8401c1d31a45",
                "8d30de49aebbd98",
                "0f8e7d31e0f0434",
                "3f066266b9f243d",
                "0d20b22b91e6666",
                "4c709ac9d82aa4f",
                "b96d306d59d27eb",
                "170ad3c8e3ebb5c",
                "9aaa18043b49918",
                "f19555c07e57afd",
                "b1c1b572866b302",
                "a8fd40ad071ad7b",
                "6bab67da0f1c47c",
                "dc59054987d5607",
                "900d053c130a640",
                "107bb130b224217",
                "8f50ce285646722",
                "c69798a31c06326",
                "3d2cf64eb731283",
                "ae2e657ec0aeb97",
                "089d3bbea8c618f",
                "c705a40044f4087",
                "59037ebf620a3fb",
                "c1f2c61a54776d0",
                "51288bf59f3182c",
                "3d7be9de0701c09",
                "10ef80c17e2b1aa",
                "bbdf30dbd507939",
                "0c57a89ec3acf1a",
                "466dea40c877fee",
                "50c5b334f5b9fc1",
                "35ad4dc929e5ee7",
                "ab6d44bab5366f1",
                "f364671daebd58b",
                "452179a1ffe3b8d",
                "798adbe1c9e3308",
                "36746a7d7fd2374",
                "f21045665f08eb5",
                "bccc9df4db67e27",
                "c027962c25c220b",
                "21e639689399139",
                "b146318fe51b95c",
                "74feccc3706ec39",
                "c41716269b81e46",
                "a00e4e950290bd3",
                "6dafcdd00368e57",
                "4cd896b6ff2cc12",
                "b093042e5a49c36",
                "75879ec891a3cda",
                "7889415ad57004d",
                "eb2368d23efdb84"
            ]
        },
        "request": [
            {
                "table": "mex_inegi_inpc_product_monthly",
                "variables": [
                    "c572db59b8cd109"
                ]
            }
        ],
        "from": from_d
    }

    response = get_tukan_api_request(payload)
    data = response["data"]

    # ----
    # The filters
    # ----
    aux_df = data.copy()
    max_date = aux_df['date'].max()
    aux_df = aux_df[aux_df['date'] == max_date]
    aux_df = aux_df.sort_values(
        'c572db59b8cd109', ascending=False).reset_index(drop=True)
    top_products = aux_df.head(3).copy()  # change to 10 for the table
    top_products = top_products['product'].unique().tolist()
    data = data[data["product"].isin(top_products)]

    # ----
    # The table
    # ----
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)

    X_max = data["date"].iloc[-1]

    sort_products = data[data["date"] == data["date"].max()].sort_values(
        by="c572db59b8cd109", ascending=False)
    products = list(sort_products['product'].unique())
    # So we don't get very light colors
    cmap = mpl.cm.get_cmap("GnBu_r", len(products) + 4)

    for index, product in enumerate(products):
        plot_data_aux = data[data["product"] == product].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["c572db59b8cd109"],
                marker="o", markevery=[-1], color=cmap(index), mec="white", ms=5)
        Y_end = plot_data_aux["c572db59b8cd109"].iloc[-1]
        ax_text(x=X_max + relativedelta(months=1), y=Y_end,
                s=f"<{product}>",
                highlight_textprops=[{"color": cmap(index)}],
                ax=ax, weight="bold", font="Dosis", ha="left", size=7)

    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.xaxis.set_major_locator(mdates.MonthLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))

    plt.savefig(
        "plots/es_yoy_agricultural_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

# %%
# ------------------------------------------------------------------
#
# CHART 8: AGRICULTURAL PRODUCTS
#
# ------------------------------------------------------------------


def plot_chart_8(from_d="2019-01-01"):

    payload = {
        "type": "data_table",
        "operation": "yoy_growth_rel",
        "language": "es",
        "group_by": [
            "product"
        ],
        "categories": {
            "product": [
                "aa7513e136a9ef7",
                "2b6594483f75fb6",
                "91f000d00e522da",
                "5673af6b6ddec2b",
                "c4a63ce96d9ea96",
                "08489177c159831",
                "4099795618656fd",
                "ee9914e30cfb90e",
                "310cec97eb65812",
                "6d208962bcada90",
                "b52772488920101",
                "7259636d172fee3",
                "0ffa6e2dd228574",
                "9d2344bdf1dd97a",
                "02b414002b0923d",
                "787bfe317e79fe7",
                "83b09ec629dc758",
                "da5aff254a26b4c",
                "732526d8719ceab",
                "0c50808b2060295",
                "93340301fa113df",
                "cbcdd524e734d47",
                "e4f91a6cd3bb389",
                "45c9665efb643c2",
                "ae07a5e99ac16fe",
                "64967d83ac5aa6d",
                "ec7f658bb8caabc",
                "0189693b7f7987f",
                "d86a62764deab74",
                "2e0f130a7321515",
                "0b67cb47af7f8ea"
            ]
        },
        "request": [
            {
                "table": "mex_inegi_inpc_product_monthly",
                "variables": [
                    "c572db59b8cd109"
                ]
            }
        ],
        "from": from_d
    }

    response = get_tukan_api_request(payload)
    data = response["data"]

    weight_payload = {
        "type": "data_table",
        "operation": "sum",
        "language": "es",
        "group_by": [
            "product"
        ],
        "categories": {
            "product": "all"
        },
        "request": [
            {
                "table": "mex_inegi_inpc_product_weights",
                "variables": "all"
            }
        ]
    }

    response_weights = get_tukan_api_request(weight_payload)
    weight_data = response_weights["data"]

    data = pd.merge(data, weight_data, how="left", on="product__ref")

    # ----
    # The filters
    # ----
    aux_df = data.copy()
    max_date = aux_df['date'].max()
    aux_df = aux_df[aux_df['date'] == max_date]
    aux_df = aux_df.sort_values(
        'c572db59b8cd109', ascending=False).reset_index(drop=True)
    top_products = aux_df.head(3).copy()  # change to 10 for the table
    top_products = top_products['product'].unique().tolist()
    data = data[data["product"].isin(top_products)]

    # ----
    # The chart (for top 3)
    # ----
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)

    ax.scatter(test_df["5993e5e787e4259"], test_df["c572db59b8cd109"])

    X_max = data["date"].iloc[-1]

    sort_products = data[data["date"] == data["date"].max()].sort_values(
        by="c572db59b8cd109", ascending=False)
    products = list(sort_products['product'].unique())
    # So we don't get very light colors
    cmap = mpl.cm.get_cmap("GnBu_r", len(products) + 4)

    for index, product in enumerate(products):
        plot_data_aux = data[data["product"] == product].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["c572db59b8cd109"],
                marker="o", markevery=[-1], color=cmap(index), mec="white", ms=5)
        Y_end = plot_data_aux["c572db59b8cd109"].iloc[-1]
        ax_text(x=X_max + relativedelta(months=1), y=Y_end,
                s=f"<{product}>",
                highlight_textprops=[{"color": cmap(index)}],
                ax=ax, weight="bold", font="Dosis", ha="left", size=7)

    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.xaxis.set_major_locator(mdates.MonthLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))

    plt.savefig(
        "plots/es_yoy_agricultural_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

# %%
# ------------------------------------------------------------------
#
# CHART 9: ENERGY PRODUCTS YOY INFLATION RATE
#
# ------------------------------------------------------------------


def plot_chart_9(from_d="2019-01-01"):

    payload = {
        "type": "data_table",
        "operation": "yoy_growth_rel",
        "language": "es",
        "group_by": [
            "product"
        ],
        "categories": {
            "product": [
                "d56f0b272070132",
                "ff51d18393e3638",
                "615caab67c0ea3d",
                "e78e4ada1e4ffff"
            ]
        },
        "request": [
            {
                "table": "mex_inegi_inpc_product_monthly",
                "variables": [
                    "c572db59b8cd109"
                ]
            }
        ],
        "from": from_d
    }

    response = get_tukan_api_request(payload)
    data = response["data"]

    # ----
    # The chart
    # ----
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)

    X_max = data["date"].iloc[-1]

    sort_products = data[data["date"] == data["date"].max()].sort_values(
        by="c572db59b8cd109", ascending=False)
    products = list(sort_products['product'].unique())
    # So we don't get very light colors
    cmap = mpl.cm.get_cmap("GnBu_r", len(products) + 4)

    for index, product in enumerate(products):
        plot_data_aux = data[data["product"] == product].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["c572db59b8cd109"],
                marker="o", markevery=[-1], color=cmap(index), mec="white", ms=5)
        Y_end = plot_data_aux["c572db59b8cd109"].iloc[-1]
        # weight_end = plot_data_aux["total_loans"].iloc[-1]
        ax_text(x=X_max + relativedelta(months=1), y=Y_end,
                s=f"<{product}>",
                highlight_textprops=[{"color": cmap(index)}],
                ax=ax, weight="bold", font="Dosis", ha="left", size=7)

    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.xaxis.set_major_locator(mdates.MonthLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))

    plt.savefig(
        "plots/es_yoy_energy_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    # # ---
    # print(
    #     f"Annual inflation from the terciary sector came in at {Y_end:.2%} during {X_max.strftime('%b-%Y')}")

# %%
# ------------------------------------------------------------------
#
# CHART 10: SERVICES YOY INFLATION RATE
#
# ------------------------------------------------------------------


def plot_chart_10(from_d="2000-01-01"):

    payload = {
        "type": "data_table",
        "operation": "yoy_growth_rel",
        "language": "es",
        "group_by": [
            "economic_activity"
        ],
        "categories": {
            "economic_activity": [
                "8fd5b02b9f891fb"
            ]
        },
        "request": [
            {
                "table": "mex_inegi_inpc_scian_monthly",
                "variables": [
                    "c572db59b8cd109"
                ]
            }
        ],
        "from": from_d
    }

    response = get_tukan_api_request(payload)
    data = response["data"]

    # ----
    # The chart
    # ----
    cmap = mpl.cm.get_cmap("GnBu_r", 5)
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)

    ax.plot(data["date"], data["c572db59b8cd109"],
            marker="o", ms=6, mec="white", markevery=[-1], color=cmap(0))

    Y_end = data["c572db59b8cd109"].iloc[-1]
    X_max = data["date"].iloc[-1]

    ax_text(x=X_max + relativedelta(months=3), y=Y_end,
            s=f"<{Y_end:.2%}>",
            highlight_textprops=[{"color": cmap(0)}],
            ax=ax, weight="bold", font="Dosis", ha="left", size=9)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))
    ax.set_ylim(0)

    plt.savefig(
        "plots/esyoy_terciary_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    # ---
    print(
        f"Annual inflation from the terciary sector came in at {Y_end:.2%} during {X_max.strftime('%b-%Y')}")


# %%
