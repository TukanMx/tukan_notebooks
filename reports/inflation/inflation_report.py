# %%
# ----------------------------------
# We load fonts and stylesheet.
# ----------------------------------
import datetime
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

# %%
# ----------------------------------
# We load the product_mapping_inegi csv file to map the products to the INEGI category
# ----------------------------------


def map_tukan_inegi_products(from_d="2019-01-01"):

    # Map to match INEGI classifications
    tukan_inegi_map = pd.read_csv(
        "product_mapping_inegi.csv", encoding="utf-8")
    tukan_inegi_map.drop(["weight"], axis=1, inplace=True)

    weight_payload = {
        "type": "data_table",
        "operation": "sum",
        "language": "en",
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
        "language": "en",
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
        "plots/yoy_cpi_change.svg",
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
        "language": "en",
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
        "language": "en",
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
    fig = plt.figure(figsize=(8, 3), dpi=200)
    ax = plt.subplot(111)

    index_ticks = np.arange(plot_df["date"].shape[0])
    width = 0.25

    ax.bar(index_ticks, plot_df["yoy_under"], color=cmap(
        0), width=width, zorder=3, label="Core inflation")
    ax.bar(index_ticks + width, plot_df["yoy_non_under"],
           color=cmap(1), width=width, zorder=3, label="Non-core inflation")

    ax.set_xticks(index_ticks + width / 2)
    ax.set_xticklabels(plot_df["date"].dt.strftime("%b-%y"), rotation=90)

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))
    ax.set_ylim(0)

    ax.legend(loc="lower center", bbox_to_anchor=(0.55, -0.45), ncol=2)

    plt.savefig(
        "plots/core_vs_non_core_change.svg",
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

    fig = plt.figure(figsize=(8, 3), dpi=200)

    ax = plt.subplot(111)

    sub_indices = list(sort_index["primary"].unique())

    for index, sub_index in enumerate(sub_indices):
        plot_data_aux = yoy_data[yoy_data["primary"] == sub_index].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["yoy_change"],
                marker="o", markevery=[-1], color=cmap(index), mec="white", ms=5)
        Y_end = plot_data_aux["yoy_change"].iloc[-1]
        weight_end = plot_data_aux["rel_weight"].iloc[-1]
        if sub_index == "Education":
            Y_end = Y_end + 0.005
        ax_text(x=X_max + datetime.timedelta(15), y= Y_end + 0.001,
                s=f"<{sub_index}>",
                highlight_textprops=[{"color": cmap(index)}],
                                    ax=ax, weight="bold", font="Dosis", ha="left", size=7.5)

    ax.set_xlim(X_min, X_max + relativedelta(months=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    if yoy_data["yoy_change"].min() < 0:
        ax.hlines(0, X_min, X_max, ls = "--", color = "black", lw = 0.75)
    else:
        ax.set_ylim(0)


    plt.savefig(
        "plots/core_subindices.svg",
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

    fig = plt.figure(figsize=(8, 3), dpi=200)

    ax = plt.subplot(111)

    sub_indices = list(sort_index["primary"].unique())

    for index, sub_index in enumerate(sub_indices):
        plot_data_aux = yoy_data[yoy_data["primary"] == sub_index].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["yoy_change"],
                marker="o", markevery=[-1], color=cmap(index), mec="white", ms=5)
        Y_end = plot_data_aux["yoy_change"].iloc[-1]
        weight_end = plot_data_aux["rel_weight"].iloc[-1]
        ax_text(x=X_max + datetime.timedelta(15), y= Y_end + 0.001,
                s=f"<{sub_index}>",
                highlight_textprops=[{"color": cmap(index)}],
                                    ax=ax, weight="bold", font="Dosis", ha="left", size=7.5)

    ax.set_xlim(X_min, X_max + relativedelta(months=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    if yoy_data["yoy_change"].min() < 0:
        ax.hlines(0, X_min, X_max, ls = "--", color = "black", lw = 0.75)
    else:
        ax.set_ylim(0)


    plt.savefig(
        "plots/non_core_subindices.svg",
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
    core_data.loc[:, "yoy_change"] = core_data["index_weighted"] - core_data["12_m_lag"]
    core_data.loc[:, "mom_change"] = core_data["index_weighted"] - core_data["1_m_lag"]

    core_inflation = core_data.groupby(["date"])[["yoy_change", "mom_change"]].sum()
    core_inflation.reset_index(inplace = True)
    core_inflation.rename(columns = {"yoy_change":"yoy_core","mom_change":"mom_core"}, inplace = True)

    core_data = pd.merge(core_data, core_inflation, how = "left", on = "date")
    core_data.loc[:,"yoy_contribution"] = core_data["yoy_change"]/core_data["yoy_core"]

    core_data = core_data[core_data["date"] == core_data["date"].max()]
    core_data = core_data.sort_values(by = "yoy_contribution", ascending=False).head(9)
    
    # True YoY and MoM Change
    core_data.loc[:,"yoy_change"] = core_data["yoy_change"]/core_data["12_m_lag"]
    core_data.loc[:,"mom_change"] = core_data["mom_change"]/core_data["1_m_lag"]

    top_10 = list(core_data["product__ref"])

    core_data.reset_index(drop = True, inplace = True)

    core_data = core_data[["product", "primary", "yoy_change", "mom_change", "yoy_contribution"]]
    core_data.replace({"product":{"Electric power transmission services": "Electric power trans."}}, inplace = True)
    core_data.set_index("product", inplace = True)
    core_data.columns = ["Group", "YoY %", "MoM %", "Cont. to\nYoY%"]


    fig = plt.figure(figsize = (6.5,6), dpi = 200)

    ax = fig.add_subplot(111)
    ax.set_ylim(-8.25,1.5)
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
            size = 8,
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
            ax.plot([-0.75,9.15], [-1*row - .25, -1*row - .25], color = color, linewidth = linewidth, ls = "-")
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
                size = 8,
                va="top",
                ha="center",
                weight="bold"
            )

    ax.plot([1.9,8.15], [.75, .75], color = "black", linewidth = 0.85)


    ax.add_patch(mpl.patches.Rectangle((1.9, .75), width=7.15, height=0.8, linewidth=1,
            color='#2B5AA5', fill=True))


    plt.savefig(
        "plots/non_core_top_10.svg",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    cpi_payload = {
        "type": "data_table",
        "operation": "yoy_growth_rel",
        "language": "en",
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
    cpi_data.replace({"product":{"Electric power transmission services": "Electric power trans."}}, inplace = True)
    

    cmap = mpl.cm.get_cmap(
        "GnBu_r", 5)

    for index, x in enumerate(top_10):
        fig = plt.figure(figsize=(3, 1.5), dpi = 200)
        ax = plt.subplot(111)
        aux_aux_df = cpi_data[cpi_data["product__ref"] == x]
        product = aux_aux_df["product"].iloc[0] 
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.plot(aux_aux_df["date"], aux_aux_df["c572db59b8cd109"], zorder = 10, lw = 1, color = cmap(0))
        ax.tick_params(axis='both', which='major', labelsize=6)
        Y_max = aux_aux_df["c572db59b8cd109"].max()
        Y_min = aux_aux_df["c572db59b8cd109"].min()
        ax.text(
                aux_aux_df["date"].min() + datetime.timedelta(days = 20),
                Y_max + (Y_max - Y_min)/4,
                f"{product}",
                horizontalalignment="left",
                verticalalignment="top",
                size = 7,
                weight = "bold",
                zorder = 3
            )
        
        if Y_min < 0:
            ax.hlines(0, xmin = aux_aux_df["date"].min(), xmax = aux_aux_df["date"].max(),
            ls = "--", color = "black", lw = 0.75)
        else:
            ax.set_ylim(0)

        plt.savefig(
            f"plots/non_core_{product}_ts.svg",
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
    core_data.loc[:, "yoy_change"] = core_data["index_weighted"] - core_data["12_m_lag"]
    core_data.loc[:, "mom_change"] = core_data["index_weighted"] - core_data["1_m_lag"]

    core_inflation = core_data.groupby(["date"])[["yoy_change", "mom_change"]].sum()
    core_inflation.reset_index(inplace = True)
    core_inflation.rename(columns = {"yoy_change":"yoy_core","mom_change":"mom_core"}, inplace = True)

    core_data = pd.merge(core_data, core_inflation, how = "left", on = "date")
    core_data.loc[:,"yoy_contribution"] = core_data["yoy_change"]/core_data["yoy_core"]

    core_data = core_data[core_data["date"] == core_data["date"].max()]
    core_data = core_data.sort_values(by = "yoy_contribution", ascending=False).head(9)
    
    # True YoY and MoM Change
    core_data.loc[:,"yoy_change"] = core_data["yoy_change"]/core_data["12_m_lag"]
    core_data.loc[:,"mom_change"] = core_data["mom_change"]/core_data["1_m_lag"]

    top_10 = list(core_data["product__ref"])

    core_data.reset_index(drop = True, inplace = True)

    core_data = core_data[["product", "primary", "yoy_change", "mom_change", "yoy_contribution"]]
    core_data.replace({"product":{"Diners, inns, torterías and taquerías": "Diners, inns & others", "Passenger air transportation": "Air transportation"}}, inplace = True)
    core_data.set_index("product", inplace = True)
    core_data.columns = ["Group", "YoY %", "MoM %", "Cont. to\nYoY%"]


    fig = plt.figure(figsize = (6.5,6), dpi = 200)

    ax = fig.add_subplot(111)
    ax.set_ylim(-8.25,1.5)
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
            size = 8,
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
            ax.plot([-0.75,9.15], [-1*row - .25, -1*row - .25], color = color, linewidth = linewidth, ls = "-")
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
                size = 8,
                va="top",
                ha="center",
                weight="bold"
            )

    ax.plot([1.9,9.15], [.75, .75], color = "black", linewidth = 0.85)


    ax.add_patch(mpl.patches.Rectangle((1.9, .75), width=7.15, height=0.8, linewidth=1,
            color='#2B5AA5', fill=True))


    plt.savefig(
        "plots/core_top_10.svg",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    cpi_payload = {
        "type": "data_table",
        "operation": "yoy_growth_rel",
        "language": "en",
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
    cpi_data.replace({"product":{"Diners, inns, torterías and taquerías": "Diners, inns & others", "Passenger air transportation": "Air transportation"}}, inplace = True)
    

    cmap = mpl.cm.get_cmap(
        "GnBu_r", 5)

    for index, x in enumerate(top_10):
        fig = plt.figure(figsize=(3, 1.5), dpi = 200)
        ax = plt.subplot(111)
        aux_aux_df = cpi_data[cpi_data["product__ref"] == x]
        product = aux_aux_df["product"].iloc[0] 
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.plot(aux_aux_df["date"], aux_aux_df["c572db59b8cd109"], zorder = 10, lw = 1, color = cmap(0))
        ax.tick_params(axis='both', which='major', labelsize=6)
        Y_max = aux_aux_df["c572db59b8cd109"].max()
        Y_min = aux_aux_df["c572db59b8cd109"].min()
        ax.text(
                aux_aux_df["date"].min() + datetime.timedelta(days = 20),
                Y_max + (Y_max - Y_min)/4,
                f"{product}",
                horizontalalignment="left",
                verticalalignment="top",
                size = 7,
                weight = "bold",
                zorder = 3
            )
        
        if Y_min < 0:
            ax.hlines(0, xmin = aux_aux_df["date"].min(), xmax = aux_aux_df["date"].max(),
            ls = "--", color = "black", lw = 0.75)
        else:
            ax.set_ylim(0)

        plt.savefig(
            f"plots/core_{product}_ts.svg",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            transparent=False,
        )

# %%
# ------------------------------------------------------------------
#
# CHART 6: SERVICES BREAKDOWN YOY INFLATION RATE
#
# ------------------------------------------------------------------
def plot_chart_6(from_d="2012-01-01"):

    payload = {
    "type": "data_table",
    "operation": "yoy_growth_rel",
    "language": "en",
    "group_by": [
        "economic_activity"
    ],
    "categories": {
        "economic_activity": [
            "f5adaadda584ca7",
            "e426cc87d0540ab",
            "990b94ebe38c9ca",
            "d35f5b82779e7d5",
            "3726993cc9fecab",
            "bbb49ae78601ab9",
            "4bc9836c2d7e60a",
            "fcb303b72a98f6c",
            "d05c3b2b73d75fc",
            "feb7bb4445c808d",
            "a07267f78158c2c",
            "44d246411040129"
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

    data.replace({'Administrative and support and waste management and remediation services':'Waste Management',
       'Real estate and rental and leasing':'Real Estate',
       'Health care and social assistance':'Healthcare',
       'Other services (except public administration)':'Other services',
       'Professional, scientific, and technical services':'Professional services',
       'Arts, entertainment, and recreation':'Entretainment', 'Finance and insurance':'Finance and insurance',
       'Accommodation and food services':'Hotels & restaurants', 'Information':'Information services',
       'Governmental, legislative activities of law enforcement and international and extraterritorial bodies':'Governmental activities',
       'Educational services':'Educational services', 'Transportation and warehousing':'Transportation & warehousing'}, inplace=True)
    
    # data = data[(data['economic_activity'] != 'Waste Management')]
    # data = data[(data['economic_activity'] != 'Governmental activities')]

    sort_data = data[data["date"] == data["date"].max()].sort_values(by = "c572db59b8cd109", ascending = False)
    cmap = mpl.cm.get_cmap(
        "GnBu_r", 5)
    
    # ----
    # The chart
    # ----
    
    fig = plt.figure(figsize=(8, 7), dpi = 500)
    gspec = mpl.gridspec.GridSpec(ncols=2, nrows=5, figure=fig)

    #Locators for axis
    month_locator = mdates.MonthLocator(interval=6)
    loc = ticker.MultipleLocator(base=0.1)

    max_date = data["date"].iloc[-1]

    index_aux = 0
    for index, x in enumerate(sort_data["economic_activity"].unique()):
        if index % 2 == 0:
            col = 0
        else:
            col = 1
        aux_aux_df = data[data["economic_activity"] == x]
        ax = plt.subplot(gspec[index_aux,col])
        ax.plot(aux_aux_df["date"], aux_aux_df["c572db59b8cd109"], linewidth = 1, zorder = 3, color = cmap(0))
        ax.set_ylim(data["c572db59b8cd109"].min() - 0.05, data["c572db59b8cd109"].max() + 0.05)
        ax.hlines(0, xmin = aux_aux_df["date"].min(), xmax = aux_aux_df["date"].max(), ls = "--", color = "black", lw = 1.1)
        for activity in data["economic_activity"].unique():
            activity_filter = data["economic_activity"] == activity
            ax.plot(data[activity_filter]["date"], data[activity_filter]["c572db59b8cd109"], linewidth = 0.95, color = "gray", alpha = 0.35, zorder = 1)
            Y_end=data[activity_filter]["c572db59b8cd109"].iloc[-1]
            

        ax.text(
            aux_aux_df["date"].min() + datetime.timedelta(days = 20),
            data["c572db59b8cd109"].max() + 0.05,
            x + " (YoY " + "{:,.1%}".format(aux_aux_df[aux_aux_df["date"] == max_date]["c572db59b8cd109"].iloc[-1]) + ")",
            horizontalalignment="left",
            verticalalignment="top",
            size = 7.5,
            weight = "bold",
            zorder = 3
        )

        ax.grid(True, "major", axis = "x", ls = "--", color = "#dddddd")
        # ax.grid(True, "minor", axis = "x", ls = "--", lw = 0.5, color = "#dddddd")
        ax.grid(True, "major", axis = "y", ls = "--", color = "#dddddd")
        ax.xaxis.set_minor_locator(month_locator)
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")) #"%b-%Y"
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_locator(loc)

        ax.tick_params("both", labelsize=6.5, which="major")
        
        if (index_aux != len(data["economic_activity"].unique())/2 - 1):
            # ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False)
            ax.xaxis.set_ticklabels([])

        if index % 2 != 0:
            index_aux += 1

    plt.tight_layout()


    plt.savefig(
        "plots/yoy_terciary_breakdown_change.svg",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )


# %%
# ------------------------------------------------------------------
#
# CHART 7: YoY Change in Primary, Secondary and Tertiary Activities
#
# ------------------------------------------------------------------
def plot_chart_7(from_d="2012-01-01"):

    payload = {
        "type": "data_table",
        "operation": "yoy_growth_rel",
        "language": "en",
        "group_by": [
            "economic_activity"
        ],
        "categories": {
            "economic_activity": [
                "7460634ca523beb",
                "761bc00426e1c48",
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
    economic_activities = ['Primary activities', 'Secondary activities', 'Tertiary activities']

    cmap = mpl.cm.get_cmap(
        "GnBu_r", 5)

    X_min = data["date"].min()
    X_max = data["date"].max()

    fig = plt.figure(figsize=(8, 3), dpi=200)

    ax = plt.subplot(111)


    for index, sub_index in enumerate(economic_activities):
        plot_data_aux = data[data["economic_activity"] == sub_index].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["c572db59b8cd109"],
                marker="o", markevery=[-1], color=cmap(index), mec="white", ms=5)
        Y_end = plot_data_aux["c572db59b8cd109"].iloc[-1]

        ax_text(x=X_max + datetime.timedelta(20), y = Y_end + 0.006,
                s=f"<{sub_index}>",
                highlight_textprops=[{"color": cmap(index)}],
                                    ax=ax, weight="bold", font="Dosis", ha="left", size=7.5)

    ax.set_xlim(X_min, X_max + relativedelta(months=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))

    if data["c572db59b8cd109"].min() < 0:
        ax.hlines(0, X_min, X_max, ls = "--", color = "black", lw = 0.75)
    else:
        ax.set_ylim(0)


    plt.savefig(
        "plots/naics_sectors_yoy.svg",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )


# %%
# ------------------------------------------------------------------
#
# CHART 8: PRIMARY YOY INFLATION RATE
#
# ------------------------------------------------------------------
def plot_chart_8(from_d="2012-01-01"):

    payload = {
        "type": "data_table",
        "operation": "yoy_growth_rel",
        "language": "en",
        "group_by": [
            "economic_activity"
        ],
        "categories": {
            "economic_activity": [
                "ae66086d7d89185",
                "3f551bcf12544f7",
                "6c39dba3c30a0a9"
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

    data.replace({"economic_activity":{'Animal production and aquaculture':"Livestock & aquaculture",
       'Fishing, hunting and trapping':"Fishing & hunting"}}, inplace = True)

    # ----
    # The chart
    # ----
    fig = plt.figure(figsize=(8, 3), dpi=200)
    ax = plt.subplot(111)
    
    X_min = data["date"].min()
    X_max = data["date"].max()

    sort_activities = data[data["date"] == data["date"].max()].sort_values(by = "c572db59b8cd109", ascending = False)
    activities = list(sort_activities['economic_activity'].unique())
    cmap = mpl.cm.get_cmap("GnBu_r", 5) # So we don't get very light colors

    for index, activity in enumerate(activities):
        plot_data_aux = data[data["economic_activity"] == activity].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["c572db59b8cd109"], marker = "o", markevery = [-1], color = cmap(index), mec = "white", ms = 5)
        Y_end = plot_data_aux["c572db59b8cd109"].iloc[-1]
        if activity == "Crop production":
            Y_end = Y_end + 0.04
        elif activity == "Fishing & hunting":
            Y_end = Y_end + 0.01
        ax_text(x = X_max + datetime.timedelta(20), y = Y_end,
                s = f"<{activity}>",
                highlight_textprops=[{"color": cmap(index)}], 
                                    ax = ax, weight = "bold", font = "Dosis", ha = "left", size = 7.5)

    ax.set_xlim(X_min, X_max + relativedelta(months=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))

    if data["c572db59b8cd109"].min() < 0:
        ax.hlines(0, X_min, X_max, ls = "--", color = "black", lw = 0.75)
    else:
        ax.set_ylim(0)

    plt.savefig(
        "plots/yoy_primary_breakdown_change.svg",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

# %%
# ------------------------------------------------------------------
#
# CHART 9: SECONDARY BREAKDOWN YOY INFLATION RATE
#
# ------------------------------------------------------------------

def plot_chart_9(from_d="2012-01-01"):

    payload = {
        "type": "data_table",
        "operation": "yoy_growth_rel",
        "language": "en",
        "group_by": [
            "economic_activity"
        ],
        "categories": {
            "economic_activity": [
                "faa2a8d0af8a72c",
                "c5b7edbae753f2b",
                "1743f1e9abaf5e0"
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
    data.replace({"economic_activity":{'Electric power generation, transmission and distribution':"Electric utility",
       'Water, sewage and other systems':"Other utilities"}}, inplace = True)

    # ----
    # The chart
    # ----
    fig = plt.figure(figsize=(8, 3), dpi=200)
    ax = plt.subplot(111)
    
    X_min = data["date"].min()
    X_max = data["date"].max()

    sort_activities = data[data["date"] == data["date"].max()].sort_values(by = "c572db59b8cd109", ascending = False)
    activities = list(sort_activities['economic_activity'].unique())
    cmap = mpl.cm.get_cmap("GnBu_r", 5) # So we don't get very light colors

    for index, activity in enumerate(activities):
        plot_data_aux = data[data["economic_activity"] == activity].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["c572db59b8cd109"], marker = "o", markevery = [-1], color = cmap(index), mec = "white", ms = 5)
        Y_end = plot_data_aux["c572db59b8cd109"].iloc[-1]
        if activity == "Crop production":
            Y_end = Y_end + 0.04
        elif activity == "Fishing & hunting":
            Y_end = Y_end + 0.01
        ax_text(x = X_max + datetime.timedelta(20), y = Y_end,
                s = f"<{activity}>",
                highlight_textprops=[{"color": cmap(index)}], 
                                    ax = ax, weight = "bold", font = "Dosis", ha = "left", size = 7.5)

    ax.set_xlim(X_min, X_max + relativedelta(months=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))

    if data["c572db59b8cd109"].min() < 0:
        ax.hlines(0, X_min, X_max, ls = "--", color = "black", lw = 0.75)
    else:
        ax.set_ylim(0)

    plt.savefig(
        "plots/yoy_secondary_breakdown_change.svg",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

# response.cookies["TS01c32642"] == response_2.cookies["TS01c32642"]