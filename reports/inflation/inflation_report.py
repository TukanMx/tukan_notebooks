# %%
from tukan_helper_functions import get_tukan_api_request
import sys
import os
from urllib import response
import pandas as pd
import numpy as np

# For creating cool charts :)
import matplotlib.pyplot as plt
from matplotlib import image
import matplotlib as mpl
from dateutil.relativedelta import relativedelta
from highlight_text import ax_text
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from highlight_text import ax_text
import datetime

# ----------------------------------
# We load fonts and stylesheet.
# ----------------------------------
module_path = os.path.abspath(os.path.join('../../'))
plt.style.use(module_path + '\\utils\\tukan_style.mpl')
if module_path not in sys.path:
    sys.path.append(module_path+"\\utils")
    sys.path.append(module_path+"\\assets")


# ----------------------------------
# We load the get_tukan_api_request to query TUKAN's API
# ----------------------------------


# ------------------------------------------------------------------
#
#                       CHARTS FOR REPORT
#
# ------------------------------------------------------------------

# ------------------------------------------------------------------
#
# CHART 1: YOY CHANGE IN CPI
#
# ------------------------------------------------------------------

# %%

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
    cpi_data_underlying.loc[:, "cpi_under_cont"] = [
        x*y for x, y in zip(cpi_data_underlying["c572db59b8cd109"], cpi_data_underlying["5993e5e787e4259"])]
    cpi_data_underlying.loc[:, "cpi_under"] = [
        x*y for x, y in zip(cpi_data_underlying["c572db59b8cd109"], cpi_data_underlying["2dac8b4b2fc8037"])]

    cpi_data_underlying = cpi_data_underlying.groupby(
        ["date"]).sum(["cpi_under_cont", "cpi_under"]).reset_index()

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
    cpi_data_non_underlying.loc[:, "cpi_non_under_cont"] = [
        x*y for x, y in zip(cpi_data_non_underlying["c572db59b8cd109"], cpi_data_non_underlying["5993e5e787e4259"])]
    cpi_data_non_underlying.loc[:, "cpi_non_under"] = [
        x*y for x, y in zip(cpi_data_non_underlying["c572db59b8cd109"], cpi_data_non_underlying["a5a09b8e3b7cbb5"])]

    cpi_data_non_underlying = cpi_data_non_underlying.groupby(
        ["date"]).sum(["cpi_non_under_cont", "cpi_non_under"]).reset_index()

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

    ax.set_xticks(index_ticks + width / 2,
                  labels=plot_df["date"].dt.strftime("%b-%y"), rotation=90)

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
