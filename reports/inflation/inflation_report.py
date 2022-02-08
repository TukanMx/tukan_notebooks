# %%
# ----------------------------------
# We load fonts and stylesheet.
# ----------------------------------
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\utils")
    sys.path.append(module_path+"\\assets")

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
plt.style.use(module_path + '\\utils\\tukan_style.mpl')


# ----------------------------------
# We load the get_tukan_api_request to query TUKAN's API
# ----------------------------------


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
# ------------------------------------------------------------------
#
# CHART 3: NON-AGRICULTURAL PRODUCTS IN BASIC BASKET
#
# ------------------------------------------------------------------

def plot_chart_3(from_d="2019-01-01"):

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
    aux_df = aux_df[aux_df['date']== max_date]
    aux_df = aux_df.sort_values('c572db59b8cd109', ascending=False).reset_index(drop=True)
    top_products = aux_df.head(3).copy() # change to 10 for the table
    top_products = top_products['product'].unique().tolist()
    data = data[data["product"].isin(top_products)]
    
    # ----
    # The table
    # ----
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)
    
    X_max = data["date"].iloc[-1]

    sort_products = data[data["date"] == data["date"].max()].sort_values(by = "c572db59b8cd109", ascending = False)
    products = list(sort_products['product'].unique())
    cmap = mpl.cm.get_cmap("GnBu_r", len(products) + 4) # So we don't get very light colors

    for index, product in enumerate(products):
        plot_data_aux = data[data["product"] == product].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["c572db59b8cd109"], marker = "o", markevery = [-1], color = cmap(index), mec = "white", ms = 5)
        Y_end = plot_data_aux["c572db59b8cd109"].iloc[-1]
        ax_text(x = X_max + relativedelta(months = 1), y = Y_end,
                s = f"<{product}>",
                highlight_textprops=[{"color": cmap(index)}], 
                                    ax = ax, weight = "bold", font = "Dosis", ha = "left", size = 7)

    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.xaxis.set_major_locator(mdates.MonthLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))

    plt.savefig(
        "plots/yoy_agricultural_change.svg",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    
# %%
# ------------------------------------------------------------------
#
# CHART 4: AGRICULTURAL PRODUCTS
#
# ------------------------------------------------------------------

def plot_chart_4(from_d="2019-01-01"):

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

    # ----
    # The filters
    # ----
    aux_df = data.copy()
    max_date = aux_df['date'].max()
    aux_df = aux_df[aux_df['date']== max_date]
    aux_df = aux_df.sort_values('c572db59b8cd109', ascending=False).reset_index(drop=True)
    top_products = aux_df.head(3).copy() # change to 10 for the table
    top_products = top_products['product'].unique().tolist()
    data = data[data["product"].isin(top_products)]
    
    # ----
    # The chart (for top 3)
    # ----
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)
    
    X_max = data["date"].iloc[-1]

    sort_products = data[data["date"] == data["date"].max()].sort_values(by = "c572db59b8cd109", ascending = False)
    products = list(sort_products['product'].unique())
    cmap = mpl.cm.get_cmap("GnBu_r", len(products) + 4) # So we don't get very light colors

    for index, product in enumerate(products):
        plot_data_aux = data[data["product"] == product].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["c572db59b8cd109"], marker = "o", markevery = [-1], color = cmap(index), mec = "white", ms = 5)
        Y_end = plot_data_aux["c572db59b8cd109"].iloc[-1]
        ax_text(x = X_max + relativedelta(months = 1), y = Y_end,
                s = f"<{product}>",
                highlight_textprops=[{"color": cmap(index)}], 
                                    ax = ax, weight = "bold", font = "Dosis", ha = "left", size = 7)

    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.xaxis.set_major_locator(mdates.MonthLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))

    plt.savefig(
        "plots/yoy_agricultural_change.svg",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
       
# %%
# ------------------------------------------------------------------
#
# CHART 5: ENERGY PRODUCTS YOY INFLATION RATE
#
# ------------------------------------------------------------------

def plot_chart_5(from_d="2019-01-01"):

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

    sort_products = data[data["date"] == data["date"].max()].sort_values(by = "c572db59b8cd109", ascending = False)
    products = list(sort_products['product'].unique())
    cmap = mpl.cm.get_cmap("GnBu_r", len(products) + 4) # So we don't get very light colors

    for index, product in enumerate(products):
        plot_data_aux = data[data["product"] == product].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["c572db59b8cd109"], marker = "o", markevery = [-1], color = cmap(index), mec = "white", ms = 5)
        Y_end = plot_data_aux["c572db59b8cd109"].iloc[-1]
        # loans_end = plot_data_aux["total_loans"].iloc[-1]
        ax_text(x = X_max + relativedelta(months = 1), y = Y_end,
                s = f"<{product}>",
                highlight_textprops=[{"color": cmap(index)}], 
                                    ax = ax, weight = "bold", font = "Dosis", ha = "left", size = 7)

    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

    


    ax.xaxis.set_major_locator(mdates.MonthLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))

    plt.savefig(
        "plots/yoy_energy_change.svg",
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
# CHART 6: SERVICES YOY INFLATION RATE
#
# ------------------------------------------------------------------
def plot_chart_6(from_d="2000-01-01"):

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
        "plots/yoy_terciary_change.svg",
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
