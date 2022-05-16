'''
Inflation Report Functions.
----------------------------------------------
Main Author: Miguel Angel Dávila

This script contains function that generate our TUKAN Monthly Mexican
inflation report.
'''

from tukan_mkt.helpers import get_tukan_api_request
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from highlight_text import ax_text
from dateutil.relativedelta import relativedelta
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime



# --- Function that helps us map INEGI products to TUKAN's catalog

def map_tukan_inegi_products(from_d="2019-01-01"):

    # Map to match INEGI classifications
    tukan_inegi_map = pd.read_csv(
        r"C:\Users\migue\Documents\TUKAN\tukan_notebooks\assets\maps\product_mapping_inegi.csv", 
        encoding="utf-8")
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



# --- Mexican annual inflation.

def mex_annual_inflation_line(f, from_d="2000-01-01"):
    '''
    This function plots Mexican annual inflation.

    Args:
        (from_d): From when do we want this information from.
        (f): the matplotlib figure.
    '''

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

    ax = f.add_subplot(111)
    
    line_ = ax.plot(data["date"], data["c572db59b8cd109"], marker="o", ms=6, mec="white", markevery=[-1])

    Y_end = data["c572db59b8cd109"].iloc[-1]
    X_max = data["date"].iloc[-1]

    ax_text(x=X_max + relativedelta(months=3), y=Y_end,
            s=f"<{Y_end:.2%}>",
            highlight_textprops=[{"color": line_[0].get_color()}],
            ax=ax, weight="bold", ha="left", size = 8)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))
    ax.set_ylim(0)

    # ---
    print(
        f"Annual inflation came in at {Y_end:.2%} during {X_max.strftime('%b-%Y')}")

    return f


# --- Core vs. non-core inflation

def mex_annual_core_vs_noncore(f, from_d="2000-01-01"):
    '''
    This function plots Mexican core vs. non-core inflation.

    Args:
        (from_d): From when do we want this information from.
        (f): the matplotlib figure.
    '''

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

    cpi_data_underlying["yoy_under"] = cpi_data_underlying["cpi_under"] / cpi_data_underlying["lag_under"] - 1

    cpi_data_underlying = cpi_data_underlying[["date", "yoy_under", "5993e5e787e4259"]]

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


    ax = f.add_subplot(111)

    index_ticks = np.arange(plot_df["date"].shape[0])
    width = 0.25

    ax.bar(index_ticks, plot_df["yoy_under"], width=width, zorder=3, label="Core inflation")
    ax.bar(index_ticks + width, plot_df["yoy_non_under"], width=width, zorder=3, label="Non-core inflation")

    ax.set_xticks(index_ticks + width / 2)
    ax.set_xticklabels(plot_df["date"].dt.strftime("%b-%y"), rotation=90)

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)


    return f

# --- Core inflation components

def mex_annual_core_components(f, from_d="2019-01-01"):
    '''
    This function plots Mexican core vs. non-core inflation.

    Args:
        (from_d): From when do we want this information from.
        (f): the matplotlib figure.
    '''

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

    ax = f.add_subplot(111)

    sub_indices = list(sort_index["primary"].unique())

    for index, sub_index in enumerate(sub_indices):
        plot_data_aux = yoy_data[yoy_data["primary"] == sub_index].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["yoy_change"],
                marker="o", markevery=[-1], color=cmap(index), mec="white", ms=5)
        Y_end = plot_data_aux["yoy_change"].iloc[-1]

        if sub_index == "Education":
            Y_end = Y_end + 0.005
        ax_text(x=X_max + datetime.timedelta(15), y= Y_end + 0.001,
                s=f"<{sub_index}>",
                highlight_textprops=[{"color": cmap(index)}],
                                    ax=ax, weight="bold", ha="left", size=8)

    ax.set_xlim(X_min, X_max + relativedelta(months=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))
    if yoy_data["yoy_change"].min() < 0:
        ax.hlines(0, X_min, X_max, ls = "--", color = "#4E616C", lw = 0.75)
    else:
        ax.set_ylim(0)

    return f


# --- Non-Core inflation components

def mex_annual_noncore_components(f, from_d="2019-01-01"):
    '''
    This function plots Mexican core vs. non-core inflation.

    Args:
        (from_d): From when do we want this information from.
        (f): the matplotlib figure.
    '''

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
        "GnBu_r", sort_index[["primary"]].drop_duplicates().shape[0] + 4)

    yoy_data = core_data.dropna()

    X_min = yoy_data["date"].min()
    X_max = yoy_data["date"].max()

    ax = f.add_subplot(111)

    sub_indices = list(sort_index["primary"].unique())

    for index, sub_index in enumerate(sub_indices):
        plot_data_aux = yoy_data[yoy_data["primary"] == sub_index].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["yoy_change"],
                marker="o", markevery=[-1], color=cmap(index), mec="white", ms=5)
        Y_end = plot_data_aux["yoy_change"].iloc[-1]
        ax_text(x=X_max + datetime.timedelta(15), y= Y_end + 0.001,
                s=f"<{sub_index}>",
                highlight_textprops=[{"color": cmap(index)}],
                                    ax=ax, weight="bold", ha="left", size=8)

    ax.set_xlim(X_min, X_max + relativedelta(months=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))
    if yoy_data["yoy_change"].min() < 0:
        ax.hlines(0, X_min, X_max, ls = "--", color = "#4E616C", lw = 0.75)
    else:
        ax.set_ylim(0)

    return f


# --- Mex top 20 core products

def mex_core_top20_products(f):
    '''
    This function plots a table of the top 20 products with the
    most contirbution to core inflation.

    Args:
        (f): the matplotlib figure.
    '''
    data = map_tukan_inegi_products("2019-01-01")

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
    core_data = core_data.sort_values(by = "yoy_contribution", ascending=True).tail(20)

    # True YoY and MoM Change
    core_data.loc[:,"yoy_change"] = core_data["yoy_change"]/core_data["12_m_lag"]
    core_data.loc[:,"mom_change"] = core_data["mom_change"]/core_data["1_m_lag"]

    top_10 = list(core_data["product__ref"])

    core_data.reset_index(drop = True, inplace = True)

    core_data = core_data[["product", "primary", "yoy_change", "mom_change", "yoy_contribution"]]
    core_data.replace({"product":{
        "Diners, inns, torterías and taquerías": "Diners, inns & others", 
        "Passenger air transportation": "Air transportation",
        "Other cultural services, entertainment services and sporting events": "Other entertainment serv."},
        "primary":{
            "Food, beverages & tobacco": "Food and beverages"
        }}, inplace = True)
   

    ax = f.add_subplot(111)
    rows = 20
    cols = 5

    # create a coordinate system based on the number of rows/columns

    # adding a bit of padding on bottom (-1), top (1), right (0.5)

    ax.set_ylim(-1, rows + 1)
    ax.set_xlim(0, cols + .5)

    for row in range(rows):
        # extract the row data from the list

        d = core_data.iloc[row,:]

        ax.text(x=.25, y=row, s=d["product"], va='center', ha='left', size = 8)
        ax.text(x=1.45, y=row, s=d['primary'], va='center', ha='left', size = 8)
        ax.text(x=3, y=row, s=f"{d['yoy_change']:.1%}", va='center', weight= "bold", ha='right', size = 8)
        ax.text(x=4, y=row, s=f"{d['mom_change']:.1%}", va='center', ha='right', size = 8)
        ax.text(x=5, y=row, s=f"{d['yoy_contribution']:.1%}", va='center', ha='right', size = 8)

    ax.text(.25, rows - .15, 'Product', weight='bold', ha='left', size = 10)
    ax.text(1.45, rows - .15, 'Group', weight='bold', ha='left', size = 10)
    ax.text(3, rows - .15, 'YoY %', weight='bold', ha='right', size = 10)
    ax.text(4, rows - .15, 'MoM %', weight='bold', ha='right', size = 10)
    ax.text(5, rows - .25, 'Cont. to\nYoY %', weight='bold', ha='right', size = 10)

    for row in range(rows):
        ax.plot(
            [0, cols + 1],
            [row -.5, row - .5],
            ls=':',
            lw='.5',
            c='#4E616C'
        )

    ax.plot([0, cols + 1], [rows - .5, rows - .5], lw='.75', c='#7D94A1')
    ax.plot([0, cols + 1], [0 - .5, 0 - .5], lw='.75', c='#7D94A1')

    rect = patches.Rectangle(
        (2.5, -.5),  # bottom left starting position (x,y)
        .65,  # width
        rows,  # height
        ec='none',
        fc='#2384ba',
        alpha=.2,
        zorder=-1
    )
    ax.add_patch(rect)
    ax.axis('off')

    return f


# --- Mex top 20 non-core products

def mex_noncore_top20_products(f):
    '''
    This function plots a table of the top 20 products with the
    most contirbution to non-core inflation.

    Args:
        (f): the matplotlib figure.
    '''
    data = map_tukan_inegi_products("2019-01-01")

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
    core_data = core_data.sort_values(by = "yoy_contribution", ascending=True).tail(20)

    # True YoY and MoM Change
    core_data.loc[:,"yoy_change"] = core_data["yoy_change"]/core_data["12_m_lag"]
    core_data.loc[:,"mom_change"] = core_data["mom_change"]/core_data["1_m_lag"]

    top_10 = list(core_data["product__ref"])

    core_data.reset_index(drop = True, inplace = True)

    core_data = core_data[["product", "primary", "yoy_change", "mom_change", "yoy_contribution"]]
    core_data.replace({"product":{
        "Electric power transmission services": "Electric power trans.", 
        "Eggs and egg substitutes": "Eggs", 
        "Liquefied natural gas LNG":"Liq. natural gas"}}, inplace = True)

    ax = f.add_subplot(111)
    rows = 20
    cols = 5

    # create a coordinate system based on the number of rows/columns

    # adding a bit of padding on bottom (-1), top (1), right (0.5)

    ax.set_ylim(-1, rows + 1)
    ax.set_xlim(0, cols + .5)

    for row in range(rows):
        # extract the row data from the list

        d = core_data.iloc[row,:]

        ax.text(x=.25, y=row, s=d["product"], va='center', ha='left', size = 8)
        ax.text(x=1.45, y=row, s=d['primary'], va='center', ha='left', size = 8)
        ax.text(x=3, y=row, s=f"{d['yoy_change']:.1%}", va='center', weight= "bold", ha='right', size = 8)
        ax.text(x=4, y=row, s=f"{d['mom_change']:.1%}", va='center', ha='right', size = 8)
        ax.text(x=5, y=row, s=f"{d['yoy_contribution']:.1%}", va='center', ha='right', size = 8)

    ax.text(.25, rows - .15, 'Product', weight='bold', ha='left', size = 10)
    ax.text(1.45, rows - .15, 'Group', weight='bold', ha='left', size = 10)
    ax.text(3, rows - .15, 'YoY %', weight='bold', ha='right', size = 10)
    ax.text(4, rows - .15, 'MoM %', weight='bold', ha='right', size = 10)
    ax.text(5, rows - .25, 'Cont. to\nYoY %', weight='bold', ha='right', size = 10)

    for row in range(rows):
        ax.plot(
            [0, cols + 1],
            [row -.5, row - .5],
            ls=':',
            lw='.5',
            c='#4E616C'
        )

    ax.plot([0, cols + 1], [rows - .5, rows - .5], lw='.75', c='#7D94A1')
    ax.plot([0, cols + 1], [0 - .5, 0 - .5], lw='.75', c='#7D94A1')

    rect = patches.Rectangle(
        (2.5, -.5),  # bottom left starting position (x,y)
        .65,  # width
        rows,  # height
        ec='none',
        fc='#2384ba',
        alpha=.2,
        zorder=-1
    )
    ax.add_patch(rect)
    ax.axis('off')

    return f


# --- Mexican Inflation by Economic Activity

def mex_activity_inflation(f, from_d = "2012-01-01"):
    '''
    This function plots inflation by economic activity.

    Args:
        (f): the matplotlib figure.
    '''

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

    ax = f.add_subplot(111)


    for index, sub_index in enumerate(economic_activities):
        plot_data_aux = data[data["economic_activity"] == sub_index].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["c572db59b8cd109"],
                marker="o", markevery=[-1], color=cmap(index), mec="white", ms=5)
        Y_end = plot_data_aux["c572db59b8cd109"].iloc[-1]

        ax_text(x=X_max + datetime.timedelta(20), y = Y_end + 0.006,
                s=f"<{sub_index}>",
                highlight_textprops=[{"color": cmap(index)}],
                                    ax=ax, weight="bold", ha="left", size=8)

    ax.set_xlim(X_min, X_max + relativedelta(months=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))

    if data["c572db59b8cd109"].min() < 0:
        ax.hlines(0, X_min, X_max, ls = "--", color = "#4E616C", lw = 0.75)
    else:
        ax.set_ylim(0)

    return f


# --- Mexican Inflation by Primary Economic Activity

def mex_primary_activity_inflation(f, from_d = "2012-01-01"):
    '''
    This function plots inflation by primary economic sectors.

    Args:
        (f): the matplotlib figure.
    '''

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
    sort_activities = data[data["date"] == data["date"].max()].sort_values(by = "c572db59b8cd109", ascending = False)
    activities = list(sort_activities['economic_activity'].unique())

    cmap = mpl.cm.get_cmap(
        "GnBu_r", 5)

    X_min = data["date"].min()
    X_max = data["date"].max()

    ax = f.add_subplot(111)


    for index, activity in enumerate(activities):
        plot_data_aux = data[data["economic_activity"] == activity].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["c572db59b8cd109"], marker = "o", markevery = [-1], color = cmap(index), mec = "white", ms = 5)
        Y_end = plot_data_aux["c572db59b8cd109"].iloc[-1]
        if activity == "Crop production":
            Y_end = Y_end
        elif activity == "Fishing & hunting":
            Y_end = Y_end + 0.05
        ax_text(x=X_max + datetime.timedelta(20), y = Y_end,
                s=f"<{activity}>",
                highlight_textprops=[{"color": cmap(index)}],
                                    ax=ax, weight="bold", ha="left", size=8)

    ax.set_xlim(X_min, X_max + relativedelta(months=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))

    if data["c572db59b8cd109"].min() < 0:
        ax.hlines(0, X_min, X_max, ls = "--", color = "#4E616C", lw = 0.75)
    else:
        ax.set_ylim(0)

    return f


# --- Mexican Inflation by Secondary Economic Activity

def mex_secondary_activity_inflation(f, from_d = "2012-01-01"):
    '''
    This function plots inflation by secondary economic sectors.

    Args:
        (f): the matplotlib figure.
    '''

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
    sort_activities = data[data["date"] == data["date"].max()].sort_values(by = "c572db59b8cd109", ascending = False)
    activities = list(sort_activities['economic_activity'].unique())

    cmap = mpl.cm.get_cmap(
        "GnBu_r", 5)

    X_min = data["date"].min()
    X_max = data["date"].max()

    ax = f.add_subplot(111)


    for index, activity in enumerate(activities):
        plot_data_aux = data[data["economic_activity"] == activity].copy()
        ax.plot(plot_data_aux["date"], plot_data_aux["c572db59b8cd109"], marker = "o", markevery = [-1], color = cmap(index), mec = "white", ms = 5)
        Y_end = plot_data_aux["c572db59b8cd109"].iloc[-1]
        ax_text(x=X_max + datetime.timedelta(20), y = Y_end,
                s=f"<{activity}>",
                highlight_textprops=[{"color": cmap(index)}],
                                    ax=ax, weight="bold", ha="left", size=8)

    ax.set_xlim(X_min, X_max + relativedelta(months=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))

    if data["c572db59b8cd109"].min() < 0:
        ax.hlines(0, X_min, X_max, ls = "--", color = "#4E616C", lw = 0.75)
    else:
        ax.set_ylim(0)

    return f


# --- Mexican Inflation by Tertiary Economic Activity

def mex_tertiary_activity_inflation(f, from_d = "2012-01-01"):
    '''
    This function plots inflation by tertiary economic sectors.

    Args:
        (f): the matplotlib figure.
    '''

    payload = {
    "type": "data_table",
    "operation": "sum",
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
    "from": "2019-01-01"
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
    
    data.loc[:, "12_m_lag"] = data.groupby(["economic_activity"])["c572db59b8cd109"].shift(12)
    data.loc[:, "1_m_lag"] = data.groupby(["economic_activity"])["c572db59b8cd109"].shift(1)

    # True YoY and MoM Change
    data.loc[:,"yoy_change"] = data["c572db59b8cd109"]/data["12_m_lag"] - 1
    data.loc[:,"mom_change"] = data["c572db59b8cd109"]/data["1_m_lag"] - 1

    data = data[data["date"] == data["date"].max()].copy()
    data = data.sort_values(by = "yoy_change", ascending=True)

    # ----
    # The chart
    # ----
    ax = f.add_subplot(111)
    rows = 12
    cols = 4

    # create a coordinate system based on the number of rows/columns

    # adding a bit of padding on bottom (-1), top (1), right (0.5)

    ax.set_ylim(-1, rows + 1)
    ax.set_xlim(0, cols + .5)

    for row in range(rows):
        # extract the row data from the list

        d = data.iloc[row,:]

        ax.text(x=.25, y=row, s=d["economic_activity"], va='center', ha='left', size = 8)
        # ax.text(x=1.45, y=row, s=d['primary'], va='center', ha='left', size = 8)
        ax.text(x=3, y=row, s=f"{d['yoy_change']:.1%}", va='center', weight= "bold", ha='right', size = 8)
        ax.text(x=4, y=row, s=f"{d['mom_change']:.1%}", va='center', ha='right', size = 8)


    ax.text(.25, rows - .25, 'Economic Activity', weight='bold', ha='left', size = 10)
    # ax.text(1.45, rows - .15, 'Group', weight='bold', ha='left', size = 10)
    ax.text(3, rows - .15, 'YoY %', weight='bold', ha='right', size = 10)
    ax.text(4, rows - .15, 'MoM %', weight='bold', ha='right', size = 10)

    for row in range(rows):
        ax.plot(
            [0, cols + 1],
            [row -.5, row - .5],
            ls=':',
            lw='.5',
            c='#4E616C'
        )

    ax.plot([0, cols + 1], [rows - .5, rows - .5], lw='.75', c='#7D94A1')
    ax.plot([0, cols + 1], [0 - .5, 0 - .5], lw='.75', c='#7D94A1')

    rect = patches.Rectangle(
        (2.5, -.5),  # bottom left starting position (x,y)
        .65,  # width
        rows,  # height
        ec='none',
        fc='#2384ba',
        alpha=.2,
        zorder=-1
    )
    ax.add_patch(rect)
    ax.axis('off')

    return f
