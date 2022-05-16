'''
Economic Activity Monitor Functions.
----------------------------------------------
Main Author: Miguel Angel Dávila

This script contains function that generate our TUKAN 
Mexican Economic Activity Monitor report.
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

'''

Section 1. Construction Prices

'''

def mex_construction_price_line(f, from_d = "2000-01-01"):
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