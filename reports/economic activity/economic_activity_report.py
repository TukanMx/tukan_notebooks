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

#%%
def get_igae_data(from_d="2000-01-15"):
    payload_yoy = {
        "type": "data_table",
        "operation": "yoy_growth_rel", 
        "language": "es",
        "group_by": [
            "adjustment_type",
            "economic_activity"
        ],
        "categories": {
            "adjustment_type": [
                "61060325ab095ed"
            ],
            "economic_activity": [
                "dfeefc621d16d0c",
                "7460634ca523beb",
                "761bc00426e1c48",
                "8fd5b02b9f891fb"
            ]
        },
        "request": [
            {
                "table": "mex_inegi_igae",
                "variables": [
                    "fa581e55c3b52cb"
                ]
            }
        ],
        "from": from_d
    }

    payload_mom = {
        "type": "data_table",
        "operation": "last_growth_rel", 
        "language": "es",
        "group_by": [
            "adjustment_type",
            "economic_activity"
        ],
        "categories": {
            "adjustment_type": [
                "61060325ab095ed"
            ],
            "economic_activity": [
                "dfeefc621d16d0c",
                "7460634ca523beb",
                "761bc00426e1c48",
                "8fd5b02b9f891fb"
            ]
        },
        "request": [
            {
                "table": "mex_inegi_igae",
                "variables": [
                    "fa581e55c3b52cb"
                ]
            }
        ],
        "from": from_d
    }

    yoy_response = get_tukan_api_request(payload_yoy)
    yoy_data = yoy_response["data"]
    yoy_data.rename(columns={'fa581e55c3b52cb':'yoy_igae'}, inplace=True)
    mom_response = get_tukan_api_request(payload_mom)
    mom_data = mom_response["data"]
    mom_data.rename(columns={'fa581e55c3b52cb':'mom_igae'}, inplace=True)
    igae_data = pd.merge(yoy_data,mom_data, how='left')
    return igae_data

def get_activities_data(from_d="2000-01-15"):
    payload_yoy = {
        "type": "data_table",
        "operation": "yoy_growth_rel", 
        "language": "es",
        "group_by": [
            "adjustment_type",
            "economic_activity"
        ],
        "categories": {
            "adjustment_type": [
                "61060325ab095ed"
            ],
            "economic_activity": "all"
        },
        "request": [
            {
                "table": "mex_inegi_igae",
                "variables": [
                    "fa581e55c3b52cb"
                ]
            }
        ],
        "from": from_d
    }

    payload_mom = {
        "type": "data_table",
        "operation": "last_growth_rel", 
        "language": "es",
        "group_by": [
            "adjustment_type",
            "economic_activity"
        ],
        "categories": {
            "adjustment_type": [
                "61060325ab095ed"
            ],
            "economic_activity": "all"
        },
        "request": [
            {
                "table": "mex_inegi_igae",
                "variables": [
                    "fa581e55c3b52cb"
                ]
            }
        ],
        "from": from_d
    }

    yoy_response = get_tukan_api_request(payload_yoy)
    yoy_data = yoy_response["data"]
    yoy_data.rename(columns={'fa581e55c3b52cb':'yoy_igae'}, inplace=True)
    mom_response = get_tukan_api_request(payload_mom)
    mom_data = mom_response["data"]
    mom_data.rename(columns={'fa581e55c3b52cb':'mom_igae'}, inplace=True)
    activities_data = pd.merge(yoy_data,mom_data, how='left')
    return activities_data

# %%
