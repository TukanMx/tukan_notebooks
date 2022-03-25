# %%
# ----------------------------------
# We load fonts and stylesheet.
# ----------------------------------
import datetime
from logging import exception
from turtle import width
import matplotlib.dates as mdates
# from matplotlib.lines import _LineStyle
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
# ----------------------------------
# Load  data
# ----------------------------------
def get_igae_data(from_d="2000-01-01", language="en"):
    payload_yoy = {
        "type": "data_table",
        "operation": "yoy_growth_rel", 
        "language": language,
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
        "language": language,
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

def get_activities_data(from_d="2000-01-01", language="en"):
    payload_yoy = {
        "type": "data_table",
        "operation": "yoy_growth_rel", 
        "language": language,
        "group_by": [
            "adjustment_type",
            "economic_activity"
        ],
        "categories": {
            "adjustment_type": [
                "61060325ab095ed"
            ],
            "economic_activity": [
            "6b36ca46b6cfd91",
            "457155464609a2f",
            "36348912d8470dd",
            "faa2a8d0af8a72c",
            "be676b5dd921cb7",
            "f5adaadda584ca7",
            "e426cc87d0540ab",
            "afcc312ccddfcc1",
            "990b94ebe38c9ca",
            "d35f5b82779e7d5",
            "3726993cc9fecab",
            "bbb49ae78601ab9",
            "4bc9836c2d7e60a",
            "fcb303b72a98f6c",
            "d05c3b2b73d75fc",
            "44d246411040129",
            "a07267f78158c2c",
            "feb7bb4445c808d",
            "169c33ccdd66d77",
            "23cf92d98dd7c11"
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
        "language": language,
        "group_by": [
            "adjustment_type",
            "economic_activity"
        ],
        "categories": {
            "adjustment_type": [
                "61060325ab095ed"
            ],
            "economic_activity": [
            "6b36ca46b6cfd91",
            "457155464609a2f",
            "36348912d8470dd",
            "faa2a8d0af8a72c",
            "be676b5dd921cb7",
            "f5adaadda584ca7",
            "e426cc87d0540ab",
            "afcc312ccddfcc1",
            "990b94ebe38c9ca",
            "d35f5b82779e7d5",
            "3726993cc9fecab",
            "bbb49ae78601ab9",
            "4bc9836c2d7e60a",
            "fcb303b72a98f6c",
            "d05c3b2b73d75fc",
            "44d246411040129",
            "a07267f78158c2c",
            "feb7bb4445c808d",
            "169c33ccdd66d77",
            "23cf92d98dd7c11"
        ]        },
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

def get_enec_data(from_d="2000-01-01", language="en"):
    payload = {
    "type": "data_table",
    "operation": "sum",
    "language": language,
    "group_by": [
        "economic_activity",
        "geography"
    ],
    "categories": {
        "economic_activity": [
            "457155464609a2f", # General level
            "14edc470d8f3e2f",
            "4382bc56abc3b3b",
            "222bc7bc27c6906"
        ],
        "geography": [
            "b815762a2c6a283"
        ]
    },
    "request": [
        {
            "table": "mex_inegi_enec_main",
            "variables": [
                "e721ea412d5cbc1"
            ]
        }
    ],
    "from": from_d
    }

    response = get_tukan_api_request(payload)
    data = response["data"]
    return data

def get_labour_enec_data(from_d="2000-01-01", language="en"):
    payload = {
    "type": "data_table",
    "operation": "sum",
    "language": language,
    "group_by": [
        "economic_activity"
    ],
    "categories": {
        "economic_activity": [
            "457155464609a2f"
        ]
    },
    "request": [
        {
            "table": "mex_inegi_enec_main",
            "variables": [
                "14066e939239b6e",
                "f12649252c82ff0",
                "74e4b5f7542fc3f",
                "62a2aecf0193aac"
            ]
        }
    ],
    "from": from_d
}

    response = get_tukan_api_request(payload)
    data = response["data"]
    return data

def get_inpc_data(from_d="2000-01-01", language="en"):
    payload = {
    "type": "data_table",
    "operation": "sum",
    "language": language,
    "group_by": [
        "economic_activity"
    ],
    "categories": {
        "economic_activity": [
            "dfeefc621d16d0c"
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
    data.rename(columns={'c572db59b8cd109':'inpc'}, inplace=True)
    return(data)

def get_emim_data(from_d="2013-01-01", language="en", activities=True):
    if activities==True:
        payload = {
        "type": "data_table",
        "operation": "last_growth_rel",
        "language": language,
        "group_by": [
            "economic_activity",
            "geography"
        ],
        "categories": {
            "economic_activity": "all",
            "geography": [
                "b815762a2c6a283"
            ]
        },
        "request": [
            {
                "table": "mex_inegi_emim_variables",
                "variables": [
                    "86336e63b802373"
                ]
            }
        ],
        "from": from_d
        }
    else:
        payload = {
        "type": "data_table",
        "operation": "sum",
        "language": language,
        "group_by": [
            "economic_activity",
            "geography"
        ],
        "categories": {
            "economic_activity": ["dfeefc621d16d0c"],
            "geography": [
                "b815762a2c6a283"
            ]
        },
        "request": [
            {
                "table": "mex_inegi_emim_variables",
                "variables": [
                    "86336e63b802373"
                ]
            }
        ],
        "from": from_d
        }
    try:
        response = get_tukan_api_request(payload)
        data = response["data"]
    except Exception as e:
        print(e)
        from_d = '2021-01-01'
        payload.update({"from":from_d})
        response = get_tukan_api_request(payload)
        data = response["data"]
        print(f"Getting data from {from_d}")
   
    data.rename(columns={'86336e63b802373':'production_value'}, inplace=True)
    data['production_value'] = data['production_value']/1000000    

    return(data)

def get_ems_data(from_d="2008-01-01", language="en", operation ="yoy_growth_rel"):
    payload = {
    "type": "data_table",
    "operation": operation,
    "language": language,
    "group_by": [
        "economic_activity"
    ],
    "categories": {
        "economic_activity": "all"
    },
    "request": [
        {
            "table": "mex_inegi_ems",
            "variables": [
                "560fe2a60684221"
            ]
        }
    ],
    "from": from_d
    }
    response = get_tukan_api_request(payload)
    data = response["data"]
    data.rename(columns={'560fe2a60684221':'income'}, inplace=True)
    
    second_level_acts = ["1d0185629b65ee3","29daaeb36fea1ab","36348912d8470dd","3726993cc9fecab","44d246411040129","457155464609a2f","4adeefc3d481a07","4bc9836c2d7e60a","6b36ca46b6cfd91","990b94ebe38c9ca","a07267f78158c2c","aa2e62d86204fee","afaceb85ed568ca","afcc312ccddfcc1","bbb49ae78601ab9","be676b5dd921cb7","c2c32762f1f055c","d05c3b2b73d75fc","d35f5b82779e7d5","e426cc87d0540ab","f5adaadda584ca7","faa2a8d0af8a72c","fcb303b72a98f6c","feb7bb4445c808d"]
    data = data[data["economic_activity__ref"].isin(second_level_acts)].reset_index(drop=True)
    
    return(data)

def get_emoe_data(from_d="2008-01-01", language="en"):
    payload={
    "type": "data_table",
    "operation": "sum",
    "language": language,
    "group_by": [
        "economic_activity"
    ],
    "categories": {
        "economic_activity": [
            "457155464609a2f",
            "faa2a8d0af8a72c",
            "1d0185629b65ee3"
        ]
    },
    "request": [
        {
            "table": "mex_inegi_emoe",
            "variables": [
                "1b0f1ce9956876e"
            ]
        }
    ],
    "from": from_d
    }
    response = get_tukan_api_request(payload)
    data = response["data"]
    data.rename(columns={'1b0f1ce9956876e':'ice'}, inplace=True)
    return(data)

def get_enco_data(from_d="2008-01-01", language="en"):
    payload={
    "type": "data_table",
    "operation": "sum",
    "language": language,
    "group_by": [
        ""
    ],
    "categories": {},
    "request": [
        {
            "table": "mex_inegi_enco_index",
            "variables": [
                "f789b42197d3c85"
            ]
        }
    ],
    "from": from_d
}
    response = get_tukan_api_request(payload)
    data = response["data"]
    data.rename(columns={'f789b42197d3c85':'icc'}, inplace=True)
    return(data)

def get_inpp_construction_data(from_d="2000-01-01", language="en"):
    payload= {
    "type": "data_table",
    "operation": "sum",
    "language": language,
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
            "table": "mex_inegi_inpp_construction_product",
            "variables": [
                "08c24466faffc9c"
            ]
        }
    ],
    "from": from_d
    }
    response = get_tukan_api_request(payload)
    data = response["data"]
    data.rename(columns={'08c24466faffc9c':'inpp'}, inplace=True)
    data.drop(columns=['product__ref','product'], inplace=True)
    return(data)

#%%
# ----------------------------------
# Deflate  functions
# ----------------------------------
# Variables
# language = "en"
# from_d = "2000-01-01"
# base_date = "2018-11-01"
# deflate_date = "2021-02-01"

def benchmark_deflate_inpc(df, id="column_name", base_date = "2018-11-01"):
    deflate = get_inpc_data(from_d = "2000-01-01", language = "en")
    def_base = deflate[deflate['date']==base_date]
    def_val = def_base.iloc[-1][3]
    deflate['def_val'] = deflate['inpc'] / def_val
    deflate = deflate[['date','def_val']].iloc[:]
    deflate.dropna(inplace=True)
    deflate.reset_index(inplace=True,drop=True)
    final_df = df.merge(deflate, how='left', on='date')
    newcol_name = 'deflated_'+str(id)
    final_df[newcol_name] = final_df[id] / final_df['def_val']
    final_df.drop(columns='def_val',inplace=True)
    # delete var def_val 
    return(final_df)

def deflate_inpp_construction(df, id="column_names", base_date="2013-01-01"):
    deflate = get_inpp_construction_data(from_d="2000-01-01", language="en")
    def_base = deflate[deflate['date']==base_date]
    def_val = def_base.iloc[-1][1]
    deflate['def_val'] = deflate['inpp'] / def_val
    deflate = deflate[['date','def_val']].iloc[:]
    deflate.dropna(inplace=True)
    deflate.reset_index(inplace=True,drop=True)
    final_df = df.merge(deflate, how='left', on='date')
    for column in id:
        newcol_name = 'deflated_'+str(column)
        final_df[newcol_name] = final_df[column] / final_df['def_val']
    
    final_df.drop(columns='def_val',inplace=True)
    
    return(final_df)

    

##########################################    
## plot sample ## 
# cmap = mpl.cm.get_cmap("GnBu_r", 5)
# fig = plt.figure(figsize=(8, 4), dpi=200)
# ax = plt.subplot(111)
# ax.plot(aux_df["date"], aux_df["production_value"],marker="o", ms=6, mec="white", markevery=[-1], color=cmap(0))
# ax.plot(aux_df["date"], aux_df["deflated_production_value"],marker="x", ms=6, mec="white", markevery=[-1], color=cmap(2))
# %%
# ------------------------------------------------------------------
#
#                       CHARTS FOR REPORT
#
# ------------------------------------------------------------------

# ------------------------------------------------------------------
#
# CHART 1: YOY CHANGE IN IGAE - LINE
#
# ------------------------------------------------------------------

def plot_chart_1(from_d="2000-01-01", language="en"):
    
    plot_data = get_igae_data(from_d, language)
    plot_data = plot_data[plot_data['economic_activity__ref']=='dfeefc621d16d0c']

    cmap = mpl.cm.get_cmap("GnBu_r", 5)
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)

    Y_mom = plot_data["mom_igae"].iloc[-1]
    Y_mom_prevmonth = plot_data["mom_igae"].iloc[-2]
    Y_end_prevmonth = plot_data["yoy_igae"].iloc[-2]
    Y_end = plot_data["yoy_igae"].iloc[-1]
    X_min = plot_data["date"].iloc[0]
    X_max = plot_data["date"].iloc[-1]

    ax.plot(plot_data["date"], plot_data["yoy_igae"],
            marker="o", ms=6, mec="white", markevery=[-1], color=cmap(0))

    ax.hlines(0, X_min, X_max, ls = "-", color = "black", lw = 0.75)


    ax_text(x=X_max + relativedelta(months=3), y=Y_end,
            s=f"<{Y_end:.1%}>",
            highlight_textprops=[{"color": cmap(0)}],
            ax=ax, weight="bold", font="Dosis", ha="left", size=9)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))

    # fig.text(
    #     0.1,
    #     1,
    #     "Indicador General de la Actividad Económica (IGAE) - YoY Change",
    #     size=14,
    #     weight = "bold"
    # )
    if language == "en":
        plt.savefig(
        "plots/yoy_igae_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    else:
        plt.savefig(
        "plots/es_yoy_igae_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )  

    # ---
    if language == "en":
        print(f"During {X_max.strftime('%b-%Y')} the annual change came in at {Y_end:.1%} and the monthly change at {Y_mom:.1%}; The previus YoY rate was {Y_end_prevmonth:.1%} while MoM was {Y_mom_prevmonth:.1%}.")
    else:
        print(f"Durante {X_max.strftime('%b-%Y')}, la variación anual fue de {Y_end:.1%} y la mensual de {Y_mom:.1%}; la variación anual anterior fue de {Y_end_prevmonth:.1%}, mientras que la mensual fue de {Y_mom_prevmonth:.1%}.")

# ------------------------------------------------------------------
#
# CHART 2: YOY CHANGE IN IGAE SECTORS - BARS
#
# ------------------------------------------------------------------

def plot_chart_2(from_d="2020-11-01", language="en"):
    
    data = get_igae_data(from_d, language)
    plot_data = data[data['economic_activity__ref']!='dfeefc621d16d0c'].copy()
    plot_data.drop(columns=['adjustment_type__ref', 'adjustment_type',
        'economic_activity__ref','mom_igae'], inplace=True)
    plot_data = plot_data.pivot(index='date',columns='economic_activity', values='yoy_igae')
    plot_data.reset_index(inplace=True)
    
    data = data[data['economic_activity__ref']!='dfeefc621d16d0c'].copy()
    data.drop(columns=['adjustment_type__ref', 'adjustment_type',
    'economic_activity__ref','yoy_igae'], inplace=True)
    data = data.pivot(index='date',columns='economic_activity', values='mom_igae')
    data.reset_index(inplace=True)

    # Plot
    cmap = mpl.cm.get_cmap("GnBu_r", 5)
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)

    
    index_ticks = np.arange(plot_data['date'].shape[0])
    width = 0.3
    
    if language =="en":
        primary = "Primary activities"
        prim = "Primary"
        secondary = "Secondary activities"
        seco = "Secondary"
        tertiary = "Tertiary activities"
        tert = "Tertiary"
    else:
        primary = "Actividades primarias"
        prim = "Primarias"
        secondary = "Actividades secundarias"
        seco = "Secundarias"
        tertiary = "Actividades terciarias"
        tert = "Terciarias"

    pri_yoy = plot_data[primary].iloc[-1] 
    sec_yoy = plot_data[secondary].iloc[-1] 
    ter_yoy = plot_data[tertiary].iloc[-1]
    pri_mom = data[primary].iloc[-1]
    pri_ann = np.power((1+pri_mom),12)-1 
    sec_mom = data[secondary].iloc[-1]
    sec_ann = np.power((1+sec_mom),12)-1 
    ter_mom = data[tertiary].iloc[-1]
    ter_ann = np.power((1+ter_mom),12)-1 
    max_date = plot_data["date"].max()

    
    X_min = index_ticks.min()
    X_max = index_ticks.max()

    ax.bar(index_ticks, plot_data[primary], color=cmap(
        0), width=width, zorder=3, label=prim)
    ax.bar(index_ticks + width, plot_data[secondary],
           color=cmap(1), width=width, zorder=3, label=seco)
    ax.bar(index_ticks + 2*width, plot_data[tertiary],
        color=cmap(2), width=width, zorder=3, label=tert)
    
    ax.hlines(0, X_min-.13, X_max+.4, ls = "-", color = "black", lw = 0.75, zorder=3)

    ax.set_xticks(index_ticks + width / 3,
                  labels=plot_data["date"].dt.strftime("%b-%y"), rotation=90)

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))
    # ax.set_ylim(0)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)

    # ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))

    # fig.text(
    #     0.1,
    #     1,
    #     "IGAE by Sector - YoY Change",
    #     size=14,
    #     weight = "bold"
    # )

    if language == "en":
        plt.savefig(
        "plots/yoy_sector_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    else:
        plt.savefig(
        "plots/es_yoy_sector_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    # # ---
    if language == "en":
        print(f"During {max_date.strftime('%b-%Y')} the annual change came in at {pri_yoy:.1%} for primary activities, at {sec_yoy:.1%} for secondary activities and at {ter_yoy:.1%} for tertiary activities.\n\nThe monthly change rates came in at {pri_mom:.1%} ({pri_ann:.1%} annualized) for primary activities, at {sec_mom:.1%} ({sec_ann:.1%} annualized) for secondary activities and at {ter_mom:.1%} ({ter_ann:.1%} annualized) for tertiary activities.")
    else:
        print(f"Durante {max_date.strftime('%b-%Y')} la variación anual de las acts. primarias fue de {pri_yoy:.1%}, la de las acts. secundarias de {sec_yoy:.1%}  y la de las acts. terciarias de {ter_yoy:.1%}.\n\nLa variación mensual fue de {pri_mom:.1%} ({pri_ann:.1%} tasa anualizada) para las acts. primarias, de {sec_mom:.1%} ({sec_ann:.1%} tasa anualizada) para las acts. secundarias y de {ter_mom:.1%} ({ter_ann:.1%} annualized) para las terciarias.")

# ------------------------------------------------------------------
#
# CHART 3: YOY CHANGE IN IGAE BY ECONOMIC ACTIVITY - LOLLIPOP
#
# ------------------------------------------------------------------

def plot_chart_3(from_d="2016-01-01", language="en"):
    plot_data = get_activities_data(from_d, language)
    plot_data = plot_data[plot_data['date'] == plot_data['date'].max()].sort_values(by="yoy_igae", ascending=True).reset_index(drop=True)
    if language == "en":
        plot_data = plot_data.replace({'Other services (except public administration)':'Other services','Governmental, legislative activities of law enforcement and international and extraterritorial bodies':'Governmental and legislative services','Administrative and support and waste management and remediation services':'Administrative and waste management services', 'Information':'Media'})
    else:
        plot_data = plot_data.replace({'Servicios de esparcimiento culturales y deportivos, y otros servicios recreativos':'Serv. de esparcimiento y recreativos','Generación, transmisión, distribución y comercialización de energía eléctrica, suministro de agua y de gas natural por ductos al consumidor final':'Energía, suministro de agua y gas natural','Servicios de alojamiento temporal y de preparación de alimentos y bebidas':'Hoteles y Restaurantes','Otros servicios excepto actividades gubernamentales':'Otros servicios','Agricultura, cría y explotación de animales, aprovechamiento forestal, pesca y caza':'Agricultura, ganadería, pesca y acts. forestales','Servicios de apoyo a los negocios y manejo de residuos, y servicios de remediación':'Serv. de apoyo a los negocios y manejo de residuos','Actividades legislativas, gubernamentales, de impartición de justicia y de organismos internacionales y extraterritoriales':'Acts. gubernamentales y legislativas','Servicios inmobiliarios y de alquiler de bienes muebles e intangibles':'Serv. inmobiliarios y alquiler de bienes','Servicios de salud y de asistencia social':'Serv. sociales y de salud','Servicios profesionales, científicos y técnicos':'Serv. profesionales, científicos y técnicos','Servicios educativos':'Serv. educativos','Servicios financieros y de seguros':'Serv. financieros y de seguros'})
    
    plot_data['p_ref'] =  range(plot_data.shape[0]) #plot_reference
    labels = plot_data['economic_activity'].tolist()

    fig = plt.figure(figsize=(10,6), dpi = 300)
    ax = plt.subplot(111)
    cmap = mpl.cm.get_cmap("GnBu_r", 5)

    # Change splines
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)

    # Data to plot
    for x in plot_data.index:
        if plot_data['yoy_igae'].loc[x]>=0:
            ax.hlines(y=plot_data['p_ref'].loc[x], xmin=0, xmax=plot_data['yoy_igae'].loc[x],lw=1, color='black' ,zorder=2)
        else:
            ax.hlines(y=plot_data['p_ref'].loc[x], xmin=plot_data['yoy_igae'].loc[x], xmax=0,lw=1, color='black' ,zorder=2)
    ax.scatter(plot_data['yoy_igae'],plot_data['p_ref'],s=50, edgecolors='black', lw=0.8, zorder=3, color=cmap(0))

    # Add gridlines and format to ticks 
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0%}'))
    ax.axvline(x=0,lw=1, color='black')
    ax.set_yticks(plot_data['p_ref'])
    ax.set_yticklabels(labels)

    # Axis ticks on right side
    ax.yaxis.tick_right()
    
    #
    acts = plot_data['economic_activity'].iloc[-3:].values.tolist()
    growth_yoy = plot_data['yoy_igae'].iloc[-3:].values.tolist()
    max_date = plot_data["date"].max()

    # fig.text(
    #     0.2,
    #     1,
    #     "IGAE by Economic Activity - Change",
    #     size=14,
    #     weight = "bold"
    # )

    if language == "en":
        plt.savefig(
        "plots/yoy_activities_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    else:
        plt.savefig(
        "plots/es_yoy_activities_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    # # ---
    if language == "en":
        print(f"During {max_date.strftime('%b-%Y')} the best perfoming economic activities were {acts[2]} ({growth_yoy[2]:.1%}), {acts[1]} ({growth_yoy[1]:.1%}) and {acts[0]} ({growth_yoy[0]:.1%}).")
    else :
        print(f"Durante {max_date.strftime('%b-%Y')} las actividades económicas con mejor desempeño fueron {acts[2]} ({growth_yoy[2]:.1%}), {acts[1]} ({growth_yoy[1]:.1%}) y {acts[0]} ({growth_yoy[0]:.1%}).")

# ------------------------------------------------------------------
#
# CHART 4: MOM CHANGE IN IGAE BY ECONOMIC ACTIVITY - LOLLIPOP
#
# ------------------------------------------------------------------

def plot_chart_4(from_d="2016-01-01", language="en"):
    plot_data = get_activities_data(from_d, language)
    plot_data = plot_data[plot_data['date'] == plot_data['date'].max()].sort_values(by="mom_igae", ascending=True).reset_index(drop=True)
    if language == "en":
        plot_data = plot_data.replace({'Other services (except public administration)':'Other services','Governmental, legislative activities of law enforcement and international and extraterritorial bodies':'Governmental and legislative services','Administrative and support and waste management and remediation services':'Administrative and waste management services', 'Information':'Media'})
    else:
        plot_data = plot_data.replace({'Servicios de esparcimiento culturales y deportivos, y otros servicios recreativos':'Serv. de esparcimiento y recreativos','Generación, transmisión, distribución y comercialización de energía eléctrica, suministro de agua y de gas natural por ductos al consumidor final':'Energía, suministro de agua y gas natural','Servicios de alojamiento temporal y de preparación de alimentos y bebidas':'Hoteles y Restaurantes','Otros servicios excepto actividades gubernamentales':'Otros servicios','Agricultura, cría y explotación de animales, aprovechamiento forestal, pesca y caza':'Agricultura, ganadería, pesca y acts. forestales','Servicios de apoyo a los negocios y manejo de residuos, y servicios de remediación':'Serv. de apoyo a los negocios y manejo de residuos','Actividades legislativas, gubernamentales, de impartición de justicia y de organismos internacionales y extraterritoriales':'Acts. gubernamentales y legislativas','Servicios inmobiliarios y de alquiler de bienes muebles e intangibles':'Serv. inmobiliarios y alquiler de bienes','Servicios de salud y de asistencia social':'Serv. sociales y de salud','Servicios profesionales, científicos y técnicos':'Serv. profesionales, científicos y técnicos','Servicios educativos':'Serv. educativos','Servicios financieros y de seguros':'Serv. financieros y de seguros'})
    
    plot_data['p_ref'] =  range(plot_data.shape[0]) #plot_reference
    labels = plot_data['economic_activity'].tolist()

    fig = plt.figure(figsize=(10,6), dpi = 300)
    ax = plt.subplot(111)
    cmap = mpl.cm.get_cmap("GnBu_r", 5)

    # Change splines
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)

    # Data to plot
    for x in plot_data.index:
        if plot_data['mom_igae'].loc[x]>=0:
            ax.hlines(y=plot_data['p_ref'].loc[x], xmin=0, xmax=plot_data['mom_igae'].loc[x],lw=1, color='black' ,zorder=2)
        else:
            ax.hlines(y=plot_data['p_ref'].loc[x], xmin=plot_data['mom_igae'].loc[x], xmax=0,lw=1, color='black' ,zorder=2)
    ax.scatter(plot_data['mom_igae'],plot_data['p_ref'],s=50, edgecolors='black', lw=0.8, zorder=3, color=cmap(2))

    # Add gridlines and format to ticks 
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0%}'))
    ax.axvline(x=0,lw=1, color='black')
    ax.set_yticks(plot_data['p_ref'])
    ax.set_yticklabels(labels)

    # Axis ticks on right side
    ax.yaxis.tick_right()
    
    #
    acts = plot_data['economic_activity'].iloc[-3:].values.tolist()
    growth_mom = plot_data['mom_igae'].iloc[-3:].values.tolist()
    max_date = plot_data["date"].max()

    # fig.text(
    #     0.2,
    #     1,
    #     "IGAE by Economic Activity - Change",
    #     size=14,
    #     weight = "bold"
    # )

    if language == "en":
        plt.savefig(
        "plots/mom_activities_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    else:
        plt.savefig(
        "plots/es_mom_activities_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    # # ---
    if language == "en":
        print(f"During {max_date.strftime('%b-%Y')} the best perfoming economic activities were {acts[2]} ({growth_mom[2]:.1%}), {acts[1]} ({growth_mom[1]:.1%}) and {acts[0]} ({growth_mom[0]:.1%}).")
    else :
        print(f"Durante {max_date.strftime('%b-%Y')} las actividades económicas con mejor desempeño fueron {acts[2]} ({growth_mom[2]:.1%}), {acts[1]} ({growth_mom[1]:.1%}) y {acts[0]} ({growth_mom[0]:.1%}).")

# ------------------------------------------------------------------
#
# CHART 5: CONSTRUCTION - BARS
#
# ------------------------------------------------------------------

def plot_chart_5(from_d="2013-01-01", language="en"):
    data = get_enec_data(from_d, language)
    data = data.rename(columns={"e721ea412d5cbc1":"production_value"})
    data['production_value'] = data['production_value'] / 1000000
    data.reset_index(inplace=True, drop=True)
    data = data.pivot(index='date', columns='economic_activity__ref')['production_value']
    data.reset_index(inplace=True)
    if language =='en':
        construction = 'Construction'
        civil = 'Civil engineering constructions'
        buildings = 'Buildings'
        specialty = 'Specialty manufacturing contractors'
        unit = 'Millions of MXN'
    else:
        construction = 'Construcción'
        civil = 'Obras de ingeniería civil'
        buildings = 'Edificación'
        specialty = 'Trabajos especializados'
        unit = 'Millones de MXN'
    data.rename(columns={'14edc470d8f3e2f':civil,'222bc7bc27c6906':specialty,'4382bc56abc3b3b':buildings,'457155464609a2f':construction}, inplace=True)
    plot_data = deflate_inpp_construction(data, id=[construction,civil,buildings,specialty], base_date="2013-01-01")
    
    mom_var = (plot_data[f"deflated_{construction}"].iloc[-1] / plot_data[f"deflated_{construction}"].iloc[-2])-1
    yoy_var = (plot_data[f"deflated_{construction}"].iloc[-1] / plot_data[f"deflated_{construction}"].iloc[-13])-1
    X_max = plot_data["date"].iloc[-1]

    
    ##### Plot
    cmap = mpl.cm.get_cmap("GnBu_r", 5)
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)

    width = 20      
    p1 = ax.bar(plot_data['date'], plot_data[f"deflated_{buildings}"], width, color=cmap(1), zorder=3)
    p2 = ax.bar(plot_data['date'], plot_data[f"deflated_{civil}"], width,  bottom= plot_data[f"deflated_{buildings}"], color=cmap(2), zorder=3)
    p3 = ax.bar(plot_data['date'], plot_data[f"deflated_{specialty}"], width,  bottom=plot_data[f"deflated_{buildings}"]+plot_data[f"deflated_{civil}"], color=cmap(0), zorder=3)

    ax.legend((p1[0], p2[0], p3[0]), (buildings, civil, specialty),loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)

    plt.ylabel(unit)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    # ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

    # fig.text(
    #     0.1,
    #     1,
    #     "Deep Dive - Construction",
    #     size=14,
    #     weight = "bold"
    # )
    if language == "en":
        plt.savefig(
        "plots/construction_value.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    else:
        plt.savefig(
        "plots/es_construction_value.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )  

    # # ---
    if language == "en":
        print(f"During {X_max.strftime('%b-%Y')} the annual change came in at {yoy_var:.1%} and the monthly change at {mom_var:.1%}.")
    else:
        print(f"Durante {X_max.strftime('%b-%Y')}, la variación anual fue de {yoy_var:.1%} y la mensual de {mom_var:.1%}.")
    
# def plot_construction_labor(from_d="2006-01-01", language="en"):
#     work_data = get_labour_enec_data(from_d,language)
#     work_data['workforce'] = work_data['14066e939239b6e'] + work_data['62a2aecf0193aac'] + + work_data['74e4b5f7542fc3f'] + work_data['f12649252c82ff0']
#     work_data = work_data[['date','workforce']].copy()
#     # work_data
    
#     min_wf = work_data.sort_values(by=['workforce'], ascending=True).iloc[0]['workforce']
#     min_wf_date = work_data.sort_values(by=['workforce'], ascending=True).iloc[0]['date']
#     last_wf = work_data.iloc[-1]['workforce']
#     last_wf_date = work_data.iloc[-1]['date']
#     diff_wf = last_wf - min_wf
    
#     print("Since the all time low "+ str(min_wf)+ " in date, a total of " +str(diff_wf)+" have people joined the construction workforce, which includes these positions...\nThe count of people working in the construction sector adds up to "+str(last_wf))
    
#     ##### Plot
#     cmap = mpl.cm.get_cmap("GnBu_r", 5)
#     fig = plt.figure(figsize=(8, 4), dpi=200)
#     ax = plt.subplot(111)


#     ax.plot(work_data["date"], work_data["workforce"],
#             marker="o", ms=6, mec="white", markevery=[-1], color=cmap(0))

#     ax.xaxis.set_major_locator(mdates.YearLocator(2))
#     ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    

# ------------------------------------------------------------------
#
# CHART 6: MANUFACTURING - LINE / BARS
#
# ------------------------------------------------------------------

def plot_chart_6(from_d="2013-01-01", language="en"):
    agg_data = get_emim_data(from_d, language, activities=False)
        
    # Aggregated data
    agg_data['yoy_production'] = (agg_data['production_value'] / agg_data['production_value'].shift(12))-1
    agg_data['mom_production'] = (agg_data['production_value'] / agg_data['production_value'].shift(1))-1
    yoy_production = agg_data['yoy_production'].iloc[-1]
    mom_production = agg_data['mom_production'].iloc[-1]
    X_max = agg_data["date"].iloc[-1]
    
    # Plot
    cmap = mpl.cm.get_cmap("GnBu_r", 5)
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)
    
    if language =='en':
        production = 'Production'
        unit = 'Millions of MXN'
    else:
        production = 'Producción'
        unit = 'Millones de MXN'

    ax.plot(agg_data["date"], agg_data["production_value"], color=cmap(0), label = production, zorder=3)
    ax.bar(agg_data["date"], agg_data["production_value"],color=cmap(2), width=20, label=production, zorder=2)
    
    # ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)

    ax.set_ylim(0)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    # ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.ylabel(unit)


    # fig.text(
    #     0.1,
    #     1,
    #     "Manufacturing",
    #     size=14,
    #     weight = "bold"
    # )
    if language == "en":
        plt.savefig(
        "plots/manufacturing.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    else:
        plt.savefig(
        "plots/es_manufacturing.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )  
    
    if language =='en':
        print(f"On {X_max.strftime('%b-%Y')}, the  production value of manufactured products changed by {yoy_production:.1%} YoY; the MoM rate was {mom_production:.1%}.")
    else:
        print(f"En {X_max.strftime('%b-%Y')}, el valor de la producción de los productos manufacturados cambió en {yoy_production:.1%} YoY; la tasa MoM fue de {mom_production:.1%}.")
       
# ------------------------------------------------------------------
#
# CHART 7: SERVICES - BARS
#
# ------------------------------------------------------------------

def plot_chart_7(from_d="2013-01-01", language="en"):
    yoy_data = get_ems_data(from_d, language, operation ="yoy_growth_rel")
    
    mom_data = get_ems_data(from_d, language, operation ="last_growth_rel")
    
    top_mom = mom_data[mom_data['date'] == mom_data['date'].max()].sort_values(by="income", ascending=False).reset_index(drop=True).head(10)
    top_mom = top_mom.sort_values(by="income").reset_index(drop=True)
    top_yoy = yoy_data[yoy_data['date'] == yoy_data['date'].max()].sort_values(by="income", ascending=False).reset_index(drop=True).head(10)
    top_yoy = top_yoy.sort_values(by="income").reset_index(drop=True)
    if language == "en":
        top_yoy = top_yoy.replace({'Other services (except public administration)':'Other services','Governmental, legislative activities of law enforcement and international and extraterritorial bodies':'Governmental and legislative services','Administrative and support and waste management and remediation services':'Administrative and waste management services', 'Information':'Media'})
        top_mom = top_mom.replace({'Other services (except public administration)':'Other services','Governmental, legislative activities of law enforcement and international and extraterritorial bodies':'Governmental and legislative services','Administrative and support and waste management and remediation services':'Administrative and waste management services', 'Information':'Media'})
    else:
        top_yoy = top_yoy.replace({'Servicios de esparcimiento culturales y deportivos, y otros servicios recreativos':'Serv. de esparcimiento y recreativos','Generación, transmisión, distribución y comercialización de energía eléctrica, suministro de agua y de gas natural por ductos al consumidor final':'Energía, suministro de agua y gas natural','Servicios de alojamiento temporal y de preparación de alimentos y bebidas':'Hoteles y Restaurantes','Otros servicios excepto actividades gubernamentales':'Otros servicios','Agricultura, cría y explotación de animales, aprovechamiento forestal, pesca y caza':'Agricultura, ganadería, pesca y acts. forestales','Servicios de apoyo a los negocios y manejo de residuos, y servicios de remediación':'Serv. de apoyo a los negocios y manejo de residuos','Actividades legislativas, gubernamentales, de impartición de justicia y de organismos internacionales y extraterritoriales':'Acts. gubernamentales y legislativas','Servicios inmobiliarios y de alquiler de bienes muebles e intangibles':'Serv. inmobiliarios y alquiler de bienes','Servicios de salud y de asistencia social':'Serv. sociales y de salud','Servicios profesionales, científicos y técnicos':'Serv. profesionales, científicos y técnicos','Servicios educativos':'Serv. educativos','Servicios financieros y de seguros':'Serv. financieros y de seguros'})
        top_mom = top_mom.replace({'Servicios de esparcimiento culturales y deportivos, y otros servicios recreativos':'Serv. de esparcimiento y recreativos','Generación, transmisión, distribución y comercialización de energía eléctrica, suministro de agua y de gas natural por ductos al consumidor final':'Energía, suministro de agua y gas natural','Servicios de alojamiento temporal y de preparación de alimentos y bebidas':'Hoteles y Restaurantes','Otros servicios excepto actividades gubernamentales':'Otros servicios','Agricultura, cría y explotación de animales, aprovechamiento forestal, pesca y caza':'Agricultura, ganadería, pesca y acts. forestales','Servicios de apoyo a los negocios y manejo de residuos, y servicios de remediación':'Serv. de apoyo a los negocios y manejo de residuos','Actividades legislativas, gubernamentales, de impartición de justicia y de organismos internacionales y extraterritoriales':'Acts. gubernamentales y legislativas','Servicios inmobiliarios y de alquiler de bienes muebles e intangibles':'Serv. inmobiliarios y alquiler de bienes','Servicios de salud y de asistencia social':'Serv. sociales y de salud','Servicios profesionales, científicos y técnicos':'Serv. profesionales, científicos y técnicos','Servicios educativos':'Serv. educativos','Servicios financieros y de seguros':'Serv. financieros y de seguros'})
    # Plot
    cmap = mpl.cm.get_cmap("GnBu_r", 5)
    fig = plt.figure(figsize=(11, 4), dpi=200)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.barh(top_yoy["economic_activity"], top_yoy["income"],color=cmap(0), label="sales", zorder=2)
    ax1.axvline(x=0,lw=1, color='black')

    ax2.barh(top_mom["economic_activity"], top_mom["income"],color=cmap(2), label="sales", zorder=2)
    ax2.axvline(x=0,lw=1, color='black')

    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,wspace=1.5,hspace=1)
    fig.tight_layout()

    ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0%}"))
    ax2.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0%}"))
    ax1.title.set_text('YoY')
    ax2.title.set_text('MoM')
    
    yoy_first_act_name =top_yoy["economic_activity"].iloc[-1]
    yoy_second_act_name =top_yoy["economic_activity"].iloc[-2]
    yoy_third_act_name =top_yoy["economic_activity"].iloc[-3]
    yoy_first_act_value =top_yoy["income"].iloc[-1]
    yoy_second_act_value =top_yoy["income"].iloc[-2]
    yoy_third_act_value =top_yoy["income"].iloc[-3]
    
    mom_first_act_name =top_mom["economic_activity"].iloc[-1]
    mom_second_act_name =top_mom["economic_activity"].iloc[-2]
    mom_third_act_name =top_mom["economic_activity"].iloc[-3]
    mom_first_act_value =top_mom["income"].iloc[-1]
    mom_second_act_value =top_mom["income"].iloc[-2]
    mom_third_act_value =top_mom["income"].iloc[-3]
    
    X_max = top_yoy["date"].iloc[-1]


    # fig.text(
    #     0.1,
    #     1,
    #     "Services",
    #     size=14,
    #     weight = "bold"
    # )
    if language == "en":
        plt.savefig(
        "plots/services.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    else:
        plt.savefig(
        "plots/es_services.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )  
    
    if language =='en':
        print(f"On {X_max.strftime('%b-%Y')}, the  economic activities with the most YoY revenue changes were {yoy_first_act_name} ({yoy_first_act_value:,.1%}), {yoy_second_act_name} ({yoy_second_act_value:,.1%}) and {yoy_third_act_name} ({yoy_third_act_value:,.1%}); while the top 3 for MoM revenue changes were {mom_first_act_name} ({mom_first_act_value:,.1%}), {mom_second_act_name} ({mom_second_act_value:,.1%}) and {mom_third_act_name} ({mom_third_act_value:,.1%}).")

    else:
        print(f"En {X_max.strftime('%b-%Y')}, las actividades económicas con mayor cambio YoY fueron {yoy_first_act_name} ({yoy_first_act_value:,.1%}), {yoy_second_act_name} ({yoy_second_act_value:,.1%}) y {yoy_third_act_name} ({yoy_third_act_value:,.1%}); mientras que el top 3 en cuanto a su variación MoM fueron {mom_first_act_name} ({mom_first_act_value:,.1%}), {mom_second_act_name} ({mom_second_act_value:,.1%}) and {mom_third_act_name} ({mom_third_act_value:,.1%}).")
       
# ------------------------------------------------------------------
#
# CHART 8: BUSINESS CONFIDENCE - BARS
#
# ------------------------------------------------------------------

def plot_chart_8(from_d="2013-01-01", language="en", previous_month =False):
    plot_data = get_emoe_data(from_d,language)
    if previous_month == True:
        plot_data = plot_data[plot_data['date']<=plot_data['date'].iloc[-4]]
    economic_activities = plot_data['economic_activity'].unique().tolist()
    ice_prev_mont = []
    for activity in economic_activities:
        aux_df = plot_data[plot_data['economic_activity']==activity]
        ice_prev_mont.append(aux_df['ice'].iloc[-2])
    
    plot_data = plot_data[plot_data['date'].dt.month == plot_data['date'].max().month].reset_index(drop=True)
    plot_data.drop(columns='economic_activity__ref', inplace=True)
    
    plot_data = plot_data.pivot(index='date',columns='economic_activity', values='ice')
    
    if language == "es":
        plot_data.rename(columns={'Industrias manufactureras':'Manufactura'},inplace=True)
    plot_data.reset_index(inplace=True)
    
    # Plot
    cmap = mpl.cm.get_cmap("GnBu_r", 5)
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)
    
    index_ticks = np.arange(plot_data['date'].shape[0])
    width = 0.3
    
    X_max = plot_data["date"].iloc[-1]
    
    if language =="en":
        trade = "Trade"
        construction = "Construction"
        manufacturing = "Manufacturing"
        source_text = f"Showing {X_max.strftime('%B')} data for each year"

    else:
        trade = "Comercio"
        construction = "Construcción"
        manufacturing = "Manufactura"
        source_text = f"Se muestran los datos de {X_max.strftime('%B')} para cada año."

    last_construction = plot_data[construction].iloc[-1]
    second_last_construction = plot_data[construction].iloc[-2]
    last_manufacturing = plot_data[manufacturing].iloc[-1]
    second_last_manufacturing = plot_data[manufacturing].iloc[-2]
    last_trade = plot_data[trade].iloc[-1]
    second_last_trade = plot_data[trade].iloc[-2]
    yoy_construction_diff = last_construction - second_last_construction
    yoy_construction_var = (last_construction / second_last_construction)-1
    yoy_manufacturing_diff = last_manufacturing - second_last_manufacturing
    yoy_manufacturing_var = (last_manufacturing / second_last_manufacturing)-1
    yoy_trade_diff = last_trade - second_last_trade
    yoy_trade_var = (last_trade / second_last_trade)-1
    mom_construction_diff = last_construction - ice_prev_mont[1]
    mom_construction_var = (last_construction / ice_prev_mont[1])-1
    mom_manufacturing_diff = last_manufacturing - ice_prev_mont[2]
    mom_manufacturing_var = (last_manufacturing / ice_prev_mont[2])-1
    mom_trade_diff = last_trade - ice_prev_mont[0]
    mom_trade_var = (last_trade / ice_prev_mont[0])-1
    
    ax.bar(index_ticks, plot_data[trade], color=cmap(
    0), width=width, zorder=3, label=trade)
    ax.bar(index_ticks + width, plot_data[construction],
        color=cmap(1), width=width, zorder=3, label=construction)
    ax.bar(index_ticks + 2*width, plot_data[manufacturing],
    color=cmap(2), width=width, zorder=3, label=manufacturing)
    
    ax.set_xticks(index_ticks + width / 3, labels=plot_data["date"].dt.strftime("%b-%y"))

    ax.set_ylim(0)
    
    # fig.text(0.13, 0.0, source_text, fontsize=9)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))

    # fig.text(
    #     0.1,
    #     1,
    #     "Business Confidence",
    #     size=14,
    #     weight = "bold"
    # )

    if language == "en":
        plt.savefig(
        "plots/business_confidence.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    else:
        plt.savefig(
        "plots/es_business_confidence.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    # # ---
    if language == "en":
        print(f"During {X_max.strftime('%b-%Y')}, the YoY change in the Business Confidence Indicator (ICE) for {economic_activities[0]} companies was {yoy_trade_diff:.1f} points ({yoy_trade_var:.1%}),{yoy_construction_diff:.1f} points ({yoy_construction_var:.1%}) for {economic_activities[1]}, and {yoy_manufacturing_diff:.1f} points ({yoy_manufacturing_var:.1%}) for {economic_activities[2]} companies.\nThe MoM change was {mom_trade_diff:.1f} points ({mom_trade_var:.1%}), {mom_construction_diff:.1f} points ({mom_construction_var:.1%}), and {mom_manufacturing_diff:.1f} points ({mom_manufacturing_var:.1%}) for {economic_activities[0]}, {economic_activities[1]} and {economic_activities[2]} companies, respectively.") 
        

    else:
        print(f"En {X_max.strftime('%b-%Y')}, la variación YoY en el Indicador de Confianza Emmpresarial fue de {yoy_trade_diff:.1f} points ({yoy_trade_var:.1%}) para las actividades de {economic_activities[0]},{yoy_construction_diff:.1f} puntos ({yoy_construction_var:.1%}) para las de {economic_activities[1]}, y {yoy_manufacturing_diff:.1f} puntos ({yoy_manufacturing_var:.1%}) para {economic_activities[2]}.\nLa variación MoM fue de {mom_trade_diff:.1f} puntos ({mom_trade_var:.1%}) para las actividades de {economic_activities[0]}, de {mom_construction_diff:.1f} puntos ({mom_construction_var:.1%}) para las de {economic_activities[1]} y de {mom_manufacturing_diff:.1f} puntos ({mom_manufacturing_var:.1%}) para {economic_activities[2]}.")
  
# ------------------------------------------------------------------
#
# CHART 9: CONSUMER CONFIDENCE - BARS
#
# ------------------------------------------------------------------

def plot_chart_9(from_d="2013-01-01", language="en", previous_month =False):
    plot_data = get_enco_data(from_d, language)
    if previous_month == True:
        plot_data = plot_data[plot_data['date']<=plot_data['date'].iloc[-2]]
    icc_prev_month = plot_data['icc'].iloc[-2]
    plot_data = plot_data[plot_data['date'].dt.month == plot_data['date'].max().month].reset_index(drop=True)
    
    ##### Plot
    cmap = mpl.cm.get_cmap("GnBu_r", 5)
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)

    ax.bar(plot_data['date'], plot_data['icc'], width=150, color=cmap(0), zorder=3, align="center")
    # ax.plot(plot_data['date'], plot_data['icc'], color=cmap(0), zorder=3)

    ax.xaxis.set_major_locator(mdates.MonthLocator(plot_data['date'].max().month))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_ylim(0)

    X_max = plot_data["date"].iloc[-1]
    icc_last = plot_data["icc"].iloc[-1]
    icc_2last = plot_data["icc"].iloc[-2]
    icc_dif =  icc_last - icc_2last
    icc_var = (icc_last / icc_2last) -1
    icc_mom_dif =  icc_last - icc_prev_month
    icc_mom_var = (icc_last / icc_prev_month) -1
        
    if language =='en':
        source_text = f"Showing {X_max.strftime('%B')} data for each year"
    else:
        source_text = f"Se muestran los datos de {X_max.strftime('%B')} para cada año."

    
    # fig.text(0.13, 0.0, source_text, fontsize=9)
    
    
    # fig.text(
    #     0.1,
    #     1,
    #     "Deep Dive - Consumer Confidence Index",
    #     size=14,
    #     weight = "bold"
    # )
    if language == "en":
        plt.savefig(
        "plots/consumer_confidence.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    else:
        plt.savefig(
        "plots/es_consumer_confidence.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )  

    # # ---
    if language == "en":
        print(f"During {X_max.strftime('%b-%Y')} the Consumer Confidence Indicator changed {icc_dif:.1f} points ({icc_var:.1%}), when comparing it to the same month of last year. The monthly change was {icc_mom_dif:.1f} points ({icc_mom_var:.1%}).")
    else:
        print(f"En {X_max.strftime('%b-%Y')} el Índice de Confianza del Consumidor cambió {icc_dif:.1f} puntos ({icc_var:.1%}), al compararlo con el mismo mes del año anterior. La variación mensual fue de {icc_mom_dif:.1f} puntos ({icc_mom_var:.1%}).")

# %%
