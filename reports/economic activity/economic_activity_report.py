# %%
# ----------------------------------
# We load fonts and stylesheet.
# ----------------------------------
import datetime
import matplotlib.dates as mdates
from matplotlib.lines import _LineStyle
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
# Load the igae_data
# ----------------------------------
def get_igae_igae_data(from_d="2000-01-01"):
    payload_yoy = {
        "type": "igae_data_table",
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

def get_activities_data(from_d="2000-01-01"):
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

activities_data = get_activities_data(from_d="2016-01-01")

# %%
# ------------------------------------------------------------------
#
#                       CHARTS FOR REPORT
#
# ------------------------------------------------------------------

# %% First part of report

# ------------------------------------------------------------------
#
# CHART 1: YOY CHANGE IN IGAE
#
# ------------------------------------------------------------------

def plot_chart_1(from_d="2000-01-01"):
    
    plot_data = get_igae_data(from_d)
    plot_data = plot_data[plot_data['economic_activity__ref']=='dfeefc621d16d0c']

    cmap = mpl.cm.get_cmap("GnBu_r", 5)
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)

    Y_mom = plot_data["mom_igae"].iloc[-1]
    Y_mom_annualized = np.power((1+Y_mom),12)-1
    Y_end = plot_data["yoy_igae"].iloc[-1]
    X_min = plot_data["date"].iloc[0]
    X_max = plot_data["date"].iloc[-1]

    ax.plot(plot_data["date"], plot_data["yoy_igae"],
            marker="o", ms=6, mec="white", markevery=[-1], color=cmap(0))

    ax.hlines(0, X_min, X_max, ls = "--", color = "black", lw = 0.75)


    ax_text(x=X_max + relativedelta(months=3), y=Y_end,
            s=f"<{Y_end:.2%}>",
            highlight_textprops=[{"color": cmap(0)}],
            ax=ax, weight="bold", font="Dosis", ha="left", size=9)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))

    fig.text(
        0.1,
        1,
        "Indicador General de la Actividad Económica (IGAE) - YoY Change",
        size=14,
        weight = "bold"
    )

    plt.savefig(
        "plots/yoy_igae_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    # ---
    print(
        f"During {X_max.strftime('%b-%Y')} the annual change came in at {Y_end:.2%} and the monthly change at {Y_mom:.2%}; The monthly rate implies an annualized change of {Y_mom_annualized:.2%}.")

# ------------------------------------------------------------------
#
# CHART 2: YOY CHANGE IN IGAE SECTORS
#
# ------------------------------------------------------------------

def plot_chart_2(from_d="2021-11-01"):
    
    data = get_igae_data(from_d)
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

    cmap = mpl.cm.get_cmap("GnBu_r", 5)
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.subplot(111)

    
    index_ticks = np.arange(plot_data['date'].shape[0])
    width = 0.3

    pri_yoy = plot_data['Actividades primarias'].iloc[-1] 
    sec_yoy = plot_data['Actividades secundarias'].iloc[-1] 
    ter_yoy = plot_data['Actividades terciarias'].iloc[-1]
    pri_mom = data['Actividades primarias'].iloc[-1]
    pri_ann = np.power((1+pri_mom),12)-1 
    sec_mom = data['Actividades secundarias'].iloc[-1]
    sec_ann = np.power((1+sec_mom),12)-1 
    ter_mom = data['Actividades terciarias'].iloc[-1]
    ter_ann = np.power((1+ter_mom),12)-1 
    max_date = plot_data["date"].max()

    
    X_min = index_ticks.min()
    X_max = index_ticks.max()

    ax.bar(index_ticks, plot_data['Actividades primarias'], color=cmap(
        0), width=width, zorder=3, label="Primary")
    ax.bar(index_ticks + width, plot_data['Actividades secundarias'],
           color=cmap(1), width=width, zorder=3, label="Secondary")
    ax.bar(index_ticks + 2*width, plot_data['Actividades terciarias'],
        color=cmap(2), width=width, zorder=3, label="Terciary")
    
    ax.hlines(0, X_min-.13, X_max+.4, ls = "-", color = "black", lw = 0.75, zorder=3)

    ax.set_xticks(index_ticks + width / 3,
                  labels=plot_data["date"].dt.strftime("%b-%y"), rotation=90)

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))
    # ax.set_ylim(0)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)

    # ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))

    fig.text(
        0.1,
        1,
        "IGAE by Sector - YoY Change",
        size=14,
        weight = "bold"
    )

    plt.savefig(
        "plots/yoy_sector_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    # # ---
    print(
        f"During {max_date.strftime('%b-%Y')} the annual change came in at {pri_yoy:.2%} for primary activities, at {sec_yoy:.2%} for secondary activities and at {ter_yoy:.2%} for terciary activities.\nThe monthly change rates came in at {pri_mom:.2%} ({pri_ann:.2%} annualized) for primary activities, at {sec_mom:.2%} ({sec_ann:.2%} annualized) for secondary activities and at {ter_mom:.2%} ({ter_ann:.2%} annualized) for terciary activities.")

# ------------------------------------------------------------------
#
# CHART 3: YOY & MOM CHANGE IN IGAE BY ECONOMIC ACTIVITY - LOLLIPOP
#
# ------------------------------------------------------------------

def plot_chart_3(from_d="2016-01-01"):
    plot_data = get_activities_data(from_d)
    plot_data = plot_data[plot_data['date'] == plot_data['date'].max()].sort_values(by="yoy_igae", ascending=True).reset_index(drop=True)
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
    ax.scatter(plot_data['yoy_igae'],plot_data['p_ref'],s=50, edgecolors='black', lw=0.8, zorder=3, color=cmap(0), label="YoY")

    for x in plot_data.index:
        if plot_data['mom_igae'].loc[x]>=0:
            ax.hlines(y=plot_data['p_ref'].loc[x], xmin=0, xmax=plot_data['mom_igae'].loc[x],lw=1, color='black' ,zorder=2)
        else:
            ax.hlines(y=plot_data['p_ref'].loc[x], xmin=plot_data['mom_igae'].loc[x], xmax=0,lw=1, color='black' ,zorder=2)
    ax.scatter(plot_data['mom_igae'],plot_data['p_ref'],s=50, edgecolors='black', lw=0.8, zorder=3, color=cmap(2), label="MoM")

    # Add gridlines and format to ticks 
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0%}'))
    ax.axvline(x=0,lw=1, color='black')
    ax.set_yticks(plot_data['p_ref'])
    ax.set_yticklabels(labels)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)

    # Axis ticks on right side
    ax.yaxis.tick_right()
    
    #
    acts = plot_data['economic_activity'].iloc[-3:].values.tolist()
    growth_yoy = plot_data['yoy_igae'].iloc[-3:].values.tolist()
    max_date = plot_data["date"].max()

    fig.text(
        0.2,
        1,
        "IGAE by Economic Activity - Change",
        size=14,
        weight = "bold"
    )

    plt.savefig(
        "plots/yoy_activities_change.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    # # ---
    print(
        f"During {max_date.strftime('%b-%Y')} the best perfoming economic activities where {acts[2]} ({growth_yoy[2]:.2%}), {acts[1]} ({growth_yoy[1]:.2%}) and {acts[0]} ({growth_yoy[0]:.2%}).")

# %%
