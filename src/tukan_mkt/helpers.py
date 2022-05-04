import requests
import pandas as pd
import json
import numpy as np
import os

# --- Global variables: TOKEN, headers and URL


token = os.environ.get('tukan_api_token')  # YOUR TOKEN goes here
url = 'http://api.tukanmx.com/v1/retrieve/'

headers = {
    "Content-Type": "application/json",
    "Authorization": "Token " + token
}

# -----

# Helper functions

# -----


def get_tukan_api_request(payload):
    '''
    This function helps on making a request to TUKAN's API by specifying a payload.

    Args:
     - payload (dic): a dictionary with the payload of the data to be requested.
    '''

    global url
    global headers
    global token

    metadata_payload = {
        "type": "data_table",
        "operation": "metadata",
        "language": payload['language'],
        "request": payload['request'],
        "categories": payload['categories']
    }

    response = requests.request(
        "POST", url, headers=headers, data=json.dumps(payload))
    md_response = requests.request(
        "POST", url="http://api.tukanmx.com/v1/control/", headers=headers, data=json.dumps(metadata_payload))
    try:
        data = pd.DataFrame(response.json()['data'])
        print("Success getting the data")
    except:
        print(response.content)

    data['date'] = pd.to_datetime(data['date'])
    data.replace({'': np.nan}, inplace=True)

    try:
        categories = md_response.json()['data']['categories']
    except:
        categories = []

    return({
        'data': data,
        'variables': md_response.json()['data']['variables'],
        'categories': categories,
        'metadata': md_response.json()['data']['data_table']
    })


def get_table_dictionary(table_id):
    '''
    This function helps on making getting a data table's indicator dictionary.

    Args:
     - table_id (str): a dictionary with the table_id of the data table.
    '''
    global url
    global headers

    payload = {
        "type": "variable_dict",
        "data_table": table_id,
        "operation": "all"
    }

    response = requests.request(
        "POST", url, headers=headers, data=json.dumps(payload))
    var_dict = pd.DataFrame(
        response.json()[table_id + '_variables_dictionary'])

    return(var_dict)