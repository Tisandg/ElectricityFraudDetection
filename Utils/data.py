import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import quantile_transform
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold
import requests
import zipfile
import pickle

def download_data():

    print("Downloading data...")
    # Download data from GitHub repository
    os.system('wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z01')
    os.system('wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z02')
    os.system('wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.zip')

    # Unzip downloaded data
    os.system('cat data.z01 data.z02 data.zip > data_compress.zip')
    os.system('unzip -n -q data_compress')

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def download_data_modified():
    try:
        print("Downloading data...")

        # Define the URLs
        urls = [
            ('https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z01', 'data.z01'),
            ('https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z02', 'data.z02'),
            ('https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.zip', 'data.zip')
        ]

        # Download the files
        for url, filename in urls:
            print(f"Downloading {filename}...")
            download_file(url, filename)

            # Check if file was downloaded correctly
            if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                print(f"File {filename} was not downloaded correctly.")
                return

        # Concatenate the files
        with open('data_compress.zip', 'wb') as f_out:
            for part in ['data.z01', 'data.z02', 'data.zip']:
                with open(part, 'rb') as f_in:
                    f_out.write(f_in.read())

        # Verify the concatenated file is a valid ZIP
        try:
            with zipfile.ZipFile('data_compress.zip', 'r') as zip_ref:
                zip_ref.extractall()
            print("Data downloaded and extracted successfully.")
        except zipfile.BadZipFile:
            print("Error: The concatenated file is not a valid ZIP file.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the files: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_dataset(filepath):
    """## Saving "flags" """

    df_raw = pd.read_csv(filepath,index_col=0)
    flags = df_raw.FLAG.copy()
 
    df_raw.drop(['FLAG'], axis=1, inplace=True)

    """## Sorting"""

    # Delete temporarily the row FLAG, transpose the column to have all the dates in a column
    # and sort them.

    df_raw = df_raw.T.copy()
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw.sort_index(inplace=True, axis=0)

    # Finally, transpose again to have the original shape, and add the FLAG column.
    df_raw = df_raw.T.copy()
    df_raw['FLAG'] = flags
    
    return df_raw

"""# Processing dataset"""
def get_processed_dataset(filepath):

    df_raw = get_dataset(filepath)
    flags = df_raw['FLAG']
    df_raw.drop(['FLAG'], axis=1, inplace=True)

    """## Quantile transform"""
    quantile = quantile_transform(df_raw.values, n_quantiles=10, random_state=0, copy=True, output_distribution='uniform')
    df__ = pd.DataFrame(data=quantile, columns=df_raw.columns, index=df_raw.index)
    df__['flags'] = flags

    return df__.iloc[:, 5:]


def save_file(data, directory, filename):

    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it does not exist
        os.makedirs(directory)

    # Save DataFrame and labels
    filename = filename + '.pkl'
    filename = os.path.join(directory, filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {filename}")

def load_file(directory, filename):

    # Load DataFrame and labels
    filename = filename + '.pkl'
    filename = os.path.join(directory, filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data