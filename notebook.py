"""
File: notebook.py
Author: Chuncheng Zhang
Date: 2024-05-29
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Follow the instruction of Colab Demo 1: Lag-Llama Zero-Shot Forecasting Demo
    url = https://colab.research.google.com/drive/1DRAzLUPxsd-0r8b-o4nlyFXrjw_ZajJJ?usp=sharing#scrollTo=wjlKnHl8-CWT

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-05-29 ------------------------
# Requirements and constants
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates

from itertools import islice
from matplotlib import pyplot as plt
from rich import inspect, print
from IPython.display import display

from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from lag_llama.gluon.estimator import LagLlamaEstimator

# %% ---- 2024-05-29 ------------------------
# Function and class


def get_lag_llama_predictions(dataset, prediction_length, device, context_length=32, use_rope_scaling=False, num_samples=100):
    # Uses GPU since in this Colab we use a GPU.
    ckpt = torch.load("lag-llama.ckpt", map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        # Lag-Llama was trained with a context length of 32, but can work with any context length
        context_length=context_length,

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,

        batch_size=1,
        num_parallel_samples=100,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss


# %% ---- 2024-05-29 ------------------------
# Play ground

# ----------------------------------------
# ---- Load example data ----

url = (
    "https://gist.githubusercontent.com/rsnirwan/a8b424085c9f44ef2598da74ce43e7a3"
    "/raw/b6fdef21fe1f654787fa0493846c546b7f9c4df2/ts_long.csv"
)

df = pd.read_csv(url, index_col=0, parse_dates=True)
print(df)

df.index

# %%

# Set numerical columns as float32
for col in df.columns:
    # Check if column is not of string type
    if df[col].dtype != 'object' and pd.api.types.is_string_dtype(df[col]) == False:
        df[col] = df[col].astype('float32')

# Create the Pandas
dataset = PandasDataset.from_long_dataframe(
    df, target="target", item_id="item_id")
print(dataset)

# %%
# ----------------------------------------
# ---- Destroy the datetime formatted index ----

# df.index = range(len(df))
# print(df)

# # Create the Pandas
# dataset = PandasDataset.from_long_dataframe(
#     df, target="target", item_id="item_id")
# print(dataset)

# %%
# ----------------------------------------
# ---- Forecast ----

backtest_dataset = dataset
# Define your prediction length. We use 24 here since the data is of hourly frequency
prediction_length = 24
# number of samples sampled from the probability distribution for each timestep
num_samples = 100
# You can switch this to CPU or other GPUs if you'd like, depending on your environment
device = torch.device("cuda:0")

forecasts, tss = get_lag_llama_predictions(
    backtest_dataset, prediction_length, device, num_samples)
display(forecasts, tss)

# %%
forecasts[0]


# %%
plt.figure(figsize=(20, 15))
date_formater = mdates.DateFormatter('%b, %d')
plt.rcParams.update({'font.size': 15})

# Iterate through the first 9 series, and plot the predicted samples
for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
    ax = plt.subplot(3, 3, idx+1)

    plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="target", )
    forecast.plot(color='g')
    plt.xticks(rotation=60)
    ax.xaxis.set_major_formatter(date_formater)
    ax.set_title(forecast.item_id)

plt.gcf().tight_layout()
plt.legend()
plt.show()

# %%
_df = df[df['item_id'] == 'A']
_df['x'] = _df.index
sns.lineplot(_df.iloc[-4*prediction_length:], x='x', y='target')
plt.show()

# %% ---- 2024-05-29 ------------------------
# Pending
display(dir(forecast))
forecast.mean.shape
plt.plot(forecast.mean)
plt.show()

# %%
forecast.quantile(0.5)

# %% ---- 2024-05-29 ------------------------
# Pending
inspect(forecast)


# %%
