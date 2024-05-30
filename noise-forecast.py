"""
File: noise-forecast.py
Author: Chuncheng Zhang
Date: 2024-05-29
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Forecasting for random-walk data

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-05-29 ------------------------
# Requirements and constants
import time
import torch
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from perlin_noise import PerlinNoise

from prophet import Prophet
from rich import print, inspect
from loguru import logger

from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from lag_llama.gluon.estimator import LagLlamaEstimator

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
logger.debug(f'Using device: {device}')

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


def convert_numerical_columns_as_float32(df):
    for col in df.columns:
        # Check if column is not of string type
        if df[col].dtype != 'object' and pd.api.types.is_string_dtype(df[col]) == False:
            df[col] = df[col].astype('float32')
    return df


def generate_data(n: int = 1000):
    # Generate random-walk dataset
    # df = pd.DataFrame()
    # dy = np.random.randn(n)
    # y = dy.copy()
    # for j in range(n):
    #     y[j] = np.sum(dy[:j+1])
    # df['value'] = y
    # df['dvalue'] = dy

    # Generate Perlin noise dataset
    p_noise = PerlinNoise(octaves=20, seed=time.time())
    y = np.array([p_noise(i/n) for i in range(n)])
    y *= 2 / np.max(np.abs(y))
    dy = y.copy()
    dy[1:] = y[1:] - y[:-1]
    df = pd.DataFrame()
    df['value'] = y
    df['dvalue'] = dy

    # Using example data
    # url = (
    #     "https://gist.githubusercontent.com/rsnirwan/a8b424085c9f44ef2598da74ce43e7a3"
    #     "/raw/b6fdef21fe1f654787fa0493846c546b7f9c4df2/ts_long.csv"
    # )
    # df = pd.read_csv(url, index_col=0, parse_dates=True).query('item_id=="A"')
    # df['value'] = df['target']
    # df['dvalue'] = df['target']

    # Generate dataset
    df['target'] = df['value']
    df['item_id'] = 'A'
    one_day = 60 * 60 * 24
    df.index = [pd.to_datetime(int(time.time()) + i * one_day, unit='s')
                for i in range(len(df))]
    df.index = pd.PeriodIndex(df.index, dtype=pd.PeriodDtype(freq='D'))

    convert_numerical_columns_as_float32(df)

    dataset = PandasDataset.from_long_dataframe(
        df, target='target', item_id='item_id')

    return df, dataset


# %% ---- 2024-05-29 ------------------------
# Play ground
# ----------------------------------------
# ---- Generate data ----
n_total = 240
generated_df, dataset = generate_data(n_total)
print(generated_df)
print(dataset)


# ----------------------------------------
# ---- Forecast ----
backtest_dataset = dataset
# Define your prediction length. We use 24 here since the data is of hourly frequency
prediction_length = 24
# number of samples sampled from the probability distribution for each timestep
num_samples = 100

forecasts, tss = get_lag_llama_predictions(
    backtest_dataset, prediction_length, device, num_samples)

forecast = forecasts[0]
ts = tss[0]

print(forecast)
print(ts)


def predict_with_prophet():
    df = generated_df.copy().to_timestamp()
    df['y'] = df["value"]
    df["ds"] = df.index
    m = Prophet()
    m.fit(df.iloc[:-prediction_length])
    future = pd.DataFrame(df.iloc[-prediction_length:]['ds'])
    forecast = m.predict(future)
    return forecast.set_index('ds')


forecast_prophet = predict_with_prophet()
print(forecast_prophet)

# %%

fig, axs = plt.subplots(3, 1, figsize=(8, 8))


# ----------------------------------------
# ---- Plot forecast ----
ax = axs[0]
ax.plot(ts.to_timestamp(), label="target")
forecast.plot(color='g', ax=ax)
ax.plot(forecast_prophet['yhat'], color='orange', label="prophet")
ax.set_title('Forecast')
ax.legend(loc='lower left')

# ----------------------------------------
# ---- Plot compare ----
ax = axs[1]
ax.plot(ts[-10-prediction_length:].to_timestamp(), label="target")
# ax.plot(pd.DataFrame(forecast.mean_ts).to_timestamp(), color='g', label='pred')
forecast.plot(color='g', ax=ax)
ax.plot(forecast_prophet['yhat'], color='orange', label="prophet")
# plt.xticks(rotation=60)
ax.set_title('Forecast (prediction only)')
ax.legend(loc='lower left')

# ----------------------------------------
# ---- Plot diff ----
ax = axs[2]
_ts = ts[-prediction_length:].to_timestamp()
target = np.array(_ts).squeeze()
_ts['pred'] = forecast.mean
_ts['diff_llama'] = forecast.mean - target
_ts['diff_prophet'] = forecast_prophet['yhat'].to_numpy() - target
ax.plot(_ts['diff_llama'], color='g', label="llama")
ax.plot(_ts['diff_prophet'], color='orange', label="prophet")
ax.set_title('Diff')
ax.legend(loc='lower left')

fig.suptitle(forecast.item_id)
fig.tight_layout()
plt.show()

# %%

# %%
# ----------------------------------------
# ---- Diff analysis ----
target = np.array(ts[-prediction_length:].to_timestamp()).squeeze()
predict = forecast.mean
df = pd.DataFrame()
df['target'] = target
df['predict'] = predict
df['diff'] = predict - target
df['dy'] = generated_df['dvalue'].to_numpy()[-len(df):]
print(df)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.scatterplot(df, x='dy', y='diff', hue='dy', ax=ax)
ax.set_title('Compare between dy with diff')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
plt.show()

# %%
# Summary
testing_index = ts.index[-prediction_length:]

res_df = generated_df.copy()
res_df['lagLlama'] = forecast.mean_ts
forecast_prophet.index = testing_index
res_df['prophet'] = forecast_prophet['yhat']
res_df['marker'] = 'train'
res_df.loc[testing_index, 'marker'] = 'pred'
print(res_df)

y_lag_llama = res_df.loc[testing_index, 'lagLlama']
y_prophet = res_df.loc[testing_index, 'prophet']
y_true = res_df.loc[testing_index, 'value']
print(np.corrcoef(y_lag_llama, y_true))
print(np.corrcoef(y_prophet, y_true))


# %% ---- 2024-05-29 ------------------------
# Pending

# %% ---- 2024-05-29 ------------------------
# Pending

# %%

# %%
