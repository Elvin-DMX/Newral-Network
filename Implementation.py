
import os
os.chdir(r'C:\Users\einoa\Desktop\Assignment\Term3\BDM_3104\Project\Final')
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from data_preprocessing import normalize_featuresDF, split_ValidationSet, split_Final_df, split_Train_Test_DF, DataSet_Graph
import Models 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("always")


# Step 1: Download Data
data = yf.download('ADBE', start='2020-01-01', end='2022-01-01')
print("Data shape:", data.shape)
print("Data head:", data.head())

# Step 2: Preprocess Data
# Add target column
data['Target'] = data['Close'].shift(-1)
# Drop any rows with NaN values
data.dropna(inplace=True)
# Normalize the feature columns
normalized_features_df = normalize_featuresDF(data.drop(['Target'], axis=1))
# Split the data into training, validation, and test sets
validation_x, validation_y = split_ValidationSet(normalized_features_df, data['Target'])
final_features_df, final_target_df = split_Final_df(normalized_features_df, data['Target'])
x_train, y_train, x_test, y_test = split_Train_Test_DF(final_features_df, final_target_df)

# Decision Tree Model
print("Decision Tree Model")

RMSE_Score, R2_Score = Models.model_validateResult(Models.model_Decision_Tree_Regressor(x_train, y_train,validation_x), model_name = "Decision Tree")
print(RMSE_Score, R2_Score)

# Linear Regression Model
print("Linear Regression Model")
RMSE_Score, R2_Score = Models.model_validateResult(Models.model_Linear_Regression(x_train, y_train,validation_x), model_name = " Linear Regression")
print(RMSE_Score, R2_Score)

# Support Vector Regressor
print("Support Vector Regressor")
RMSE_Score, R2_Score = Models.model_validateResult(Models.model_SVR(x_train, y_train,validation_x), model_name = "Support Vector Regressor")
print(RMSE_Score, R2_Score)

# Tuned Support Vector Regressor
print("Tuned Support Vector Regressor")
RMSE_Score, R2_Score = Models.model_validateResult(Models.model_SVRTuning(x_train, y_train,validation_x), model_name = "Tuned Support Vector Regressor")
print(RMSE_Score, R2_Score)

# Stochastic Gradient Descent Model
print("Stochastic Gradient Descent Model")
RMSE_Score, R2_Score = Models.model_validateResult(Models.Stochastic_Gradient_Descent_model(x_train, y_train,validation_x), model_name = "Stochastic Gradient Descent")
print(RMSE_Score, R2_Score)

# Multi-Layer Perceptron Regression model
print("Multi-Layer Perceptron Regression model")
RMSE_Score, R2_Score = Models.model_validateResult(Models.model_MLPRegressor(x_train, y_train,validation_x), model_name = "Multi-Layer Perceptron Regression")
print(RMSE_Score, R2_Score)


# Visualize Results for LSTM
import LSTM_MODEL
import os
import time
from tensorflow.keras.layers import LSTM
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
# Window size or the sequence length
N_STEPS = 50
# Lookup step, 1 is the next day
LOOKUP_STEP = 15
# whether to scale feature columns & output price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"
# whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"
# whether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")
### model parameters
N_LAYERS = 2
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False
### training parameters
# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 100
EPOCHS = 100
# Amazon stock market
ticker = "ADBE"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"
    
# set seed, so we can get the same results after rerunning several times
import random 
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")
ticker = "ADBE"
# load the data
data = LSTM_MODEL.load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, 
                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, 
                feature_columns=FEATURE_COLUMNS)
# save the dataframe
data["df"].to_csv(ticker_data_filename)
# construct the model
model = LSTM_MODEL.create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
# a new optimal model using ModelCheckpoint
history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)
# load optimal model weights from results folder
model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

# evaluate the model
loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
if SCALE:
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
else:
    mean_absolute_error = mae
    
# evaluate the model
loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)

# get the final dataframe for the testing set
final_df = LSTM_MODEL.get_final_df(model, data)

# predict the future price
future_price = LSTM_MODEL.predict(model, data)

# we calculate the accuracy by counting the number of positive profits
accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
# calculating total buy & sell profit
total_buy_profit  = final_df["buy_profit"].sum()
total_sell_profit = final_df["sell_profit"].sum()
# total profit by adding sell & buy together
total_profit = total_buy_profit + total_sell_profit
# dividing total profit by number of testing samples (number of trades)
profit_per_trade = total_profit / len(final_df)

# printing metrics
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
print(f"{LOSS} loss:", loss)
print("Mean Absolute Error:", mean_absolute_error)
print("Accuracy score:", accuracy_score)
print("Total buy profit:", total_buy_profit)
print("Total sell profit:", total_sell_profit)
print("Total profit:", total_profit)
print("Profit per trade:", profit_per_trade)


# plot true/pred prices graph
LSTM_MODEL.plot_graph(final_df)
