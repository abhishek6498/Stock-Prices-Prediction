# %% [code]
# This Pythom 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# Problem statement: Predict stock price using the LSTM neural network

# %% [markdown]
# 1. Imports:

# %% [code]
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
%matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

from sklearn.preprocessing import MinMaxScaler

# %% [markdown]
# 2. Read the dataset:

# %% [code]
df = pd.read_csv("../input/nsetataglobal/NSE-TATAGLOBAL11.csv")
df.head()

# %% [code]
df.info()

# %% [markdown]
# 3. Analyze the closing prices from dataframe:

# %% [code]
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df.index = df["Date"]

plt.figure(figsize=(16,8))
plt.plot(df["Close"], label='Closing Price History')

# %% [markdown]
# 4. Sort the dataset on date time and filter “Close” column:

# %% [code]
sorted_index_df = df.sort_index(ascending = True, axis = 0)
filtered_df = pd.DataFrame(index=sorted_index_df.index, columns=["Close"])

for i in range(0,len(df)):
    filtered_df["Close"][i] = sorted_index_df["Close"][i]
filtered_df.head()

# %% [markdown]
# 5. Normalize the new filtered dataset:

# %% [code]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(filtered_df)
scaled_data[:5]

# %% [markdown]
# 6.Prepare train data for LSTM

# %% [code]
train_data = list(scaled_data[:987,0])
time_step = 60

X_train = []
y_train = []
for i in range(time_step, len(train_data)):
    X_train.append(train_data[i-time_step:i])
    y_train.append(train_data[i])
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# %% [markdown]
# 7. Build and train the LSTM model:

# %% [code]
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))

lstm_model.compile(loss="mean_squared_error", optimizer="adam")
lstm_model.fit(X_train, y_train, epochs= 50,batch_size=64, verbose=2)

# %% [code]
lstm_model.summary()

# %% [markdown]
# 8. Prepare test data to make predictions

# %% [code]
validation_data = list(scaled_data[987 - time_step:,0])

X_test = []
y_test = []
for i in range(time_step, len(validation_data)):
    X_test.append(validation_data[i-time_step:i])
    y_test.append(validation_data[i])
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_close_price = lstm_model.predict(X_test)
predicted_close_price = scaler.inverse_transform(predicted_close_price)


# %% [markdown]
# 9. Visualize the predicted stock costs with actual stock costs:

# %% [code]
train_data = filtered_df[:987]
validation_data = filtered_df[987:]
validation_data["Predicted_close_price"] = predicted_close_price
plt.plot(train_data["Close"], label="train_close_price")
plt.plot(validation_data["Close"], label="test_actual_close_price")
plt.plot(validation_data["Predicted_close_price"], label="test_predicted_close_price")
plt.legend()
plt.show()

# %% [code]
from sklearn.metrics import mean_absolute_percentage_error as mape

mape_score = mape(list(validation_data["Close"]), list(validation_data["Predicted_close_price"]))
print(mape_score)

# %% [code]
print(np.mean(list(validation_data["Close"])))