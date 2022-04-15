from typing import final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from keras.models import load_model
import yfinance as yfin
from tensorflow.keras.models import load_model
import streamlit as st


startD = '2002-01-01'
endD = '2022-01-01'
symbol = 'AAPL'
yfin.pdr_override()

st.title("Stock Price Prediction")
user_input = st.text_input("Enter Stock Ticker", "GOOG")

df = pdr.get_data_yahoo(user_input, start=startD, end=endD)
df = df.reset_index()


st.subheader("Data from " + startD + " to " + endD)
st.write(df.head())


st.subheader("Data Description and Info")
st.write(df.describe())


st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(18, 9))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100 days moving average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(18, 9))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100 and 200 days moving averages')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(18, 9))
plt.plot(ma100, 'r')
plt.plot(ma200, 'b')
plt.plot(df.Close, 'g')
st.pyplot(fig)


#splitting the data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#loading the LSTM model
model = load_model('keras_model.h5')

#testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []


for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_


scaled_factor = 1/scaler[0]
y_predicted = y_predicted * scaled_factor
y_test = y_test * scaled_factor


#writting the actual data for test and prediction part which is 30%

#values of testing actual data
# st.subheader("Actual Closing Prices of the Data for testing part.")
df_actual = pd.DataFrame({ 'Actual Closing Price': y_test[-10: ]})
# df_actual = df_actual.reset_index()
# st.write(df_actual)

#values of testing predicted data
# st.subheader("Predicted Closing Prices of the Data for testing part.")
data_predicted_10 = y_predicted[-10: ]
data_predicted_10 = np.squeeze(data_predicted_10)
df_pred = pd.DataFrame({ 'Predicted Closing Price': data_predicted_10})
# df_pred = df_pred.reset_index()
# st.write(df_pred)


#adding the date column
df['Date'] = pd.to_datetime(df.Date)
df_date = df['Date']
df_date_tail10 = np.array(df_date.tail(10))


df_date_dataframe = pd.DataFrame({ 'Date': df_date_tail10})
# st.write(df_date_dataframe)


# final_result_array = np.concatenate((y_test[-10: ], y_predicted[-10:]), axis=1)
st.subheader("Actual and Predicted Prices Numeric values of last 10 days.")
# df3 = pd.merge( df_actual, df_pred)
# st.write(df3)

frames = [df_date_dataframe, df_actual, df_pred]
result = pd.concat(frames, axis=1)
st.write(result)

#plotting the visualization final prediction graph
st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize=(18, 9))
plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)