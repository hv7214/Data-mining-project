import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# load data
bakery = pd.read_csv('preprocessed_BreadBasket_DMS.csv')
bakery['Time'] = pd.to_datetime(bakery['Time']).dt.hour

# filter data for coffee and remove transactions column
coffee = bakery[bakery['Item'] == "Coffee"].drop(['Transaction'], axis=1)
coffee_date_count = coffee.groupby(["Date"]).size().reset_index(name="Coffee")

# split data into test and train part
data_len = len(coffee_date_count.Coffee)
test = coffee_date_count.Coffee.loc[data_len*0.8:]
train = coffee_date_count.Coffee.loc[:data_len*0.8]

# MODEL

# normalize sales to 0-1 range
train_scaled = train.values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_scaled)

# create X_train and y_train datas with timesteps = 7 for feed of LSTM model
X_train = []
y_train = []
timesteps = 7
for i in range(timesteps, len(train)):
    X_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# lstm model
model = Sequential()
model.add(LSTM(40, input_shape=(timesteps, 1))) # 40 LSTM layer
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=300, batch_size=6, verbose=2)

'''We must prepare "inputs" to create "X_train". This "inputs" contain "test" and last 7 items of "train".
So why we add last 7 items of "train" to beginning of the "inputs"?
because when we use "train" to create "X_train", we couldn't use last 7 item to create new array step.
So our "inputs" size equal to (lenght of "test" + "timestep")
When we start to create "X_test" we will see "X_test" size will be ("inputs" size - 7)
because again we won't use last 7 item of "inputs"'''

total = pd.concat((train, test), axis=0)
inputs = total[len(total) - len(test) - timesteps:].values.reshape(-1,1)
# Lastly we normalize our inputs data.
inputs = scaler.transform(inputs)

print("Inputs shape ->",inputs.shape, "  Test shape ->", test.shape)

# And we will create X_test data from inputs data.
# Timestep is same

X_test = []
for i in range(timesteps, len(inputs)):
    X_test.append(inputs[i-timesteps:i, 0])

# We turn it list to np.array
X_test = np.array(X_test)

# Then transform it's shape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# We give X_test to our model to predict predicted_coffe_sales
predicted_coffe_sales = model.predict(X_test)
# Inverse transform will transform it from (0,1) to it's original range
predicted_coffe_sales = scaler.inverse_transform(predicted_coffe_sales)

# We will compare Predicted Coffe Sales and Real Coffee Sales
real_coffe_sales = test.values

plt.plot(real_coffe_sales, color = 'red', label = 'Real Coffe Sales')
plt.plot(predicted_coffe_sales, color = 'blue', label = 'Predicted Coffe Sales')
plt.title('Coffee Sales Prediction For Last Month')
plt.xlabel('Days')
plt.ylabel('Coffee Sale per Day')
plt.legend()
plt.show()
