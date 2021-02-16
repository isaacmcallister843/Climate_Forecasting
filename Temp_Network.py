# ------------------------------ Sources:
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# https://www.rs-online.com/designspark/predicting-weather-using-lstm
# https://machinelearningmastery.com/exploding-gradients-in-neural-networks/
# TODO: Maybe add a csv with progress reports? like accuracy of each prediction +
# size of validation set?
# Also need todo the entire gam thing but no worries should be easy...
# ------------------------------ Libraries
# ------------- Data Prep and Processing
import pandas as pd
import numpy as np
from numpy import array
from numpy import hstack
from sklearn.preprocessing import MinMaxScaler
# ------------- Tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
# ------------- Corrections
from sklearn.ensemble import RandomForestRegressor
# -------------- Plotting
import matplotlib.pyplot
# -------------- Os and Pathing
import os
# -------------- Stats
import statistics
from statistics import mode

# ------------------------------ Parameters
epoc = 80
long_mem = 100
n_steps_out = 7
n_features = 1
modeltrain = False
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
sc1 = MinMaxScaler(feature_range=(0, 1))
# Directory for project
os.chdir("\\Users\\dmcallister\\Desktop\\Past Projects\\Temp_mapping")
# Location of Temp_data, need a column V2 with temperature values
path = 'Temp_data'

# ------------------------------ Modeling
# ------------- Split Function
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# ------------- Creating Model
if modeltrain:
    # ------------- Training set
    training = pd.read_csv("Temp_Network_Sup/training_temp.csv")
    training_list = list(training["Temp"].values)

    # -------------- Transform
    training_list = sc1.fit_transform(array(training_list).reshape(len(training_list), 1))

    # ------------- Splitting
    X_train, y_train = split_sequence(training_list, long_mem, n_steps_out)

    # ------------------------------ The Model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(long_mem, n_features)))
    model.add(LSTM(50, activation='relu', input_shape=(long_mem, n_features)))
    # model.add(tf.keras.layers.RepeatVector(n_steps_out))
    # model.add(LSTM(90, activation='relu', input_shape=(long_mem, n_features)))
    model.add(Dense(units=30, activation="relu"))
    model.add(Dense(units=n_steps_out))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=.5, clipvalue=0.5)
    model.compile(optimizer=sgd, loss='mse')
    model.fit(X_train, y_train, epochs=epoc, batch_size=32)

    # ------------- Save the model
    model.save('Temp_Network_Sup/Models')
else:
    model = keras.models.load_model('Temp_Network_Sup/Models')
    model.summary()

# ------------------------------ Model Implimentation

def predict_future(csv, name, graph=False, return_model=False, iterations=30):
    test_list = list(csv["V2"].values)
    test_list = sc1.fit_transform(array(test_list).reshape(len(test_list), 1))
    y_test_grad = test_list[long_mem:]

    # ------------- Graduated Learning / Creating and Training Validation Set
    num_steps = test_list[long_mem:].shape[0] // n_steps_out
    grad_vec = test_list[:long_mem]
    input_vec = grad_vec
    for x in range(num_steps+1):
        y_pred = model.predict(array([input_vec]))
        y_pred = y_pred.reshape(y_pred.shape[1], 1)
        grad_vec = np.vstack((grad_vec, y_pred))
        input_vec = grad_vec[(len(grad_vec) - long_mem):]

    grad_vec = grad_vec[long_mem:(len(y_test_grad)+long_mem)]

    # ------------- Random Forest Correction
    # n_estimator is the number of trees
    regressor = RandomForestRegressor(n_estimators=30, random_state=0)
    regressor.fit(grad_vec, y_test_grad.ravel())

# ------------------------------ Long Predictions
    # ------------- Long Term Prediction
    future_vec = test_list[:long_mem]
    input_vec = future_vec
    for i in range(1, iterations):
        y_pred = model.predict(array([input_vec]))
        y_pred = y_pred[0].reshape(y_pred.shape[1], 1)
        future_vec = np.vstack((future_vec, y_pred))
        input_vec = future_vec[(len(future_vec) - long_mem):]
    future_vec = future_vec[long_mem:]
    random_forest = regressor.predict(future_vec)

    # ------------- Check data to make sure the regressor predicted correctly
    count = 0
    for item in random_forest:
        if item == mode(random_forest):
            count = count + 1
        if count > int(iterations / 1.8):
            print("Too many repeat values using default regressor")
            random_forest = defualt.predict(future_vec)
            break

    # ------------- Data Transformation
    randomforest = sc1.inverse_transform(random_forest.reshape(random_forest.shape[0], 1))

    # ------------- Plotting
    if graph:
        # ------------- Data For plotting
        future_vec = sc1.inverse_transform(future_vec)
        future_pred_flat = [float(item) for item in future_vec]
        future_randomforest_flat = [float(item) for sublist in randomforest for item in sublist]

        y_test_grad = sc1.inverse_transform(y_test_grad)
        data = [float(item) for sublist in y_test_grad for item in sublist]

        # ------------- Generate a date list
        datelist = []
        for i in range(len(future_pred_flat)):
            datelist.append(i)

        matplotlib.pyplot.scatter(datelist, future_pred_flat, label="LTSM Prediction")
        matplotlib.pyplot.scatter(datelist, future_randomforest_flat, label="Random Forest Correction")
        matplotlib.pyplot.scatter(datelist[0:len(data)], data, label = "Data")
        matplotlib.pyplot.legend(loc='upper left', frameon=False)
        matplotlib.pyplot.title('Temperature Predictions - Future')
        matplotlib.pyplot.xlabel('Time')
        matplotlib.pyplot.ylabel('Temperature (C)')
        matplotlib.pyplot.show()

    # --------------- Output Selection
    if not return_model:
        # -------- Create a new dataframe and write to file
        name_path = "Predicted Data/" + "Predicted " + name
        pd.DataFrame(randomforest[(len(test_list) - long_mem):]).to_csv(name_path)
    else:
        # -------- Return the model
        return regressor

# -------------------------- Full Implementation
# ----------- Initialization / get defualt regressor function
max_len=0
for filename in os.listdir(path):
    if filename.endswith('.csv'):
        DataFrame = pd.read_csv(path + "/" + filename)
        if len(DataFrame.index) > max_len:
            longest_frame = DataFrame
            max_len = len(DataFrame.index)

defualt = predict_future(longest_frame, filename, return_model=True)

# ----------- Implementation

for filename in os.listdir(path):
    if filename.endswith('.csv'):
        DataFrame = pd.read_csv(path + "/" + filename)
        if len(DataFrame.index)+1 > (long_mem * 1.10):
            print(filename)
            predict_future(pd.read_csv(path + "/" + filename), filename, iterations=29)
        else:
            print("not enough data for " + filename)

# CHECK THIS SHOULD BE ENOUGH DATA
predict_future(pd.read_csv(path + "/" + "Temperature 1925.csv"), "Temperature 1925.csv", iterations=100, graph=True)
