from data_preparation.series_to_supervised import series_to_supervised
from matplotlib import pyplot
from pandas import read_csv
import numpy as np
from numpy import concatenate
import os
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime
from pandas import DataFrame
from keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def lstm_multi(data_folder, train_file, file_name):
    model_name = file_name

    dataset = read_csv(os.path.join(data_folder, train_file))
    dataset.drop('Unnamed: 0', axis=1, inplace=True)
    dataset.index.name = 'date'

    print(dataset.head(5))

    del(dataset["Delivery_Date_week"])
    del(dataset["Delivery_year"])

    values = dataset.values

    # integer encode direction
    for i in range(3, 9):
        encoder = LabelEncoder()
        values[:, i] = encoder.fit_transform(values[:, i])

    # ensure all data is float
    values.astype('float32')

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, 2, 1)

    print(reframed.head())

    # split into train and test sets
    values = reframed.values
    n_train = 400000
    train = values[:n_train, :]
    test = values[n_train:, :]

    # split into input and outputs
    train_X, train_y = train[:, :20], train[:, 20:]
    test_X, test_y = test[:, :20], test[:, 20:]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(10))
    model.compile(loss='mae', optimizer='adam')

    # fit network
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    history = model.fit(train_X, train_y, epochs=15, batch_size=72, validation_data=(test_X, test_y),
                        verbose=2,
                        shuffle=False,
                        callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 20:]), axis=1)
    print(inv_yhat.shape, "inv_yhat")
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    #invert scaling for actual
    test_y = test_y.reshape((len(test_y), 10))
    inv_y = concatenate((test_y, test_X[:, 20:]), axis=1)
    print(inv_y.shape, "inv_y")
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    # calculate r2 score
    r2 = r2_score(test_y, yhat)
    print('Test r2 score: %.3f' % r2)
    print(f'{model_name}, test_r2_score = {r2}')

    with open(f'results/{model_name}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'test_r2_score': r2}, f)