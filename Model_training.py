import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:1000, :1800].values
Q = dataset.iloc[:1000, 1800:3600].values
Y = dataset.iloc[:1000, 3600:].values

os.makedirs('Scaler_parameter', exist_ok=True)
os.makedirs('Trained_model', exist_ok=True)

X_heating = []
X_reaction = []
T_reaction = []
X_cooling = []
Y_heating = Y[:, [11]]
Y_reaction = Y[:, 0:11]
Y_reaction = np.delete(Y_reaction, 5, axis=1)  # delete T_peak
Y_cooling = Y[:, 12:15]

temp_threshold = 69
target_length_cooling: int = 500
target_length_reaction: int = 1700
target_length_heating: int = 200


def pad_or_truncate(seq, length):
    if len(seq) > length:
        return seq[:length]
    elif len(seq) < length:
        return np.pad(seq, (0, length - len(seq)), mode='constant')
    else:
        return seq


for i in range(X.shape[0]):
    sequence = X[i]
    dX = np.diff(sequence)
    Q_sequence = Q[i]
    reaction_index = np.argmax(sequence >= temp_threshold)
    time_onset = np.argmax(dX >= 1)
    max_index = np.argmax(sequence)
    T_onset = sequence[time_onset]

    sequence_heating = sequence[:reaction_index + 1]
    sequence_heating = pad_or_truncate(sequence_heating, target_length_heating)

    sequence_cooling = sequence[max_index:]
    sequence_cooling = pad_or_truncate(sequence_cooling, target_length_cooling)

    sequence_reaction = sequence[reaction_index: max_index]
    sequence_reaction = pad_or_truncate(sequence_reaction, target_length_reaction)
    Q_sequence_reaction = Q_sequence[reaction_index: max_index]
    Q_sequence_reaction = np.log10(Q_sequence_reaction)
    Q_sequence_reaction = Q_sequence_reaction[Q_sequence_reaction >= 0]
    Q_sequence_reaction = pad_or_truncate(Q_sequence_reaction, target_length_reaction)
    mix_reaction = np.column_stack((sequence_reaction, Q_sequence_reaction))

    X_heating.append(sequence_heating)
    X_reaction.append(mix_reaction)
    X_cooling.append(sequence_cooling)
    T_reaction.append(T_onset)

T_reaction = np.array(T_reaction)
T_reaction = T_reaction.reshape(-1, 1)
X_reaction = np.array(X_reaction)

# X_train and Y_train for battery cooling stage
X_cooling_train, X_cooling_test, Y_cooling_train, Y_cooling_test = train_test_split(X_cooling, Y_cooling, test_size=0.2, shuffle=False)

scaler_X_cooling = MinMaxScaler(feature_range=(0, 1))
X_cooling_train = scaler_X_cooling.fit_transform(X_cooling_train)
X_cooling_test = scaler_X_cooling.transform(X_cooling_test)
X_cooling_predict = scaler_X_cooling.transform(X_cooling)  # used for predicting Y_cooling_predict

scaler_Y_cooling = MinMaxScaler(feature_range=(0, 1))
Y_cooling_train = scaler_Y_cooling.fit_transform(Y_cooling_train)
Y_cooling_test = scaler_Y_cooling.transform(Y_cooling_test)

X_cooling_train = X_cooling_train.reshape(X_cooling_train.shape[0], X_cooling_train.shape[1], 1)
X_cooling_test = X_cooling_test.reshape(X_cooling_test.shape[0], X_cooling_test.shape[1], 1)
X_cooling_predict = X_cooling_predict.reshape(X_cooling_predict.shape[0], X_cooling_predict.shape[1], 1)

if not os.path.exists('Trained_model/Model_cooling.h5'):

    model_cooling = tf.keras.Sequential((
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu',
                               input_shape=(target_length_cooling, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128, return_sequences=False),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='linear')
    ))

    model_cooling.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss='mean_squared_error')

    model_cooling.fit(X_cooling_train,
                      Y_cooling_train,
                      epochs=500,
                      batch_size=64,
                      validation_data=(X_cooling_test, Y_cooling_test),
                      shuffle=True,
                      validation_freq=1)

    model_cooling.summary()

    model_cooling.save('Trained_model/Model_cooling.h5')

else:
    model_cooling = tf.keras.models.load_model('Trained_model/Model_cooling.h5')

Y_cooling_predict_scaled = model_cooling.predict(X_cooling_predict)
Y_cooling_predict = scaler_Y_cooling.inverse_transform(Y_cooling_predict_scaled)

# Training Q_heat
X_heating_train, X_heating_test, Y_cooling_train, Y_cooling_test, Y_heating_train, Y_heating_test = train_test_split(
    X_heating, Y_cooling_predict_scaled, Y_heating, test_size=0.2, shuffle=False)

scaler_X_heating = MinMaxScaler(feature_range=(0, 1))
X_heating_train = scaler_X_heating.fit_transform(X_heating_train)
X_heating_test = scaler_X_heating.transform(X_heating_test)

scaler_Y_heating = MinMaxScaler(feature_range=(0, 1))
Y_heating_train = scaler_Y_heating.fit_transform(Y_heating_train)
Y_heating_test = scaler_Y_heating.transform(Y_heating_test)

X_heating_train = X_heating_train.reshape(X_heating_train.shape[0], X_heating_train.shape[1], 1)
X_heating_test = X_heating_test.reshape(X_heating_test.shape[0], X_heating_test.shape[1], 1)

if not os.path.exists('Trained_model/Model_heating.h5'):

    time_input = layers.Input(shape=(target_length_heating, 1))
    CNN1 = layers.Conv1D(64, 5, padding='same', activation='relu')(time_input)
    CNN1 = layers.MaxPooling1D(2)(CNN1)
    CNN2 = layers.Conv1D(128, 5, padding='same', activation='relu')(CNN1)
    CNN2 = layers.MaxPooling1D(2)(CNN2)
    CNN3 = layers.Conv1D(256, 5, padding='same', activation='relu')(CNN2)
    CNN3 = layers.MaxPooling1D(2)(CNN3)
    LSTM1 = layers.LSTM(64, return_sequences=True)(CNN3)
    LSTM2 = layers.LSTM(64, return_sequences=False)(LSTM1)

    features_input = layers.Input(shape=(3,))
    Dense1 = layers.Dense(32, activation='relu')(features_input)
    combined = layers.concatenate([LSTM2, Dense1])

    Dense2 = layers.Dense(64, activation='relu')(combined)
    Dense2 = layers.Dense(32, activation='relu')(Dense2)
    output = layers.Dense(1, activation='linear')(Dense2)
    model_heating = models.Model(inputs=[time_input, features_input], outputs=output)

    model_heating.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mean_squared_error')

    model_heating.fit(
        [X_heating_train, Y_cooling_train],
        Y_heating_train,
        epochs=500,
        validation_data=([X_heating_test, Y_cooling_test], Y_heating_test),
        batch_size=64,
        shuffle=True,
        validation_freq=1)

    model_heating.summary()

    model_heating.save('Trained_model/Model_heating.h5')

else:
    model_heating = tf.keras.models.load_model('Trained_model/Model_heating.h5')

Y_heating_predict_scaled = model_heating.predict([X_heating_train, Y_cooling_train])
Y_heating_predict = scaler_Y_heating.inverse_transform(Y_heating_predict_scaled)

# Training T_onset and dr
[X_reaction_train, X_reaction_test, T_reaction_train, T_reaction_test, Y_reaction_train,
 Y_reaction_test] = train_test_split(X_reaction, T_reaction, Y_reaction,
                                     test_size=0.2, shuffle=False)

X_reaction_scaled = np.zeros_like(X_reaction)
X_reaction_train_scaled = np.zeros_like(X_reaction_train)
X_reaction_test_scaled = np.zeros_like(X_reaction_test)
for i in range(2):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_feature = X_reaction_train[:, :, i].reshape(-1, 1)
    test_feature = X_reaction_test[:, :, i].reshape(-1, 1)

    train_feature_scaled = scaler.fit_transform(train_feature)
    test_feature_scaled = scaler.transform(test_feature)

    X_reaction_train[:, :, i] = train_feature_scaled.reshape(X_reaction_train.shape[0], X_reaction_train.shape[1])
    X_reaction_test[:, :, i] = test_feature_scaled.reshape(X_reaction_test.shape[0], X_reaction_test.shape[1])

    joblib.dump(scaler, f'Scaler_parameter/scaler_reaction_{i}.pkl')

scaler_T_reaction = MinMaxScaler(feature_range=(0, 1))
T_reaction_train = scaler_T_reaction.fit_transform(T_reaction_train)
T_reaction_test = scaler_T_reaction.transform(T_reaction_test)

scaler_Y_reaction = MinMaxScaler(feature_range=(0, 1))
Y_reaction_train = scaler_Y_reaction.fit_transform(Y_reaction_train)
Y_reaction_test = scaler_Y_reaction.transform(Y_reaction_test)

if not os.path.exists('Trained_model/Model_reaction.h5'):

    time_input = layers.Input(shape=(target_length_reaction, 2))
    CNN1 = layers.Conv1D(64, 5, padding='same', activation='relu')(time_input)
    CNN1 = layers.MaxPooling1D(2)(CNN1)
    CNN2 = layers.Conv1D(128, 5, padding='same', activation='relu')(CNN1)
    CNN2 = layers.MaxPooling1D(2)(CNN2)
    CNN3 = layers.Conv1D(256, 5, padding='same', activation='relu')(CNN2)
    CNN3 = layers.MaxPooling1D(2)(CNN3)

    LSTM1 = layers.LSTM(128, return_sequences=True)(CNN3)
    LSTM2 = layers.LSTM(128, return_sequences=False)(LSTM1)

    features_input = layers.Input(shape=(1,))
    dense_feature = layers.Dense(32, activation='relu')(features_input)
    combined = layers.concatenate([LSTM2, dense_feature])

    Dense1 = layers.Dense(128, activation='relu')(combined)
    Dense1 = layers.Dense(64, activation='linear')(Dense1)
    output = layers.Dense(10, activation='linear')(Dense1)
    model_reaction = models.Model(inputs=[time_input, features_input], outputs=output)

    model_reaction.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss='mean_squared_error')

    model_reaction.fit([X_reaction_train, T_reaction_train],
                       Y_reaction_train,
                       epochs=300,
                       batch_size=64,
                       validation_data=([X_reaction_test, T_reaction_test], Y_reaction_test),
                       shuffle=True,
                       validation_freq=1)

    model_reaction.summary()

    model_reaction.save('Trained_model/Model_reaction.h5')

else:
    model_reaction = tf.keras.models.load_model('Trained_model/Model_reaction.h5')

Y_reaction_predict_scaled = model_reaction.predict([X_reaction_train, T_reaction_train])
Y_reaction_predict = scaler_Y_reaction.inverse_transform(Y_reaction_predict_scaled)
Y_reaction_va = scaler_Y_reaction.inverse_transform(Y_reaction_train)
for i in range(50, 60):
    for j in range(0, 5):
        print("{:.2e}".format(Y_reaction_predict[i][j]), "{:.2e}".format(Y_reaction_va[i][j]))

joblib.dump(scaler_X_cooling, 'Scaler_parameter/scaler_X_cooling.pkl')
joblib.dump(scaler_Y_cooling, 'Scaler_parameter/scaler_Y_cooling.pkl')
joblib.dump(scaler_X_heating, 'Scaler_parameter/scaler_X_heating.pkl')
joblib.dump(scaler_Y_heating, 'Scaler_parameter/scaler_Y_heating.pkl')
joblib.dump(scaler_Y_reaction, 'Scaler_parameter/scaler_Y_reaction.pkl')
joblib.dump(scaler_T_reaction, 'Scaler_parameter/scaler_T_reaction.pkl')