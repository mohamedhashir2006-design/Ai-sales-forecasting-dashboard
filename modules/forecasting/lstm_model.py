import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


def train_lstm(series):
    # ==============================
    # 1. PREPARE DATA
    # ==============================
    data = series.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # ==============================
    # 2. CREATE SEQUENCES
    # ==============================
    sequence_length = 20  # 🔥 better than 10

    X, y = [], []

    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)

    # ==============================
    # 3. BUILD MODEL
    # ==============================
    model = Sequential()

    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))  # 🔥 reduces noise
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    # ==============================
    # 4. TRAIN MODEL
    # ==============================
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    # ==============================
    # 5. PREDICT
    # ==============================
    predictions_scaled = model.predict(X)

    # Convert back to original scale
    predictions = scaler.inverse_transform(predictions_scaled)

    # ==============================
    # 6. ALIGN WITH ORIGINAL LENGTH
    # ==============================
    # pad beginning with NaN so it matches original timeline
    padding = [np.nan] * sequence_length
    final_predictions = np.concatenate([padding, predictions.flatten()])

    return final_predictions