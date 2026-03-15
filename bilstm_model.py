import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout


class BiLSTMPredictor:
    def __init__(self, seq_len=30):
        self.seq_len = seq_len
        self.scaler = MinMaxScaler()
        self.model = None
        
    def prepare_data(self, data):
        scaled = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.seq_len, len(scaled)):
            X.append(scaled[i-self.seq_len:i])
            y.append(scaled[i])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(32), input_shape=input_shape),
            Dropout(0.2),
            Dense(input_shape[1])
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X_train, y_train, epochs=20, batch_size=32, verbose=1):
        if self.model is None:
            self.model = self.build_model(X_train.shape[1:])
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filepath):
        self.model.save(filepath)
    
    def load(self, filepath):
        self.model = load_model(filepath)
