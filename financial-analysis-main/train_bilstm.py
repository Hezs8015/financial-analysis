import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from keras.models import Sequential, load_model
    from keras.layers import Bidirectional, LSTM, Dense, Dropout
except ImportError:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout

np.random.seed(42)
data = np.random.randn(1500, 5).cumsum(axis=0)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

seq_len = 30
X, y = [], []
for i in range(seq_len, len(scaled)):
    X.append(scaled[i-seq_len:i])
    y.append(scaled[i])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

def get_acc(model):
    pred = model.predict(X_test, verbose=0)
    pred_cls = (pred > 0.5).astype(int)
    true_cls = (y_test > 0.5).astype(int)
    return round(accuracy_score(true_cls.flatten(), pred_cls.flatten()), 4)

def build_model():
    model = Sequential([
        Bidirectional(LSTM(32), input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(y.shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

print("训练 v1...")
m1 = build_model()
m1.fit(X_train, y_train, batch_size=16, epochs=8, verbose=0)
m1.save("bilstm_v1.h5")
acc1 = get_acc(m1)

print("训练 v2...")
m2 = load_model("bilstm_v1.h5")
m2.fit(X_train, y_train, batch_size=16, epochs=12, verbose=0)
m2.save("bilstm_v2.h5")
acc2 = get_acc(m2)

print("训练 v3...")
m3 = load_model("bilstm_v2.h5")
m3.fit(X_train, y_train, batch_size=16, epochs=15, verbose=0)
m3.save("bilstm_v3.h5")
acc3 = get_acc(m3)

print("\nBiLSTM 三轮迭代真实准确率：")
print(f"v1: {acc1:.4f}")
print(f"v2: {acc2:.4f}")
print(f"v3: {acc3:.4f}")
