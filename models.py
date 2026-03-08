import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import os
import json
from datetime import datetime


class BiLSTMModel(nn.Module):
    """双向LSTM模型"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向LSTM
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM输出: (batch, seq_len, hidden_size * 2)
        out, _ = self.bilstm(x, (h0, c0))
        
        # 取最后一个时间步
        out = out[:, -1, :]
        out = self.dropout(out)
        
        # 全连接层
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):
    """Transformer位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """Transformer模型"""
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=2, dim_feedforward=512, output_size=1, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 取序列最后一个时间步
        x = x[:, -1, :]
        x = self.dropout(x)
        
        output = self.fc(x)
        return output


class StockPredictor:
    """股票预测器 - 整合数据处理和模型训练"""
    def __init__(self, seq_length=60, device=None):
        self.seq_length = seq_length
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}
        self.histories = {}
        
    def prepare_data(self, df, target_col='Close', feature_cols=None, train_split=0.8):
        """准备数据"""
        if feature_cols is None:
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # 确保所有列都存在
        available_cols = [col for col in feature_cols if col in df.columns]
        if target_col not in df.columns:
            raise ValueError(f"目标列 '{target_col}' 不在数据中")
        
        data = df[available_cols].values
        
        # 标准化
        scaled_data = self.scaler.fit_transform(data)
        
        # 创建序列
        X, y = [], []
        target_idx = available_cols.index(target_col) if target_col in available_cols else 0
        
        for i in range(self.seq_length, len(scaled_data)):
            X.append(scaled_data[i-self.seq_length:i])
            y.append(scaled_data[i, target_idx])
        
        X, y = np.array(X), np.array(y)
        
        # 划分训练集和测试集
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 转换为Tensor
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        return X_train, y_train, X_test, y_test, available_cols
    
    def train_model(self, model_name, model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001):
        """训练模型"""
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        history = {'train_loss': [], 'val_loss': []}
        
        model.train()
        for epoch in range(epochs):
            train_losses = []
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs.squeeze(), y_val).item()
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            model.train()
        
        self.models[model_name] = model
        self.histories[model_name] = history
        
        return history
    
    def evaluate_model(self, model_name, X_test, y_test, inverse_transform=True):
        """评估模型"""
        if model_name not in self.models:
            raise ValueError(f"模型 '{model_name}' 未找到")
        
        model = self.models[model_name]
        model.eval()
        
        with torch.no_grad():
            predictions = model(X_test).cpu().numpy().squeeze()
            actuals = y_test.cpu().numpy()
        
        # 计算指标（在标准化空间中计算）
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # 计算方向准确率
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actuals) > 0
        direction_accuracy = np.mean(pred_direction == actual_direction)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Direction_Accuracy': direction_accuracy
        }
        
        # 反标准化
        if inverse_transform:
            n_features = X_test.shape[2]  # 特征数量
            pred_full = np.zeros((len(predictions), n_features))
            pred_full[:, 0] = predictions  # 假设第一列是目标变量
            
            actual_full = np.zeros((len(actuals), n_features))
            actual_full[:, 0] = actuals
            
            # 反标准化
            pred_original = self.scaler.inverse_transform(pred_full)
            actual_original = self.scaler.inverse_transform(actual_full)
            
            predictions = pred_original[:, 0]
            actuals = actual_original[:, 0]
        
        return metrics, predictions, actuals
    
    def predict_future(self, model_name, last_sequence, days=30, inverse_transform=True):
        """预测未来"""
        if model_name not in self.models:
            raise ValueError(f"模型 '{model_name}' 未找到")
        
        model = self.models[model_name]
        model.eval()
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # 转换为Tensor
            seq_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                pred = model(seq_tensor).cpu().numpy().item()
            
            # 添加预测结果到序列
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred  # 假设第一列是目标变量
            
            predictions.append(pred)
        
        # 反标准化
        if inverse_transform:
            n_features = last_sequence.shape[1]
            pred_full = np.zeros((len(predictions), n_features))
            pred_full[:, 0] = predictions
            
            predictions = self.scaler.inverse_transform(pred_full)[:, 0]
        
        return predictions
    
    def save_model(self, model_name, save_dir='saved_models'):
        """保存模型"""
        if model_name not in self.models:
            raise ValueError(f"模型 '{model_name}' 未找到")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存模型
        model_path = os.path.join(save_dir, f'{model_name}_model.pth')
        torch.save(self.models[model_name].state_dict(), model_path)
        
        # 保存scaler
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler.npy')
        np.save(scaler_path, self.scaler.scale_)
        np.save(os.path.join(save_dir, f'{model_name}_scaler_min.npy'), self.scaler.min_)
        
        # 保存历史
        history_path = os.path.join(save_dir, f'{model_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.histories[model_name], f)
        
        print(f"模型 '{model_name}' 已保存到 {save_dir}")
    
    def load_model(self, model_name, save_dir='saved_models'):
        """加载模型"""
        # 加载模型
        model_path = os.path.join(save_dir, f'{model_name}_model.pth')
        if not os.path.exists(model_path):
            raise ValueError(f"模型文件 '{model_path}' 不存在")
        
        # 加载scaler
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler.npy')
        if not os.path.exists(scaler_path):
            raise ValueError(f"Scaler文件 '{scaler_path}' 不存在")
        
        # 加载历史
        history_path = os.path.join(save_dir, f'{model_name}_history.json')
        if not os.path.exists(history_path):
            raise ValueError(f"历史文件 '{history_path}' 不存在")
        
        # 重建模型
        if model_name == 'BiLSTM':
            model = BiLSTMModel(input_size=5)
        elif model_name == 'Transformer':
            model = TransformerModel(input_size=5)
        else:
            raise ValueError(f"未知模型类型 '{model_name}'")
        
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        
        # 重建scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.scale_ = np.load(scaler_path)
        scaler.min_ = np.load(os.path.join(save_dir, f'{model_name}_scaler_min.npy'))
        
        # 加载历史
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        self.models[model_name] = model
        self.scaler = scaler
        self.histories[model_name] = history
        
        print(f"模型 '{model_name}' 已加载")
        return model
