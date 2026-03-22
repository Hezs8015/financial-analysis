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


class BiLSTMModelV1(nn.Module):
    """BiLSTM模型 v1 (基础版)"""
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1, dropout=0.1):
        super(BiLSTMModelV1, self).__init__()
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


class BiLSTMModelV2(nn.Module):
    """BiLSTM模型 v2 (增强版)"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(BiLSTMModelV2, self).__init__()
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
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
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
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class BiLSTMModelV3(nn.Module):
    """BiLSTM模型 v3 (高级版)"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, output_size=1, dropout=0.3):
        super(BiLSTMModelV3, self).__init__()
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
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
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
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class TransformerModelV1(nn.Module):
    """Transformer模型 v1 (基础版)"""
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=1, dim_feedforward=128, output_size=1, dropout=0.1):
        super(TransformerModelV1, self).__init__()
        
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


class TransformerModelV2(nn.Module):
    """Transformer模型 v2 (增强版)"""
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, output_size=1, dropout=0.2):
        super(TransformerModelV2, self).__init__()
        
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
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 取序列最后一个时间步
        x = x[:, -1, :]
        x = self.dropout(x)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        return output


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
    """Transformer模型（优化版）"""
    def __init__(self, input_size, d_model=64, nhead=8, num_layers=2, dim_feedforward=256, output_size=1, dropout=0.2):
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
        
        # 选择数值列并清理数据
        data = df[available_cols].copy()
        
        # 将所有列转换为数值类型，无法转换的设为NaN
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 删除包含NaN的行
        data = data.dropna()
        
        # 转换为numpy数组
        data = data.values
        
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
    
    def train_model(self, model_name, model, X_train, y_train, X_val, y_val, X_test=None, y_test=None, epochs=50, batch_size=32, lr=0.001, early_stopping_patience=10, verbose=True, progress_callback=None):
        """训练模型（优化版）- 支持测试损失记录"""
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        history = {'train_loss': [], 'val_loss': [], 'test_loss': [], 'lr': []}
        has_test_data = X_test is not None and y_test is not None
        
        model.train()
        for epoch in range(epochs):
            train_losses = []
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs.squeeze(), y_val).item()
                
                test_loss = None
                if has_test_data:
                    test_outputs = model(X_test)
                    test_loss = criterion(test_outputs.squeeze(), y_test).item()
            
            avg_train_loss = np.mean(train_losses)
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['test_loss'].append(test_loss if test_loss is not None else 0.0)
            history['lr'].append(current_lr)
            
            scheduler.step()
            
            if progress_callback is not None:
                progress_callback(epoch + 1, epochs, avg_train_loss, val_loss, model_name)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if verbose and patience_counter % 3 == 0:
                    print(f"  {model_name}: 验证损失未改善 {patience_counter} 轮")
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"  {model_name}: 早停于第 {epoch + 1} 轮")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
            
            if verbose and (epoch + 1) % 5 == 0:
                test_info = f", Test Loss: {test_loss:.6f}" if test_loss is not None else ""
                print(f"  {model_name} Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}{test_info}, LR: {current_lr:.6f}")
            
            model.train()
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
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
        
        # 计算 MAPE (平均绝对百分比误差)
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-9))) * 100
        
        # 计算方向准确率
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actuals) > 0
        direction_accuracy = np.mean(pred_direction == actual_direction)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
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


class ARMAModel:
    """ARMA 模型 - 自回归移动平均模型"""
    def __init__(self, order=(1, 1)):
        from statsmodels.tsa.arima.model import ARIMA
        self.order = order
        self.model = None
        self.fitted = False
        
    def fit(self, y):
        from statsmodels.tsa.arima.model import ARIMA
        self.model = ARIMA(y, order=(self.order[0], 0, self.order[1]))
        self.model = self.model.fit()
        self.fitted = True
        return self
    
    def predict(self, steps=1):
        if not self.fitted:
            raise ValueError("模型尚未训练")
        return self.model.forecast(steps=steps)
    
    def predict_in_sample(self, start=0):
        if not self.fitted:
            raise ValueError("模型尚未训练")
        return self.model.predict(start=start)


class GARCHModel:
    """GARCH 模型 - 广义自回归条件异方差模型"""
    def __init__(self, p=1, q=1):
        from arch import arch_model
        self.p = p
        self.q = q
        self.model = None
        self.fitted = False
        
    def fit(self, y):
        from arch import arch_model
        self.model = arch_model(y, vol='Garch', p=self.p, q=self.q)
        self.model = self.model.fit(disp='off')
        self.fitted = True
        return self
    
    def predict(self, horizon=1):
        if not self.fitted:
            raise ValueError("模型尚未训练")
        forecast = self.model.forecast(horizon=horizon)
        return forecast.mean.values[-1] if hasattr(forecast, 'mean') else forecast
    
    def predict_in_sample(self):
        if not self.fitted:
            raise ValueError("模型尚未训练")
        return self.model.conditional_volatility


class TimeSeriesPredictor:
    """时间序列预测器 - 支持 ARMA 和 GARCH 模型"""
    def __init__(self):
        self.models = {}
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, df, target_col='Close'):
        """准备数据"""
        data = df[[target_col]].dropna().values
        return data
    
    def train_arma(self, model_name, data, order=(1, 1), test_size=0.2):
        """训练 ARMA 模型"""
        split = int(len(data) * (1 - test_size))
        train_data = data[:split]
        test_data = data[split:]
        
        model = ARMAModel(order=order)
        model.fit(train_data)
        
        # 预测
        train_pred = model.predict_in_sample()
        test_pred = []
        
        for i in range(len(test_data)):
            # 使用历史数据逐步预测
            hist_data = np.concatenate([train_data, test_data[:i]])
            temp_model = ARMAModel(order=order)
            temp_model.fit(hist_data)
            pred = temp_model.predict(steps=1)[0]
            test_pred.append(pred)
        
        test_pred = np.array(test_pred)
        
        # 计算指标
        train_metrics = self._calculate_metrics(train_data, train_pred)
        test_metrics = self._calculate_metrics(test_data, test_pred)
        
        self.models[model_name] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_actual': train_data,
            'test_actual': test_data
        }
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': test_pred,
            'actuals': test_data
        }
    
    def train_garch(self, model_name, data, p=1, q=1, test_size=0.2):
        """训练 GARCH 模型"""
        split = int(len(data) * (1 - test_size))
        train_data = data[:split]
        test_data = data[split:]
        
        # GARCH 模型通常用于收益率
        returns = np.diff(np.log(data + 1e-9)) * 100
        split_ret = int(len(returns) * (1 - test_size))
        train_ret = returns[:split_ret]
        test_ret = returns[split_ret:]
        
        model = GARCHModel(p=p, q=q)
        model.fit(train_ret)
        
        # 预测波动率
        train_vol = model.predict_in_sample()
        
        # 测试集预测
        test_pred = []
        for i in range(len(test_ret)):
            hist_ret = np.concatenate([train_ret, test_ret[:i]])
            temp_model = GARCHModel(p=p, q=q)
            temp_model.fit(hist_ret)
            pred = temp_model.predict(horizon=1)
            test_pred.append(pred)
        
        test_pred = np.array(test_pred)
        
        # 计算指标
        train_metrics = self._calculate_metrics(train_ret, train_vol)
        test_metrics = self._calculate_metrics(test_ret, test_pred)
        
        self.models[model_name] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_pred': train_vol,
            'test_pred': test_pred,
            'train_actual': train_ret,
            'test_actual': test_ret
        }
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': test_pred,
            'actuals': test_ret
        }
    
    def _calculate_metrics(self, actuals, predictions):
        """计算评估指标"""
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # MAPE
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-9))) * 100
        
        # 方向准确率
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actuals) > 0
        direction_accuracy = np.mean(pred_direction == actual_direction) if len(pred_direction) > 0 else 0
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2,
            'Direction_Accuracy': direction_accuracy
        }
