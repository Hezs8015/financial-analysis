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


def add_technical_indicators(df):
    """添加技术指标"""
    df = df.copy()
    
    # 1. 移动平均线 (Moving Averages)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # 价格与均线的关系（趋势指标）
    df['Close_MA5_ratio'] = df['Close'] / df['MA5']
    df['Close_MA20_ratio'] = df['Close'] / df['MA20']
    df['MA5_MA20_ratio'] = df['MA5'] / df['MA20']
    
    # 2. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # 4. 布林带 (Bollinger Bands)
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # 5. 价格波动率
    df['Price_range'] = (df['High'] - df['Low']) / df['Close']
    df['Price_change'] = df['Close'].pct_change()
    df['Volatility_5'] = df['Price_change'].rolling(window=5).std()
    df['Volatility_20'] = df['Price_change'].rolling(window=20).std()
    
    # 6. 成交量指标
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA20']
    
    # 7. 价格动量
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    # 8. 日内价格位置
    df['Price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    # 删除NaN值
    df = df.dropna()
    
    return df


class BiLSTMModel(nn.Module):
    """增强版双向LSTM模型，带有注意力机制"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, output_size=1, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # 多层BiLSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 自注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)
        
        # LSTM输出: (batch, seq_len, hidden_size * 2)
        lstm_out, (hidden, cell) = self.bilstm(x)
        
        # 自注意力
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后时刻的输出
        final_output = attn_out[:, -1, :]
        
        # 全连接层
        output = self.fc_layers(final_output)
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
    """增强版Transformer模型"""
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, output_size=1, dropout=0.3):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影和层归一化
        self.input_projection = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 全局平均池化 + 最后时刻
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # 输出层（更深的网络）
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 取序列最后一个时间步
        last_output = x[:, -1, :]  # (batch, d_model)
        
        # 全局平均池化
        pooled = self.pooling(x.transpose(1, 2)).squeeze(-1)  # (batch, d_model)
        
        # 合并特征
        combined = torch.cat([last_output, pooled], dim=1)  # (batch, d_model * 2)
        
        # 全连接层
        output = self.fc_layers(combined)
        return output


class StockPredictor:
    """股票预测器 - 整合数据处理和模型训练"""
    def __init__(self, seq_length=60, device=None):
        self.seq_length = seq_length
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}
        self.histories = {}
        
    def prepare_data(self, df, target_col='Close', feature_cols=None, train_split=0.8, use_technical_indicators=True):
        """准备数据"""
        # 添加技术指标
        if use_technical_indicators:
            df = add_technical_indicators(df)
            # 使用所有可用的特征列
            feature_cols = [col for col in df.columns if col not in ['Date', target_col]]
        
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
    
    def train_model(self, model_name, model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, lr=0.001, early_stopping_patience=15):
        """训练模型（带早停和学习率调度）"""
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # 早停设置
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        model.train()
        for epoch in range(epochs):
            # 训练阶段
            train_losses = []
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs.squeeze(), y_val).item()
            
            avg_train_loss = np.mean(train_losses)
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(current_lr)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"早停于第 {epoch + 1} 轮")
                    # 恢复最佳模型
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
            
            model.train()
        
        # 确保使用最佳模型
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
        
        with torch.no_grad():
            for _ in range(days):
                # 准备输入
                x = torch.FloatTensor(current_sequence[-self.seq_length:]).unsqueeze(0).to(self.device)
                
                # 预测
                pred = model(x).cpu().numpy().squeeze()
                predictions.append(pred)
                
                # 更新序列 (假设单变量预测)
                new_step = current_sequence[-1].copy()
                new_step[0] = pred  # 假设第一列是目标变量
                current_sequence = np.vstack([current_sequence, new_step])
        
        predictions = np.array(predictions)
        
        # 反标准化
            if inverse_transform:
                # 创建一个与原始数据相同形状的数组，填充预测值
                n_features = last_sequence.shape[1]
                pred_full = np.zeros((len(predictions), n_features))
                pred_full[:, 0] = predictions  # 假设第一列是目标变量
                
                # 反标准化
                pred_original = self.scaler.inverse_transform(pred_full)
                predictions = pred_original[:, 0]
            
            return predictions
    
    def save_model(self, model_name, save_dir='saved_models'):
        """保存模型和训练历史"""
        if model_name not in self.models:
            raise ValueError(f"模型 '{model_name}' 未找到")
        
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存模型参数
        model_path = os.path.join(save_dir, f'{model_name}_model.pth')
        torch.save(self.models[model_name].state_dict(), model_path)
        
        # 保存标准化器
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler.npy')
        np.save(scaler_path, {
            'min': self.scaler.min_,
            'scale': self.scaler.scale_,
            'data_min': self.scaler.data_min_,
            'data_max': self.scaler.data_max_
        })
        
        # 保存训练历史
        history_path = os.path.join(save_dir, f'{model_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.histories[model_name], f)
        
        return model_path, scaler_path, history_path
    
    def load_model(self, model_name, model_class, model_params, save_dir='saved_models'):
        """加载已保存的模型"""
        model_path = os.path.join(save_dir, f'{model_name}_model.pth')
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler.npy')
        history_path = os.path.join(save_dir, f'{model_name}_history.json')
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            return False, f"模型文件不存在: {model_path}"
        
        # 创建模型并加载参数
        model = model_class(**model_params).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        self.models[model_name] = model
        
        # 加载标准化器
        if os.path.exists(scaler_path):
            scaler_data = np.load(scaler_path, allow_pickle=True).item()
            self.scaler.min_ = scaler_data['min']
            self.scaler.scale_ = scaler_data['scale']
            self.scaler.data_min_ = scaler_data['data_min']
            self.scaler.data_max_ = scaler_data['data_max']
        
        # 加载训练历史
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.histories[model_name] = json.load(f)
        
        return True, f"模型 '{model_name}' 加载成功"
    
    def get_training_summary(self):
        """获取训练摘要"""
        summary = []
        for model_name in self.models.keys():
            if model_name in self.histories:
                history = self.histories[model_name]
                final_train_loss = history['train_loss'][-1] if history['train_loss'] else None
                final_val_loss = history['val_loss'][-1] if history['val_loss'] else None
                summary.append({
                    'model_name': model_name,
                    'epochs_trained': len(history['train_loss']),
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss
                })
        return summary
