"""
多模型对比系统
整合多个版本的BiLSTM和Transformer模型
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 尝试导入Keras
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("警告: Keras未安装，BiLSTM-Keras模型将不可用")


# ============== PyTorch 模型 ==============

class BiLSTMModelPyTorch(nn.Module):
    """PyTorch版BiLSTM"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(BiLSTMModelPyTorch, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out


class TransformerModelPyTorch(nn.Module):
    """PyTorch版Transformer"""
    def __init__(self, input_size, d_model=64, nhead=8, num_layers=2, dim_feedforward=256, output_size=1, dropout=0.2):
        super(TransformerModelPyTorch, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 输出层
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class GatedResidual(nn.Module):
    """门控残差：增强非线性表达，改善梯度流。"""
    def __init__(self, d_model: int):
        super(GatedResidual, self).__init__()
        self.fc   = nn.Linear(d_model, d_model * 2)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        gate, val = self.fc(self.norm(x)).chunk(2, dim=-1)
        return x + torch.sigmoid(gate) * val


class TransformerModelPyTorchV2(nn.Module):
    """PyTorch版Transformer v2（增强版）"""
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, output_size=1, dropout=0.1):
        super(TransformerModelPyTorchV2, self).__init__()
        self.d_model = d_model
        
        # 输入嵌入
        self.proj = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_enc = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True   # Pre-LN：训练更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 门控残差
        self.gate = GatedResidual(d_model)
        
        # 多尺度聚合：最后时间步 + 全局均值池化
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, output_size)
        )

    def _causal_mask(self, seq_len: int, device):
        """上三角掩码：每个时间步只能看到过去。"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, x):                          # (B, T, F)
        x = self.proj(x)
        x = self.pos_enc(x)
        mask = self._causal_mask(x.size(1), x.device)
        x = self.encoder(x, mask=mask)             # (B, T, d)
        x = self.gate(x)
        last = x[:, -1, :]                         # 最后时间步
        avg  = x.mean(dim=1)                       # 全局均值
        x    = torch.cat([last, avg], dim=-1)      # 多尺度
        return self.head(x).squeeze(-1)            # (B,)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============== Keras 模型 ==============

class BiLSTMModelKeras:
    """Keras版BiLSTM"""
    def __init__(self, seq_len, n_features, hidden_size=32):
        if not KERAS_AVAILABLE:
            raise ImportError("Keras未安装，无法使用BiLSTM-Keras模型")
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_model(self):
        """构建模型"""
        self.model = Sequential([
            Bidirectional(LSTM(self.hidden_size), input_shape=(self.seq_len, self.n_features)),
            Dropout(0.2),
            Dense(self.n_features)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        return self.model
    
    def train(self, X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=verbose
        )
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )
        return history
    
    def predict(self, X):
        """预测"""
        return self.model.predict(X, verbose=0)
    
    def save(self, filepath):
        """保存模型"""
        self.model.save(filepath)
    
    def load(self, filepath):
        """加载模型"""
        self.model = load_model(filepath)


# ============== 统一预测器 ==============

class MultiModelPredictor:
    """多模型统一预测器"""
    
    def __init__(self, seq_len=60, device='cpu'):
        self.seq_len = seq_len
        self.device = device
        self.models = {}
        self.histories = {}
        self.scalers = {}
        
    def prepare_data(self, df, feature_cols=None, target_col='Close'):
        """准备数据"""
        if feature_cols is None:
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        available_cols = [col for col in feature_cols if col in df.columns]
        if target_col not in df.columns:
            raise ValueError(f"目标列 '{target_col}' 不在数据中")
        
        data = df[available_cols].copy()
        
        # 转换为数值类型
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data = data.dropna()
        
        if len(data) < self.seq_len + 10:
            raise ValueError(f"数据量不足，需要至少 {self.seq_len + 10} 行")
        
        # 标准化
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # 创建序列
        X, y = [], []
        target_idx = available_cols.index(target_col) if target_col in available_cols else 3
        
        for i in range(self.seq_len, len(scaled_data)):
            X.append(scaled_data[i-self.seq_len:i])
            y.append(scaled_data[i, target_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, scaler, available_cols, target_idx
    
    def train_pytorch_model(self, model_name, model_class, X_train, y_train, X_val, y_val, 
                           model_kwargs, epochs=50, batch_size=32, lr=0.001, verbose=True, base_model=None):
        """训练PyTorch模型（支持增量训练）"""
        
        # 转换为Tensor
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        # 创建或加载模型
        if base_model is not None:
            model = base_model
        else:
            model = model_class(**model_kwargs).to(self.device)
        
        # 训练设置
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
        
        # 早停设置
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
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
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs.squeeze(), y_val_t).item()
            
            avg_train_loss = np.mean(train_losses)
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(current_lr)
            
            scheduler.step()
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    if verbose:
                        print(f"  {model_name}: 早停于第 {epoch + 1} 轮")
                    break
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  {model_name} Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.6f}, Val: {val_loss:.6f}")
            
            model.train()
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        self.models[model_name] = model
        self.histories[model_name] = history
        
        return history
    
    def train_keras_model(self, model_name, X_train, y_train, epochs=20, batch_size=32, verbose=0):
        """训练Keras模型"""
        if not KERAS_AVAILABLE:
            raise ImportError("Keras未安装")
        
        n_features = X_train.shape[2]
        model = BiLSTMModelKeras(self.seq_len, n_features)
        model.build_model()
        
        # 从训练集划分验证集
        val_size = int(0.1 * len(X_train))
        if val_size < 10:
            val_size = min(10, len(X_train) // 5)
        
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train_sub = X_train[:-val_size]
        y_train_sub = y_train[:-val_size]
        
        history = model.train(X_train_sub, y_train_sub, epochs=epochs, 
                             batch_size=batch_size, validation_split=0.1, verbose=verbose)
        
        self.models[model_name] = model
        self.histories[model_name] = {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
        
        return history
    
    def evaluate_model(self, model_name, X_test, y_test, target_idx=3, feature_cols=None):
        """评估模型"""
        if model_name not in self.models:
            raise ValueError(f"模型 '{model_name}' 未找到")
        
        model = self.models[model_name]
        
        if 'Keras' in model_name:
            # Keras模型
            predictions = model.predict(X_test).squeeze()
            actuals = y_test
        else:
            # PyTorch模型
            X_test_t = torch.FloatTensor(X_test).to(self.device)
            model.eval()
            with torch.no_grad():
                predictions = model(X_test_t).cpu().numpy().squeeze()
            actuals = y_test
        
        # 计算指标
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # 方向准确率
        pred_diff = np.diff(predictions)
        actual_diff = np.diff(actuals)
        direction_acc = np.mean((pred_diff > 0) == (actual_diff > 0))
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            '方向准确率': direction_acc
        }
        
        return metrics, predictions, actuals
    
    def compare_models(self, X_test, y_test):
        """对比所有模型"""
        results = {}
        
        for model_name in self.models.keys():
            try:
                metrics, predictions, actuals = self.evaluate_model(model_name, X_test, y_test)
                results[model_name] = {
                    'metrics': metrics,
                    'predictions': predictions,
                    'actuals': actuals
                }
            except Exception as e:
                print(f"评估 {model_name} 时出错: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def get_best_model(self, metric='方向准确率'):
        """获取最佳模型"""
        best_model = None
        best_score = -float('inf')
        
        for model_name, history in self.histories.items():
            if 'val_loss' in history and len(history['val_loss']) > 0:
                # 使用最后的验证损失作为指标
                score = -history['val_loss'][-1]  # 负损失，越小越好
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model


# ============== 便捷函数 ==============

def train_all_models(df, seq_len=60, epochs=30, feature_cols=None, target_col='Close', device='cpu'):
    """训练所有可用模型"""
    predictor = MultiModelPredictor(seq_len=seq_len, device=device)
    
    # 准备数据
    X, y, scaler, available_cols, target_idx = predictor.prepare_data(df, feature_cols, target_col)
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 划分验证集
    val_size = int(0.1 * len(X_train))
    if val_size < 10:
        val_size = min(10, len(X_train) // 5)
    
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_sub = X_train[:-val_size]
    y_train_sub = y_train[:-val_size]
    
    print(f"数据划分: 训练 {len(X_train_sub)}, 验证 {len(X_val)}, 测试 {len(X_test)}")
    
    results = {}
    
    # 1. BiLSTM-PyTorch
    print("\n训练 BiLSTM-PyTorch...")
    try:
        history = predictor.train_pytorch_model(
            'BiLSTM-PyTorch', BiLSTMModelPyTorch,
            X_train_sub, y_train_sub, X_val, y_val,
            model_kwargs={'input_size': len(available_cols), 'hidden_size': 64},
            epochs=epochs, batch_size=32, lr=0.001
        )
        results['BiLSTM-PyTorch'] = history
    except Exception as e:
        print(f"  失败: {str(e)}")
    
    # 2. Transformer-PyTorch
    print("\n训练 Transformer-PyTorch...")
    try:
        history = predictor.train_pytorch_model(
            'Transformer-PyTorch', TransformerModelPyTorch,
            X_train_sub, y_train_sub, X_val, y_val,
            model_kwargs={'input_size': len(available_cols), 'd_model': 64},
            epochs=epochs, batch_size=32, lr=0.001
        )
        results['Transformer-PyTorch'] = history
    except Exception as e:
        print(f"  失败: {str(e)}")
    
    # 3. BiLSTM-Keras
    if KERAS_AVAILABLE:
        print("\n训练 BiLSTM-Keras...")
        try:
            history = predictor.train_keras_model(
                'BiLSTM-Keras', X_train, y_train,
                epochs=epochs, batch_size=32
            )
            results['BiLSTM-Keras'] = history
        except Exception as e:
            print(f"  失败: {str(e)}")
    else:
        print("\n跳过 BiLSTM-Keras (Keras未安装)")
    
    # 对比结果
    print("\n" + "="*50)
    print("模型对比结果")
    print("="*50)
    
    comparison = predictor.compare_models(X_test, y_test)
    
    for model_name, result in comparison.items():
        if 'error' not in result:
            metrics = result['metrics']
            print(f"\n{model_name}:")
            print(f"  方向准确率: {metrics['方向准确率']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.6f}")
            print(f"  MAE: {metrics['MAE']:.6f}")
            print(f"  R²: {metrics['R²']:.4f}")
    
    best_model = predictor.get_best_model()
    print(f"\n🏆 最佳模型: {best_model}")
    
    return predictor, comparison, scaler


if __name__ == "__main__":
    # 测试代码
    print("多模型对比系统测试")
    print("="*50)
    
    # 生成测试数据
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    price = 100.0
    prices = []
    for i in range(1000):
        change = np.random.normal(0, 2)
        price = max(10, price + change)
        prices.append(price)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * 1.02 for p in prices],
        'Low': [p * 0.98 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 1000)
    })
    
    # 训练所有模型
    predictor, comparison, scaler = train_all_models(df, seq_len=30, epochs=10)
