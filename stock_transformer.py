"""
Stock Price Prediction with Transformer (PyTorch)
===================================================
使用 Transformer 对股票收盘价进行时序预测。

依赖:
    pip install torch numpy pandas yfinance scikit-learn matplotlib

用法:
    python stock_transformer.py
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 0. 超参数
# ──────────────────────────────────────────────
TICKER      = "AAPL"        # 股票代码
SEQ_LEN     = 60            # 输入序列长度（天）
PRED_LEN    = 1             # 预测步数
FEATURES    = ["Close", "Volume", "Open", "High", "Low"]  # 特征列
TARGET      = "Close"       # 预测目标

D_MODEL     = 64            # Transformer 隐层维度
N_HEADS     = 4             # 多头注意力头数
N_LAYERS    = 2             # Encoder 层数
D_FF        = 256           # 前馈网络维度
DROPOUT     = 0.1

BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
SPLIT       = 0.8           # 训练集比例
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────
# 1. 下载数据
# ──────────────────────────────────────────────
def load_data(ticker: str, start: str = "2018-01-01", end: str = "2024-01-01") -> pd.DataFrame:
    """下载雅虎财经历史数据，失败时生成模拟数据。"""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError("empty")
        # yfinance 返回 MultiIndex columns，展平
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        print(f"Downloaded {len(df)} rows from Yahoo Finance.")
        return df[FEATURES].dropna()
    except Exception as e:
        print(f"yfinance unavailable ({e}), generating synthetic data...")
        return _synthetic_data()

def _synthetic_data(n: int = 1500) -> pd.DataFrame:
    """生成合成 OHLCV 数据，用于离线测试。"""
    np.random.seed(42)
    price = 150.0
    closes, opens, highs, lows, vols = [], [], [], [], []
    for _ in range(n):
        ret    = np.random.normal(0.0003, 0.015)
        price *= (1 + ret)
        noise  = price * 0.005
        opens.append(price + np.random.uniform(-noise, noise))
        highs.append(price + abs(np.random.normal(0, noise)))
        lows.append(price  - abs(np.random.normal(0, noise)))
        closes.append(price)
        vols.append(np.random.randint(5_000_000, 50_000_000))
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Close": closes, "Volume": vols, "Open": opens, "High": highs, "Low": lows},
        index=dates
    )

# ──────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────
class StockDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, target_idx: int):
        """
        data      : (T, F) scaled numpy array
        seq_len   : 滑动窗口长度
        target_idx: 目标列索引
        """
        self.data       = torch.tensor(data, dtype=torch.float32)
        self.seq_len    = seq_len
        self.target_idx = target_idx

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]           # (seq_len, F)
        y = self.data[idx + self.seq_len, self.target_idx] # scalar
        return x, y

# ──────────────────────────────────────────────
# 3. 位置编码
# ──────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):          # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

# ──────────────────────────────────────────────
# 4. Transformer 模型
# ──────────────────────────────────────────────
class StockTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head       = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):           # x: (B, T, F)
        x = self.input_proj(x)      # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)         # (B, T, d_model)
        x = x[:, -1, :]             # 取最后时间步
        return self.head(x).squeeze(-1)  # (B,)

# ──────────────────────────────────────────────
# 5. 训练 / 评估工具
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item() * len(x)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total_loss / len(loader.dataset), preds, trues

# ──────────────────────────────────────────────
# 6. 主流程
# ──────────────────────────────────────────────
def main():
    # ── 数据准备 ──────────────────────────────
    df = load_data(TICKER)
    raw = df[FEATURES].values.astype(np.float32)

    split_idx = int(len(raw) * SPLIT)
    train_raw = raw[:split_idx]
    test_raw  = raw[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)

    target_idx = FEATURES.index(TARGET)

    train_ds = StockDataset(train_scaled, SEQ_LEN, target_idx)
    test_ds  = StockDataset(test_scaled,  SEQ_LEN, target_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ── 模型 ──────────────────────────────────
    model = StockTransformer(
        n_features=len(FEATURES),
        d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF,
        dropout=DROPOUT
    ).to(DEVICE)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    # ── 训练循环 ──────────────────────────────
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        va_loss, _, _ = eval_epoch(model, test_loader, criterion, DEVICE)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), "best_stock_transformer.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | train={tr_loss:.6f} | val={va_loss:.6f}")

    # ── 测试集评估 ────────────────────────────
    model.load_state_dict(torch.load("best_stock_transformer.pt", map_location=DEVICE))
    _, preds_scaled, trues_scaled = eval_epoch(model, test_loader, criterion, DEVICE)

    # 反归一化（只还原 target 列）
    dummy = np.zeros((len(preds_scaled), len(FEATURES)), dtype=np.float32)
    dummy[:, target_idx] = preds_scaled
    preds_price = scaler.inverse_transform(dummy)[:, target_idx]

    dummy[:, target_idx] = trues_scaled
    trues_price = scaler.inverse_transform(dummy)[:, target_idx]

    mae  = np.mean(np.abs(preds_price - trues_price))
    rmse = np.sqrt(np.mean((preds_price - trues_price) ** 2))
    mape = np.mean(np.abs((preds_price - trues_price) / trues_price)) * 100
    print(f"\nTest  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")

    # ── 可视化 ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses,   label="Val Loss")
    axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(trues_price, label="Actual",    alpha=0.8)
    axes[1].plot(preds_price, label="Predicted", alpha=0.8)
    axes[1].set_title(f"{TICKER} Price Prediction (Test Set)")
    axes[1].legend(); axes[1].set_xlabel("Day")

    plt.tight_layout()
    plt.savefig("stock_transformer_result.png", dpi=150)
    print("Figure saved → stock_transformer_result.png")
    plt.show()


if __name__ == "__main__":
    main()
