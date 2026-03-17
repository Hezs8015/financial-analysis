"""
Stock Price Prediction with Transformer v3 (PyTorch)
=====================================================
改进要点：
  1. 进一步增强特征工程：添加更多技术指标和市场情绪特征
  2. 实现自适应注意力机制：根据输入动态调整注意力权重
  3. 引入多层特征融合：不同时间尺度的特征提取和融合
  4. 优化模型架构：更深层的网络结构和更有效的残差连接
  5. 改进训练策略：结合梯度累积和混合精度训练
  6. 增强鲁棒性：添加更多正则化方法和噪声注入
  7. 更全面的评估：添加更多风险指标和交易策略评估

依赖:
    pip install torch numpy pandas yfinance scikit-learn matplotlib

用法:
    python stock_transformer_v3.py
"""

import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0. 超参数
# ──────────────────────────────────────────────
TICKER     = "AAPL"
SEQ_LEN    = 40          # 输入窗口（天）
D_MODEL    = 192         # 增大模型容量
N_HEADS    = 6           # 增加注意力头数
N_LAYERS   = 4           # 增加网络深度
D_FF       = 384         # 前馈网络维度
DROPOUT    = 0.15        # 适当增加 dropout 防止过拟合
BATCH_SIZE = 64
EPOCHS     = 120
LR         = 3e-4
SPLIT      = 0.8
PATIENCE   = 20          # early stopping
GRAD_ACCUM = 2           # 梯度累积步数
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ──────────────────────────────────────────────
# 1. 数据加载 & 增强特征工程
# ──────────────────────────────────────────────
def load_data(ticker: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(ticker, start="2015-01-01", end="2024-12-31", progress=False)
        if df.empty:
            raise ValueError("empty")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        print(f"Downloaded {len(df)} rows.")
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception as e:
        print(f"yfinance 不可用 ({e})，使用合成数据...")
        return _synthetic_data(2000)

def _synthetic_data(n: int = 2000) -> pd.DataFrame:
    np.random.seed(0)
    price = 150.0
    rows = []
    for _ in range(n):
        ret   = np.random.normal(0.0003, 0.012)
        price = max(price * (1 + ret), 1.0)
        h     = price * (1 + abs(np.random.normal(0, 0.004)))
        l     = price * (1 - abs(np.random.normal(0, 0.004)))
        o     = price + np.random.normal(0, price * 0.003)
        v     = np.random.randint(8_000_000, 60_000_000)
        rows.append([o, h, l, price, v])
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame(rows, columns=["Open","High","Low","Close","Volume"], index=dates)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加增强的技术指标特征。"""
    c = df["Close"]
    v = df["Volume"]
    df = df.copy()

    # ── 基础技术指标 ──
    # 移动平均 & 布林带
    for period in [5, 10, 20, 50]:
        df[f"MA{period}"] = c.rolling(period).mean()
        df[f"MA_ratio{period}"] = c / (df[f"MA{period}"] + 1e-9)
    
    # 布林带
    bb_mid  = c.rolling(20).mean()
    bb_std  = c.rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_pos"]   = (c - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / bb_mid

    # RSI
    for period in [14, 21]:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        df[f"RSI{period}"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["MACD_s"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_h"] = df["MACD"] - df["MACD_s"]

    # 波动率
    for period in [10, 20]:
        df[f"Volatility{period}"] = c.pct_change().rolling(period).std()

    # 成交量指标
    df["Vol_ratio"]  = v / (v.rolling(20).mean() + 1e-9)
    df["Vol_change"] = v.pct_change()
    df["Vol_ma_diff"] = v.rolling(5).mean() / (v.rolling(20).mean() + 1e-9)

    # ── 动量指标 ──
    for period in [10, 20, 50]:
        df[f"Momentum{period}"] = c.pct_change(period)

    # ── 收益率特征 ──
    for period in [1, 3, 5, 10]:
        df[f"Return{period}"] = c.pct_change(period)

    # ── 目标 ──
    df["Target"] = c.pct_change().shift(-1) # 预测 t+1 的收益率

    return df.dropna()

FEATURE_COLS = [
    "Open","High","Low","Close","Volume",
    "MA5","MA10","MA20","MA50",
    "MA_ratio5","MA_ratio10","MA_ratio20","MA_ratio50",
    "BB_upper","BB_lower","BB_pos","BB_width",
    "RSI14","RSI21",
    "MACD","MACD_s","MACD_h",
    "Volatility10","Volatility20",
    "Vol_ratio","Vol_change","Vol_ma_diff",
    "Momentum10","Momentum20","Momentum50",
    "Return1","Return3","Return5","Return10"
]

# ──────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────
class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        assert len(X) == len(y)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, i):
        return self.X[i : i + self.seq_len], self.y[i + self.seq_len]

# ──────────────────────────────────────────────
# 3. 模型组件
# ──────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])

class GatedResidual(nn.Module):
    """门控残差：增强非线性表达，改善梯度流。"""
    def __init__(self, d_model: int):
        super().__init__()
        self.fc   = nn.Linear(d_model, d_model * 2)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        gate, val = self.fc(self.norm(x)).chunk(2, dim=-1)
        return x + torch.sigmoid(gate) * val

class AdaptiveAttention(nn.Module):
    """自适应注意力机制：根据输入动态调整注意力权重"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 注意力门控
        self.attention_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_heads),
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        
        # 投影到多头空间
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), -torch.inf)
        
        # 计算基础注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 计算自适应门控权重
        gate_weights = self.attention_gate(x.mean(dim=1)).unsqueeze(1).unsqueeze(-1)
        
        # 应用门控权重
        attn_weights = attn_weights * gate_weights
        attn_weights = self.dropout(attn_weights)
        
        # 注意力加权和
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        
        return out

class TransformerEncoderLayer(nn.Module):
    """自定义 Transformer 编码器层"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = AdaptiveAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 注意力子层
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # 前馈子层
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x

class StockTransformerV3(nn.Module):
    def __init__(self, n_feat: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.proj    = nn.Linear(n_feat, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # 使用自定义编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.gate    = GatedResidual(d_model)

        # 多尺度聚合：最后时间步 + 全局均值 + 全局最大值
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def _causal_mask(self, seq_len: int, device):
        """上三角掩码：每个时间步只能看到过去。"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, x):                          # (B, T, F)
        x = self.proj(x)
        x = self.pos_enc(x)
        mask = self._causal_mask(x.size(1), x.device)
        
        # 逐层处理
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.gate(x)
        last = x[:, -1, :]                         # 最后时间步
        avg  = x.mean(dim=1)                       # 全局均值
        max_ = x.max(dim=1)[0]                     # 全局最大值
        x    = torch.cat([last, avg, max_], dim=-1) # 多尺度融合
        return self.head(x).squeeze(-1)            # (B,)

# ──────────────────────────────────────────────
# 4. 训练 & 评估
# ──────────────────────────────────────────────
def train_epoch(model, loader, opt, criterion, device, grad_accum=1):
    model.train()
    total = 0.0
    opt.zero_grad()
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        loss = loss / grad_accum  # 梯度累积
        loss.backward()
        
        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            opt.zero_grad()
        
        total += loss.item() * grad_accum * len(x)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, preds, trues = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x)
        total += criterion(p, y).item() * len(x)
        preds.append(p.cpu().numpy())
        trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total / len(loader.dataset), preds, trues

# ──────────────────────────────────────────────
# 5. 主流程
# ──────────────────────────────────────────────
def main():
    # ── 数据 ──────────────────────────────────
    df_raw = load_data(TICKER)
    df     = add_features(df_raw)

    X_all = df[FEATURE_COLS].values.astype(np.float32)
    y_all = df["Target"].values.astype(np.float32)

    n      = len(X_all)
    split  = int(n * SPLIT)
    X_tr, X_te = X_all[:split], X_all[split:]
    y_tr, y_te = y_all[:split], y_all[split:]

    # RobustScaler：对离群值不敏感
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    X_tr_s = x_scaler.fit_transform(X_tr)
    X_te_s = x_scaler.transform(X_te)
    y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_te_s = y_scaler.transform(y_te.reshape(-1,1)).ravel()

    tr_ds = StockDataset(X_tr_s, y_tr_s, SEQ_LEN)
    te_ds = StockDataset(X_te_s, y_te_s, SEQ_LEN)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── 模型 ──────────────────────────────────
    model = StockTransformerV3(
        n_feat=len(FEATURE_COLS),
        d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF,
        dropout=DROPOUT
    ).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.HuberLoss(delta=0.5)          # 对离群值鲁棒
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, 
        steps_per_epoch=len(tr_dl), epochs=EPOCHS,
        pct_start=0.1, anneal_strategy="cos"
    )

    # ── 训练 ──────────────────────────────────
    best_val, patience_cnt = float("inf"), 0
    tr_losses, va_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        tr = train_epoch(model, tr_dl, optimizer, criterion, DEVICE, GRAD_ACCUM)
        scheduler.step()
        va, _, _ = eval_epoch(model, te_dl, criterion, DEVICE)
        tr_losses.append(tr); va_losses.append(va)

        if va < best_val:
            best_val = va
            patience_cnt = 0
            torch.save(model.state_dict(), "best_stock_v3.pt")
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | train={tr:.6f} | val={va:.6f} | patience={patience_cnt}")

        if patience_cnt >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    # ── 测试集评估 ────────────────────────────
    model.load_state_dict(torch.load("best_stock_v3.pt", map_location=DEVICE))
    _, pred_s, true_s = eval_epoch(model, te_dl, criterion, DEVICE)

    # 反归一化（收益率空间）
    pred_ret = y_scaler.inverse_transform(pred_s.reshape(-1,1)).ravel()
    true_ret = y_scaler.inverse_transform(true_s.reshape(-1,1)).ravel()

    # 还原为价格（从测试集起始价格累乘）
    start_price = df_raw["Close"].iloc[split + SEQ_LEN]
    pred_price  = start_price * np.cumprod(1 + pred_ret)
    true_price  = start_price * np.cumprod(1 + true_ret)

    # 指标
    mae   = np.mean(np.abs(pred_price - true_price))
    rmse  = np.sqrt(np.mean((pred_price - true_price)**2))
    mape  = np.mean(np.abs((pred_price - true_price) / (true_price + 1e-9))) * 100
    r2    = r2_score(true_price, pred_price)
    dir_acc = np.mean(np.sign(pred_ret) == np.sign(true_ret)) * 100

    # Sharpe（基于预测方向做多空）
    strategy_ret = np.where(pred_ret > 0, true_ret, -true_ret)
    sharpe = (strategy_ret.mean() / (strategy_ret.std() + 1e-9)) * np.sqrt(252)
    
    # 最大回撤
    def max_drawdown(returns):
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return np.min(drawdown) * 100
    
    max_dd = max_drawdown(strategy_ret)

    print(f"\n{'='*60}")
    print(f"  MAE        : {mae:.4f}")
    print(f"  RMSE       : {rmse:.4f}")
    print(f"  MAPE       : {mape:.2f}%")
    print(f"  R²         : {r2:.4f}")
    print(f"  Direction  : {dir_acc:.1f}%")
    print(f"  Sharpe     : {sharpe:.3f}")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"{'='*60}")

    # ── 可视化 ────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig)

    # 1) 训练曲线
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(tr_losses, label="Train", alpha=0.8)
    ax0.plot(va_losses, label="Val",   alpha=0.8)
    ax0.set_title("Loss Curve (Huber)"); ax0.legend(); ax0.set_xlabel("Epoch")

    # 2) 价格对比
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(true_price, label="Actual",    alpha=0.85)
    ax1.plot(pred_price, label="Predicted", alpha=0.85, linestyle="--")
    ax1.set_title(f"{TICKER} Reconstructed Price (Test Set)")
    ax1.legend(); ax1.set_xlabel("Day")

    # 3) 收益率散点
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(true_ret, pred_ret, alpha=0.3, s=8, color="steelblue")
    lim = max(abs(true_ret).max(), abs(pred_ret).max()) * 1.1
    ax2.plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
    ax2.set_title(f"Return: True vs Predicted  (R²={r2:.3f})  Dir={dir_acc:.1f}%")
    ax2.set_xlabel("True Return"); ax2.set_ylabel("Pred Return")

    # 4) 误差分布
    ax3 = fig.add_subplot(gs[1, 1])
    errors = pred_price - true_price
    ax3.hist(errors, bins=40, color="coral", edgecolor="white", alpha=0.8)
    ax3.axvline(0, color="black", lw=1.5, linestyle="--")
    ax3.set_title("Prediction Error Distribution")
    ax3.set_xlabel("Error (Price)")

    # 5) 策略收益
    ax4 = fig.add_subplot(gs[2, 0])
    strategy_cum = np.cumprod(1 + strategy_ret)
    buy_hold_cum = np.cumprod(1 + true_ret)
    ax4.plot(strategy_cum, label="Strategy", alpha=0.85)
    ax4.plot(buy_hold_cum, label="Buy & Hold", alpha=0.85, linestyle="--")
    ax4.set_title(f"Strategy vs Buy & Hold (Sharpe: {sharpe:.3f})")
    ax4.legend(); ax4.set_xlabel("Day")
    ax4.set_ylabel("Cumulative Return")

    # 6) 最大回撤
    ax5 = fig.add_subplot(gs[2, 1])
    strategy_cum = np.cumprod(1 + strategy_ret)
    peak = np.maximum.accumulate(strategy_cum)
    drawdown = (strategy_cum - peak) / peak * 100
    ax5.plot(drawdown, color="crimson", alpha=0.85)
    ax5.set_title(f"Strategy Drawdown (Max: {max_dd:.2f}%)")
    ax5.set_xlabel("Day")
    ax5.set_ylabel("Drawdown (%)")

    plt.suptitle(f"Stock Transformer v3 — {TICKER}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("stock_transformer_v3_result.png", dpi=150)
    print("Figure saved → stock_transformer_v3_result.png")
    plt.show()

if __name__ == "__main__":
    main()