import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from models import BiLSTMModel, TransformerModel, StockPredictor
import io

st.set_page_config(
    page_title="股市预测模型对比 - BiLSTM vs Transformer",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .model-card {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 股市预测模型对比分析</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">双向LSTM vs Transformer 深度学习模型性能比较</div>', unsafe_allow_html=True)

st.sidebar.header("📖 模型介绍")

if st.sidebar.button("🔍 查看模型原理", use_container_width=True):
    st.sidebar.markdown("""
    ---
    ### 🧠 BiLSTM (双向长短期记忆网络)
    
    **核心思想**：
    - 同时捕捉时间序列的**前向**和**后向**依赖关系
    - 适合处理具有时序特征的股票数据
    
    **工作原理**：
    ```
    输入序列: [Day1] → [Day2] → [Day3] → [Day4] → [Day5]
                 ↓        ↓        ↓        ↓        ↓
              ┌─────────────────────────────────────────┐
              │  前向LSTM: Day1 → Day2 → Day3 → ...    │
              │  后向LSTM: Day5 → Day4 → Day3 → ...    │
              └─────────────────────────────────────────┘
                              ↓
                    合并双向信息 → 预测Day6
    ```
    
    **优点**：
    - ✅ 能捕捉长期依赖关系
    - ✅ 双向信息融合更全面
    - ✅ 对时序数据效果好
    
    **缺点**：
    - ❌ 串行计算，速度较慢
    - ❌ 难以捕捉非常长期的依赖
    
    ---
    ### 🤖 Transformer (自注意力机制)
    
    **核心思想**：
    - 通过**自注意力机制**直接建模序列中任意两个位置的关系
    - 并行计算，训练速度快
    
    **工作原理**：
    ```
    输入序列: [Day1] [Day2] [Day3] [Day4] [Day5]
                 ↓     ↓     ↓     ↓     ↓
              ┌─────────────────────────────────┐
              │      自注意力计算               │
              │  Day1 关注 [Day1-5] 的权重     │
              │  Day2 关注 [Day1-5] 的权重     │
              │  ...                           │
              └─────────────────────────────────┘
                          ↓
                    加权融合 → 预测Day6
    ```
    
    **关键组件**：
    1. **位置编码**：给序列添加位置信息
    2. **多头注意力**：从多个角度捕捉关系
    3. **前馈网络**：进一步处理特征
    
    **优点**：
    - ✅ 并行计算，训练速度快
    - ✅ 能捕捉长距离依赖
    - ✅ 注意力权重可解释
    
    **缺点**：
    - ❌ 需要更多数据
    - ❌ 计算复杂度高
    
    ---
    ### 📊 对比总结
    
    | 特性 | BiLSTM | Transformer |
    |------|--------|-------------|
    | 计算方式 | 串行 | 并行 |
    | 长期依赖 | 一般 | 优秀 |
    | 训练速度 | 慢 | 快 |
    | 数据需求 | 较少 | 较多 |
    | 可解释性 | 一般 | 好(注意力) |
    
    ---
    """)

st.sidebar.header("📂 模型管理")

# 加载已保存的模型
if st.sidebar.checkbox("📥 加载已保存的模型"):
    st.sidebar.info("""
    **加载模型功能**：
    如果之前保存过模型，可以直接加载使用，无需重新训练。
    
    ⚠️ 注意：加载模型后，请确保使用相同的数据和参数设置。
    """)
    
    if st.sidebar.button("🔍 检查已保存的模型"):
        import os
        save_dir = 'saved_models'
        if os.path.exists(save_dir):
            files = os.listdir(save_dir)
            if files:
                st.sidebar.success(f"找到 {len(files)//3} 个保存的模型")
                for f in sorted(set([f.replace('_model.pth', '').replace('_scaler.npy', '').replace('_history.json', '') for f in files])):
                    st.sidebar.write(f"- {f}")
            else:
                st.sidebar.warning("暂无保存的模型")
        else:
            st.sidebar.warning("保存目录不存在")

st.sidebar.header("⚙️ 配置参数")

st.sidebar.subheader("📁 数据上传")
uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=['csv'])

use_sample = st.sidebar.checkbox("使用示例数据", value=False)

st.sidebar.subheader("🔧 模型参数")
seq_length = st.sidebar.slider("序列长度", min_value=10, max_value=120, value=60, step=10)
if st.sidebar.checkbox("ℹ️ 序列长度是什么？"):
    st.sidebar.info("""
    **序列长度**：用过去多少天的数据来预测下一天。
    
    📌 示例：设置为60表示用过去60天的股价预测第61天
    
    💡 建议：
    - 短期预测：20-40天
    - 中期预测：60-90天
    - 长期预测：100-120天
    """)

train_split = st.sidebar.slider("训练集比例", min_value=0.5, max_value=0.9, value=0.8, step=0.05)
if st.sidebar.checkbox("ℹ️ 训练集比例是什么？"):
    st.sidebar.info("""
    **训练集比例**：多少数据用于训练，剩余用于测试。
    
    📌 示例：0.8表示80%数据训练，20%数据测试
    
    💡 建议：
    - 数据量大：0.8-0.9
    - 数据量小：0.7-0.8
    """)

st.sidebar.subheader("🧠 BiLSTM参数")
bilstm_hidden = st.sidebar.slider("隐藏层大小", min_value=32, max_value=512, value=256, step=32)
if st.sidebar.checkbox("ℹ️ 隐藏层大小是什么？"):
    st.sidebar.info("""
    **隐藏层大小**：LSTM神经元的数量，决定模型的学习能力。
    
    📌 通俗解释：
    - 值越大 → 模型越复杂 → 能学习更复杂的模式
    - 值越小 → 模型越简单 → 训练更快，不易过拟合
    
    💡 建议：
    - 简单数据：64-128
    - 复杂数据：128-256
    """)

bilstm_layers = st.sidebar.slider("LSTM层数", min_value=1, max_value=4, value=3)
if st.sidebar.checkbox("ℹ️ LSTM层数是什么？"):
    st.sidebar.info("""
    **LSTM层数**：堆叠多少个LSTM层。
    
    📌 通俗解释：
    - 层数越多 → 能学习更深层特征 → 但容易过拟合
    - 层数越少 → 训练更快 → 但可能欠拟合
    
    💡 建议：
    - 新手推荐：1-2层
    - 经验丰富：2-3层
    """)

st.sidebar.subheader("🤖 Transformer参数")
trans_d_model = st.sidebar.slider("模型维度", min_value=32, max_value=512, value=256, step=32)
if st.sidebar.checkbox("ℹ️ 模型维度是什么？"):
    st.sidebar.info("""
    **模型维度(d_model)**：Transformer内部特征向量的维度。
    
    📌 通俗解释：
    - 维度越高 → 表达能力越强 → 但需要更多数据
    - 维度越低 → 计算越快 → 但可能表达能力不足
    
    💡 建议：
    - 小数据集：64-128
    - 大数据集：128-256
    """)

trans_heads = st.sidebar.slider("注意力头数", min_value=2, max_value=16, value=8, step=2)
if st.sidebar.checkbox("ℹ️ 注意力头数是什么？"):
    st.sidebar.info("""
    **注意力头数**：同时进行多少次注意力计算。
    
    📌 通俗解释：
    - 多个头 → 从不同角度观察数据 → 捕捉更多关系
    - 类似多人从不同角度看同一张图
    
    💡 建议：
    - 必须能被模型维度整除
    - 常用值：4, 8, 16
    """)

trans_layers = st.sidebar.slider("编码器层数", min_value=1, max_value=6, value=4)
if st.sidebar.checkbox("ℹ️ 编码器层数是什么？"):
    st.sidebar.info("""
    **编码器层数**：堆叠多少个Transformer编码器块。
    
    📌 通俗解释：
    - 层数越多 → 特征提取能力越强 → 但训练更慢
    - 类似深度神经网络的概念
    
    💡 建议：
    - 新手推荐：1-2层
    - 追求精度：3-6层
    """)

st.sidebar.subheader("⚡ 训练参数")
epochs = st.sidebar.slider("训练轮数", min_value=10, max_value=300, value=100, step=10)
if st.sidebar.checkbox("ℹ️ 训练轮数是什么？"):
    st.sidebar.info("""
    **训练轮数**：模型遍历整个数据集的次数。
    
    📌 通俗解释：
    - 轮数太少 → 模型学不够 → 欠拟合
    - 轮数太多 → 模型死记硬背 → 过拟合
    
    💡 建议：
    - 快速测试：20-50轮
    - 正式训练：50-100轮
    """)

batch_size = st.sidebar.slider("批次大小", min_value=8, max_value=128, value=32, step=8)
if st.sidebar.checkbox("ℹ️ 批次大小是什么？"):
    st.sidebar.info("""
    **批次大小**：每次训练使用多少个样本。
    
    📌 通俗解释：
    - 批次大 → 训练稳定 → 但需要更多显存
    - 批次小 → 训练波动大 → 但可能找到更好的解
    
    💡 建议：
    - 显存小：8-16
    - 显存大：32-64
    """)

learning_rate = st.sidebar.select_slider("学习率", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
if st.sidebar.checkbox("ℹ️ 学习率是什么？"):
    st.sidebar.info("""
    **学习率**：模型学习的步长大小。
    
    📌 通俗解释：
    - 学习率大 → 学得快 → 但可能错过最优解
    - 学习率小 → 学得慢 → 但更精细
    
    💡 建议：
    - 保守选择：0.0001-0.001
    - 快速收敛：0.001-0.01
    """)

st.sidebar.subheader("🔮 预测设置")
prediction_days = st.sidebar.slider("预测未来天数", min_value=7, max_value=90, value=30, step=7)
if st.sidebar.checkbox("ℹ️ 预测天数建议"):
    st.sidebar.info("""
    **预测未来天数**：预测未来多少天的股价。
    
    ⚠️ 重要提示：
    - 短期预测（7-14天）相对准确
    - 中期预测（30天）仅供参考
    - 长期预测（60-90天）不确定性很高
    
    💡 股市有风险，预测结果仅供学习参考！
    """)

@st.cache_data
def load_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    n = len(dates)
    
    trend = np.linspace(100, 200, n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.normal(0, 5, n)
    
    close = trend + seasonal + noise
    high = close + np.abs(np.random.normal(2, 1, n))
    low = close - np.abs(np.random.normal(2, 1, n))
    open_price = close + np.random.normal(0, 1, n)
    volume = np.random.randint(1000000, 10000000, n)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # 数据清洗：将数值列转换为float类型
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    for col in numeric_columns:
        if col in df.columns:
            # 移除可能的逗号和货币符号，然后转换为float
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('¥', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除包含NaN的行
    df = df.dropna()
    
    st.success(f"✅ 成功加载数据: {len(df)} 条记录")
elif use_sample:
    df = load_sample_data()
    st.info("📊 使用示例数据进行演示")
else:
    st.info("👈 请在侧边栏上传CSV文件或选择使用示例数据")
    st.markdown("""
    ### 📋 数据格式要求
    CSV文件应包含以下列：
    - `Date`: 日期
    - `Open`: 开盘价
    - `High`: 最高价
    - `Low`: 最低价
    - `Close`: 收盘价
    - `Volume`: 成交量
    
    ### 🎯 功能特点
    1. **双向LSTM**: 捕捉时间序列的前后向依赖关系
    2. **Transformer**: 利用自注意力机制建模长期依赖
    3. **模型对比**: 全面的性能指标对比分析
    4. **未来预测**: 基于训练好的模型预测未来走势
    """)
    st.stop()

st.header("📈 数据概览")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("数据条数", len(df))
with col2:
    if 'Close' in df.columns and len(df) > 0:
        try:
            close_value = float(df['Close'].iloc[-1])
            st.metric("最新收盘价", f"{close_value:.2f}")
        except (ValueError, TypeError):
            st.metric("最新收盘价", "N/A")
with col3:
    if 'Close' in df.columns and len(df) > 0:
        try:
            close_last = float(df['Close'].iloc[-1])
            close_first = float(df['Close'].iloc[0])
            change = ((close_last - close_first) / close_first) * 100
            st.metric("总涨跌幅", f"{change:.2f}%")
        except (ValueError, TypeError, ZeroDivisionError):
            st.metric("总涨跌幅", "N/A")
with col4:
    if 'Date' in df.columns and len(df) > 0:
        try:
            days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
            st.metric("时间跨度", f"{days} 天")
        except (ValueError, TypeError):
            st.metric("时间跨度", "N/A")

st.subheader("股价走势图")
fig = go.Figure()
if 'Date' in df.columns and 'Close' in df.columns:
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Close'],
        mode='lines',
        name='收盘价',
        line=dict(color='#1f77b4', width=2)
    ))
    if 'High' in df.columns and 'Low' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['High'],
            mode='lines',
            name='最高价',
            line=dict(color='green', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['Low'],
            mode='lines',
            name='最低价',
            line=dict(color='red', width=1)
        ))
    
    fig.update_layout(
        title='股票价格走势',
        xaxis_title='日期',
        yaxis_title='价格',
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

if st.button("🚀 开始训练模型", type="primary"):
    with st.spinner("正在准备数据和训练模型..."):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"使用设备: {device}")
        
        predictor = StockPredictor(seq_length=seq_length, device=device)
        
        try:
            X_train, y_train, X_test, y_test, feature_cols = predictor.prepare_data(
                df, train_split=train_split
            )
            
            st.success(f"数据准备完成! 训练集: {len(X_train)}, 测试集: {len(X_test)}")
            st.write(f"使用的特征数: {len(feature_cols)}")
            
            # 显示使用的技术指标
            with st.expander("📊 点击查看使用的技术指标"):
                st.markdown("""
                **基础价格数据**: Open, High, Low, Close, Volume
                
                **趋势指标**:
                - MA5, MA10, MA20, MA60 (移动平均线)
                - Close_MA5_ratio, Close_MA20_ratio (价格与均线关系)
                
                **动量指标**:
                - RSI (相对强弱指数)
                - Momentum_5, Momentum_10, Momentum_20 (价格动量)
                
                **趋势跟踪**:
                - MACD, MACD_signal, MACD_hist
                
                **波动率指标**:
                - BB_upper, BB_middle, BB_lower, BB_position (布林带)
                - Volatility_5, Volatility_20 (波动率)
                
                **成交量指标**:
                - Volume_MA5, Volume_MA20, Volume_ratio
                
                **价格特征**:
                - Price_range (日内波动)
                - Price_change (价格变化率)
                - Price_position (日内位置)
                """)
                st.info(f"总计 {len(feature_cols)} 个特征用于训练")
            
            input_size = len(feature_cols)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("训练 BiLSTM 模型...")
            bilstm_model = BiLSTMModel(
                input_size=input_size,
                hidden_size=bilstm_hidden,
                num_layers=bilstm_layers,
                output_size=1
            ).to(device)
            
            bilstm_history = predictor.train_model(
                'BiLSTM', bilstm_model, X_train, y_train, X_test, y_test,
                epochs=epochs, batch_size=batch_size, lr=learning_rate
            )
            progress_bar.progress(50)
            
            status_text.text("训练 Transformer 模型...")
            transformer_model = TransformerModel(
                input_size=input_size,
                d_model=trans_d_model,
                nhead=trans_heads,
                num_layers=trans_layers,
                output_size=1
            ).to(device)
            
            trans_history = predictor.train_model(
                'Transformer', transformer_model, X_train, y_train, X_test, y_test,
                epochs=epochs, batch_size=batch_size, lr=learning_rate
            )
            progress_bar.progress(100)
            status_text.text("训练完成!")
            
            st.header("📊 模型性能对比")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🧠 BiLSTM 评估结果")
                bilstm_metrics, bilstm_preds, actuals = predictor.evaluate_model('BiLSTM', X_test, y_test)
                
                for metric, value in bilstm_metrics.items():
                    display_name = {
                        'MSE': '均方误差 (MSE)',
                        'RMSE': '均方根误差 (RMSE)',
                        'MAE': '平均绝对误差 (MAE)',
                        'R²': '决定系数 (R²)',
                        'Direction_Accuracy': '方向准确率'
                    }.get(metric, metric)
                    st.metric(display_name, f"{value:.4f}")
            
            with col2:
                st.subheader("🤖 Transformer 评估结果")
                trans_metrics, trans_preds, _ = predictor.evaluate_model('Transformer', X_test, y_test)
                
                for metric, value in trans_metrics.items():
                    display_name = {
                        'MSE': '均方误差 (MSE)',
                        'RMSE': '均方根误差 (RMSE)',
                        'MAE': '平均绝对误差 (MAE)',
                        'R²': '决定系数 (R²)',
                        'Direction_Accuracy': '方向准确率'
                    }.get(metric, metric)
                    st.metric(display_name, f"{value:.4f}")
            
            st.subheader("📈 性能指标对比")
            comparison_df = pd.DataFrame({
                'BiLSTM': list(bilstm_metrics.values()),
                'Transformer': list(trans_metrics.values())
            }, index=list(bilstm_metrics.keys()))
            
            fig_comparison = go.Figure()
            for model in ['BiLSTM', 'Transformer']:
                fig_comparison.add_trace(go.Bar(
                    name=model,
                    x=comparison_df.index,
                    y=comparison_df[model],
                    text=[f"{v:.4f}" for v in comparison_df[model]],
                    textposition='auto'
                ))
            
            fig_comparison.update_layout(
                title='模型性能指标对比',
                barmode='group',
                template='plotly_white'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            st.subheader("📉 训练过程对比")
            fig_training = make_subplots(rows=1, cols=2, subplot_titles=('训练损失', '验证损失'))
            
            fig_training.add_trace(
                go.Scatter(y=bilstm_history['train_loss'], name='BiLSTM Train', line=dict(color='blue')),
                row=1, col=1
            )
            fig_training.add_trace(
                go.Scatter(y=trans_history['train_loss'], name='Transformer Train', line=dict(color='red')),
                row=1, col=1
            )
            
            fig_training.add_trace(
                go.Scatter(y=bilstm_history['val_loss'], name='BiLSTM Val', line=dict(color='blue', dash='dash')),
                row=1, col=2
            )
            fig_training.add_trace(
                go.Scatter(y=trans_history['val_loss'], name='Transformer Val', line=dict(color='red', dash='dash')),
                row=1, col=2
            )
            
            fig_training.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig_training, use_container_width=True)
            
            st.subheader("🔮 预测结果可视化")
            fig_pred = go.Figure()
            
            test_dates = df['Date'].iloc[-len(actuals):] if 'Date' in df.columns else range(len(actuals))
            
            fig_pred.add_trace(go.Scatter(
                x=test_dates, y=actuals,
                mode='lines', name='实际值',
                line=dict(color='white', width=2)
            ))
            fig_pred.add_trace(go.Scatter(
                x=test_dates, y=bilstm_preds,
                mode='lines', name='BiLSTM预测',
                line=dict(color='blue', width=2)
            ))
            fig_pred.add_trace(go.Scatter(
                x=test_dates, y=trans_preds,
                mode='lines', name='Transformer预测',
                line=dict(color='red', width=2)
            ))
            
            fig_pred.update_layout(
                title='模型预测对比',
                xaxis_title='时间',
                yaxis_title='价格',
                hovermode='x unified',
                template='plotly_dark',
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e'
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.header("🔮 未来预测")
            
            last_sequence = predictor.scaler.transform(df[feature_cols].values)[-seq_length:]
            
            bilstm_future = predictor.predict_future('BiLSTM', last_sequence, days=prediction_days)
            trans_future = predictor.predict_future('Transformer', last_sequence, days=prediction_days)
            
            future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=prediction_days) if 'Date' in df.columns else range(prediction_days)
            
            fig_future = go.Figure()
            
            historical_dates = df['Date'].iloc[-60:] if 'Date' in df.columns else range(len(df)-60, len(df))
            fig_future.add_trace(go.Scatter(
                x=historical_dates,
                y=df['Close'].iloc[-60:],
                mode='lines',
                name='历史数据',
                line=dict(color='white', width=2)
            ))
            
            fig_future.add_trace(go.Scatter(
                x=future_dates,
                y=bilstm_future,
                mode='lines+markers',
                name='BiLSTM预测',
                line=dict(color='blue', width=2, dash='dash')
            ))
            
            fig_future.add_trace(go.Scatter(
                x=future_dates,
                y=trans_future,
                mode='lines+markers',
                name='Transformer预测',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_future.update_layout(
                title=f'未来{prediction_days}天股价预测',
                xaxis_title='日期',
                yaxis_title='价格',
                hovermode='x unified',
                template='plotly_dark',
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e'
            )
            st.plotly_chart(fig_future, use_container_width=True)
            
            st.subheader("📋 预测结果数据")
            future_df = pd.DataFrame({
                '日期': future_dates,
                'BiLSTM预测': bilstm_future,
                'Transformer预测': trans_future,
                '预测差异': np.abs(bilstm_future - trans_future)
            })
            st.dataframe(future_df)
            
            csv = future_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下载预测结果",
                data=csv,
                file_name='stock_prediction_results.csv',
                mime='text/csv'
            )
            
            # 模型推荐
            st.header("🏆 模型推荐")
            
            # 计算综合得分（越低越好）
            bilstm_score = (bilstm_metrics['RMSE'] + bilstm_metrics['MAE']) / 2 + (1 - bilstm_metrics['R²']) + (1 - bilstm_metrics['Direction_Accuracy'])
            trans_score = (trans_metrics['RMSE'] + trans_metrics['MAE']) / 2 + (1 - trans_metrics['R²']) + (1 - trans_metrics['Direction_Accuracy'])
            
            if trans_score < bilstm_score:
                winner = "Transformer"
                winner_metrics = trans_metrics
                loser_metrics = bilstm_metrics
                reason = "Transformer模型在捕捉长期依赖关系方面表现更优"
                color = "#ff6b6b"
            else:
                winner = "BiLSTM"
                winner_metrics = bilstm_metrics
                loser_metrics = trans_metrics
                reason = "BiLSTM模型在捕捉时序特征方面表现更优"
                color = "#4ecdc4"
            
            # 计算优势百分比
            rmse_improvement = ((loser_metrics['RMSE'] - winner_metrics['RMSE']) / loser_metrics['RMSE']) * 100
            mae_improvement = ((loser_metrics['MAE'] - winner_metrics['MAE']) / loser_metrics['MAE']) * 100
            r2_improvement = ((winner_metrics['R²'] - loser_metrics['R²']) / abs(loser_metrics['R²'])) * 100 if loser_metrics['R²'] != 0 else 0
            dir_improvement = ((winner_metrics['Direction_Accuracy'] - loser_metrics['Direction_Accuracy']) / loser_metrics['Direction_Accuracy']) * 100
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}22, {color}11); 
                        border-left: 5px solid {color}; 
                        padding: 20px; 
                        border-radius: 10px; 
                        margin: 20px 0;">
                <h3 style="color: {color}; margin-top: 0;">🎯 推荐模型: {winner}</h3>
                <p style="font-size: 16px; margin: 10px 0;"><strong>{reason}</strong></p>
                
                <h4 style="margin-top: 20px;">📊 性能优势对比:</h4>
                <ul style="font-size: 14px; line-height: 1.8;">
                    <li>RMSE 降低: <strong>{rmse_improvement:.2f}%</strong> ({loser_metrics['RMSE']:.4f} → {winner_metrics['RMSE']:.4f})</li>
                    <li>MAE 降低: <strong>{mae_improvement:.2f}%</strong> ({loser_metrics['MAE']:.4f} → {winner_metrics['MAE']:.4f})</li>
                    <li>R² 提升: <strong>{r2_improvement:.2f}%</strong> ({loser_metrics['R²']:.4f} → {winner_metrics['R²']:.4f})</li>
                    <li>方向准确率提升: <strong>{dir_improvement:.2f}%</strong> ({loser_metrics['Direction_Accuracy']:.2%} → {winner_metrics['Direction_Accuracy']:.2%})</li>
                </ul>
                
                <div style="background-color: rgba(255,255,255,0.5); padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <h4 style="margin-top: 0;">💡 使用建议:</h4>
                    <p style="margin: 5px 0;">基于当前数据集，建议使用 <strong>{winner}</strong> 模型进行股价预测。</p>
                    <p style="margin: 5px 0; font-size: 13px; color: #666;">
                        注意：模型性能会因不同的股票和时间周期而变化，建议定期重新评估模型表现。
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 详细对比表格
            st.subheader("📈 详细性能对比")
            comparison_detail = pd.DataFrame({
                '指标': ['RMSE (越低越好)', 'MAE (越低越好)', 'R² (越接近1越好)', '方向准确率 (越高越好)'],
                'BiLSTM': [f"{bilstm_metrics['RMSE']:.4f}", f"{bilstm_metrics['MAE']:.4f}", 
                          f"{bilstm_metrics['R²']:.4f}", f"{bilstm_metrics['Direction_Accuracy']:.2%}"],
                'Transformer': [f"{trans_metrics['RMSE']:.4f}", f"{trans_metrics['MAE']:.4f}", 
                               f"{trans_metrics['R²']:.4f}", f"{trans_metrics['Direction_Accuracy']:.2%}"],
                '胜出者': [
                    'Transformer' if trans_metrics['RMSE'] < bilstm_metrics['RMSE'] else 'BiLSTM',
                    'Transformer' if trans_metrics['MAE'] < bilstm_metrics['MAE'] else 'BiLSTM',
                    'Transformer' if trans_metrics['R²'] > bilstm_metrics['R²'] else 'BiLSTM',
                    'Transformer' if trans_metrics['Direction_Accuracy'] > bilstm_metrics['Direction_Accuracy'] else 'BiLSTM'
                ]
            })
            st.table(comparison_detail)
            
            # 模型保存功能
            st.header("💾 模型管理")
            
            col_save1, col_save2 = st.columns(2)
            
            with col_save1:
                if st.button("💾 保存训练好的模型", use_container_width=True):
                    try:
                        bilstm_path, bilstm_scaler, bilstm_hist = predictor.save_model('BiLSTM')
                        trans_path, trans_scaler, trans_hist = predictor.save_model('Transformer')
                        st.success(f"✅ 模型已保存！\n\nBiLSTM: {bilstm_path}\nTransformer: {trans_path}")
                    except Exception as e:
                        st.error(f"保存失败: {str(e)}")
            
            with col_save2:
                if st.button("📊 查看训练历史", use_container_width=True):
                    summary = predictor.get_training_summary()
                    if summary:
                        st.subheader("训练摘要")
                        for item in summary:
                            st.write(f"**{item['model_name']}**:")
                            st.write(f"- 训练轮数: {item['epochs_trained']}")
                            st.write(f"- 最终训练损失: {item['final_train_loss']:.6f}" if item['final_train_loss'] else "- 最终训练损失: N/A")
                            st.write(f"- 最终验证损失: {item['final_val_loss']:.6f}" if item['final_val_loss'] else "- 最终验证损失: N/A")
                    else:
                        st.info("暂无训练记录")
            
        except Exception as e:
            st.error(f"训练过程中出现错误: {str(e)}")
            st.exception(e)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Made with ❤️ using Streamlit | 股市预测模型对比分析</div>", unsafe_allow_html=True)
