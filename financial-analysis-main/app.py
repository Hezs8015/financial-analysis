import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from models import (
    BiLSTMModelV1, BiLSTMModelV2, BiLSTMModelV3,
    TransformerModelV1, TransformerModelV2,
    StockPredictor
)
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

st.sidebar.subheader("🧠 模型选择")

# BiLSTM版本选择
bilstm_version = st.sidebar.selectbox(
    "BiLSTM 版本",
    options=["v1 (基础版)", "v2 (增强版)", "v3 (高级版)"],
    index=1
)
if st.sidebar.checkbox("ℹ️ BiLSTM版本说明"):
    st.sidebar.info("""
    **BiLSTM版本对比：**
    - **v1 (基础版)**: 1层LSTM，64隐藏单元，适合简单数据
    - **v2 (增强版)**: 2层LSTM，128隐藏单元，适合一般数据
    - **v3 (高级版)**: 3层LSTM，256隐藏单元，适合复杂数据
    """)

# Transformer版本选择
transformer_version = st.sidebar.selectbox(
    "Transformer 版本",
    options=["v1 (基础版)", "v2 (增强版)"],
    index=1
)
if st.sidebar.checkbox("ℹ️ Transformer版本说明"):
    st.sidebar.info("""
    **Transformer版本对比：**
    - **v1 (基础版)**: 1层编码器，64模型维度，4注意力头
    - **v2 (增强版)**: 2层编码器，128模型维度，8注意力头
    """)

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
bilstm_hidden = st.sidebar.slider("隐藏层大小", min_value=32, max_value=256, value=128, step=32)
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

bilstm_layers = st.sidebar.slider("LSTM层数", min_value=1, max_value=4, value=2)
if st.sidebar.checkbox("ℹ️ LSTM层数是什么？"):
    st.sidebar.info("""
    **LSTM层数**：堆叠多少个LSTM层。
    
    📌 通俗解释：
    - 层数越多 → 特征提取能力越强 → 但训练更慢
    - 层数越少 → 训练越快 → 但可能学习能力不足
    
    💡 建议：
    - 新手推荐：1-2层
    - 经验丰富：2-3层
    """)

st.sidebar.subheader("🤖 Transformer参数")
trans_d_model = st.sidebar.slider("模型维度", min_value=32, max_value=256, value=128, step=32)
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

trans_layers = st.sidebar.slider("编码器层数", min_value=1, max_value=6, value=2)
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
epochs = st.sidebar.slider("训练轮数", min_value=10, max_value=200, value=50, step=10)
if st.sidebar.checkbox("ℹ️ 训练轮数是什么？"):
    st.sidebar.info("""
    **训练轮数**：模型在训练数据上迭代的次数。
    
    📌 通俗解释：
    - 轮数越多 → 模型学习越充分 → 但可能过拟合
    - 轮数越少 → 训练更快 → 但可能欠拟合
    
    💡 建议：
    - 简单模型：30-50轮
    - 复杂模型：50-100轮
    """)

batch_size = st.sidebar.slider("批次大小", min_value=8, max_value=128, value=32, step=8)
if st.sidebar.checkbox("ℹ️ 批次大小是什么？"):
    st.sidebar.info("""
    **批次大小**：每次训练时处理的数据量。
    
    📌 通俗解释：
    - 批次越大 → 训练速度越快 → 内存需求越大
    - 批次越小 → 内存需求越小 → 训练速度较慢
    
    💡 建议：
    - 小内存：8-16
    - 一般内存：32-64
    - 大内存：64-128
    """)

learning_rate = st.sidebar.slider("学习率", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)
if st.sidebar.checkbox("ℹ️ 学习率是什么？"):
    st.sidebar.info("""
    **学习率**：模型参数更新的步长。
    
    📌 通俗解释：
    - 学习率大 → 收敛快 → 可能错过最优解
    - 学习率小 → 收敛慢 → 但更精确
    
    💡 建议：
    - 初始：0.001
    - 微调：0.0001-0.001
    """)

# 主内容区域
main_container = st.container()

with main_container:
    if uploaded_file is not None or use_sample:
        # 加载数据
        if use_sample:
            # 生成示例数据
            dates = pd.date_range('2020-01-01', '2024-01-01')
            price = 100.0
            prices = []
            volumes = []
            for i, date in enumerate(dates):
                # 生成随机价格波动
                change = np.random.normal(0, 2)
                price = max(10, price + change)
                prices.append(price)
                # 生成随机成交量
                volume = np.random.normal(1000000, 500000)
                volumes.append(int(volume))
            
            df = pd.DataFrame({
                'Date': dates,
                'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
                'High': [p * (1 + np.random.normal(0, 0.02)) for p in prices],
                'Low': [p * (1 - np.random.normal(0, 0.02)) for p in prices],
                'Close': prices,
                'Volume': volumes
            })
            st.success("✅ 示例数据加载成功！")
        else:
            # 加载用户上传的数据
            df = pd.read_csv(uploaded_file)
            # 确保日期列存在
            if 'Date' not in df.columns:
                st.error("❌ 数据中缺少 'Date' 列")
                st.stop()
            # 转换日期格式
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                st.error("❌ 日期格式不正确，请确保 'Date' 列格式正确")
                st.stop()
            
            # 清理数值列：将所有数值列转换为正确的类型
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除数值列中包含NaN的行
            df = df.dropna(subset=[col for col in numeric_cols if col in df.columns])
            
            st.success("✅ 数据加载成功！")
        
        # 显示数据概览
        st.header("📋 数据概览")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("数据行数", len(df))
        with col2:
            st.metric("时间范围", f"{df['Date'].min().strftime('%Y-%m-%d')} 至 {df['Date'].max().strftime('%Y-%m-%d')}")
        with col3:
            try:
                st.metric("最新收盘价", f"{df['Close'].iloc[-1]:.2f}")
            except:
                st.metric("最新收盘价", "N/A")
        
        # 数据预览
        st.subheader("📊 数据预览")
        st.dataframe(df.tail(10))
        
        # 绘制历史价格图
        st.subheader("📈 历史价格走势")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'],
            name='收盘价',
            line=dict(color='white', width=2)
        ))
        fig.update_layout(
            xaxis_title='日期',
            yaxis_title='价格',
            title='历史收盘价走势',
            template='plotly_dark',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 准备数据
        st.header("🧠 模型训练")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 处理数据
        try:
            predictor = StockPredictor(seq_length=seq_length, device=device)
            X_train, y_train, X_test, y_test, feature_cols = predictor.prepare_data(df)
            
            # 检查数据量是否足够
            if len(X_train) < 10:
                st.error(f"❌ 训练数据不足！当前只有 {len(X_train)} 条训练数据，至少需要 10 条。")
                st.info(f"💡 建议：增加数据量或减少序列长度（当前序列长度: {seq_length}）")
                st.info(f"📊 原始数据行数: {len(df)}, 清理后可用数据: {len(X_train) + len(X_test)}")
                st.stop()
            
            st.success(f"数据准备完成! 训练集: {len(X_train)}, 测试集: {len(X_test)}")
            st.write(f"使用的特征: {feature_cols}")
            st.info(f"📊 数据形状 - X_train: {X_train.shape}, X_test: {X_test.shape}")
            
            input_size = len(feature_cols)
            
            # 训练模型
            if st.button("🚀 开始训练模型", use_container_width=True):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 选择BiLSTM模型版本
                    if bilstm_version == "v1 (基础版)":
                        bilstm_model_class = BiLSTMModelV1
                        bilstm_model_name = "BiLSTM-PyTorch-v1"
                    elif bilstm_version == "v2 (增强版)":
                        bilstm_model_class = BiLSTMModelV2
                        bilstm_model_name = "BiLSTM-PyTorch-v2"
                    else:  # v3
                        bilstm_model_class = BiLSTMModelV3
                        bilstm_model_name = "BiLSTM-PyTorch-v3"
                    
                    # 选择Transformer模型版本
                    if transformer_version == "v1 (基础版)":
                        transformer_model_class = TransformerModelV1
                        transformer_model_name = "Transformer-PyTorch-v1"
                    else:  # v2
                        transformer_model_class = TransformerModelV2
                        transformer_model_name = "Transformer-PyTorch-v2"
                    
                    status_text.text(f"训练 {bilstm_version} 模型...")
                    bilstm_model = bilstm_model_class(
                        input_size=input_size,
                        hidden_size=bilstm_hidden,
                        num_layers=bilstm_layers,
                        output_size=1
                    ).to(device)
                    
                    bilstm_history = predictor.train_model(
                        bilstm_model_name, bilstm_model, X_train, y_train, X_test, y_test,
                        epochs=epochs, batch_size=batch_size, lr=learning_rate
                    )
                    progress_bar.progress(50)
                    
                    status_text.text(f"训练 {transformer_version} 模型...")
                    transformer_model = transformer_model_class(
                        input_size=input_size,
                        d_model=trans_d_model,
                        nhead=trans_heads,
                        num_layers=trans_layers,
                        output_size=1
                    ).to(device)
                    
                    trans_history = predictor.train_model(
                        transformer_model_name, transformer_model, X_train, y_train, X_test, y_test,
                        epochs=epochs, batch_size=batch_size, lr=learning_rate
                    )
                    progress_bar.progress(100)
                    status_text.text("训练完成!")
                except Exception as e:
                    st.error(f"❌ 训练过程中出错: {str(e)}")
                    import traceback
                    st.error(f"详细错误信息: {traceback.format_exc()}")
                
                st.header("📊 模型性能对比")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"🧠 BiLSTM {bilstm_version} 评估结果")
                    bilstm_metrics, bilstm_preds, actuals = predictor.evaluate_model(bilstm_model_name, X_test, y_test)
                    
                    for metric, value in bilstm_metrics.items():
                        display_name = {
                            'MSE': '均方误差 (MSE)',
                            'RMSE': '均方根误差 (RMSE)',
                            'MAE': '平均绝对误差 (MAE)',
                            'R²': 'R² 评分',
                            'Direction_Accuracy': '方向准确率'
                        }.get(metric, metric)
                        st.metric(display_name, f"{value:.4f}")
                
                with col2:
                    st.subheader(f"🤖 Transformer {transformer_version} 评估结果")
                    trans_metrics, trans_preds, _ = predictor.evaluate_model(transformer_model_name, X_test, y_test)
                    
                    for metric, value in trans_metrics.items():
                        display_name = {
                            'MSE': '均方误差 (MSE)',
                            'RMSE': '均方根误差 (RMSE)',
                            'MAE': '平均绝对误差 (MAE)',
                            'R²': 'R² 评分',
                            'Direction_Accuracy': '方向准确率'
                        }.get(metric, metric)
                        st.metric(display_name, f"{value:.4f}")
                
                # 绘制预测对比图
                st.subheader("📈 预测效果对比")
                fig = go.Figure()
                
                # 实际值
                fig.add_trace(go.Scatter(
                    x=df['Date'].iloc[-len(actuals):],
                    y=actuals,
                    name='实际值',
                    line=dict(color='white', width=2)
                ))
                
                # BiLSTM预测值
                fig.add_trace(go.Scatter(
                    x=df['Date'].iloc[-len(bilstm_preds):],
                    y=bilstm_preds,
                    name=f'BiLSTM {bilstm_version} 预测',
                    line=dict(color='green', width=2, dash='dash')
                ))
                
                # Transformer预测值
                fig.add_trace(go.Scatter(
                    x=df['Date'].iloc[-len(trans_preds):],
                    y=trans_preds,
                    name=f'Transformer {transformer_version} 预测',
                    line=dict(color='blue', width=2, dash='dot')
                ))
                
                fig.update_layout(
                    xaxis_title='日期',
                    yaxis_title='价格',
                    title='实际值 vs 预测值',
                    template='plotly_dark',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 模型推荐
                st.header("🏆 模型推荐")
                
                # 比较方向准确率
                bilstm_dir_acc = bilstm_metrics['Direction_Accuracy']
                trans_dir_acc = trans_metrics['Direction_Accuracy']
                
                if bilstm_dir_acc > trans_dir_acc:
                    st.success(f"🧠 **BiLSTM 模型**表现更好！")
                    st.info(f"方向准确率: {bilstm_dir_acc:.2f} vs {trans_dir_acc:.2f}")
                elif trans_dir_acc > bilstm_dir_acc:
                    st.success(f"🤖 **Transformer 模型**表现更好！")
                    st.info(f"方向准确率: {trans_dir_acc:.2f} vs {bilstm_dir_acc:.2f}")
                else:
                    st.info(f"两个模型表现相当！方向准确率: {bilstm_dir_acc:.2f}")
                
                # 预测未来
                st.header("🔮 未来预测")
                days_to_predict = st.slider("预测未来天数", min_value=1, max_value=60, value=30, step=1)
                
                if st.button("📊 生成未来预测", use_container_width=True):
                    # 准备最后一个序列
                    last_data = df[feature_cols].values[-seq_length:]
                    scaled_last = predictor.scaler.transform(last_data)
                    
                    # 预测未来
                    bilstm_future = predictor.predict_future('BiLSTM', scaled_last, days=days_to_predict)
                    trans_future = predictor.predict_future('Transformer', scaled_last, days=days_to_predict)
                    
                    # 生成未来日期
                    last_date = df['Date'].iloc[-1]
                    future_dates = pd.date_range(last_date, periods=days_to_predict+1)[1:]
                    
                    # 绘制未来预测图
                    fig = go.Figure()
                    
                    # 历史数据
                    fig.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df['Close'],
                        name='历史数据',
                        line=dict(color='white', width=2)
                    ))
                    
                    # BiLSTM预测
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=bilstm_future,
                        name='BiLSTM 预测',
                        line=dict(color='green', width=2, dash='dash')
                    ))
                    
                    # Transformer预测
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=trans_future,
                        name='Transformer 预测',
                        line=dict(color='blue', width=2, dash='dot')
                    ))
                    
                    fig.update_layout(
                        xaxis_title='日期',
                        yaxis_title='价格',
                        title='未来价格预测',
                        template='plotly_dark',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示预测值
                    st.subheader("📋 预测结果")
                    pred_df = pd.DataFrame({
                        '日期': future_dates,
                        'BiLSTM 预测': bilstm_future,
                        'Transformer 预测': trans_future
                    })
                    st.dataframe(pred_df)
                    
                    # 保存模型
                    st.header("💾 模型保存")
                    model_name = st.text_input("模型名称", f"stock_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
                    if st.button("💾 保存模型", use_container_width=True):
                        try:
                            predictor.save_model('BiLSTM', save_dir='saved_models')
                            predictor.save_model('Transformer', save_dir='saved_models')
                            st.success("✅ 模型保存成功！")
                        except Exception as e:
                            st.error(f"❌ 保存失败: {str(e)}")
        except Exception as e:
            st.error(f"❌ 处理数据时出错: {str(e)}")
    else:
        st.info("请上传CSV文件或使用示例数据开始分析")
