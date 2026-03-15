import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from models import BiLSTMModel, TransformerModel, StockPredictor
from model_comparison import MultiModelPredictor, BiLSTMModelPyTorch, TransformerModelPyTorch
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

# 模型选择
st.sidebar.markdown("**选择要训练的模型：**")
train_bilstm_pytorch = st.sidebar.checkbox("BiLSTM-PyTorch", value=True)
train_transformer_pytorch = st.sidebar.checkbox("Transformer-PyTorch", value=True)

# BiLSTM版本选择（增量训练）
if train_bilstm_pytorch:
    st.sidebar.markdown("**BiLSTM训练版本：**")
    bilstm_version = st.sidebar.selectbox(
        "选择版本",
        ["v1 (8轮)", "v2 (20轮)", "v3 (35轮)"],
        index=0,
        help="v1: 训练8轮 | v2: 基于v1继续训练12轮 | v3: 基于v2继续训练15轮"
    )
else:
    bilstm_version = "v1 (8轮)"

# Keras模型可选（需要TensorFlow）
keras_available = False
train_bilstm_keras = False

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
            
            # 从训练集中划分验证集（用于早停）
            val_size = int(0.1 * len(X_train))  # 10%作为验证集
            if val_size < 10:
                val_size = min(10, len(X_train) // 5)  # 至少10个样本
            
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train_sub = X_train[:-val_size]
            y_train_sub = y_train[:-val_size]
            
            st.info(f"📊 训练子集: {len(X_train_sub)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
            
            # 训练模型
            if st.button("🚀 开始训练模型", use_container_width=True):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    epoch_text = st.empty()
                    
                    # 计算选中的模型数量
                    selected_models = []
                    if train_bilstm_pytorch:
                        # 添加版本信息
                        version_suffix = bilstm_version.replace(" (", "-").replace(")", "")
                        selected_models.append(f'BiLSTM-PyTorch-{version_suffix}')
                    if train_transformer_pytorch:
                        selected_models.append('Transformer-PyTorch')
                    if keras_available and train_bilstm_keras:
                        selected_models.append('BiLSTM-Keras')
                    
                    if len(selected_models) == 0:
                        st.warning("⚠️ 请至少选择一个模型进行训练")
                        st.stop()
                    
                    total_models = len(selected_models)
                    model_results = {}
                    
                    # 创建多模型预测器
                    multi_predictor = MultiModelPredictor(seq_len=seq_length, device=device)
                    
                    for idx, model_name in enumerate(selected_models):
                        status_text.text(f"训练 {model_name}... ({idx+1}/{total_models})")
                        
                        def make_progress_callback(model_idx, total_m, model_n):
                            def callback(epoch, total_epochs, train_loss, val_loss, name):
                                base_progress = (model_idx / total_m) * 100
                                epoch_progress = (epoch / total_epochs) * (100 / total_m)
                                progress = int(base_progress + epoch_progress)
                                progress_bar.progress(min(progress, 99))
                                epoch_text.text(f"{model_n} - Epoch {epoch}/{total_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
                            return callback
                        
                        progress_cb = make_progress_callback(idx, total_models, model_name)
                        
                        try:
                            if 'BiLSTM-PyTorch' in model_name:
                                # 根据版本选择训练轮数
                                if bilstm_version == "v1 (8轮)":
                                    bilstm_epochs = 8
                                elif bilstm_version == "v2 (20轮)":
                                    bilstm_epochs = 20
                                else:  # v3 (35轮)
                                    bilstm_epochs = 35
                                
                                # 增量训练：v2和v3需要加载之前训练的模型
                                if bilstm_version in ["v2 (20轮)", "v3 (35轮)"]:
                                    # 先训练基础模型（v1）
                                    base_model = BiLSTMModelPyTorch(
                                        input_size=input_size,
                                        hidden_size=bilstm_hidden,
                                        num_layers=bilstm_layers,
                                        output_size=1
                                    ).to(device)
                                    
                                    # 训练8轮作为基础
                                    base_history = multi_predictor.train_pytorch_model(
                                        f'{model_name}-base', BiLSTMModelPyTorch,
                                        X_train_sub, y_train_sub, X_val, y_val,
                                        model_kwargs={'input_size': input_size, 'hidden_size': bilstm_hidden, 'num_layers': bilstm_layers},
                                        epochs=8, batch_size=batch_size, lr=learning_rate,
                                        verbose=False
                                    )
                                    
                                    # 加载基础模型继续训练
                                    model_kwargs = {'input_size': input_size, 'hidden_size': bilstm_hidden, 'num_layers': bilstm_layers}
                                    additional_epochs = bilstm_epochs - 8
                                    
                                    history = multi_predictor.train_pytorch_model(
                                        model_name, BiLSTMModelPyTorch,
                                        X_train_sub, y_train_sub, X_val, y_val,
                                        model_kwargs=model_kwargs,
                                        epochs=additional_epochs, batch_size=batch_size, lr=learning_rate,
                                        verbose=False,
                                        base_model=multi_predictor.models[f'{model_name}-base']
                                    )
                                else:
                                    # v1：直接训练8轮
                                    history = multi_predictor.train_pytorch_model(
                                        model_name, BiLSTMModelPyTorch,
                                        X_train_sub, y_train_sub, X_val, y_val,
                                        model_kwargs={'input_size': input_size, 'hidden_size': bilstm_hidden, 'num_layers': bilstm_layers},
                                        epochs=bilstm_epochs, batch_size=batch_size, lr=learning_rate,
                                        verbose=False
                                    )
                                
                                model_results[model_name] = history
                                
                            elif model_name == 'Transformer-PyTorch':
                                history = multi_predictor.train_pytorch_model(
                                    model_name, TransformerModelPyTorch,
                                    X_train_sub, y_train_sub, X_val, y_val,
                                    model_kwargs={'input_size': input_size, 'd_model': trans_d_model, 'nhead': trans_heads, 'num_layers': trans_layers},
                                    epochs=epochs, batch_size=batch_size, lr=learning_rate,
                                    verbose=False
                                )
                                model_results[model_name] = history
                                
                            elif model_name == 'BiLSTM-Keras' and keras_available:
                                # 使用Keras训练
                                from model_comparison import BiLSTMModelKeras
                                keras_model = BiLSTMModelKeras(seq_length, input_size, hidden_size=bilstm_hidden)
                                keras_model.build_model()
                                
                                val_size_keras = int(0.1 * len(X_train_sub))
                                if val_size_keras < 10:
                                    val_size_keras = min(10, len(X_train_sub) // 5)
                                
                                X_val_keras = X_train_sub[-val_size_keras:]
                                y_val_keras = y_train_sub[-val_size_keras:]
                                X_train_keras = X_train_sub[:-val_size_keras]
                                y_train_keras = y_train_sub[:-val_size_keras]
                                
                                history = keras_model.model.fit(
                                    X_train_keras, y_train_keras,
                                    validation_data=(X_val_keras, y_val_keras),
                                    epochs=epochs, batch_size=batch_size,
                                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                                    verbose=0
                                )
                                
                                multi_predictor.models[model_name] = keras_model
                                multi_predictor.histories[model_name] = {
                                    'train_loss': history.history['loss'],
                                    'val_loss': history.history['val_loss']
                                }
                                model_results[model_name] = multi_predictor.histories[model_name]
                            
                        except Exception as e:
                            st.error(f"❌ {model_name} 训练失败: {str(e)}")
                            continue
                    
                    progress_bar.progress(100)
                    epoch_text.empty()
                    status_text.text(f"训练完成! 成功训练 {len(model_results)}/{total_models} 个模型")
                    
                    # 保存到session state
                    st.session_state['multi_predictor'] = multi_predictor
                    st.session_state['model_results'] = model_results
                    
                except Exception as e:
                    st.error(f"❌ 训练过程中出错: {str(e)}")
                    import traceback
                    st.error(f"详细错误信息: {traceback.format_exc()}")
                except Exception as e:
                    st.error(f"❌ 训练过程中出错: {str(e)}")
                    import traceback
                    st.error(f"详细错误信息: {traceback.format_exc()}")
                
                st.header("📊 模型性能对比")
                
                # 获取所有模型的评估结果
                all_metrics = {}
                all_predictions = {}
                
                if 'multi_predictor' in st.session_state:
                    multi_predictor = st.session_state['multi_predictor']
                    comparison_results = multi_predictor.compare_models(X_test, y_test)
                    
                    for model_name, result in comparison_results.items():
                        if 'error' not in result:
                            all_metrics[model_name] = result['metrics']
                            all_predictions[model_name] = result['predictions']
                
                if len(all_metrics) == 0:
                    st.error("❌ 没有可用的模型评估结果")
                    st.stop()
                
                # 创建对比表格
                st.subheader("📋 评估指标对比表")
                
                # 准备对比数据
                metrics_df_data = []
                for model_name, metrics in all_metrics.items():
                    row = {'模型': model_name}
                    row.update(metrics)
                    metrics_df_data.append(row)
                
                metrics_df = pd.DataFrame(metrics_df_data)
                
                # 高亮最佳值
                def highlight_best(s):
                    if s.name == '模型':
                        return [''] * len(s)
                    # 对于R²和方向准确率，越高越好
                    if s.name in ['R²', '方向准确率']:
                        is_best = s == s.max()
                    else:  # 对于误差指标，越低越好
                        is_best = s == s.min()
                    return ['background-color: rgba(0, 255, 0, 0.3)' if v else '' for v in is_best]
                
                st.dataframe(
                    metrics_df.style.apply(highlight_best, axis=0).format({
                        'MSE': '{:.6f}',
                        'RMSE': '{:.6f}',
                        'MAE': '{:.6f}',
                        'R²': '{:.4f}',
                        '方向准确率': '{:.4f}'
                    }),
                    use_container_width=True
                )
                
                # 找出最佳模型
                best_direction_model = max(all_metrics.items(), key=lambda x: x[1]['方向准确率'])
                best_r2_model = max(all_metrics.items(), key=lambda x: x[1]['R²'])
                best_rmse_model = min(all_metrics.items(), key=lambda x: x[1]['RMSE'])
                
                # 显示最佳模型
                st.subheader("🏆 最佳模型")
                
                best_cols = st.columns(3)
                with best_cols[0]:
                    st.metric(
                        "🎯 最佳方向预测",
                        best_direction_model[0],
                        f"{best_direction_model[1]['方向准确率']:.2%}"
                    )
                with best_cols[1]:
                    st.metric(
                        "📈 最佳拟合度 (R²)",
                        best_r2_model[0],
                        f"{best_r2_model[1]['R²']:.4f}"
                    )
                with best_cols[2]:
                    st.metric(
                        "📉 最小误差 (RMSE)",
                        best_rmse_model[0],
                        f"{best_rmse_model[1]['RMSE']:.4f}"
                    )
                
                # 绘制预测对比图
                st.subheader("📈 预测效果对比")
                fig = go.Figure()
                
                # 获取实际值（使用第一个模型的实际值）
                first_model = list(all_predictions.keys())[0]
                actuals = comparison_results[first_model]['actuals']
                
                # 实际值
                fig.add_trace(go.Scatter(
                    x=df['Date'].iloc[-len(actuals):],
                    y=actuals,
                    name='实际值',
                    line=dict(color='white', width=3)
                ))
                
                # 颜色映射
                colors = {
                    'BiLSTM-PyTorch-v1': '#00ff00',
                    'BiLSTM-PyTorch-v2': '#00cc00',
                    'BiLSTM-PyTorch-v3': '#009900',
                    'Transformer-PyTorch': '#0080ff',
                    'BiLSTM-Keras': '#ff8000'
                }
                
                # 添加每个模型的预测值
                for model_name, preds in all_predictions.items():
                    color = colors.get(model_name, '#808080')
                    fig.add_trace(go.Scatter(
                        x=df['Date'].iloc[-len(preds):],
                        y=preds,
                        name=f'{model_name} 预测',
                        line=dict(color=color, width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    xaxis_title='日期',
                    yaxis_title='价格',
                    title='实际值 vs 各模型预测值',
                    template='plotly_dark',
                    height=500,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 训练历史对比
                if 'model_results' in st.session_state:
                    st.subheader("📉 训练历史对比")
                    
                    history_fig = go.Figure()
                    
                    for model_name, history in st.session_state['model_results'].items():
                        if 'val_loss' in history and len(history['val_loss']) > 0:
                            color = colors.get(model_name, '#808080')
                            history_fig.add_trace(go.Scatter(
                                y=history['val_loss'],
                                name=f'{model_name} 验证损失',
                                line=dict(color=color, width=2)
                            ))
                    
                    history_fig.update_layout(
                        xaxis_title='Epoch',
                        yaxis_title='验证损失 (MSE)',
                        title='模型训练收敛对比',
                        template='plotly_dark',
                        height=400
                    )
                    st.plotly_chart(history_fig, use_container_width=True)
                
                # 模型推荐总结
                st.header("💡 模型选择建议")
                
                # 综合评分
                st.markdown("### 📊 综合评分")
                
                for model_name, metrics in all_metrics.items():
                    # 计算综合得分 (方向准确率权重最高)
                    score = (
                        metrics['方向准确率'] * 0.5 +
                        metrics['R²'] * 0.3 +
                        (1 - min(metrics['RMSE'] / 100, 1)) * 0.2
                    )
                    
                    if score >= 0.7:
                        emoji = "🟢"
                        recommendation = "强烈推荐"
                    elif score >= 0.5:
                        emoji = "🟡"
                        recommendation = "推荐使用"
                    else:
                        emoji = "🔴"
                        recommendation = "建议优化"
                    
                    with st.expander(f"{emoji} {model_name} - {recommendation} (得分: {score:.2f})"):
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("方向准确率", f"{metrics['方向准确率']:.2%}")
                        with cols[1]:
                            st.metric("R²", f"{metrics['R²']:.4f}")
                        with cols[2]:
                            st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                        with cols[3]:
                            st.metric("MAE", f"{metrics['MAE']:.4f}")
                
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
        st.info("请上传CSV文件、使用示例数据或获取实时数据开始分析")
