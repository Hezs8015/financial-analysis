import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from models import BiLSTMModel, TransformerModel, StockPredictor
from model_comparison import MultiModelPredictor, BiLSTMModelPyTorch, TransformerModelPyTorch, TransformerModelPyTorchV2
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
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📊 股市预测模型对比平台</p>', unsafe_allow_html=True)

st.sidebar.title("⚙️ 配置面板")

st.sidebar.header("📚 使用说明")
if st.sidebar.checkbox("📖 查看使用指南"):
    st.sidebar.markdown("""
    **快速开始：**
    1. 上传CSV格式的股票数据
    2. 选择要训练的模型
    3. 调整模型参数（可选）
    4. 点击"开始训练"
    5. 查看对比结果
    
    **数据格式要求：**
    - 必须包含列：Date, Open, High, Low, Close, Volume
    - 日期格式：YYYY-MM-DD
    - 数据行数：建议至少200行
    """)

st.sidebar.header("🔧 模型选择")

st.sidebar.markdown("**BiLSTM版本：**")
bilstm_version = st.sidebar.selectbox(
    "选择BiLSTM版本",
    ["v1 (基础版)", "v2 (增强版)", "v3 (高级版)"],
    index=0,
    help="v1: 基础BiLSTM | v2: 增强版 | v3: 高级版"
)

train_transformer_pytorch = st.sidebar.checkbox("✅ 训练 Transformer-PyTorch", value=True)

if train_transformer_pytorch:
    st.sidebar.markdown("**Transformer版本：**")
    transformer_version = st.sidebar.selectbox(
        "选择版本",
        ["v1 (基础版)", "v2 (增强版)"],
        index=0,
        help="v1: 基础版Transformer | v2: 增强版（带门控残差、多尺度聚合、因果掩码）"
    )
else:
    transformer_version = "v1 (基础版)"

seq_length = st.sidebar.slider("序列长度", min_value=10, max_value=120, value=60, step=10)
epochs = st.sidebar.slider("训练轮数", min_value=10, max_value=200, value=50, step=10)

st.sidebar.header("📊 模型说明")
model_info = st.sidebar.radio("选择模型查看详情", ["BiLSTM", "Transformer"])

if model_info == "BiLSTM":
    st.sidebar.markdown("""
    **BiLSTM（双向长短期记忆网络）**
    
    🧠 **运作原理：**
    - 同时学习过去和未来的时间依赖
    - 通过门控机制控制信息流
    - 适合捕捉长期时序模式
    
    📊 **版本差异：**
    - **v1**: 基础BiLSTM结构
    - **v2**: 增加Dropout和正则化
    - **v3**: 多层堆叠+注意力机制
    """)
else:
    st.sidebar.markdown("""
    **Transformer（自注意力机制）**
    
    🧠 **运作原理：**
    - 自注意力机制捕捉全局依赖
    - 并行计算，训练效率高
    - 位置编码保留时序信息
    
    📊 **版本差异：**
    - **v1**: 基础Transformer编码器
    - **v2**: 门控残差+多尺度聚合+因果掩码
    """)

st.header("📁 数据上传")

uploaded_file = st.file_uploader("上传股票数据CSV文件", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ 数据加载成功！共 {len(df)} 行")
        
        st.subheader("📋 数据预览")
        st.dataframe(df.head(10))
        
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ 缺少必要的列: {', '.join(missing_cols)}")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            
            st.subheader("📈 数据可视化")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Close'],
                name='收盘价',
                line=dict(color='#FF00FF', width=2)
            ))
            fig.update_layout(
                xaxis_title='日期',
                yaxis_title='价格',
                title='股票价格走势',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("🚀 开始训练", use_container_width=True):
                with st.spinner("正在训练模型..."):
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    predictor = MultiModelPredictor(seq_len=seq_length, device=device)
                    
                    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    X, y, scaler, available_cols, target_idx = predictor.prepare_data(df, feature_cols, 'Close')
                    
                    train_size = int(0.8 * len(X))
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    val_size = int(0.1 * len(X_train))
                    if val_size < 10:
                        val_size = min(10, len(X_train) // 5)
                    
                    X_val = X_train[-val_size:]
                    y_val = y_train[-val_size:]
                    X_train_sub = X_train[:-val_size]
                    y_train_sub = y_train[:-val_size]
                    
                    st.info(f"📊 数据划分: 训练 {len(X_train_sub)}, 验证 {len(X_val)}, 测试 {len(X_test)}")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = {}
                    all_predictions = {}
                    
                    # 训练BiLSTM
                    status_text.text("🔄 正在训练 BiLSTM...")
                    try:
                        input_size = len(available_cols)
                        if bilstm_version == "v1 (基础版)":
                            model_kwargs = {'input_size': input_size, 'hidden_size': 64, 'num_layers': 2}
                        elif bilstm_version == "v2 (增强版)":
                            model_kwargs = {'input_size': input_size, 'hidden_size': 128, 'num_layers': 3}
                        else:  # v3
                            model_kwargs = {'input_size': input_size, 'hidden_size': 256, 'num_layers': 4}
                        
                        history = predictor.train_pytorch_model(
                            f'BiLSTM-PyTorch-{bilstm_version.split()[0]}', BiLSTMModelPyTorch,
                            X_train_sub, y_train_sub, X_val, y_val,
                            model_kwargs=model_kwargs,
                            epochs=epochs, batch_size=32, lr=0.001, verbose=False
                        )
                        results[f'BiLSTM-PyTorch-{bilstm_version.split()[0]}'] = history
                        st.success(f"✅ BiLSTM {bilstm_version} 训练完成！")
                    except Exception as e:
                        st.error(f"❌ BiLSTM 训练失败: {str(e)}")
                    
                    progress_bar.progress(50)
                    
                    # 训练Transformer
                    if train_transformer_pytorch:
                        status_text.text("🔄 正在训练 Transformer...")
                        try:
                            input_size = len(available_cols)
                            if transformer_version == "v1 (基础版)":
                                model_kwargs = {'input_size': input_size, 'd_model': 64, 'nhead': 8, 'num_layers': 2}
                            else:  # v2
                                model_kwargs = {'input_size': input_size, 'd_model': 128, 'nhead': 4, 'num_layers': 3}
                            
                            history = predictor.train_pytorch_model(
                                f'Transformer-PyTorch-{transformer_version.split()[0]}', 
                                TransformerModelPyTorch if transformer_version == "v1 (基础版)" else TransformerModelPyTorchV2,
                                X_train_sub, y_train_sub, X_val, y_val,
                                model_kwargs=model_kwargs,
                                epochs=epochs, batch_size=32, lr=0.001, verbose=False
                            )
                            results[f'Transformer-PyTorch-{transformer_version.split()[0]}'] = history
                            st.success(f"✅ Transformer {transformer_version} 训练完成！")
                        except Exception as e:
                            st.error(f"❌ Transformer 训练失败: {str(e)}")
                    
                    progress_bar.progress(100)
                    status_text.text("✅ 所有模型训练完成！")
                    
                    # 评估模型
                    st.header("📊 模型评估对比")
                    
                    comparison_results = predictor.compare_models(X_test, y_test)
                    
                    metrics_data = []
                    for model_name, result in comparison_results.items():
                        if 'error' not in result:
                            metrics = result['metrics']
                            all_predictions[model_name] = result['predictions']
                            metrics_data.append({
                                '模型': model_name,
                                '方向准确率': f"{metrics['方向准确率']:.2%}",
                                'R²': f"{metrics['R²']:.4f}",
                                'RMSE': f"{metrics['RMSE']:.4f}",
                                'MAE': f"{metrics['MAE']:.4f}"
                            })
                    
                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # 预测对比图
                        st.subheader("📈 预测结果对比")
                        
                        fig = go.Figure()
                        
                        # 实际值
                        fig.add_trace(go.Scatter(
                            x=df['Date'].iloc[-len(y_test):],
                            y=y_test,
                            name='实际值',
                            line=dict(color='#FF00FF', width=3)
                        ))
                        
                        # 高对比度颜色方案
                        colors = {
                            'BiLSTM-PyTorch-v1': '#FF0000',  # 红色
                            'BiLSTM-PyTorch-v2': '#FF4500',  # 橙红色
                            'BiLSTM-PyTorch-v3': '#FF8C00',  # 深橙色
                            'Transformer-PyTorch-v1': '#0000FF',  # 蓝色
                            'Transformer-PyTorch-v2': '#4B0082',  # 靛蓝色
                            'BiLSTM-Keras': '#00FF00'  # 绿色
                        }
                        
                        for model_name, preds in all_predictions.items():
                            color = colors.get(model_name, '#FFFF00')
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
                        if results:
                            st.subheader("📉 训练历史对比")
                            
                            history_fig = go.Figure()
                            
                            for model_name, history in results.items():
                                if 'val_loss' in history and len(history['val_loss']) > 0:
                                    color = colors.get(model_name, '#FFFF00')
                                    history_fig.add_trace(go.Scatter(
                                        y=history['val_loss'],
                                        name=f'{model_name} 验证损失',
                                        line=dict(color=color, width=2)
                                    ))
                            
                            history_fig.update_layout(
                                xaxis_title='轮次',
                                yaxis_title='验证损失',
                                title='训练过程对比',
                                template='plotly_dark',
                                height=400
                            )
                            st.plotly_chart(history_fig, use_container_width=True)
                        
                        # 保存结果到session state
                        st.session_state['predictor'] = predictor
                        st.session_state['model_results'] = results
                        st.session_state['all_predictions'] = all_predictions
                        st.session_state['df'] = df
                        st.session_state['y_test'] = y_test
                        st.session_state['feature_cols'] = feature_cols
                        st.session_state['seq_length'] = seq_length
                        st.session_state['bilstm_version'] = bilstm_version
                        st.session_state['transformer_version'] = transformer_version
                        
    except Exception as e:
        st.error(f"❌ 处理数据时出错: {str(e)}")
else:
    st.info("👆 请上传CSV文件开始分析")
    
    st.markdown("""
    ### 📋 示例数据格式
    
    您的CSV文件应包含以下列：
    
    | Date | Open | High | Low | Close | Volume |
    |------|------|------|-----|-------|--------|
    | 2023-01-01 | 100.0 | 105.0 | 98.0 | 102.0 | 1000000 |
    | 2023-01-02 | 102.0 | 108.0 | 101.0 | 106.0 | 1200000 |
    
    **提示：** 数据越多（建议至少200行），模型训练效果越好。
    """)

# 未来预测部分
if 'predictor' in st.session_state:
    st.header("🔮 未来预测")
    days_to_predict = st.slider("预测未来天数", min_value=1, max_value=60, value=30, step=1)
    
    if st.button("📊 生成未来预测", use_container_width=True):
        predictor = st.session_state['predictor']
        df = st.session_state['df']
        feature_cols = st.session_state['feature_cols']
        seq_length = st.session_state['seq_length']
        bilstm_version = st.session_state.get('bilstm_version', 'v1 (基础版)')
        transformer_version = st.session_state.get('transformer_version', 'v1 (基础版)')
        
        # 准备最后一个序列
        last_data = df[feature_cols].values[-seq_length:]
        scaled_last = predictor.scalers[list(predictor.scalers.keys())[0]].transform(last_data)
        
        # 获取模型预测
        bilstm_model_name = f'BiLSTM-PyTorch-{bilstm_version.split()[0]}'
        trans_model_name = f'Transformer-PyTorch-{transformer_version.split()[0]}'
        
        bilstm_future = []
        trans_future = []
        
        if bilstm_model_name in predictor.models:
            bilstm_future = predictor.predict_future(bilstm_model_name, scaled_last, days=days_to_predict)
        if trans_model_name in predictor.models:
            trans_future = predictor.predict_future(trans_model_name, scaled_last, days=days_to_predict)
        
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
            line=dict(color='#FF00FF', width=2)
        ))
        
        # 高对比度颜色 - BiLSTM使用暖色调，Transformer使用冷色调
        bilstm_colors = {
            'v1 (基础版)': '#FF0000',    # 红色
            'v2 (增强版)': '#FF4500',    # 橙红色
            'v3 (高级版)': '#FF8C00'     # 深橙色
        }
        
        trans_colors = {
            'v1 (基础版)': '#0000FF',    # 蓝色
            'v2 (增强版)': '#4B0082'     # 靛蓝色
        }
        
        bilstm_color = bilstm_colors.get(bilstm_version, '#FF0000')
        trans_color = trans_colors.get(transformer_version, '#0000FF')
        
        # BiLSTM预测
        if bilstm_future:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=bilstm_future,
                name=f'BiLSTM {bilstm_version} 预测',
                line=dict(color=bilstm_color, width=2, dash='dash')
            ))
        
        # Transformer预测
        if trans_future:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=trans_future,
                name=f'Transformer {transformer_version} 预测',
                line=dict(color=trans_color, width=2, dash='dot')
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
        if bilstm_future or trans_future:
            st.subheader("📋 预测结果")
            pred_data = {'日期': future_dates}
            if bilstm_future:
                pred_data[f'BiLSTM {bilstm_version} 预测'] = bilstm_future
            if trans_future:
                pred_data[f'Transformer {transformer_version} 预测'] = trans_future
            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>股市预测模型对比平台 © 2024</p>", unsafe_allow_html=True)
