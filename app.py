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

st.sidebar.header("⚙️ 配置参数")

st.sidebar.subheader("📁 数据上传")
uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=['csv'])

use_sample = st.sidebar.checkbox("使用示例数据", value=False)

st.sidebar.subheader("🔧 模型参数")
seq_length = st.sidebar.slider("序列长度", min_value=10, max_value=120, value=60, step=10)
train_split = st.sidebar.slider("训练集比例", min_value=0.5, max_value=0.9, value=0.8, step=0.05)

st.sidebar.subheader("🧠 BiLSTM参数")
bilstm_hidden = st.sidebar.slider("隐藏层大小", min_value=32, max_value=256, value=128, step=32)
bilstm_layers = st.sidebar.slider("LSTM层数", min_value=1, max_value=4, value=2)

st.sidebar.subheader("🤖 Transformer参数")
trans_d_model = st.sidebar.slider("模型维度", min_value=32, max_value=256, value=128, step=32)
trans_heads = st.sidebar.slider("注意力头数", min_value=2, max_value=16, value=8, step=2)
trans_layers = st.sidebar.slider("编码器层数", min_value=1, max_value=6, value=2)

st.sidebar.subheader("⚡ 训练参数")
epochs = st.sidebar.slider("训练轮数", min_value=10, max_value=200, value=50, step=10)
batch_size = st.sidebar.slider("批次大小", min_value=8, max_value=128, value=32, step=8)
learning_rate = st.sidebar.select_slider("学习率", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)

st.sidebar.subheader("🔮 预测设置")
prediction_days = st.sidebar.slider("预测未来天数", min_value=7, max_value=90, value=30, step=7)

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
            st.write(f"使用的特征: {feature_cols}")
            
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
                line=dict(color='black', width=2)
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
                template='plotly_white'
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
                line=dict(color='black', width=2)
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
                template='plotly_white'
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
            
        except Exception as e:
            st.error(f"训练过程中出现错误: {str(e)}")
            st.exception(e)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Made with ❤️ using Streamlit | 股市预测模型对比分析</div>", unsafe_allow_html=True)
