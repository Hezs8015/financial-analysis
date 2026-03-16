"""
股票预测模型验证损失+测试损失对比曲线 - Plotly版本
生成交互式图表，适用于Streamlit、HTML页面嵌入等场景
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ==================== 数据定义（可替换为真实实验数据） ====================
# 训练轮次范围（可修改接口）
EPOCHS = 20  # 总训练轮次
train_epochs = list(range(1, EPOCHS + 1))  # 生成1到20的轮次列表

# BiLSTM模型损失数据（示例数据，实际使用时替换为真实训练日志）
# 模拟收敛曲线：前期快速下降，后期趋于平稳
bilstm_val_loss = [
    0.085, 0.072, 0.065, 0.058, 0.052, 0.048, 0.045, 0.042, 0.040, 0.038,
    0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.031, 0.030, 0.030, 0.029
]
bilstm_test_loss = [
    0.088, 0.075, 0.068, 0.061, 0.055, 0.051, 0.047, 0.045, 0.042, 0.040,
    0.039, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.031, 0.031
]

# Transformer模型损失数据（示例数据，实际使用时替换为真实训练日志）
# 通常Transformer收敛更快但最终损失可能略高
transformer_val_loss = [
    0.082, 0.068, 0.058, 0.050, 0.045, 0.041, 0.038, 0.035, 0.033, 0.031,
    0.030, 0.029, 0.028, 0.027, 0.026, 0.025, 0.025, 0.024, 0.024, 0.023
]
transformer_test_loss = [
    0.085, 0.071, 0.061, 0.053, 0.048, 0.044, 0.040, 0.037, 0.035, 0.033,
    0.032, 0.031, 0.030, 0.029, 0.028, 0.027, 0.027, 0.026, 0.026, 0.025
]

# ==================== 配色方案 ====================
colors = {
    'bilstm_val': '#FF7F0E',      # BiLSTM验证损失 - 深橙色
    'bilstm_test': '#FFBC70',     # BiLSTM测试损失 - 浅橙色
    'transformer_val': '#1F77B4', # Transformer验证损失 - 深蓝色
    'transformer_test': '#72A8D8' # Transformer测试损失 - 浅蓝色
}

# ==================== 创建交互式图表 ====================
fig = go.Figure()

# 线条宽度配置
LINE_WIDTH = 2.5
MARKER_SIZE = 8

# 添加BiLSTM验证损失曲线（实线+圆形标记）
fig.add_trace(go.Scatter(
    x=train_epochs,
    y=bilstm_val_loss,
    mode='lines+markers',
    name='BiLSTM - 验证损失',
    line=dict(color=colors['bilstm_val'], width=LINE_WIDTH, dash='solid'),
    marker=dict(
        symbol='circle',
        size=MARKER_SIZE,
        color=colors['bilstm_val'],
        line=dict(color='white', width=1)
    ),
    hovertemplate='<b>BiLSTM 验证损失</b><br>' +
                  '轮次: %{x}<br>' +
                  '损失值: %{y:.4f}<br>' +
                  '<extra></extra>'
))

# 添加BiLSTM测试损失曲线（虚线+圆形标记）
fig.add_trace(go.Scatter(
    x=train_epochs,
    y=bilstm_test_loss,
    mode='lines+markers',
    name='BiLSTM - 测试损失',
    line=dict(color=colors['bilstm_test'], width=LINE_WIDTH, dash='dash'),
    marker=dict(
        symbol='circle',
        size=MARKER_SIZE,
        color=colors['bilstm_test'],
        line=dict(color='white', width=1)
    ),
    hovertemplate='<b>BiLSTM 测试损失</b><br>' +
                  '轮次: %{x}<br>' +
                  '损失值: %{y:.4f}<br>' +
                  '<extra></extra>'
))

# 添加Transformer验证损失曲线（实线+方形标记）
fig.add_trace(go.Scatter(
    x=train_epochs,
    y=transformer_val_loss,
    mode='lines+markers',
    name='Transformer - 验证损失',
    line=dict(color=colors['transformer_val'], width=LINE_WIDTH, dash='solid'),
    marker=dict(
        symbol='square',
        size=MARKER_SIZE,
        color=colors['transformer_val'],
        line=dict(color='white', width=1)
    ),
    hovertemplate='<b>Transformer 验证损失</b><br>' +
                  '轮次: %{x}<br>' +
                  '损失值: %{y:.4f}<br>' +
                  '<extra></extra>'
))

# 添加Transformer测试损失曲线（虚线+方形标记）
fig.add_trace(go.Scatter(
    x=train_epochs,
    y=transformer_test_loss,
    mode='lines+markers',
    name='Transformer - 测试损失',
    line=dict(color=colors['transformer_test'], width=LINE_WIDTH, dash='dash'),
    marker=dict(
        symbol='square',
        size=MARKER_SIZE,
        color=colors['transformer_test'],
        line=dict(color='white', width=1)
    ),
    hovertemplate='<b>Transformer 测试损失</b><br>' +
                  '轮次: %{x}<br>' +
                  '损失值: %{y:.4f}<br>' +
                  '<extra></extra>'
))

# ==================== 图表布局配置 ====================
fig.update_layout(
    # 图表标题（突出股票预测场景）
    title=dict(
        text='<b>股票价格预测：BiLSTM vs Transformer 损失对比</b><br>' +
             '<sup>Stock Price Prediction: Model Loss Comparison</sup>',
        font=dict(size=18, color='#333333'),
        x=0.5,  # 居中
        xanchor='center'
    ),

    # 坐标轴配置
    xaxis=dict(
        title=dict(
            text='训练轮次 (Epochs)',
            font=dict(size=14, color='#333333')
        ),
        tickmode='linear',
        tick0=0,
        dtick=2,  # 每隔2轮显示刻度
        range=[0.5, EPOCHS + 0.5],
        gridcolor='rgba(128, 128, 128, 0.3)',  # 半透明网格线
        gridwidth=1,
        showgrid=True,
        zeroline=False
    ),

    yaxis=dict(
        title=dict(
            text='损失值 (Loss)',
            font=dict(size=14, color='#333333')
        ),
        range=[0.02, 0.10],  # 适配金融时序预测场景
        dtick=0.01,  # 刻度步长0.01
        gridcolor='rgba(128, 128, 128, 0.3)',  # 半透明网格线
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        tickformat='.3f'  # 显示3位小数
    ),

    # 图例配置
    legend=dict(
        x=0.99,
        y=0.99,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='gray',
        borderwidth=1,
        font=dict(size=11)
    ),

    # 画布配置
    plot_bgcolor='white',
    paper_bgcolor='white',
    width=1000,
    height=600,

    # 悬停配置
    hovermode='x unified',  # 统一显示同一x值的所有数据点

    # 边距配置
    margin=dict(l=80, r=50, t=100, b=60)
)

# ==================== 导出与显示 ====================

# 1. 保存为HTML文件（可嵌入网页）
html_output = 'stock_prediction_loss_comparison.html'
fig.write_html(html_output, include_plotlyjs='cdn')
print(f"交互式HTML图表已保存至: {html_output}")

# 2. 保存为静态图片（PNG格式）- 需要安装kaleido包
# 如需导出静态图片，请运行: pip install kaleido
try:
    image_output = 'stock_prediction_loss_comparison_plotly.png'
    fig.write_image(image_output, scale=2)  # 2倍分辨率
    print(f"静态图片已保存至: {image_output}")
except Exception as e:
    print(f"静态图片导出失败（可选功能）: {e}")
    print("提示: 运行 'pip install kaleido' 可启用静态图片导出功能")

# 3. 显示图表（在Jupyter Notebook或支持的环境中）
fig.show()

# ==================== Streamlit集成示例 ====================
"""
# 在Streamlit中使用的示例代码：

import streamlit as st

st.title("股票预测模型损失对比")
st.plotly_chart(fig, use_container_width=True)

# 或者使用HTML嵌入：
with open('stock_prediction_loss_comparison.html', 'r', encoding='utf-8') as f:
    html_content = f.read()
st.components.v1.html(html_content, height=600)
"""

print("\n" + "="*60)
print("使用说明：")
print("1. 修改 EPOCHS 变量可调整训练轮次范围")
print("2. 替换 bilstm_val_loss, bilstm_test_loss 等列表为真实实验数据")
print("3. 调整 yaxis.range 可适配不同的损失值范围")
print("4. HTML文件可直接在浏览器中打开，支持交互操作")
print("5. 支持悬停查看具体数值、缩放、平移等交互功能")
print("="*60)
