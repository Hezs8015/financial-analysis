"""
股票预测模型验证损失+测试损失对比曲线 - Matplotlib版本
生成高清静态图表，适用于论文、报告等场景
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ==================== 配置参数 ====================
# 设置中文字体支持（Windows系统）
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 数据定义（可替换为真实实验数据） ====================
# 训练轮次范围（可修改接口）
EPOCHS = 20  # 总训练轮次
train_epochs = np.arange(1, EPOCHS + 1)  # 生成1到20的轮次数组

# BiLSTM模型损失数据（示例数据，实际使用时替换为真实训练日志）
# 模拟收敛曲线：前期快速下降，后期趋于平稳
bilstm_val_loss = np.array([
    0.085, 0.072, 0.065, 0.058, 0.052, 0.048, 0.045, 0.042, 0.040, 0.038,
    0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.031, 0.030, 0.030, 0.029
])
bilstm_test_loss = np.array([
    0.088, 0.075, 0.068, 0.061, 0.055, 0.051, 0.047, 0.045, 0.042, 0.040,
    0.039, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.031, 0.031
])

# Transformer模型损失数据（示例数据，实际使用时替换为真实训练日志）
# 通常Transformer收敛更快但最终损失可能略高
transformer_val_loss = np.array([
    0.082, 0.068, 0.058, 0.050, 0.045, 0.041, 0.038, 0.035, 0.033, 0.031,
    0.030, 0.029, 0.028, 0.027, 0.026, 0.025, 0.025, 0.024, 0.024, 0.023
])
transformer_test_loss = np.array([
    0.085, 0.071, 0.061, 0.053, 0.048, 0.044, 0.040, 0.037, 0.035, 0.033,
    0.032, 0.031, 0.030, 0.029, 0.028, 0.027, 0.027, 0.026, 0.026, 0.025
])

# ==================== 配色方案 ====================
# BiLSTM：橙色系
colors = {
    'bilstm_val': '#FF7F0E',      # 验证损失 - 深橙色
    'bilstm_test': '#FFBC70',     # 测试损失 - 浅橙色
    'transformer_val': '#1F77B4', # 验证损失 - 深蓝色
    'transformer_test': '#72A8D8' # 测试损失 - 浅蓝色
}

# ==================== 创建图表 ====================
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)  # 高清输出设置

# 线条宽度配置
LINE_WIDTH = 2.5

# 绘制BiLSTM曲线（橙色系）
ax.plot(train_epochs, bilstm_val_loss,
        color=colors['bilstm_val'],
        linewidth=LINE_WIDTH,
        linestyle='-',  # 实线表示验证损失
        marker='o',     # 圆形标记
        markersize=6,
        markerfacecolor=colors['bilstm_val'],
        markeredgecolor='white',
        markeredgewidth=1,
        label='BiLSTM - Validation Loss',
        zorder=3)

ax.plot(train_epochs, bilstm_test_loss,
        color=colors['bilstm_test'],
        linewidth=LINE_WIDTH,
        linestyle='--',  # 虚线表示测试损失
        marker='o',
        markersize=6,
        markerfacecolor=colors['bilstm_test'],
        markeredgecolor='white',
        markeredgewidth=1,
        label='BiLSTM - Test Loss',
        zorder=3)

# 绘制Transformer曲线（蓝色系）
ax.plot(train_epochs, transformer_val_loss,
        color=colors['transformer_val'],
        linewidth=LINE_WIDTH,
        linestyle='-',  # 实线表示验证损失
        marker='s',     # 方形标记（与BiLSTM区分）
        markersize=6,
        markerfacecolor=colors['transformer_val'],
        markeredgecolor='white',
        markeredgewidth=1,
        label='Transformer - Validation Loss',
        zorder=3)

ax.plot(train_epochs, transformer_test_loss,
        color=colors['transformer_test'],
        linewidth=LINE_WIDTH,
        linestyle='--',  # 虚线表示测试损失
        marker='s',
        markersize=6,
        markerfacecolor=colors['transformer_test'],
        markeredgecolor='white',
        markeredgewidth=1,
        label='Transformer - Test Loss',
        zorder=3)

# ==================== 图表美化 ====================
# 设置坐标轴范围（适配金融时序预测场景，0-0.1区间）
ax.set_xlim(0.5, EPOCHS + 0.5)
ax.set_ylim(0.02, 0.10)

# 设置刻度
ax.set_xticks(np.arange(0, EPOCHS + 1, 2))  # 每隔2轮显示刻度
ax.set_yticks(np.arange(0.02, 0.11, 0.01))  # 损失值刻度，步长0.01

# 坐标轴标签
ax.set_xlabel('Training Epochs (轮次)', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss Value (损失值)', fontsize=12, fontweight='bold')

# 图表标题（突出股票预测场景）
ax.set_title('Stock Price Prediction: Model Loss Comparison\n'
             '股票价格预测：BiLSTM vs Transformer 损失对比',
             fontsize=14, fontweight='bold', pad=20)

# 网格线设置（半透明）
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
ax.set_axisbelow(True)  # 网格线置于数据下方

# 图例设置
legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.95,
                   edgecolor='gray', fancybox=True, shadow=True)
legend.get_frame().set_facecolor('white')

# 调整布局
plt.tight_layout()

# ==================== 保存与显示 ====================
# 保存高清图片
output_path = 'stock_prediction_loss_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"图表已保存至: {output_path}")

# 显示图表
plt.show()

print("\n" + "="*60)
print("使用说明：")
print("1. 修改 EPOCHS 变量可调整训练轮次范围")
print("2. 替换 bilstm_val_loss, bilstm_test_loss 等数组为真实实验数据")
print("3. 调整 ax.set_ylim() 可适配不同的损失值范围")
print("4. 输出图片分辨率为300DPI，适合论文插入")
print("="*60)
