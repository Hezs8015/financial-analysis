# 📊 股市预测模型对比 - BiLSTM vs Transformer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

这是一个基于Streamlit构建的交互式Web应用，用于比较双向LSTM（BiLSTM）和Transformer两种深度学习模型在股市预测任务中的性能表现。

## 🎯 功能特点

- **数据上传**: 支持上传CSV格式的股票数据
- **示例数据**: 内置示例数据用于快速体验
- **模型对比**: 同时训练并对比BiLSTM和Transformer模型
- **可视化分析**: 丰富的图表展示预测结果和性能指标
- **未来预测**: 基于训练好的模型预测未来股价走势

## 🧠 模型架构

### 双向LSTM (BiLSTM)
- 捕捉时间序列的前后向依赖关系
- 适用于具有时序依赖性的金融数据

### Transformer
- 利用自注意力机制建模长期依赖
- 并行计算能力强，训练效率高

## 📋 数据格式

上传的CSV文件应包含以下列：
- `Date`: 日期
- `Open`: 开盘价
- `High`: 最高价
- `Low`: 最低价
- `Close`: 收盘价
- `Volume`: 成交量

## 🚀 本地运行

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动应用
```bash
streamlit run app.py
```

## 🌐 在线部署

本项目已配置为可在Streamlit Cloud上直接部署：

1. 将代码推送到GitHub仓库
2. 访问 [Streamlit Cloud](https://streamlit.io/cloud)
3. 连接GitHub仓库并部署

## 📊 评估指标

- **MSE**: 均方误差
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **R²**: 决定系数
- **Direction Accuracy**: 涨跌方向预测准确率

## 🛠️ 技术栈

- **前端**: Streamlit
- **深度学习**: PyTorch
- **数据处理**: Pandas, NumPy, Scikit-learn
- **可视化**: Plotly

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！
