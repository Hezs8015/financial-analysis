import streamlit as st

st.set_page_config(
    page_title="模型介绍与参考文献",
    page_icon="📚",
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
    .model-card {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .section-divider {
        border-top: 2px solid #1f77b4;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📚 模型介绍与参考文献</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ARMA · GARCH · BiLSTM · Transformer</div>', unsafe_allow_html=True)

st.markdown("""
本项目使用 **BiLSTM** 和 **Transformer** 两种深度学习模型进行股票价格预测，
并以传统统计模型 **ARMA** 和 **GARCH** 作为基准对比。以下介绍各模型的原理及其在股票预测任务中的优劣势。
""")

# ─────────────────────────────────────────────
# 1. ARMA
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📈 1. ARMA (Autoregressive Moving Average)")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **ARMA** 是由 Box & Jenkins（1970）提出的经典时间序列统计模型，由两部分组成：

    - **AR（自回归）**：用过去若干期的值预测当前值，假设当前值是历史值的线性组合
    - **MA（移动平均）**：用过去的预测误差修正当前预测，降低随机噪声的影响

    ARMA(p, q) 中，p 表示自回归阶数，q 表示移动平均阶数。该模型假设时间序列是**平稳**的，
    即均值和方差不随时间变化。
    """)

with col2:
    st.info("""
    **适用场景**

    ✅ 平稳时间序列预测  
    ✅ 数据量较少时表现稳定  
    ✅ 计算轻量，解释性强  
    """)

st.markdown("""
**在股票预测中的局限性**

股票价格本质上是非平稳序列，波动性随时间变化，ARMA 无法捕捉价格的非线性规律和突发事件，
预测精度有限。通常需要先对价格取对数差分（log return）使其平稳后才能使用。
""")

# ─────────────────────────────────────────────
# 2. GARCH
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📉 2. GARCH (Generalized Autoregressive Conditional Heteroskedasticity)")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **GARCH** 由 Bollerslev（1986）在 Engle（1982）ARCH 模型基础上提出，专门用于建模时间序列的**波动率（Volatility）**。

    核心思想：金融资产的波动不是恒定的，而是具有**波动聚集性（Volatility Clustering）**——
    大幅波动后往往紧跟大幅波动，平静期后紧跟平静期。

    GARCH(p, q) 将当前条件方差建模为：
    - 过去 q 期残差平方的加权和（ARCH 项）
    - 过去 p 期条件方差的加权和（GARCH 项）
    """)

with col2:
    st.info("""
    **适用场景**

    ✅ 金融资产波动率估计  
    ✅ 风险管理与 VaR 计算  
    ✅ 期权定价辅助  
    """)

st.markdown("""
**在股票预测中的局限性**

GARCH 擅长描述**波动率**，而非价格方向或绝对价格水平。它无法捕捉非线性的价格趋势，
在预测股票涨跌方向时能力有限，通常与其他模型配合使用。
""")

# ─────────────────────────────────────────────
# 3. BiLSTM
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🧠 3. BiLSTM (Bidirectional Long Short-Term Memory)")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **BiLSTM** 由 Schuster & Paliwal（1997）提出，是对单向 LSTM 的扩展。
    LSTM 通过引入**门控机制**（输入门、遗忘门、输出门）解决了传统 RNN 的梯度消失问题，
    能够记住长期依赖关系。

    BiLSTM 在此基础上**同时从正向和反向**处理序列：

    ```
    前向 LSTM：Day1 → Day2 → Day3 → ... → DayN
    后向 LSTM：DayN → ... → Day3 → Day2 → Day1
                              ↓
                      拼接双向隐藏状态 → 预测
    ```

    这意味着每个时间步的输出同时包含了该点**之前**和**之后**的上下文信息，
    比单向 LSTM 捕捉的信息更全面。
    """)

with col2:
    st.info("""
    **优势**

    ✅ 捕捉前后向时序依赖  
    ✅ 门控机制缓解梯度消失  
    ✅ 对中短期股价序列效果好  
    ✅ 所需数据量相对较少  
    """)
    st.warning("""
    **局限**

    ❌ 串行计算，训练较慢  
    ❌ 极长序列依赖仍有瓶颈  
    """)

# ─────────────────────────────────────────────
# 4. Transformer
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🤖 4. Transformer (Self-Attention Model)")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Transformer** 由 Vaswani et al.（2017）在论文 *Attention is All You Need* 中提出，
    完全摒弃了循环结构，改用**自注意力机制（Self-Attention）**直接建模序列中任意两点的关系。

    核心组件：
    - **Multi-Head Attention（多头注意力）**：从多个子空间同时计算注意力权重，捕捉不同类型的依赖
    - **Positional Encoding（位置编码）**：为序列注入位置信息，弥补无循环结构的缺失
    - **Feed-Forward Network（前馈网络）**：对注意力输出进行非线性变换

    每个时间步可以直接关注序列中所有其他时间步，无需逐步传递信息。
    """)

with col2:
    st.info("""
    **优势**

    ✅ 并行计算，训练速度快  
    ✅ 任意距离依赖建模能力强  
    ✅ 注意力权重具有可解释性  
    """)
    st.warning("""
    **局限**

    ❌ 需要较大数据量  
    ❌ 计算复杂度为序列长度的平方  
    """)

# ─────────────────────────────────────────────
# 对比表格
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📊 模型对比总结")

st.markdown("""
| 特性 | ARMA | GARCH | BiLSTM | Transformer |
|------|------|-------|--------|-------------|
| 模型类型 | 统计 | 统计 | 深度学习 | 深度学习 |
| 主要用途 | 价格预测 | 波动率建模 | 价格预测 | 价格预测 |
| 非线性建模 | ❌ | ❌ | ✅ | ✅ |
| 长期依赖 | 有限 | 有限 | 良好 | 优秀 |
| 并行计算 | ✅ | ✅ | ❌ | ✅ |
| 数据需求 | 少 | 少 | 中 | 多 |
| 可解释性 | 高 | 高 | 低 | 中（注意力）|
| 训练复杂度 | 低 | 低 | 中 | 高 |
""")

st.markdown("""
**总结**：ARMA 和 GARCH 作为传统统计基准，优点是解释性强、数据需求低，
但对股票价格的非线性特征建模能力有限。BiLSTM 和 Transformer 作为深度学习模型，
能够自动学习复杂的非线性时序规律，在足够数据的支持下通常表现更优，
但代价是更高的计算成本和更低的可解释性。
""")

# ─────────────────────────────────────────────
# 参考文献
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📖 References")

st.markdown("""
[1] Box, G. E. P., & Jenkins, G. M. (1970). *Time series analysis: Forecasting and control*. Holden-Day.

[2] Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics, 31*(3), 307–327. https://doi.org/10.1016/0304-4076(86)90063-1

[3] Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing, 45*, 2673–2681. https://doi.org/10.1109/78.650093

[4] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*. https://arxiv.org/abs/1706.03762
""")
