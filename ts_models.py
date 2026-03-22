import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ARMAModel:
    """ARMA 模型 - 自回归移动平均模型"""
    def __init__(self, order=(1, 1)):
        self.order = order
        self.model = None
        self.fitted = False

    def fit(self, y):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(y, order=(self.order[0], 0, self.order[1]))
            self.model = self.model.fit()
            self.fitted = True
            return self
        except Exception as e:
            raise ImportError(f"ARMA 模型训练失败: {str(e)}。请确保已安装 statsmodels: pip install statsmodels")

    def predict(self, steps=1):
        if not self.fitted:
            raise ValueError("模型尚未训练")
        return self.model.forecast(steps=steps)

    def predict_in_sample(self, start=0):
        if not self.fitted:
            raise ValueError("模型尚未训练")
        return self.model.predict(start=start)


class GARCHModel:
    """GARCH 模型 - 广义自回归条件异方差模型"""
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model = None
        self.fitted = False

    def fit(self, y):
        try:
            from arch import arch_model
            self.model = arch_model(y, vol='Garch', p=self.p, q=self.q)
            self.model = self.model.fit(disp='off')
            self.fitted = True
            return self
        except Exception as e:
            raise ImportError(f"GARCH 模型训练失败: {str(e)}。请确保已安装 arch: pip install arch")

    def predict(self, horizon=1):
        if not self.fitted:
            raise ValueError("模型尚未训练")
        forecast = self.model.forecast(horizon=horizon)
        return forecast.mean.values[-1] if hasattr(forecast, 'mean') else forecast

    def predict_in_sample(self):
        if not self.fitted:
            raise ValueError("模型尚未训练")
        return self.model.conditional_volatility


class TimeSeriesPredictor:
    """时间序列预测器 - 支持 ARMA 和 GARCH 模型"""
    def __init__(self):
        self.models = {}
        self.scaler = MinMaxScaler()

    def prepare_data(self, df, target_col='Close'):
        """准备数据"""
        data = df[[target_col]].dropna().values
        return data

    def train_arma(self, model_name, data, order=(1, 1), test_size=0.2):
        """训练 ARMA 模型"""
        split = int(len(data) * (1 - test_size))
        train_data = data[:split]
        test_data = data[split:]

        model = ARMAModel(order=order)
        model.fit(train_data)

        # 预测
        train_pred = model.predict_in_sample()
        test_pred = []

        for i in range(len(test_data)):
            # 使用历史数据逐步预测
            hist_data = np.concatenate([train_data, test_data[:i]])
            temp_model = ARMAModel(order=order)
            temp_model.fit(hist_data)
            pred = temp_model.predict(steps=1)[0]
            test_pred.append(pred)

        test_pred = np.array(test_pred)

        # 计算指标
        train_metrics = self._calculate_metrics(train_data, train_pred)
        test_metrics = self._calculate_metrics(test_data, test_pred)

        self.models[model_name] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_actual': train_data,
            'test_actual': test_data
        }

        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': test_pred,
            'actuals': test_data
        }

    def train_garch(self, model_name, data, p=1, q=1, test_size=0.2):
        """训练 GARCH 模型"""
        split = int(len(data) * (1 - test_size))
        train_data = data[:split]
        test_data = data[split:]

        # GARCH 模型通常用于收益率
        returns = np.diff(np.log(data + 1e-9)) * 100
        split_ret = int(len(returns) * (1 - test_size))
        train_ret = returns[:split_ret]
        test_ret = returns[split_ret:]

        model = GARCHModel(p=p, q=q)
        model.fit(train_ret)

        # 预测波动率
        train_vol = model.predict_in_sample()

        # 测试集预测
        test_pred = []
        for i in range(len(test_ret)):
            hist_ret = np.concatenate([train_ret, test_ret[:i]])
            temp_model = GARCHModel(p=p, q=q)
            temp_model.fit(hist_ret)
            pred = temp_model.predict(horizon=1)
            test_pred.append(pred)

        test_pred = np.array(test_pred)

        # 计算指标
        train_metrics = self._calculate_metrics(train_ret, train_vol)
        test_metrics = self._calculate_metrics(test_ret, test_pred)

        self.models[model_name] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_pred': train_vol,
            'test_pred': test_pred,
            'train_actual': train_ret,
            'test_actual': test_ret
        }

        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': test_pred,
            'actuals': test_ret
        }

    def _calculate_metrics(self, actuals, predictions):
        """计算评估指标"""
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        # MAPE
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-9))) * 100

        # 方向准确率
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actuals) > 0
        direction_accuracy = np.mean(pred_direction == actual_direction) if len(pred_direction) > 0 else 0

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2,
            'Direction_Accuracy': direction_accuracy
        }
