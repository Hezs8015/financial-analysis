# 测试导入
try:
    from models import StockPredictor
    print("✅ models.py 导入成功")
except Exception as e:
    print(f"❌ models.py 导入失败: {e}")

try:
    from ts_models import TimeSeriesPredictor
    print("✅ ts_models.py 导入成功")
except Exception as e:
    print(f"❌ ts_models.py 导入失败: {e}")
