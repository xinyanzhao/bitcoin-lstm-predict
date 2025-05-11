import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 下载历史数据（比特币）
data = yf.download('BTC-USD', start='2021-01-01', end='2025-04-23', interval='1d')

# 只用收盘价
df = data[['Close']].copy()
scaler = MinMaxScaler()
df['scaled'] = scaler.fit_transform(df[['Close']])

# 构建序列数据（用前60天预测第61天）
def create_dataset(series, lookback=60):
    X, y = [], []
    for i in range(len(series) - lookback - 1):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    return np.array(X), np.array(y)

lookback = 60
X, y = create_dataset(df['scaled'].values, lookback)

# reshape成LSTM接受的格式：(samples, time_steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 拆训练测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 构建模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练
model.fit(X_train, y_train, epochs=20, batch_size=32)

# 预测
predicted = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted)
real_price = scaler.inverse_transform(y_test.reshape(-1, 1))

# 可视化
plt.plot(real_price, label='Real BTC Price')
plt.plot(predicted_price, label='Predicted')
plt.title('LSTM Prediction vs Actual')
plt.legend()
plt.show()

# 输出明天的预测方向
last_60 = df['scaled'].values[-lookback:]
X_input = np.reshape(last_60, (1, lookback, 1))
pred = model.predict(X_input)[0][0]
pred_price = scaler.inverse_transform([[pred]])[0][0]
last_price = df['Close'].values[-1]

last_price = float(df['Close'].values[-1])  # 保证它是float
print(f"当前BTC价格: {last_price:.2f}")
print(f"预测明日收盘价: {pred_price:.2f}")

if pred_price > last_price:
    print(">>> 预测上涨，建议BUY")
else:
    print(">>> 预测下跌，建议空仓或SELL")
input("预测完成，按回车退出...")