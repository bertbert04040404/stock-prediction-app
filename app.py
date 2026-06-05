import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import optuna

# 設定網頁標題
st.set_page_config(page_title="AI 股價預測系統", layout="wide")
st.title("📈 AI 股價預測互動網頁")
st.markdown("輸入股票代號，系統將即時抓取數據，並透過 LSTM 模型進行自動調參與預測。")

# 1. 側邊欄控制面板
st.sidebar.header("⚙️ 參數設定")
stock_code = st.sidebar.text_input("輸入股票代號 (例如: AAPL 或 2330.TW)", value="AAPL")
window_size = st.sidebar.slider("Window Size (參考過去天數)", min_value=3, max_value=10, value=5)
optuna_trials = st.sidebar.number_input("Optuna 調參次數 (Trials)", min_value=3, max_value=20, value=5)

start_button = st.sidebar.button("🚀 開始執行預測")

def split_windows(data, window):
    X, Y = [], []
    for i in range(len(data) - window):
        X.append(data[i:(i + window)])
        Y.append(data[i + window])
    return np.array(X), np.array(Y)

# 2. 主要執行邏輯
if start_button:
    st.info(f"正在透過 Yahoo Finance 抓取 {stock_code} 的歷史數據...")
    
    try:
        stock_data = yf.download(stock_code, start="2019-01-01", end="2024-01-01")
        if stock_data.empty:
            st.error("找不到該股票代號，請確認代號是否正確。")
            st.stop()
    except Exception as e:
        st.error(f"數據抓取失敗: {e}")
        st.stop()
        
    df = stock_data[['Close']].dropna()
    df['Year'] = df.index.year
    
    max_val, min_val = float(df['Close'].max()), float(df['Close'].min())
    df['Close_norm'] = (df['Close'] - min_val) / (max_val - min_val)
    
    test_year = df['Year'].max()
    train_df = df[df['Year'] < test_year]
    test_df = df[df['Year'] == test_year]
    
    X_train, y_train = split_windows(train_df['Close_norm'].values, window_size)
    X_test, y_test = split_windows(test_df['Close_norm'].values, window_size)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    status_text = st.empty()
    status_text.warning("🤖 Optuna 正在雲端尋找最佳 LSTM 參數組合...")
    progress_bar = st.progress(0)
    
    def objective(trial):
        model = Sequential()
        neurons = trial.suggest_categorical('neurons', [16, 32])
        dropout_rate = trial.suggest_float('dropout', 0.2, 0.3)
        
        model.add(LSTM(units=neurons, input_shape=(window_size, 1)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
        
        model.fit(X_train, y_train, epochs=2, batch_size=64, verbose=0)
        score = model.evaluate(X_test, y_test, verbose=0)
        return score

    study = optuna.create_study(direction='minimize')
    for i in range(int(optuna_trials)):
        study.optimize(objective, n_trials=1)
        progress_bar.progress((i + 1) / int(optuna_trials))
        
    status_text.success("🎉 調參完成！")
    st.write("📋 **最佳參數組合：**", study.best_params)
    
    best_model = Sequential()
    best_model.add(LSTM(units=study.best_params['neurons'], input_shape=(window_size, 1)))
    best_model.add(Dropout(study.best_params['dropout']))
    best_model.add(Dense(1))
    best_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
    
    best_model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)
    predictions = best_model.predict(X_test)
    
    predictions_recovered = predictions * (max_val - min_val) + min_val
    actual_recovered = y_test * (max_val - min_val) + min_val
    
    st.subheader("📊 預測結果對比圖")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual_recovered, label="Actual Price", color="blue", linewidth=1.5)
    ax.plot(predictions_recovered, label="Predicted Price", color="orange", linestyle="--", linewidth=1.5)
    ax.set_title(f"{stock_code} Stock Prediction")
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)
else:
    st.info("💡 請在左側面板輸入代號（例如美股 AAPL 或台股 2330.TW），並點擊「開始執行預測」。")
