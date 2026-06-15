import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def prepare_ml_dataset(df):
    """
    輸入 DataFrame 必須包含：'date', 'stock_id', 'close', 'Trading_Volume', 'foreign_buy_sell'
    且 df 必須已經依照 ['stock_id', 'date'] 進行排序。
    """
    # 確保資料依照日期排序
    df = df.sort_values(['stock_id', 'date']).reset_index(drop=True)
    
    # --- 1. 特徵工程 (Feature Engineering) ---
    # 計算 MA5
    df['ma5'] = df.groupby('stock_id')['close'].transform(lambda x: x.rolling(5).mean())
    
    # 價格與 MA5 的乖離率，乖離率公式 = (收盤價 - 均線) / 均線
    df['bias_ma5'] = df['bias_ma5'] = (df['close'] - df['ma5']) / df['ma5']

    # 計算成交量 5 日變化率
    df['Trading_Volume_ma5'] = df.groupby('stock_id')['Trading_Volume'].transform(lambda x: x.rolling(5).mean())
    df['Trading_Volume_change_rate'] = (df['Trading_Volume'] - df['Trading_Volume_ma5']) / (df['Trading_Volume_ma5'] + 1e-5) # 加 1e-5 防止除以 0
    
    # Label Engineering
    # 預測未來 5 日報酬是否為正
    # 我們需要用到 pandas 的 .shift() 功能，將未來的資料「往回拉」到今天
    
    df['future_close_5d'] = df.groupby('stock_id')['close'].shift(-5)
    
    # 計算未來 5 日報酬率
    df['return_5d'] = (df['future_close_5d'] - df['close']) / df['close']
    
    # 定義二元標籤：報酬率 > 0 為 1，其餘為 0
    df['label'] = np.where(df['return_5d'] > 0, 1, 0)
    
    # 清理缺失值（因為 rolling 和 shift 會產生 NaN）
    df = df.dropna().reset_index(drop=True)
    
    return df

def train_baseline_model(df):
    """
    輸入已經做好特徵與標籤的 df
    進行時間序列切分，並訓練隨機森林模型
    """
    # 確保資料是按時間排列的
    df = df.sort_values('date').reset_index(drop=True)
    
    # 定義特徵欄位與標籤欄位
    feature_cols = ['bias_ma5', 'Trading_Volume_change_rate']
    target_col = 'label'
    
    # --- 1. 時間序列切分 (防止 Look-ahead bias) ---
    # 我們用前 80% 的資料當訓練，後 20% 當測試
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # --- 2. 初始化模型 ---
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    
    # --- 3. 訓練模型 ---
    # 訓練模型固定都是.fit
    model.fit(X_train, y_train)
    
    # --- 4. 預測與評估 ---
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 Baseline Model Accuracy: {accuracy:.4f}")
    print("\n詳細分類報告:")
    print(classification_report(y_test, y_pred))
    import joblib
    joblib.dump(model, 'baseline_rf.pkl')
    
    return model