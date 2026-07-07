import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def optimize_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'Trading_Volume': 'volume', 'Trading_Money': 'money'})
    df = df.sort_values('date').reset_index(drop=True)
    df['return'] = df['close'].pct_change()
    df['volatility_20'] = df['return'].rolling(window=20).std()
    df['volume_ratio_5'] = df['volume'] / (df['volume'].rolling(window=5).mean() + 1e-9)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_diff = ema_12 - ema_26
    macd_signal = macd_diff.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd_diff - macd_signal
    
    return df.dropna().reset_index(drop=True)

def evaluate_model_upgrade(df: pd.DataFrame):
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna().reset_index(drop=True)
    
    old_features = ['close', 'Trading_turnover', 'MA5', 'MA20']
    new_features = old_features + ['return', 'volatility_20', 'volume_ratio_5', 'rsi_14', 'macd_hist']
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train_old = train_df[old_features]
    X_test_old = test_df[old_features]
    y_train = train_df['target']
    y_test = test_df['target']
    
    rf_old = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_old.fit(X_train_old, y_train)
    acc_old = accuracy_score(y_test, rf_old.predict(X_test_old))
    
    X_train_new = train_df[new_features]
    X_test_new = test_df[new_features]
    rf_new = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_new.fit(X_train_new, y_train)
    acc_new = accuracy_score(y_test, rf_new.predict(X_test_new))
    
    print("\n=========================================")
    print(f"-原始- 低階特徵模型 Accuracy: {acc_old:.4f}")
    print(f"-新增- 高階動能指標 Accuracy: {acc_new:.4f}")
    print("=========================================")
    
    feat_importance_df = pd.DataFrame({
        'Feature_Name': new_features,
        'Importance': rf_new.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print("\n最終模型特徵重要性排行：")
    print(feat_importance_df)
    return acc_old, acc_new, feat_importance_df

def rank_stock_portfolio(industry_csv_path: str, all_predictions: dict, sort_by: str) -> pd.DataFrame:
    if not os.path.exists(industry_csv_path):
        df_mock = pd.DataFrame({
            'stock_id': ['2330', '2303', '1101', '1102'],
            'stock_name': ['台積電', '聯電', '台泥', '亞泥'],
            'industry': ['24', '24', '01', '01'],
            'custom_tag': ['常規產業', '常規產業', '常規產業', '常規產業'],
            'revenue_share': [0.85, 0.60, 0.34, 0.95]
        })
        df_mock.to_csv(industry_csv_path, index=False, encoding='utf-8')

    df_industry = pd.read_csv(industry_csv_path)
    
    # 證交所官方代碼與中文產業名稱對照字典
    industry_translation = {
        '01': '水泥工業', '02': '食品工業', '03': '塑料工業', '04': '紡織纖維', '05': '電機機械',
        '06': '電器電纜', '07': '化學工業', '08': '生技醫療業', '09': '玻璃陶瓷', '10': '造紙工業',
        '11': '鋼鐵工業', '12': '橡膠工業', '13': '汽車工業', '14': '電子零組件業', '15': '電機機械',
        '21': '化學工業', '22': '生技醫療業', '23': '油電燃氣業', '24': '半導體業', '25': '電腦及週邊設備業',
        '26': '光電業', '27': '通信網路業', '28': '電子零組件業', '29': '電子通路業', '30': '資訊服務業', '31': '其他電子業'
    }
    
    # 確保 industry 有正確對照中文
    if 'industry' in df_industry.columns:
        df_industry['industry'] = df_industry['industry'].apply(lambda x: industry_translation.get(str(x).strip().zfill(2), str(x)))
    
    # 不管型態是 float 還是 int，一律轉換成純數字
    df_industry['clean_id'] = df_industry['stock_id'].astype(str).str.strip()
    
    # 建立一個字串型態的字典
    str_predictions = {str(k).strip(): v for k, v in all_predictions.items()}
    new_scores = df_industry['clean_id'].map(str_predictions)

    # 排序
    if sort_by == 'revenue_share':
        df_ranked = df_industry.sort_values(by='revenue_share', ascending=False)
    elif sort_by == 'model_score':
        df_ranked = df_industry.sort_values(by='model_score', ascending=False)
    else:
        df_ranked = df_industry
        
    return df_ranked.reset_index(drop=True)

if __name__ == "__main__":
    print("=== 開始測試特徵更新 ===")
    csv_files = [f for f in os.listdir('.') if f.endswith('_tech.csv')]
    
    if csv_files:
        test_file = csv_files[0]
        df_raw = pd.read_csv(test_file)
        df_new = optimize_stock_features(df_raw)
        evaluate_model_upgrade(df_new)

        print("\n=== 測試分析頁多項條件排序邏輯 ===")
        # 建立真實測試字典（涵蓋多種型態）
        mock_predictions = {'2330': 0.88, '2303': 0.40, '1101': 0.53, '1102': 0.95, '3017': 0.92, '2382': 0.65}
        
        display_cols = ['stock_id', 'stock_name', 'industry', 'custom_tag', 'revenue_share', 'model_score']

        print("\n[模式 A] 依據「題材營收佔比」由高到低排序：")
        df_rank_rev = rank_stock_portfolio("stock_list_with_industry.csv", mock_predictions, sort_by="revenue_share")
        
        # 把最右邊那個真正對接出來的欄位數據，洗回 model_score 裡
        if df_rank_rev.shape[1] > 6:  # 如果欄位數大於標準的 6 欄
            df_rank_rev['model_score'] = df_rank_rev.iloc[:, 6]  # 拿第 7 欄的真正分數蓋過去
        print(df_rank_rev[display_cols].head(5))
        
        print("\n[模式 B] 依據「模型預測分數」由高到低排序：")
        df_rank_model = rank_stock_portfolio("stock_list_with_industry.csv", mock_predictions, sort_by="model_score")
        
        # 同理修正模式 B
        if df_rank_model.shape[1] > 6:
            df_rank_model['model_score'] = df_rank_model.iloc[:, 6]
        
        # 在模式 B 的表格內，根據這個分數重新執行一次排序
        df_rank_model = df_rank_model.sort_values(by='model_score', ascending=False).reset_index(drop=True)
        print(df_rank_model[display_cols].head(5))
        
        # 將最終的 6 欄位覆寫回原 CSV
        df_rank_model[display_cols].to_csv("stock_list_with_industry.csv", index=False, encoding='utf-8')
        print("\n🎉")
        
    else:
        print("❌ 沒在根目錄找到任何 _tech.csv 檔案。")