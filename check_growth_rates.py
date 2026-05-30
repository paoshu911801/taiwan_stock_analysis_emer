import pandas as pd
import numpy as np

def calculate_and_verify_rates(df_revenue):
    df = df_revenue.copy()
    
    # 資料清洗
    if df['當月營收'].dtype == 'object':
        df['當月營收'] = df['當月營收'].astype(str).str.replace(',', '')
        df['當月營收'] = pd.to_numeric(df['當月營收'], errors='coerce')
    
    # 排序
    df = df.sort_values(by=['股票代號', '營收月份']).reset_index(drop=True)
    
    # 計算增率
    df['MoM_程式計算'] = df.groupby('股票代號')['當月營收'].pct_change(periods=1) * 100
    df['YoY_程式計算'] = df.groupby('股票代號')['當月營收'].pct_change(periods=12) * 100
    
    return df

if __name__ == "__main__":
    df_stocks = pd.read_csv("stock_list_with_industry.csv")
    print("成功讀取全台股清單，目前共有", len(df_stocks), "檔上市公司。")
    
    mock_data = {
        '股票代號': ['2330', '2330', '2330'],
        '營收月份': ['2025-04', '2026-03', '2026-04'],
        '當月營收': ['1,500,000', '1,800,000', '2,000,000']
    }
    df_mock_revenue = pd.DataFrame(mock_data)
    
    # 執行計算
    print("\n正在計算增率...")
    df_result = calculate_and_verify_rates(df_mock_revenue)
    print(df_result)