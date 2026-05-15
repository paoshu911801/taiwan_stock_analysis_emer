import streamlit as st
import pandas as pd
from FinMind.data import DataLoader

# 1. 介面設定
st.title("個股基本查詢系統")
stock_id = st.text_input("請輸入個股代號", value="2330")

if stock_id:
    dl = DataLoader()
    df_info = dl.taiwan_stock_info()
    stock_info = df_info[df_info['stock_id'] == stock_id]
    if not stock_info.empty:
        stock_name = stock_info['stock_name'].values[0]
    else:
        stock_name = "未知公司"

    # 2. 抓取技術面資料 (K線)
    df_k = dl.taiwan_stock_daily(stock_id=stock_id, start_date='2025-01-01')
    
    # 計算 MA5, MA20 (簡易版)
    df_k['MA5'] = df_k['close'].rolling(window=5).mean()
    df_k['MA20'] = df_k['close'].rolling(window=20).mean()
    
    # 3. 抓取籌碼面 (三大法人)
    df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date='2024-01-01')
    
    # 4. 顯示結果
    st.subheader(f"個股：{stock_name} ({stock_id})")
    st.write(f"### 最新收盤價: {df_k['close'].iloc[-1]}")
    
    st.write("最近 5 日技術指標")
    column_mapping_avg = {
        'date': '日期',
        'close': '本日收盤價',
        'MA5': '5日均線',
        'MA20': '20日均線'
    }

    df_display_avg = df_k[['date', 'close', 'MA5', 'MA20']].tail(5)
    df_display_avg = df_display_avg.rename(columns=column_mapping_avg)
    st.dataframe(df_display_avg)
        
    st.write("最近 5 日外資/投信買賣超")
    column_mapping_foreign = {
        'date': '日期',
        'name': '法人名稱',
        'buy': '買進張數',
        'sell': '賣出張數'
    }
    df_display_foreign = df_inst[['date', 'name', 'buy', 'sell']].tail(10)
    df_display_foreign = df_display_foreign.rename(columns=column_mapping_foreign)
    st.dataframe(df_display_foreign)
        # 5. 驗證資料並存成 CSV
    df_k.to_csv(f"{stock_id}_tech.csv", index=False)

    st.write("---") # 畫一條分隔線
    st.subheader("營收結構與題材分析")

    # 抓取營收
    df_revenue = dl.taiwan_stock_month_revenue(stock_id=stock_id, start_date='2024-01-01')
    if not df_revenue.empty:
        # 1. 單位轉換：轉成「億」
        df_revenue['營收(億)'] = (df_revenue['revenue'] / 100000000).round(2)
        
        # 2. 準備顯示用的表格
        rev_display = df_revenue.tail(6).rename(columns={
            'date': '月份',
            'revenue_month': '月增率(%)',
            'revenue_year': '年增率(%)'
        })
        
        # 3. 漂亮呈現
        st.write("### 月營收趨勢")
        
        # 用一行顯示最新資訊 (Metric)
    col1, col2, col3 = st.columns(3)
    latest = df_revenue.iloc[-1]
    col1.metric("最新月營收", f"{latest['營收(億)']} 億")
    col2.metric("月增率", f"{latest['revenue_month']}%")
    col3.metric("年增率", f"{latest['revenue_year']}%")

        # 畫出長條圖
    st.bar_chart(df_revenue.tail(12).set_index('date')['營收(億)'])

    # 題材分類標籤
    ai_list = ['2330', '2382', '6669']
    semi_list = ['2330', '2454', '2303']
    power_list = ['1503', '1513']
    cooling_list = ['3017', '3324']

    if stock_id in ai_list:
        st.button("# AI")
    if stock_id in semi_list:
        st.button("# Semi")
    if stock_id in power_list:
        st.button("# Power")
    if stock_id in cooling_list:
        st.button("# Cooling")