import streamlit as st
import pandas as pd
import numpy as np
import os
from FinMind.data import DataLoader
from fetch_stock_list import import_all_taiwan_stocks
from check_growth_rates import calculate_and_verify_rates

st.set_page_config(page_title="個股基本查詢系統", page_icon="📈", layout="wide")
st.title("個股基本查詢系統")

st.sidebar.header("系統管理")
if st.sidebar.button("🔄 更新證交所全台股名冊"):
    with st.spinner("正在連線證交所 API 並重新分類中..."):
        import_all_taiwan_stocks()
        st.sidebar.success("更新成功！")

stock_id = st.text_input("請輸入個股代號", value="2330")

if stock_id:
    dl = DataLoader()
    df_info = dl.taiwan_stock_info()
    stock_info = df_info[df_info['stock_id'] == stock_id]
    if not stock_info.empty:
        stock_name = stock_info['stock_name'].values[0]
    else:
        stock_name = "未知公司"

    df_k = dl.taiwan_stock_daily(stock_id=stock_id, start_date='2025-01-01')
    
    df_k['MA5'] = df_k['close'].rolling(window=5).mean()
    df_k['MA20'] = df_k['close'].rolling(window=20).mean()
    

    df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date='2024-01-01')
    
    st.subheader(f"個股：{stock_name} ({stock_id})")
    
    if os.path.exists("stock_list_with_industry.csv"):
        df_stocks = pd.read_csv("stock_list_with_industry.csv")
        current_stock_info = df_stocks[df_stocks['stock_id'].astype(str) == str(stock_id)]
        if not current_stock_info.empty:
            industry_name = current_stock_info['industry'].values[0]
            custom_tag_name = current_stock_info['custom_tag'].values[0]
            
            col_tag1, col_tag2 = st.columns(2)
            with col_tag1:
                st.button(f"🏛️ 官方產業別：{industry_name}", key="ind_btn")
            with col_tag2:
                st.button(f"🎯 自訂題材：{custom_tag_name}", key="tag_btn")
    else:
        st.warning("提示：請點擊左側「更新證交所全台股名冊」按鈕以啟用智慧分類標籤功能。")

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
        df_revenue['營收(億)'] = (df_revenue['revenue'] / 100000000).round(2)
        
        df_for_calc = df_revenue[['stock_id', 'date', 'revenue']].copy()
        df_for_calc.columns = ['股票代號', '營收月份', '當月營收']
        
        df_verified = calculate_and_verify_rates(df_for_calc)
        
        df_revenue['MoM_程式計算(%)'] = df_verified['MoM_程式計算'].round(2)
        df_revenue['YoY_程式計算(%)'] = df_verified['YoY_程式計算'].round(2)
        
        rev_display = df_revenue.tail(6).rename(columns={
            'date': '月份',
            'MoM_程式計算(%)': '驗證月增率(%)',
            'YoY_程式計算(%)': '驗證年增率(%)'
        })
        
        st.write("### 月營收趨勢")
        
        latest = df_revenue.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("最新月營收", f"{latest['營收(億)']} 億")
        
        col2.metric("驗證月增率", f"{latest['MoM_程式計算(%)']}%")
        
        col3.metric("驗證年增率", f"{latest['YoY_程式計算(%)']}%")

        st.bar_chart(df_revenue.tail(12).set_index('date')['營收(億)'])