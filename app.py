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
        # ====================================================================
        # 🤖 導入機器學習模型與介面展示
        # ====================================================================
        st.write("---")
        st.subheader("🤖 機器學習 AI 趨勢預測（Baseline Model）")

        import joblib
        # 從ml_pipeline.py中，匯入資料
        from ml_pipeline import prepare_ml_dataset

        # 1. 檢查並載入本地端已經由 run_ml.py 訓練好的模型大腦
        if os.path.exists('baseline_rf.pkl'):
            model = joblib.load('baseline_rf.pkl')
            
            try:
                # 複製一份目前的個股 K 線資料，避免污染原本要做圖表的資料
                df_ml_input = df_k.copy()
                
                # 2. 啟動 ml_pipeline.py ，自動把這檔股票的 close / Trading_Volume 算出 ML 特徵
                processed_df = prepare_ml_dataset(df_ml_input)
                
                if not processed_df.empty:
                    # 3. 拿出最新一天的特徵資料
                    latest_data = processed_df.iloc[-1]
                    
                    # 提取最新的特徵數值
                    current_bias = float(latest_data['bias_ma5'])
                    current_volume_change = float(latest_data['Trading_Volume_change_rate'])
                    
                    # 建立符合 Scikit-learn 輸入格式的特徵矩陣 [ [特徵1, 特徵2] ]
                    features = [[current_bias, current_volume_change]]
                    
                    # 4. 進行預測
                    prediction = model.predict(features)[0] # 拿到 0 (跌) 或 1 (漲)
                    pred_prob = model.predict_proba(features)[0] # 拿到 [跌的機率, 漲的機率]
                    
                    # 5. 畫面的排版與呈現
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.markdown("#### 當前主要特徵數值")
                        st.metric("5 日均線乖離率", f"{current_bias*100:.2f}%")
                        st.metric("成交量 5 日變化率", f"{current_volume_change*100:.2f}%")
                        
                    with col_m2:
                        st.markdown("#### AI 趨勢預測結果")
                        if prediction == 1:
                            st.success(f"📈 【模型看漲】\n\n未來 5 日累積報酬率預測為【正】\n\n(AI 信心度：{pred_prob[1]*100:.1f}%)")
                        else:
                            st.error(f"📉 【模型看跌】\n\n未來 5 日累積報酬率預測為【負】\n\n(AI 信心度：{pred_prob[0]*100:.1f}%)")
                            
                    st.caption("註：此預測為 Baseline Random Forest 模型根據技術指標計算之統計機率，供量化專案展示。")
                else:
                    st.warning("提示：該個股歷史資料天數太短，不足以計算機器學習特徵，無法進行預測。")
                    
            except Exception as e:
                st.error(f"❌ 呼叫機器學習管線時發生未知錯誤: {e}")
        else:
            st.error("❌ 找不到 `baseline_rf.pkl` 模型檔案！請先在終端機執行 `python run_ml.py` 進行模型訓練與儲存。")