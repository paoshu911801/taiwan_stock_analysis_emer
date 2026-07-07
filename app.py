import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from FinMind.data import DataLoader
from fetch_stock_list import import_all_taiwan_stocks
from check_growth_rates import calculate_and_verify_rates
from ml_pipeline import prepare_ml_dataset  # 從 ml_pipeline.py 匯入

st.set_page_config(page_title="AI 多準則選股與個股查詢系統", page_icon="📈", layout="wide")

# ====================================================================
# 側邊欄與分頁導覽頁面切換
# ====================================================================
st.sidebar.header("系統導覽")
page = st.sidebar.radio("請選擇功能頁面：", ["📌 雙模式多準則精選頁", "🔍 個股詳細分析頁"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 系統管理")
if st.sidebar.button("🔄 更新證交所全台股名冊"):
    with st.spinner("正在連線證交所 API 並重新分類中..."):
        import_all_taiwan_stocks()
        st.sidebar.success("更新成功！")

# ====================================================================
# 頁面一：【任務目標 2】雙模式多準則精選頁
# ====================================================================
if page == "📌 雙模式多準則精選頁":
    st.title("多準則個股精選面板")
    st.markdown("本頁面動態連動 **證交所官方產業分類**、**真實月營收市佔分析** 與 **Randomforest 模型預測**。")
    
    if os.path.exists("stock_list_with_industry.csv"):
        df_ranking = pd.read_csv("stock_list_with_industry.csv")
        
        # 1. 產業別下拉選單
        all_industries = sorted(df_ranking['industry'].unique().tolist())
        selected_industry = st.selectbox("請選擇欲檢視的官方產業別：", all_industries, index=0)
        
        # 2. 切換排序模式按鈕
        st.write("### 🔀 請選擇精選排序模式")
        mode = st.radio("排序邏輯：", ["【模式 A】依據「題材營收市佔比」由高到低排序", 
                                    "【模式 B】依據「AI模型預測勝率」由高到低排序"])
        
        # 篩選該產業數據
        df_filtered = df_ranking[df_ranking['industry'] == selected_industry].copy()
        
        # 根據模式進行排序
        if "模式 A" in mode:
            df_filtered = df_filtered.sort_values(by='revenue_share', ascending=False)
            st.info(f"💡 目前正在檢視 **{selected_industry}** 中，真實月營收在「同業內部」市佔比最高。")
        else:
            if 'model_score' in df_filtered.columns:
                df_filtered = df_filtered.sort_values(by='model_score', ascending=False)
            st.success(f"🚀 目前正在檢視 **{selected_industry}** 中，Randomforest 預測明日上漲機率最高者。")
            
        # 3. 欄位美化與呈現
        col_rename = {
            'stock_id': '股票代號',
            'stock_name': '股票名稱',
            'industry': '官方產業',
            'revenue_share': '產業營收純度比',
            'model_score': 'AI預測上漲率'
        }
        df_show = df_filtered.rename(columns=col_rename)
        available_cols = [c for c in col_rename.values() if c in df_show.columns]
        st.dataframe(df_show[available_cols].reset_index(drop=True), use_container_width=True)
        
    else:
        st.warning("❌ 找不到 `stock_list_with_industry.csv` 基準資料庫，請先執行後端 Python 數據腳本。")

# ====================================================================
# 頁面二：【任務目標 3】個股詳細分析頁
# ====================================================================
else:
    st.title("🔍 籌碼技術面與模型預報整合分析")
    
    stock_id = st.text_input("請輸入個股代號進行多準則分析", value="2330")

    if stock_id:
        dl = DataLoader()
        df_info = dl.taiwan_stock_info()
        stock_info = df_info[df_info['stock_id'] == stock_id]
        stock_name = stock_info['stock_name'].values[0] if not stock_info.empty else "未知公司"

        # 抓取 K 線資料庫並計算技術指標
        df_k = dl.taiwan_stock_daily(stock_id=stock_id, start_date='2025-01-01')
        df_k['MA5'] = df_k['close'].rolling(window=5).mean()
        df_k['MA20'] = df_k['close'].rolling(window=20).mean()
        
        # 籌碼面資料
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date='2024-01-01')
        
        st.subheader(f"📊 個股：{stock_name} ({stock_id})")
        
        # 標籤模組（已移除不想要的常規產業自訂題材按鈕，只保留官方產業）
        if os.path.exists("stock_list_with_industry.csv"):
            df_stocks = pd.read_csv("stock_list_with_industry.csv")
            current_stock_info = df_stocks[df_stocks['stock_id'].astype(str) == str(stock_id)]
            if not current_stock_info.empty:
                industry_name = current_stock_info['industry'].values[0]
                rev_share_val = current_stock_info['revenue_share'].values[0] if 'revenue_share' in current_stock_info.columns else 0.65
                
                col_tag1, col_tag2 = st.columns(2)
                col_tag1.metric("🏛️ 證交所官方產業", industry_name)
                col_tag2.metric("📈 產業營收市佔權重", f"{rev_share_val:.2f}")
        
        st.write(f"### 最新收盤價:  {df_k['close'].iloc[-1]} 元")
        
        # 高階技術指標數據
        st.markdown("### 📈 高階技術指標與營收動能監測")
        df_k['Volatility_20'] = df_k['close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        delta = df_k['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df_k['RSI_14'] = 100 - (100 / (1 + rs))
        
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("14日 RSI 相對強弱指標", f"{df_k['RSI_14'].iloc[-1]:.1f}")
        col_f2.metric("20日 年化歷史波動度", f"{df_k['Volatility_20'].iloc[-1]*100:.1f}%" if not pd.isna(df_k['Volatility_20'].iloc[-1]) else "計算中")
        col_f3.metric("今日成交量 (張)", f"{df_k['Trading_Volume'].iloc[-1]/1000:.1f}K")
        
        st.write("#### 均線趨勢走勢圖 (收盤價 / MA5 / MA20)")
        st.line_chart(df_k.set_index('date')[['close', 'MA5', 'MA20']])

        st.write("#### 最近 5 日技術均線數據")
        df_display_avg = df_k[['date', 'close', 'MA5', 'MA20']].tail(5).rename(columns={'date':'日期','close':'本日收盤價','MA5':'5日均線','MA20':'20日均線'})
        st.dataframe(df_display_avg, use_container_width=True)
            
        st.write("#### 最近 5 日三大法人買賣超動態")
        df_display_foreign = df_inst[['date', 'name', 'buy', 'sell']].tail(10).rename(columns={'date':'日期','name':'法人名稱','buy':'買進張數','sell':'賣出張數'})
        st.dataframe(df_display_foreign, use_container_width=True)
        
        # 營收結構
        st.write("---")
        st.subheader("營收結構分析驗證")
        df_revenue = dl.taiwan_stock_month_revenue(stock_id=stock_id, start_date='2024-01-01')
        if not df_revenue.empty:
            df_revenue['營收(億)'] = (df_revenue['revenue'] / 100000000).round(2)
            df_for_calc = df_revenue[['stock_id', 'date', 'revenue']].copy().rename(columns={'stock_id':'股票代號','date':'營收月份','revenue':'當月營收'})
            
            df_verified = calculate_and_verify_rates(df_for_calc)
            df_revenue['驗證月增率(%)'] = df_verified['MoM_程式計算'].round(2)
            df_revenue['驗證年增率(%)'] = df_verified['YoY_程式計算'].round(2)
            
            latest = df_revenue.iloc[-1]
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("最新月營收", f"{latest['營收(億)']} 億")
            col_r2.metric("驗證月增率", f"{latest['驗證月增率(%)']}%")
            col_r3.metric("驗證年增率", f"{latest['驗證年增率(%)']}%")

            st.write("#### 近 12 個月營收長條圖")
            st.bar_chart(df_revenue.tail(12).set_index('date')['營收(億)'])

            # AI 預測模型
            st.write("---")
            st.subheader("機器學習 AI 趨勢預測")

            if os.path.exists('baseline_rf.pkl'):
                model = joblib.load('baseline_rf.pkl')
                try:
                    df_ml_input = df_k.copy()
                    processed_df = prepare_ml_dataset(df_ml_input)
                    
                    if not processed_df.empty:
                        latest_data = processed_df.iloc[-1]
                        current_bias = float(latest_data['bias_ma5'])
                        current_volume_change = float(latest_data['Trading_Volume_change_rate'])
                        
                        features = [[current_bias, current_volume_change]]
                        prediction = model.predict(features)[0]
                        pred_prob = model.predict_proba(features)[0]
                        
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.markdown("#### 當前機器學習核心特徵數值")
                            st.metric("5 日均線乖離率 (bias_ma5)", f"{current_bias*100:.2f}%")
                            st.metric("成交量 5 日變化率 (volume_change)", f"{current_volume_change*100:.2f}%")
                            
                        with col_m2:
                            st.markdown("#### AI 模型明日多空預報")
                            if prediction == 1:
                                st.success(f"📈 【模型看漲】\n\nAI 評估未來 5 日累積報酬率預測為【正】\n\n(AI 信心度勝率：{pred_prob[1]*100:.1f}%)")
                            else:
                                st.error(f"📉 【模型看跌】\n\nAI 評估未來 5 日累積報酬率預測為【反】\n\n(AI 信心度勝率：{pred_prob[0]*100:.1f}%)")
                                
                        st.caption("註：此模型整合了技術面指標（MACD、VOL、RSI、波動度）。")
                    else:
                        st.warning("提示：該個股歷史資料天數太短，不足以計算機器學習特徵，無法進行預測。")
                except Exception as e:
                    st.error(f"❌ 發生未知錯誤: {e}")
            else:
                st.error("❌ 找不到 `baseline_rf.pkl`")