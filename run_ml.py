import pandas as pd
import os
# 從你的 ml_pipeline.py 匯入寫好的大腦加工管線與訓練函數
from ml_pipeline import prepare_ml_dataset, train_baseline_model

def main():
    print("==================================================")
    print("🚀 台股機器學習 Baseline 訓練控制台啟動...")
    print("==================================================")
    
    # 1. 檢查並讀取本地端的個股 CSV 資料
    target_csv = "2330_tech.csv" # 拿你目錄下的台積電資料來做 Baseline 測試
    
    if not os.path.exists(target_csv):
        print(f"❌ 錯誤：在專案目錄下找不到 {target_csv} 檔案！")
        print("💡 請確認檔名是否完全正確，或是先執行抓取資料的腳本。")
        return

    print(f"📊 步驟一：讀取原始資料 {target_csv}...")
    raw_df = pd.read_csv(target_csv)
    print(f"   -> 成功讀取，原始數據共 {len(raw_df)} 筆。")
    
    # 2. 啟動大腦工廠的特徵與標籤工程管線
    print("\n🔧 步驟二：啟動 ml_pipeline 自動化資料加工流水線...")
    try:
        processed_df = prepare_ml_dataset(raw_df)
        print(f"   -> 加工清洗完畢！切除頭尾 NaN 後，剩餘有效訓練資料：{len(processed_df)} 筆。")
    except KeyError as e:
        print(f"❌ KeyError 錯誤：你的 CSV 欄位名稱與大腦定義的不符！找不到欄位: {e}")
        print("💡 請檢查你的 CSV 欄位（例如外資買賣超的英文拼法），並去 ml_pipeline.py 修改成對應的名稱。")
        return

    # 3. 進行時間序列切分並訓練隨機森林模型
    print("\n🤖 步驟三：把加工好的資料送入隨機森林模型進行訓練與模擬考...")
    trained_model = train_baseline_model(processed_df)
    
    print("\n==================================================")
    print("🎉 恭喜！本地端 Baseline 模型訓練與儲存流程全部順利跑通！")
    print("💾 檢查專案目錄下是否已經多出一個 'baseline_rf.pkl' 檔案。")
    print("==================================================")

if __name__ == "__main__":
    main()