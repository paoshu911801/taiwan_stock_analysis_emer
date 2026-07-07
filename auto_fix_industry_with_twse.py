import pandas as pd
import requests
import os

def auto_update_industry_by_twse(csv_path="stock_list_with_industry.csv"):
    if not os.path.exists(csv_path):
        print(f"❌ 找不到檔案：{csv_path}")
        return

    df_my_list = pd.read_csv(csv_path)
    print(f"📖 讀取現有檔案成功，目前股票數量: {len(df_my_list)}")

    twse_industry_dict = {
        "01": "水泥工業", "02": "食品工業", "03": "塑膠工業", "04": "紡織纖維",
        "05": "電機機械", "06": "電器電纜", "07": "化學工業", "21": "化學工業",
        "08": "玻璃陶瓷", "09": "造紙工業", "10": "鋼鐵工業", "11": "橡膠工業",
        "12": "汽車工業", "13": "電子工業", "24": "半導體業", "25": "電腦及週邊設備業",
        "26": "光電業", "27": "通信網路業", "28": "電子零組件業", "29": "電子通路業",
        "30": "資訊服務業", "31": "其他電子業", "14": "建材營造", "15": "航運業",
        "16": "觀光餐旅", "17": "金融保險", "18": "貿易百貨", "22": "油電燃氣業",
        "23": "存託憑證", "35": "綠能環保", "36": "數位雲端", "37": "運動休閒",
        "38": "居家生活", "91": "存託憑證", "99": "其他"
    }

    print("🚀 正在從台灣證交所官方 OpenAPI 下載最新股票產業對照表...")
    twse_url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    
    try:
        response = requests.get(twse_url, timeout=10)
        data = response.json()
        
        industry_map = {}
        for item in data:
            stock_code = str(item.get('公司代號', '')).strip()
            industry_code = str(item.get('產業別', '')).strip() 
            
            real_industry_name = twse_industry_dict.get(industry_code, "其他業")
                
            if stock_code:
                industry_map[stock_code] = real_industry_name
                
        # 轉換型態
        df_my_list['stock_id'] = df_my_list['stock_id'].astype(float).astype(int).astype(str)
        
        print("🔄 正在利用官方數據將「常規產業」全面修正為正確中文類別...")
        df_my_list['industry'] = df_my_list['stock_id'].map(industry_map)
        df_my_list['industry'] = df_my_list['industry'].fillna("其他業")
        
        # 覆寫存檔
        df_my_list.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"🎉 修正成功！ {csv_path} 產業別已經更新完畢！")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    auto_update_industry_by_twse()