import requests
import pandas as pd

def import_all_taiwan_stocks():
    # 證交所開放資料：上市公司基本資料 API
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    
    print("正在從證交所 OpenAPI 獲取全台股名冊...")
    response = requests.get(url)
    
    if response.status_code != 200:
        print("無法連線至證交所 API")
        return None
        
    data = response.json()
    df = pd.DataFrame(data)
    
    df_clean = df[['公司代號', '公司簡稱', '產業別']].copy()
    df_clean.columns = ['stock_id', 'stock_name', 'industry']
    
    def get_custom_tag(row):
        ind = row['industry']
        name = row['stock_name']
        if '半導體' in ind:
            return '半導體供應鏈'
        elif '航運' in ind or '航空' in name:
            return '航運與觀光題材'
        elif '電子零組件' in ind:
            return '電子關鍵零組件'
        else:
            return '常規產業'
            
    df_clean['custom_tag'] = df_clean.apply(get_custom_tag, axis=1)
    
    output_path = "stock_list_with_industry.csv"
    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"全台股清單已匯出成功！儲存路徑為: {output_path}")
    
    return df_clean

if __name__ == "__main__":
    import_all_taiwan_stocks()