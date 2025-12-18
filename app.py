import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
import plotly.express as px
import os

# --- 頁面基本設定 ---
st.set_page_config(page_title="Steam 評論 AI 分析器 Pro", layout="wide", page_icon="🎮")

# --- 1. 載入 AI 模型 (快取優化) ---
@st.cache_resource
def load_models():
    # 優先載入微調後的模型
    local_model_path = "./fine_tuned_model"
    
    if os.path.exists(local_model_path):
        print(f"📦 載入微調模型: {local_model_path}")
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model=local_model_path,
            top_k=None
        )
    else:
        print("⚠️ 未找到微調模型，使用預設模型")
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=None
        )
    return sentiment_analyzer

sentiment_analyzer = load_models()

# --- 2. 工具函式：搜尋遊戲 ID ---
def get_game_id(game_name):
    try:
        url = "https://store.steampowered.com/api/storesearch/"
        params = {'term': game_name, 'l': 'english', 'cc': 'US'}
        r = requests.get(url, params=params)
        data = r.json()
        if data['total'] > 0:
            item = data['items'][0]
            return item['id'], item['name'], item.get('tiny_image', '')
        return None, None, None
    except Exception as e:
        return None, None, None

# --- 3. 工具函式：抓取評論 (帶進度顯示) ---
def fetch_reviews_with_progress(app_id, limit=100, language='english', status_obj=None):
    """
    使用 Steam API 手動分頁抓取評論，支援即時顯示下載進度。
    language='all' 時會分別抓取 english, schinese, tchinese
    """
    reviews_data = []
    
    # 如果選擇 'all'，分別抓取三種語言
    if language == 'all':
        languages = ['english', 'schinese', 'tchinese']
        per_lang_limit = limit // 3
        for lang in languages:
            if status_obj:
                status_obj.update(label=f"📥 正在下載 {lang} 評論...")
            lang_reviews = fetch_single_language(app_id, per_lang_limit, lang, status_obj)
            reviews_data.extend(lang_reviews)
        return reviews_data
    else:
        return fetch_single_language(app_id, limit, language, status_obj)

def fetch_single_language(app_id, limit, language, status_obj=None):
    """抓取單一語言的評論"""
    reviews_data = []
    cursor = '*'
    seen_texts = {}  # 用於追蹤重複評論
    
    base_url = f"https://store.steampowered.com/appreviews/{app_id}"
    
    while len(reviews_data) < limit:
        params = {
            'json': 1,
            'language': language,
            'filter': 'recent',
            'num_per_page': min(100, limit - len(reviews_data)),
            'cursor': cursor,
            'purchase_type': 'all'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            data = response.json()
            
            if not data.get('success') or 'reviews' not in data:
                break
            
            reviews = data['reviews']
            if not reviews:
                break  # 沒有更多評論了
            
            for r in reviews:
                if len(reviews_data) >= limit:
                    break
                    
                review_text = r.get('review', '')
                
                # 資料清洗：如果是空字串，就跳過
                if not review_text or len(str(review_text).strip()) == 0:
                    continue
                
                # 過濾重複評論（同樣內容最多 5 則）
                text_key = review_text[:100]  # 用前100字作為key
                if text_key in seen_texts:
                    seen_texts[text_key] += 1
                    if seen_texts[text_key] > 5:
                        continue
                else:
                    seen_texts[text_key] = 1
                
                reviews_data.append({
                    'text': review_text,
                    'votes_up': r.get('votes_up', 0),
                    'author_playtime': r.get('author', {}).get('playtime_forever', 0) // 60
                })
            
            # 更新進度顯示 (使用 status.update 更新標籤)
            if status_obj:
                status_obj.update(label=f"📥 正在下載評論... ({len(reviews_data)}/{limit})")
            
            # 取得下一頁的 cursor
            cursor = data.get('cursor')
            if not cursor:
                break
                
        except Exception as e:
            if status_obj:
                status_obj.write(f"⚠️ 下載時發生錯誤: {e}")
            break
    
    return reviews_data

# 快取版本 (用於儲存已下載的資料)
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_reviews_cached(app_id, limit=100, language='english'):
    """快取版本，不顯示進度但會儲存結果"""
    return fetch_reviews_with_progress(app_id, limit, language, None)

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定面板")
    target_language = st.selectbox("評論語言", ["all", "english", "schinese", "tchinese"], index=0, help="選擇 'all' 以抓取所有語言的評論 (包含中文)")
    review_count = st.slider("抓取評論數量", min_value=50, max_value=5000, value=200, step=50)
    st.info("💡 提示：數量越多，AI 分析時間會越長。選擇 'all' 可以抓到最多資料。")
    
    st.divider()
    st.subheader("🗄️ 快取管理")
    st.caption("評論資料會快取 24 小時，點擊下方按鈕可手動清除快取以重新下載。")
    if st.button("🗑️ 清除評論快取", width="stretch"):
        fetch_reviews_cached.clear()  # 清除快取
        st.success("✅ 快取已清除！下次分析時會重新下載評論。")

# --- 主程式介面 ---
st.title("🎮 Steam 評論 AI 分析器 Pro")
st.markdown("### 運用 NLP 技術，一鍵洞察玩家真實反饋")

# 搜尋區塊
col_search, col_btn = st.columns([4, 1])
with col_search:
    game_name_input = st.text_input("輸入遊戲名稱 (英文)", placeholder="例如: Palworld, Elden Ring")
with col_btn:
    st.write("") # 排版佔位用
    st.write("")
    analyze_btn = st.button("🚀 開始分析", width="stretch")

if analyze_btn and game_name_input:
    # 1. 搜尋遊戲
    with st.spinner(f"正在搜尋 '{game_name_input}' ..."):
        app_id, official_name, img_url = get_game_id(game_name_input)
    
    if not app_id:
        st.error("❌ 找不到該遊戲，請檢查拼字 (請輸入英文名稱)。")
    else:
        # 顯示遊戲資訊
        st.divider()
        head_col1, head_col2 = st.columns([1, 5])
        with head_col1:
            if img_url:
                st.image(img_url)
        with head_col2:
            st.subheader(f"{official_name} (ID: {app_id})")
            st.caption(f"正在分析最近的 {review_count} 條評論...")

        # 使用 st.status 來包裝整個過程，讓使用者知道進度
        with st.status("🚀 正在執行任務...", expanded=True) as status:
            
            # 2. 抓取資料 (使用快取)
            reviews_data = fetch_reviews_cached(app_id, limit=review_count, language=target_language)
            
            if not reviews_data:
                # 快取沒資料，嘗試即時下載
                status.write("📥 快取無資料，正在下載...")
                reviews_data = fetch_reviews_with_progress(app_id, limit=review_count, language=target_language, status_obj=status)
            
            if not reviews_data:
                status.update(label="⚠️ 任務中止：無法抓取數據", state="error")
                st.warning("⚠️ 無法抓取到足夠的評論數據。")
            else:
                status.write(f"✅ 已成功抓取 {len(reviews_data)} 條評論。")
                
                # --- AI 分析階段 (批次處理優化) ---
                import time
                import math

                texts = [r['text'] for r in reviews_data]
                total_reviews = len(texts)
                BATCH_SIZE = 10 
                
                status.write("🤖 AI 正在閱讀並分析評論中...")
                progress_bar = st.progress(0)
                # progress_text = st.empty() # 改用 progress_bar 的 caption 或者直接在 status 顯示
                
                predictions = []
                start_time = time.time()
                
                # 批次推論迴圈
                num_batches = math.ceil(total_reviews / BATCH_SIZE)
                
                for i in range(num_batches):
                    batch_start = i * BATCH_SIZE
                    batch_end = min((i + 1) * BATCH_SIZE, total_reviews)
                    batch_texts = texts[batch_start:batch_end]
                    
                    # 執行推論
                    batch_preds = sentiment_analyzer(batch_texts, truncation=True, max_length=512)
                    predictions.extend(batch_preds)
                    
                    # 計算進度
                    current_count = batch_end
                    progress = current_count / total_reviews
                    
                    # 計算時間與 ETA
                    elapsed_time = time.time() - start_time
                    avg_time_per_item = elapsed_time / current_count if current_count > 0 else 0
                    remaining_items = total_reviews - current_count
                    eta_seconds = remaining_items * avg_time_per_item
                    
                    # 更新進度條與文字
                    progress_bar.progress(progress, text=f"進度: {int(progress*100)}% ({current_count}/{total_reviews}) - 預估剩餘: {eta_seconds:.0f}s")
                
                total_time = time.time() - start_time
                status.write(f"✅ AI 分析完成！共耗時 {total_time:.1f} 秒")
                status.update(label="🚀 分析完成！", state="complete", expanded=False)
                time.sleep(1) 
                progress_bar.empty()
            
                # 整理結果
                final_results = []
                positive_count = 0
                
                for i, pred in enumerate(predictions):
                    # 找出分數最高的標籤
                    best_label = max(pred, key=lambda x: x['score'])
                    label = best_label['label']
                    score = best_label['score']
                    
                    is_positive = label == 'POSITIVE'
                    if is_positive:
                        positive_count += 1
                    
                    final_results.append({
                        "評論內容": texts[i],
                        "AI 判斷": "正面 (Good)" if is_positive else "負面 (Bad)",
                        "信心分數": score,
                        "遊玩時數(hr)": reviews_data[i]['author_playtime'],
                        "按讚數": reviews_data[i]['votes_up']
                    })
                
                df = pd.DataFrame(final_results)
                
                # --- 結果儀表板 ---
                
                # [區域 1] 關鍵指標 (KPI)
                kpi1, kpi2, kpi3 = st.columns(3)
                pos_rate = (positive_count / len(df)) * 100
                kpi1.metric("總評論數", f"{len(df)} 條")
                kpi2.metric("AI 好評率", f"{pos_rate:.1f}%")
                kpi3.metric("平均遊玩時數", f"{df['遊玩時數(hr)'].mean():.1f} 小時")
                
                st.divider()
                
                # [區域 2] 圖表區
                # 圖表區
                st.subheader("📊 好壞評比分佈")
                fig_pie = px.pie(
                    df, 
                    names="AI 判斷", 
                    color="AI 判斷",
                    color_discrete_map={"正面 (Good)": "#66c2a5", "負面 (Bad)": "#ef553b"},
                    hole=0.4
                )
                st.plotly_chart(fig_pie, width="stretch")


                
                # [區域 4] 詳細資料表
                with st.expander("點擊查看詳細評論數據表"):
                    st.dataframe(df, width="stretch")
