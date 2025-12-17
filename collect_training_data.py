# collect_training_data.py
# 收集更多訓練資料 - 目標 50,000 條

import requests
import pandas as pd
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUTPUT_FILE = "training_data.csv"

# 目標數量
TARGETS = {
    'english': {'positive': 40000, 'negative': 40000},    # 英文
    'schinese': {'positive': 40000, 'negative': 40000},   # 簡體
    'tchinese': {'positive': 40000, 'negative': 40000},   # 繁體
}

GAMES = [
    # 原有遊戲
    {"appid": 730, "name": "Counter-Strike 2"},
    {"appid": 1245620, "name": "Elden Ring"},
    {"appid": 1086940, "name": "Baldur's Gate 3"},
    {"appid": 1091500, "name": "Cyberpunk 2077"},
    {"appid": 1172470, "name": "Apex Legends"},
    {"appid": 271590, "name": "GTA V"},
    {"appid": 570, "name": "Dota 2"},
    {"appid": 892970, "name": "Valheim"},
    {"appid": 1174180, "name": "Red Dead Redemption 2"},
    {"appid": 413150, "name": "Stardew Valley"},
    {"appid": 1599340, "name": "Lost Ark"},
    {"appid": 1938090, "name": "Call of Duty"},
    {"appid": 578080, "name": "PUBG"},
    {"appid": 252490, "name": "Rust"},
    {"appid": 1203220, "name": "Naraka Bladepoint"},
    {"appid": 1517290, "name": "Battlefield 2042"},
    {"appid": 1817070, "name": "Monster Hunter Rise"},
    {"appid": 105600, "name": "Terraria"},
    {"appid": 367520, "name": "Hollow Knight"},
    {"appid": 1145360, "name": "Hades"},
    {"appid": 1817190, "name": "Marvel Rivals"},
    {"appid": 2358720, "name": "Black Myth: Wukong"},
    {"appid": 1623730, "name": "Palworld"},
    {"appid": 1426210, "name": "It Takes Two"},
    {"appid": 1290000, "name": "Nioh 2"},
    
    # 中文玩家多的遊戲
    {"appid": 1468810, "name": "鬼谷八荒"},
    {"appid": 1366540, "name": "Dyson Sphere Program"},
    {"appid": 1288310, "name": "煙火"},
    {"appid": 1279960, "name": "覓長生"},
    {"appid": 838350, "name": "太吾繪卷"},
    {"appid": 736190, "name": "Chinese Parents"},
    {"appid": 1178270, "name": "港詭實錄"},
    {"appid": 1293730, "name": "暖雪"},
    {"appid": 1794680, "name": "Vampire Survivors"},
    
    # 策略遊戲
    {"appid": 289070, "name": "Civilization VI"},
    {"appid": 1158310, "name": "Crusader Kings III"},
    {"appid": 281990, "name": "Stellaris"},
    {"appid": 394360, "name": "Hearts of Iron IV"},
    {"appid": 1142710, "name": "Age of Empires IV"},
    
    # 模擬遊戲
    {"appid": 255710, "name": "Cities: Skylines"},
    {"appid": 493340, "name": "Planet Coaster"},
    {"appid": 1336490, "name": "Euro Truck Simulator 2"},
    {"appid": 313080, "name": "The Sims 4"},
    
    # 恐怖遊戲
    {"appid": 739630, "name": "Phasmophobia"},
    {"appid": 1196590, "name": "Resident Evil Village"},
    {"appid": 1382330, "name": "Persona 5 Royal"},
    {"appid": 1817020, "name": "Marvel's Spider-Man"},
    
    # 獨立遊戲
    {"appid": 1057090, "name": "Lethal Company"},
    {"appid": 814380, "name": "Sekiro"},
    {"appid": 1238840, "name": "Armored Core VI"},
    {"appid": 1593500, "name": "God of War"},
    {"appid": 1151640, "name": "Horizon Zero Dawn"},
    {"appid": 1118310, "name": "Mother's Garden"},
    {"appid": 632360, "name": "Risk of Rain 2"},
    {"appid": 526870, "name": "Satisfactory"},
    {"appid": 960090, "name": "Bloons TD 6"},
]

def create_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    return session

def fetch_reviews(session, app_id, language, review_type, limit=500):
    """抓取特定類型的評論"""
    reviews = []
    cursor = '*'
    base_url = f"https://store.steampowered.com/appreviews/{app_id}"
    max_retries = 3
    
    while len(reviews) < limit:
        params = {
            'json': 1,
            'language': language,
            'filter': 'recent',
            'review_type': review_type,
            'num_per_page': 100,
            'cursor': cursor,
            'purchase_type': 'all'
        }
        
        for attempt in range(max_retries):
            try:
                response = session.get(base_url, params=params, timeout=30)
                data = response.json()
                
                if not data.get('success') or 'reviews' not in data:
                    return reviews
                
                batch = data['reviews']
                if not batch:
                    return reviews
                
                for r in batch:
                    if len(reviews) >= limit:
                        break
                    review_text = r.get('review', '').strip()
                    voted_up = r.get('voted_up', True)
                    
                    # === 品質過濾 ===
                    
                    # 1. 長度限制
                    if len(review_text) < 10:    # 至少10字
                        continue
                    if len(review_text) > 512:   # 最多512字
                        review_text = review_text[:512]
                    
                    # 2. 過濾過多換行 (ASCII art 通常有很多換行)
                    newline_count = review_text.count('\n')
                    if newline_count > 5:  # 超過5個換行就跳過
                        continue
                    
                    # 3. 清理多餘換行和空白
                    review_text = ' '.join(review_text.split())
                    
                    # 4. 過濾重複字元 (如 "aaaaa" 或 "!!!!!!")
                    import re
                    if re.search(r'(.)\1{4,}', review_text):  # 同一字元重複5次以上
                        continue
                    
                    # 5. 過濾純符號/表情評論
                    alpha_count = sum(1 for c in review_text if c.isalnum())
                    if alpha_count < len(review_text) * 0.3:  # 文字比例低於30%
                        continue
                    
                    # 6. 過濾 ASCII art (檢測特殊符號密度)
                    art_chars = set('─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬░▒▓█▀▄■□▪▫●○◆◇★☆♠♣♥♦')
                    art_count = sum(1 for c in review_text if c in art_chars)
                    if art_count > 3:  # 超過3個ASCII art字元
                        continue
                    
                    # 7. 過濾含網址的評論 (廣告/外連)
                    if re.search(r'https?://|www\.|\.com|\.net|\.org', review_text, re.IGNORECASE):
                        continue
                    
                    reviews.append({
                        'text': review_text,
                        'label': 1 if voted_up else 0,
                        'language': language
                    })
                
                cursor = data.get('cursor')
                if not cursor:
                    return reviews
                
                time.sleep(0.5)
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    return reviews
    
    return reviews

def save_to_csv(reviews, filename):
    if not reviews:
        return
    df = pd.DataFrame(reviews)
    header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, encoding='utf-8', header=header)

def count_data(filename):
    """統計現有資料"""
    if not os.path.exists(filename):
        return {}
    
    df = pd.read_csv(filename)
    stats = {}
    for lang in ['english', 'schinese', 'tchinese']:
        lang_df = df[df['language'] == lang]
        stats[lang] = {
            'positive': len(lang_df[lang_df['label'] == 1]),
            'negative': len(lang_df[lang_df['label'] == 0])
        }
    stats['total'] = len(df)
    return stats

def collect_reviews(session, games, language, review_type, target, lang_name, filename):
    """收集特定語言和類型的評論"""
    stats = count_data(filename)
    current = stats.get(language, {}).get(review_type, 0)
    
    if current >= target:
        print(f"  ✅ {lang_name} {review_type} 已達標 ({current}/{target})")
        return current
    
    needed = target - current
    print(f"  📥 收集 {lang_name} {review_type}... (目標: {target}, 已有: {current}, 需要: {needed})")
    
    collected = 0
    per_game = max(needed // len(games) + 50, 200)
    
    for game in games:
        if collected >= needed:
            break
        
        limit = min(per_game, needed - collected)
        print(f"    {game['name']}...", end=" ", flush=True)
        
        reviews = fetch_reviews(session, game['appid'], language, review_type, limit)
        
        if reviews:
            save_to_csv(reviews, filename)
            collected += len(reviews)
        
        print(f"得到 {len(reviews)} 條 (總計: {current + collected})")
        time.sleep(1)
    
    return current + collected

def main():
    print("=" * 60)
    print("📊 Steam 評論訓練資料收集器")
    print("=" * 60)
    print("目標:")
    print("  英文: 正負各 5,000 = 10,000")
    print("  簡體: 正負各 10,000 = 20,000")
    print("  繁體: 正負各 10,000 = 20,000")
    print("  總計: 50,000 條")
    print("=" * 60)
    
    # 顯示現有資料
    stats = count_data(OUTPUT_FILE)
    if stats.get('total', 0) > 0:
        print(f"\n📂 現有資料: {stats['total']} 條")
        for lang, name in [('english', '英文'), ('schinese', '簡體'), ('tchinese', '繁體')]:
            s = stats.get(lang, {'positive': 0, 'negative': 0})
            print(f"   {name}: 正面 {s['positive']}, 負面 {s['negative']}")
    
    session = create_session()
    
    # 收集各語言各類型
    for lang, name in [('english', '英文'), ('schinese', '簡體中文'), ('tchinese', '繁體中文')]:
        print(f"\n{'='*40}")
        print(f"🔍 {name}")
        print('='*40)
        
        for review_type in ['positive', 'negative']:
            target = TARGETS[lang][review_type]
            collect_reviews(session, GAMES, lang, review_type, target, name, OUTPUT_FILE)
    
    # 最終統計
    final = count_data(OUTPUT_FILE)
    print("\n" + "=" * 60)
    print("📈 最終統計")
    print("=" * 60)
    print(f"總計: {final.get('total', 0)} 條")
    for lang, name in [('english', '英文'), ('schinese', '簡體'), ('tchinese', '繁體')]:
        s = final.get(lang, {'positive': 0, 'negative': 0})
        print(f"  {name}: 正面 {s['positive']}, 負面 {s['negative']}, 小計 {s['positive']+s['negative']}")
    
    if os.path.exists(OUTPUT_FILE):
        print(f"\n✅ 資料已儲存至 {OUTPUT_FILE}")
        print(f"   檔案大小: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
