# Steam 遊戲評論情感分析系統

## 一、動機與目的

### 動機

隨著遊戲市場蓬勃發展，Steam 平台上每天產生大量玩家評論。這些評論包含了玩家對遊戲的真實反饋，對於開發者、玩家和研究者都具有重要價值。然而，手動閱讀大量評論既耗時又低效，因此需要一個自動化的情感分析工具來快速理解玩家的整體評價傾向。

此外，Steam 評論涵蓋多種語言，尤其是中文（繁體和簡體）與英文，現有的情感分析模型多針對英文設計，對中文遊戲評論的分析效果不佳。因此，本專案旨在建立一個專門針對遊戲領域、支援多語言的情感分析系統。

### 目的

1. **建立遊戲領域專用的情感分析模型**：針對 Steam 遊戲評論進行微調，提升對遊戲領域用語的理解能力
2. **支援多語言分析**：涵蓋英文、繁體中文、簡體中文三種語言
3. **開發視覺化分析工具**：提供直覺的 Web 介面，讓使用者能快速了解遊戲評價分佈
4. **實現高效能推論**：利用 GPU 加速，達到即時分析大量評論的能力

---

## 二、資料來源

### 主要資料來源

- **Steam Web API**：透過 Steam 官方 API 抓取遊戲評論
  - API 端點：`https://store.steampowered.com/appreviews/{appid}`
  - 可取得欄位：評論內容、推薦與否（voted_up）、語言、遊玩時數等

### 資料集規模

| 語言 | 數量 |
|------|------|
| 簡體中文 | 89,691 |
| 繁體中文 | 64,776 |
| 英文 | 60,598 |
| **總計** | **215,065** |

### 資料來源遊戲

涵蓋 50+ 款不同類型的遊戲，包括：
- 3A 大作：Elden Ring、Cyberpunk 2077、GTA V
- 獨立遊戲：Hollow Knight、Hades、Stardew Valley
- 中文遊戲：黑神話：悟空、鬼谷八荒、太吾繪卷
- 線上遊戲：CS2、Dota 2、Apex Legends

### 資料品質控制

收集資料時進行以下過濾：
- 長度限制：10-512 字元
- 過濾 ASCII art 和過多換行
- 過濾重複字元和純符號評論
- 過濾含網址的評論

---

## 三、使用方法與技術

### 模型架構

- **基底模型**：XLM-RoBERTa-base
  - 參數量：2.7 億
  - 預訓練語言：100+ 種語言
  - 選用原因：多語言效能優於 mBERT，適合處理中英文混合場景

### 訓練流程

```
Steam 評論資料 → 資料清洗 → 分詞處理 → 模型微調 → 儲存模型
                    ↓
              XLM-RoBERTa-base
                    ↓
              二分類（正面/負面）
```

### 訓練參數

| 參數 | 設定值 |
|------|--------|
| Epochs | 7 |
| Batch Size | 32 |
| Learning Rate | 2e-5 |
| Warmup Ratio | 0.1 |
| 優化器 | AdamW |
| 混合精度訓練 | FP16 |
| Early Stopping | 3 epochs |

### 技術棧

| 層次 | 技術 |
|------|------|
| 前端介面 | Streamlit |
| 深度學習框架 | PyTorch + CUDA |
| NLP 模型 | Hugging Face Transformers |
| 資料處理 | Pandas |
| 資料視覺化 | Plotly |
| API 請求 | Requests |

### 硬體環境

- GPU：NVIDIA RTX 5070 (12GB VRAM)
- CUDA 版本：12.8

### 系統架構

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Steam API     │────▶│  資料收集腳本     │────▶│  training_data  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   使用者介面    │◀────│  微調後模型       │◀────│  模型訓練腳本    │
│   (Streamlit)   │     │  (XLM-RoBERTa)   │     │  (train_model)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 四、實際成果

1. 情感分析準確度：**89.44%**
2. 支援即時分析數百條評論（< 10 秒）
3. 提供直覺的視覺化分析報告
4. 開源完整程式碼與訓練好的模型

---

## 五、安裝與使用

### 環境需求

- Python 3.10+
- NVIDIA GPU（建議 8GB+ VRAM）
- CUDA 12.x

### 安裝步驟

```bash
# 1. 建立虛擬環境
python -m venv .venv
.\venv\Scripts\Activate.ps1

# 2. 安裝 PyTorch（CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. 安裝其他依賴
pip install transformers datasets evaluate scikit-learn accelerate pandas streamlit plotly requests

# 4. 下載模型檔案
# 從 Google Drive 下載 model.safetensors 並放到 fine_tuned_model/ 資料夾
# 下載連結：https://drive.google.com/file/d/1uZRswafEu4GSNScYoc996ahvZLvvA9wv/view?usp=drive_link
```

### 使用方式

#### 執行應用程式
```bash
streamlit run app.py
```

#### 收集訓練資料（可選）
```bash
python collect_training_data.py
```

#### 訓練模型（可選）
```bash
python train_model.py
```

---

## 六、專案結構

```
├── app.py                  # Streamlit 主應用程式
├── collect_training_data.py # Steam 評論資料收集腳本
├── train_model.py          # 模型微調訓練腳本
├── training_data.csv       # 訓練資料集 (~240,000 條)
├── fine_tuned_model/       # 微調後的模型檔案
├── requirements.txt        # Python 依賴清單
└── README.md               # 專案說明文件
```
