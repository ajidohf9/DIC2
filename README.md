# DIC-2: Kaggle 50 Startups & Boston Housing 機器學習迴歸分析與特徵挑選

## 專案公開展示網址 (Live Website)
👉 **[https://ajidohf9.github.io/DIC2/](https://ajidohf9.github.io/DIC2/)**

## GitHub 程式碼倉庫 (Repository)
👉 **[https://github.com/ajidohf9/DIC2](https://github.com/ajidohf9/DIC2)**

---

## 專案概述 (Project Overview)
本專案為 **物聯網數據分析與應用** 課程之 **DIC-2 (Do In Class 2)** 實作。
依據 **CRISP-DM** 規範，針對經典迴歸資料集系統性評估 5 種特徵選取演算法之表現：
1. **Lasso (L1 Regularization Ranking)**
2. **Forward SFS (Sequential Feature Selector)**
3. **RFE (Recursive Feature Elimination)**
4. **ElasticNet Ranking (L1 + L2 Hybrid)**
5. **Exhaustive Search (Best Subset Selection)**

---

## ★ 加分項完成項目 (Bonus Features Completed)

### 1. 加分項 1：下拉視窗呈現兩種 Problem (加總分 1 分)
- 頂部導覽列提供「**題目切換下拉選單**」，可即時切換：
  - **📊 題目一：50 Startups（新創獲利預測，5 個特徵）**
  - **🏡 題目二：Boston Housing（波士頓房價預測，13 個特徵）**
- 切換時，整頁之特徵選擇階數、KPI 卡片、動態迴歸方程式、領域洞見說明、動態 Chart.js 折線圖、即時目標值預測模擬器全部動態聯動切換！

### 2. 加分項 2：可任意輸入 Dataset URL 載入並分析 (加總分 2 分)
- 點擊「**自訂 URL (加分項 2)**」按鈕，可輸入任何公開 Raw CSV 網址（預設包含老師示範之 `https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`）。
- 內建前端即時解析與迴歸計算引擎（PapaParse + 線性代數 OLS Solver），一鍵動態計算特徵相關度排序、特徵階數 RMSE 與 R² 走勢，並自動加入下拉選單切換展示！

---

## 核心研究發現 (Key Findings & Parsimony Analysis)

1. **50 Startups 最佳簡約甜蜜點 (Optimal Sweet Spot, $k = 2$)**:
   - 最佳選取特徵：`R&D Spend` + `Marketing Spend`
   - **Test RMSE 最低**：**$8,198.80**
   - **決定係數 ($R^2$) 最高**：**0.9474 (94.74%)**
   - **Adjusted $R^2$**：**0.9324**
   - **F-Statistic**：**349.00** ($p < 0.0001$)

2. **全特徵過擬合懸崖 (Overfitting Cliff, $k = 5$)**:
   - 納入全部 5 個變數（含行政支出與州別虛擬變數）時：
   - **Test RMSE 惡化反彈至 $9,137.99**（誤差暴增 **+11.5%**）
   - **Adjusted $R^2$ 驟降至 0.8530**
   - 實證支持奧坎剃刀原則（Occam's Razor）：冗餘變數只會引入噪音並損害模型的泛化可靠度。

---

## 專案檔案清單 (Deliverables)

| 檔案名稱 | 說明 |
| :--- | :--- |
| `index.html` | 現代太空暗黑玻璃擬態儀表板（支援雙題切換 + 任意 CSV 載入） |
| `style.css` | 玻璃擬態科技感樣式表（自適應排版與微光動態效果） |
| `app.js` | 核心互動邏輯、Chart.js 即時圖表、自訂 CSV 即時計算引擎 |
| `model_data.js` / `model_data.json` | 50 Startups 與 Boston Housing 完整特徵階數結構化數據 |
| `50_Startups.csv` | 50 Startups 原始數據檔 |
| `Boston_Housing.csv` | Boston Housing 原始數據檔 |
| `feature_progression_analysis.py` | 50 Startups 特徵選取分析腳本 |
| `process_both_datasets.py` | 雙資料集批次特徵計算腳本 |
| `all_5_algorithms_comparison_plot.png` | 50 Startups 特徵選取雙子圖 (RMSE & R²) |
| `boston_housing_all_5_algorithms_comparison_plot.png` | Boston Housing 特徵選取雙子圖 |
| `.github/workflows/pages.yml` | GitHub Pages 自動部屬工作流 |
| `requirements.txt` | Python 環境相依套件清單 |
