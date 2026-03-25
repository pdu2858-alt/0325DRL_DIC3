DEMO：https://0325drldic3-nhpzccge9hv6mjyot3vprs.streamlit.app/

# 🎰 Bandit Strategies Comparison (Multi-Armed Bandits)

這是一個基於 Streamlit 開發的互動式應用程式，旨在模擬並比較六種常見的多臂老虎機（Multi-Armed Bandit, MAB）策略。透過此工具，使用者可以直觀地觀察不同演算法在累積遺憾（Cumulative Regret）與總獎勵上的表現差異。

## 🚀 專案特點

*   **視覺化比較**：使用 Matplotlib 繪製累積遺憾曲線，支援對數尺度觀察。
*   **互動式模擬**：可調整模擬次數以取得平均結果，減少隨機誤差。
*   **深度討論**：內建針對不同演算法在實際場景（如廣告系統、臨床試驗）應用的分析與探討。

## 🧠 包含的策略 (Strategies)

本專案實作了以下六種策略：

1.  **A/B Testing**: 固定的探索期（Exploration），隨後切換至最佳路徑。簡單但較為浪費資源。
2.  **Optimistic Initial Values**: 透過設定極高的初始期望值來驅動早期的探索。
3.  **ε-Greedy (ε=0.1)**: 以固定的機率進行隨機探索，其餘時間選擇目前表現最佳的臂。
4.  **Softmax (τ=0.1)**: 基於機率分佈選擇臂，表現越好的臂被選中的機率越高。
5.  **UCB (Upper Confidence Bound)**: 考慮不確定性，優先選擇潛在上限較高的臂。
6.  **Thompson Sampling**: 基於貝氏機率（Beta 分佈）的隨機策略，是實務上表現最平衡的演算法之一。

## 📊 問題設定 (Problem Setup)

*   **總預算 (Total Budget)**: 10,000 輪 (Rounds)
*   **老虎機真實勝率 (True Means)**:
    *   Arm A: **0.8**
    *   Arm B: **0.7**
    *   Arm C: **0.5**
*   **目標**: 極大化總期望獎勵 / 極小化累積遺憾。

## 🛠️ 如何執行

1.  **安裝依賴項目**:
    確保你已安裝 Python，並執行以下命令安裝必要的套件：
    ```bash
    pip install streamlit numpy pandas matplotlib
    ```

2.  **啟動應用程式**:
    在專案根目錄下執行：
    ```bash
    streamlit run app.py
    ```

3.  **使用介面**:
    *   在側邊欄調整 **Number of Simulations to Average**（建議 10-20 次以獲得平滑曲線）。
    *   點擊 **Run Simulation** 開始模擬。
    *   向下捲動查看視覺化圖表、比較表格以及詳細的策略討論。

## 📝 結論摘要

*   **A/B Testing** 最簡單但效率最低，因為它在探索階段會固定浪費預算在較差的選擇上。
*   **UCB** 與 **Thompson Sampling** 通常能獲得最低的遺憾值，其中 Thompson Sampling 在許多現實應用中（如推薦系統）表現最為優異。
