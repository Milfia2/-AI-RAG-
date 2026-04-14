# AI 模型標註與訓練協議 (AI Training & Labeling Protocol)

## 1. 數據來源與預處理
- **資料源**：收集 [PROVIDER_NAME] 之多中心去識別化 NCCT 影像。
- **前處理**：統一調整為 1.0x1.0x5.0 mm 之 Resampling，並執行腦部窗格 (Brain Window) 轉換。

## 2. 標註規範 (Ground Truth)
- **標註人員**：由三名放射科醫師進行獨立標註。
- **共識機制**：若標註區域重疊率 (IoU) < 0.7，則由第四名資深醫師進行最終審核。
- **標註工具**：使用 ITK-SNAP 或 CVAT 進行 3D Segmentation 標註。

## 3. 訓練與驗證集劃分
- **比例**：Training (70%) / Validation (10%) / Testing (20%)。
- **分層抽樣**：依據出血類型 (IPH, IVH, SAH) 與體積大小進行分層，確保樣本分佈均衡。

## 4. 偏差緩解 (Bias Mitigation)
- 定期檢查模型在不同年齡、性別與不同品牌 CT 機台上的性能差異。
- 數據增強 (Data Augmentation)：加入隨機旋轉、縮放與雜訊，提升模型泛化能力。