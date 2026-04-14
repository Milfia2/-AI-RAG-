# 軟體架構設計說明 (Software Architecture Design, SAD)

## 1. 系統架構圖 (邏輯視圖)
系統由以下三大模組構成：
1. **Data Ingest 模組**：負責 DICOM 接收、De-identification 與影像預處理 (Normalization)。
2. **AI Inference 引擎**：搭載 3D U-Net 模型，執行影像分割與特徵提取。
3. **API & UI 模組**：基於 FastAPI 提供後端服務，前端以 React 呈現視覺化結果。

## 2. 元件說明
- **3D AI Model**：部署於 NVIDIA Triton Inference Server，支援模型平行運算。
- **Database**：使用 PostgreSQL 儲存病患詮釋資料與推論紀錄；MinIO 儲存原始與遮罩影像。

## 3. 軟體項目 (Software Items)
- **SOUP-01**：SimpleITK (用於影像讀取與空間變換)。
- **SOUP-02**：PyTorch (用於模型權重載入與運算)。

## 4. 異常處理與冗餘
- 當 GPU 資源不足時，系統應自動排隊或切換至 CPU 推論模式（雖速度變慢但不可當機）。
- 紀錄所有 API Error Logs 至 ELK Stack 進行監控。