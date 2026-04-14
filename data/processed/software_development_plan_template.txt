# 軟體開發計畫書 (Software Development Plan, SDP)

## 1. 專案概述
- **產品名稱**：[PRODUCT_NAME] 影像 AI 輔助診斷系統
- **軟體安全性等級**：Class B (依據 IEC 62304)
- **目的**：本文件定義軟體開發之生命週期流程、資源分配與品質保證活動。

## 2. 軟體開發生命週期 (SDLC)
本專案採用 **V 模型 (V-Model)** 開發架構，確保每一階段的開發活動皆對應相應的驗證活動：
1. 需求分析 (SRS) -> 系統測試 (System Testing)
2. 架構設計 (SAD) -> 整合測試 (Integration Testing)
3. 詳細設計與編碼 -> 單元測試 (Unit Testing)

## 3. 資源與工具
- **程式語言**：Python 3.10+
- **深度學習框架**：PyTorch / TensorFlow
- **版本控制**：Git (使用 GitLab/GitHub 進行管理)
- **專案管理**：Jira / Notion

## 4. 風險管理活動
- 依據 ISO 14971 進行初步風險分析。
- 軟體失效分析 (SOUP)：針對第三方函式庫 (如 NumPy, SimpleITK) 進行評估。

## 5. 配置管理
- 建立基準線 (Baseline)：於需求完成與 Release 前進行基準線標定。
- 變更控制：所有程式碼變更須經由 Pull Request 與 Peer Review。