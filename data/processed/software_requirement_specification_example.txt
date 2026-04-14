# 軟體需求規格書 (Software Requirement Specification, SRS)

## 1. 產品功能需求 (Functional Requirements)
- **FR-01 影像讀取**：系統應支援讀取 DICOM 格式之非對比電腦斷層 (NCCT) 影像。
- **FR-02 自動辨識**：系統應能在 60 秒內自動標註疑似腦出血 (ICH) 之區域。
- **FR-03 結果呈現**：系統應產出包含出血體積估計與關鍵切面標記之 PDF 報告。

## 2. 性能需求 (Performance Requirements)
- **PR-01 準確性**：於驗證集之 Sensitivity 應大於 90%，Specificity 應大於 85%。
- **PR-02 推論速度**：單個 Case 的 GPU 推論時間應小於 5 秒。

## 3. 介面需求 (Interface Requirements)
- **IR-01 PACS 整合**：支援透過 DICOM C-STORE 協議將結果回傳至醫院影像系統。
- **IR-02 使用者介面**：網頁端檢視器需符合 Web Accessibility 指引，支援縮放、對比度調整。

## 4. 網路安全需求 (Cybersecurity)
- **CR-01 資料加密**：靜態與傳輸中之患者資料須以 AES-256 進行加密。
- **CR-02 存取控制**：具備角色基礎權限管理 (RBAC)，僅授權醫事人員存取。