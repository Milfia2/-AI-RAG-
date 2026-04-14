# 軟體驗證計畫 (Software Verification Plan, SVP)

## 1. 驗證策略
驗證活動分為三個層級，確保軟體實作符合設計規格：
- **單元測試 (Unit Testing)**：測試個別 Python 函數與類別，覆蓋率要求 > 80%。
- **整合測試 (Integration Testing)**：測試模組間介面 (API) 的溝通與數據一致性。
- **系統測試 (System Testing)**：完整端對端測試，模擬臨床情境。

## 2. 測試環境
- **作業系統**：Ubuntu 22.04 LTS
- **硬體**：NVIDIA RTX 3090 (24GB VRAM) / 64GB RAM
- **網路**：醫院內部區域網路環境模擬

## 3. 通過標準 (Acceptance Criteria)
- 所有測試案例 (Test Cases) 執行率達 100%。
- 關鍵 Bug (Critical/Major) 修正完畢且無 Regression Issue。
- 性能需求 (PR-01, PR-02) 於測試集驗證通過。

## 4. 回歸測試 (Regression Testing)
當程式碼進行任何修補或功能增減後，必須重新執行自動化測試腳本，確保既有功能未受損。