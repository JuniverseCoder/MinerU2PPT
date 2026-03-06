# Technical Blueprint: MinerU + PaddleOCR 文本补全融合

**Functional Spec:** `docs/interview/spec-mineru-paddleocr-text-merge.md`

## 1. Technical Approach Overview

目标是在不破坏现有 MinerU 主流程的前提下，引入“整页 OCR 补全 + 重叠冲突合并”能力，并通过 CLI/GUI 开关控制启停。

实现策略：
1. 在每页处理阶段执行 PaddleOCR 全页识别，提取 OCR 文本与 bbox。
2. 以 MinerU 文本元素作为主基准集合。
3. 应用合并规则：
   - OCR bbox 与任一 MinerU 文本 bbox 任意像素相交 => 丢弃 OCR 文本。
   - OCR bbox 与所有 MinerU 文本 bbox 不相交 => 保留 OCR 文本。
4. 将保留的 OCR 文本作为文本元素并入现有 `elements`，继续复用 `_process_element -> _process_text -> render` 链路。
5. OCR 异常时回退 MinerU-only 路径，并输出诊断日志，不中断批量任务。

关键落点（现有代码）：
- CLI 参数入口：`main.py:8-13`, `main.py:24`
- GUI 单模式参数：`gui.py:197`, `gui.py:257-260`, `gui.py:405-407`
- GUI 批模式参数：`gui.py:118`, `gui.py:151`, `gui.py:186`, `gui.py:410-418`
- 页面处理主流程：`converter/generator.py:741-770`
- bbox 相交逻辑：`converter/generator.py:107-110`
- 元素渲染顺序：`converter/generator.py:62-79`

## 2. File Impact Analysis

- **Created:**
  - `converter/ocr_merge.py`（OCR 适配、标准化与合并逻辑）
  - `tests/unit/test_ocr_merge.py`
  - `tests/integration/test_generator_ocr_merge.py`
  - `tests/integration/test_cli_ocr_option.py`

- **Modified:**
  - `converter/generator.py`
    - `convert_mineru_to_ppt` 新增 OCR 融合开关参数并透传
    - `process_page` 新增 OCR 执行与合并插入点
  - `main.py`
    - 新增 CLI 开关（建议 `--ocr-merge`）并传入 `convert_mineru_to_ppt`
  - `gui.py`
    - 新增中英文开关文案
    - 单模式新增 OCR 开关与参数透传
    - 批模式 `AddTaskDialog` 新增 per-task OCR 开关并透传
  - `requirements.txt`
    - 新增 PaddleOCR 相关依赖

- **Deleted:**
  - 无

## 3. Task Breakdown by User Story

### US-001: 漏识别文本补全
**Business Acceptance Criteria:**
- 开启功能后，样本集漏识别文本被补齐并提升召回。
- 补齐文本进入最终 PPT 生成流程。

**Technical Tasks:**
- [ ] **T-001.1 (OCR Adapter)**：在 `converter/ocr_merge.py` 封装整页 OCR 调用与输出结构标准化。
- [ ] **T-001.2 (Coordinate Mapping)**：实现 OCR 像素坐标到 JSON 坐标系的统一转换。
- [ ] **T-001.3 (Pipeline Integration)**：在 `process_page`（`converter/generator.py:741-770`）中接入 OCR + 合并，输出统一文本集合。
- [ ] **T-001.4 (Fallback)**：OCR 失败时回退 MinerU-only，记录错误并继续后续页/任务。

### US-002: 融合冲突可预期
**Business Acceptance Criteria:**
- OCR 与 MinerU 重叠时优先 MinerU。
- 无重叠时补充 OCR 文本。

**Technical Tasks:**
- [ ] **T-002.1 (Overlap Rule)**：复用 bbox 相交判定（等价 `x1 < x2 && y1 < y2`）。
- [ ] **T-002.2 (Merge Engine)**：实现 `merge_ocr_with_mineru_text`，输出“无重叠 OCR + 全量 MinerU”。
- [ ] **T-002.3 (Metrics)**：统计并输出 OCR 候选数、重叠过滤数、最终补充数。

### US-003: 功能可控启停
**Business Acceptance Criteria:**
- CLI/GUI 都可启停。
- 关闭开关时与当前行为一致。

**Technical Tasks:**
- [ ] **T-003.1 (CLI Wiring)**：`main.py` 增加 `--ocr-merge` 并透传。
- [ ] **T-003.2 (GUI Single Mode)**：新增单模式 `BooleanVar` + checkbox + 参数透传。
- [ ] **T-003.3 (GUI Batch Mode)**：`AddTaskDialog` 和任务结构增加 per-task OCR 开关并透传。
- [ ] **T-003.4 (Compatibility Guard)**：默认关闭 OCR 合并，确保关闭时输出一致。

## 4. Test Plan

### Unit Tests
- [ ] `tests/unit/test_ocr_merge.py`：
  - 重叠过滤（有交集丢弃 OCR）
  - 非重叠保留
  - 多 bbox 组合稳定性
  - 坐标转换边界（页边缘、极小框、非法框过滤）

### Integration Tests
- [ ] `tests/integration/test_generator_ocr_merge.py`：
  - 开启开关时，OCR 文本进入 `process_page` 后续渲染链
  - 关闭开关时，不进入 OCR 路径
  - OCR 异常时回退不终止流程
- [ ] `tests/integration/test_cli_ocr_option.py`：
  - CLI 参数解析和透传正确
  - 默认不启用 OCR 融合

### Manual / E2E Validation
- [ ] 使用真实漏识别样本对比开关前后 PPT 文本结果。
- [ ] 记录并评估成功指标：
  - 漏识别召回提升 ≥20%
  - 新增误识别率 ≤5%
  - 单页耗时增幅 ≤25%

## 5. Technical Considerations & Risks

- **依赖与部署风险**：PaddleOCR 在 Windows 环境安装复杂，需保证依赖缺失可回退且报错可诊断。
- **坐标一致性风险**：OCR 多边形/像素坐标需稳定映射到当前 JSON 坐标，否则会产生错误重叠判断。
- **性能风险**：整页 OCR 引入额外计算，需通过统计日志持续观察并优化。
- **误识别风险**：当前方案优先召回，未引入低置信度过滤阈值（保留为后续可配置项）。
- **样式一致性**：新增 OCR 文本将复用现有文本渲染路径，样式保真度低于 MinerU，但满足补全目标。
