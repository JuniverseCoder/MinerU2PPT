# Technical Blueprint: Unified IR Refactor with Mandatory TextRunIR.bbox

**Functional Spec:** `docs/interview/spec-unified-ir-text-run-bbox-required.md`

## 1. Technical Approach Overview

本次实现采用“**强类型 IR 契约先行 -> 双适配器输出统一 IR -> 统一合并 -> 生成器只消费统一 IR**”的路径，直接替换当前 dict 主导的松散结构。

### 1.1 Target IR Contract
- 定义并统一使用三类模型：
  - `ImageIR`
  - `TextIR`
  - `TextRunIR`
- `TextIR` 必含：`bbox`, `text`, `source`, `order`，`text_runs` 允许为 `None`。
- `TextRunIR` 必含：`text`, `bbox`（强制必填且合法矩形），并携带行归属/样式信息。

### 1.2 Boundary Validation Strategy
在两个边界做强校验并显式失败（不做静默纠错）：
1. Adapter 输出边界（MinerU/OCR -> IR）
2. Merge 输出边界（Merged IR -> Generator）

校验至少覆盖：
- `bbox` 长度=4、数值、且满足 `x2 > x1`, `y2 > y1`
- `TextRunIR.bbox` 缺失直接失败
- `TextIR.text` 与 `text_runs` 可按统一规则互相校验

### 1.3 Source-specific text_runs policy
- OCR 路径：默认输出完整 `text_runs`，每个 run 必有 `bbox`。
- MinerU 路径：默认 `text_runs=None`，仅在下游确实需要 run 级能力时懒生成。
- MinerU 懒生成优先使用已有字/span 几何；无字级信息时按最小可用粒度生成并继续满足 `bbox` 强校验。

### 1.4 Merge strategy (OCR-first + text/runs synchronized)
- 继续保持 OCR 优先覆盖重叠文本。
- 合并时同步处理 `TextIR.text` 与 `TextIR.text_runs`：
  - 先按行归属聚合
  - 再按几何前后顺序稳定排序
- 统一重建规则：按行分组后用 `\n` 连接，保证 `text` 与 `text_runs` 语义一致。

### 1.5 Rendering scope
- Generator 保持双通道消费：`text` / `image`。
- 不在基础 IR/合并层保留 `list` 语义，`list` 留给未来布局分析层。

---

## 2. File Impact Analysis

- **Created:**
  - `tests/unit/test_ir_text_run_validation.py`
  - `tests/unit/test_mineru_adapter_text_runs_lazy.py`
  - `tests/unit/test_ocr_adapter_text_runs_required.py`
  - `tests/unit/test_ir_merge_text_runs_sync.py`
  - `tests/integration/test_generator_unified_ir_validation_boundary.py`

- **Modified:**
  - `converter/ir.py`
    - 从当前元素级 dict 规范化扩展为 `ImageIR/TextIR/TextRunIR` 强约束模型与验证入口。
  - `converter/adapters/mineru_adapter.py`
    - 输出 `TextIR/ImageIR`，默认 `text_runs=None`，提供懒生成挂钩。
  - `converter/adapters/ocr_adapter.py`
    - 输出携带完整 `text_runs` 的 `TextIR`，并保证 run 级 `bbox` 必填。
  - `converter/ir_merge.py`
    - 将当前基于 `text/lines` 的合并扩展为 `text + text_runs` 同步合并与一致性校验。
  - `converter/generator.py`
    - 只消费统一 IR（text/image），不再依赖 legacy `lines/spans` 分支作为主契约。
  - `tests/unit/test_ir_merge.py`
    - 补充/调整 OCR 片段聚合后 `text_runs` 与 `text` 一致性断言。
  - `tests/integration/test_case1_ir_merge_line_merge.py`
    - 增加对合并后 `text_runs` 结构与重建文本一致性的断言。

- **Deleted:**
  - 本期不强制删除文件；迁移完成后清理不再使用的 legacy 路径（尤其是生成器中依赖 `lines/spans` 的旧分支）。

---

## 3. Task Breakdown by User Story

### US-001: 统一文本与图像元素类型
**Business Acceptance Criteria:**
- [ ] 页面 IR 元素仅包含 `ImageIR` 与 `TextIR`。
- [ ] 基础 IR 层不再引入/依赖 `list` 语义类型。
- [ ] 现有渲染流程可消费新的统一 IR。

**Technical Tasks:**
- [ ] **T-001.1 (IR Core)**: 在 `converter/ir.py` 定义 `ImageIR/TextIR/TextRunIR` 与统一校验入口。
- [ ] **T-001.2 (Adapter Contract)**: 改造 `mineru_adapter.py` 与 `ocr_adapter.py` 只输出 `ImageIR/TextIR`。
- [ ] **T-001.3 (Generator Contract)**: `generator.py` 输入与处理逻辑仅依赖统一 IR 的 text/image 双通道。
- [ ] **T-001.4 (List removal at base layer)**: 移除基础 IR/合并层的 `list` 语义分支。

### US-002: 强制 TextRunIR.bbox 必填
**Business Acceptance Criteria:**
- [ ] `TextRunIR.bbox` 缺失时校验失败。
- [ ] `bbox` 非法（长度不为4、坐标非数值、非矩形）时校验失败。
- [ ] 相关测试覆盖必填与非法输入场景。

**Technical Tasks:**
- [ ] **T-002.1 (Run validator)**: 实现 `TextRunIR.bbox` 必填与几何合法性校验。
- [ ] **T-002.2 (Adapter enforcement)**: 在 OCR/MinerU 输出边界强制执行 run 级校验。
- [ ] **T-002.3 (Merge enforcement)**: merge 输出后二次校验，发现违规直接失败。

### US-003: 区分 OCR 与 MinerU 的 text_runs 生成策略
**Business Acceptance Criteria:**
- [ ] OCR 产出的 `TextIR` 默认带 `text_runs`，且每个 run 都有必填 `bbox`。
- [ ] MinerU 产出的 `TextIR` 默认 `text_runs=None`。
- [ ] 当下游需要 run 级信息时，MinerU 可按需懒生成。

**Technical Tasks:**
- [ ] **T-003.1 (OCR default runs)**: OCR adapter 默认构建完整 `text_runs`，并填充行序信息。
- [ ] **T-003.2 (MinerU lazy runs)**: MinerU adapter 默认不构建 runs，提供懒生成函数并统一校验输出。
- [ ] **T-003.3 (Consumer trigger)**: 在需要 run 级能力的节点触发 MinerU 懒生成（避免全链路强制字级开销）。

### US-004: 合并时同时合并 text 与 text_runs
**Business Acceptance Criteria:**
- [ ] 默认 OCR 优先覆盖重叠文本。
- [ ] 多 OCR 片段可在同一文本框内按行与前后关系合并。
- [ ] 合并后 `TextIR.text` 与 `TextIR.text_runs` 语义一致（可互相还原验证）。

**Technical Tasks:**
- [ ] **T-004.1 (Merge model update)**: 将 `ir_merge.py` 从文本替换升级为 `TextIR` 对象级合并。
- [ ] **T-004.2 (Run-level merge)**: 对重叠 OCR 片段执行“行归属 + 几何顺序”稳定合并。
- [ ] **T-004.3 (Text rebuild rule)**: 统一按行重建 `TextIR.text`（`line_index` 分组、`\n` 连接）。
- [ ] **T-004.4 (Consistency assert)**: 合并结果强制执行 `text == rebuild(text_runs)` 一致性断言。

---

## 4. Test Plan

### Testing for US-001
- **Unit Tests:**
  - [ ] `test_ir_text_run_validation.py`: IR 类型与字段约束（只允许 `ImageIR/TextIR` 元素）。
  - [ ] 现有 `test_mineru_adapter.py` / `test_ocr_adapter.py` 升级到新契约断言。
- **Integration Tests:**
  - [ ] 端到端页面链路可从 adapters -> merge -> generator 跑通且仅消费统一 IR。

### Testing for US-002
- **Unit Tests:**
  - [ ] 缺失 `TextRunIR.bbox` 必须失败。
  - [ ] 非法 `bbox`（长度、类型、几何）必须失败。
- **Integration Tests:**
  - [ ] 在 adapter 输出边界与 merge 输出边界分别验证失败行为（显式异常）。

### Testing for US-003
- **Unit Tests:**
  - [ ] OCR adapter 默认输出 runs 且 run.bbox 全合法。
  - [ ] MinerU adapter 默认 `text_runs=None`。
  - [ ] MinerU 懒生成触发后输出满足 run 校验。
- **Integration Tests:**
  - [ ] 仅在需要 run 级信息的流程触发 MinerU 懒生成，其余流程保持 `None`。

### Testing for US-004
- **Unit Tests:**
  - [ ] `ir_merge` 对多 OCR 片段合并后，run 顺序稳定且文本可重建一致。
- **Integration Tests:**
  - [ ] `tests/integration/test_case1_ir_merge_line_merge.py`：
    - “Design as Code, Asset as Service” 合并为单文本框。
    - 合并后 `text_runs` 与 `text` 一致。
- **Regression:**
  - [ ] 现有 case1/case2/case3 OCR 关键回归用例全部通过。

---

## 5. Technical Considerations & Risks

- **风险1：旧结构兼容期的双轨复杂度**
  - 现状仍有 `lines/spans` legacy 数据路径，迁移期间容易出现“新旧契约混用”。
  - **策略**：以边界校验作为硬门，逐步收敛到统一 IR。

- **风险2：TextIR.text 与 text_runs 不一致**
  - OCR 分片与合并后顺序不稳定会导致文本重建偏差。
  - **策略**：固定“行归属 + 几何排序 + `\n` 连接”规则，并在 merge 后强制一致性断言。

- **风险3：MinerU 懒生成粒度与开销平衡**
  - 字级生成在复杂页面可能增加处理成本。
  - **策略**：默认不生成，按需触发；触发后仅生成当前节点所需 runs。

- **风险4：边界校验失败导致回归暴露**
  - 强校验会把历史隐性脏数据转为显性失败。
  - **策略**：补齐失败样例测试与错误信息，确保可定位并快速修复。

- **风险5：渲染层仍依赖局部 legacy 字段**
  - `generator.py` 当前仍有旧的 run 构建分支。
  - **策略**：将渲染入参统一到 `TextIR.text_runs`，把 legacy 分支降为迁移期兜底并计划后续删除。
