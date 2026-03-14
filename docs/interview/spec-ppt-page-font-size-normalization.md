# Product Requirements Document: PPT Page-Level Font Size Normalization Pre-Render

## 0. Document Control
- Version: v0.1
- Status: Approved
- Source: requirements interview conversation
- Last Updated: 2026-03-14

## 1. Decision Summary

### 1.1 Confirmed Decisions
- D1: 在**每页渲染前最后一步**执行文本字号后处理（元素已完成处理/覆盖之后）。
- D2: 仅在**当前页内**处理，不跨页统计或回写。
- D3: 分组一级条件仅按 `bold`（粗体/非粗体）区分，不按语义类型（标题/正文/脚注）额外拆分。
- D4: 在每个 `bold` 类内，字号分组要求组内满足 `max_size / min_size <= 1.1`（10%）。
- D5: 分组使用“最终有效字号”：`run.style.font_size -> elem.style.font_size -> bbox估算`。
- D6: 仅对成员数 `>=2` 的组执行替换。
- D7: 目标字号为组内中位数，回写时采用**四舍五入整数 pt**。
- D8: 回写优先级：有 runs 则回写到 run 级；无 runs 的文本元素回写到 elem 级。
- D9: `font_name` 不作为分组约束条件。

### 1.2 Out of Scope / Non-Goals
- 不改变文本内容、位置、颜色、字体名等非字号属性。
- 不引入跨页统一字号策略。
- 不新增 GUI/CLI 参数开关（默认开启本后处理）。

## 2. Scope & Boundaries

### 2.1 In Scope
- 在现有页面处理流程中增加一个“字号归一化”后处理函数。
- 对可渲染文本元素及其 runs 执行分组与中位数替换。
- 保持现有渲染链路兼容（无需改动主渲染接口行为）。

### 2.2 Constraints & Dependencies
- 插入点应位于 `converter/generator.py` 的 `PPTGenerator.process_page` 渲染前流程中（`elements` 已 materialize 后，`render_to_slide` 前）。
- 必须兼容当前 IR 数据结构和 OCR/MinerU merge 后结果。
- 计算与回写应避免影响非文本元素流程。

## 3. Final-State Process Flow

### 3.1 End-to-End Happy Path
1. 页面元素 merge/materialize 完成，获得当前页最终文本元素集合。
2. 提取每个文本 run（或无 run 时 elem）的“最终有效字号”与 bold 状态。
3. 先按 `bold` 分桶。
4. 在每个桶内按字号进行分组，确保每组满足 `max/min <= 1.1`。
5. 对成员数 `>=2` 的组计算中位数，四舍五入为整数 pt。
6. 回写字号（优先 run，fallback elem）。
7. 进入原有渲染流程，输出 PPT。

### 3.2 Key Exception Flows
- EX-1: 某页无文本元素 -> 跳过后处理，直接渲染。
- EX-2: 某元素无可用字号来源 -> 该条目不参与分组，不中断页面处理。
- EX-3: 某组仅 1 个成员 -> 不替换，保持原字号。
- EX-4: 组内中位数与现字号相同 -> 可不写回或写回同值，结果等价。

## 4. Functional Requirements

### FR-001 页面级后处理插入
- Description: 在每页文本渲染前执行字号归一化。
- Trigger/Input: `process_page` 中完成 `elements` 标准化后。
- Processing rules: 调用新增后处理函数；函数仅处理当前页文本相关数据。
- Output/Result: 返回更新后的 `elements`。
- Error/Failure behavior: 无文本或无有效样本时安全跳过。
- Priority: Must

### FR-002 字号样本提取
- Description: 为每个可处理文本单元确定字号样本与 bold 属性。
- Trigger/Input: 当前页 `elements`。
- Processing rules: 字号来源优先级为 run.style -> elem.style -> bbox估算；bold 取最终可用布尔值。
- Output/Result: 形成可分组样本集合。
- Error/Failure behavior: 单条目缺失数据则跳过该条目。
- Priority: Must

### FR-003 分组策略
- Description: 按 bold + 10% 阈值分组。
- Trigger/Input: 样本集合。
- Processing rules: 先按 bold 分桶；桶内分组需满足 `max/min <= 1.1`。
- Output/Result: 一组或多组候选组。
- Error/Failure behavior: 无法归组样本保留原字号。
- Priority: Must

### FR-004 中位数替换
- Description: 对有效组执行中位数字号统一。
- Trigger/Input: 已分组结果。
- Processing rules: 仅组大小>=2；目标字号=中位数并四舍五入整数 pt。
- Output/Result: 组内成员字号统一。
- Error/Failure behavior: 计算异常时该组跳过，不影响其他组。
- Priority: Must

### FR-005 回写层级
- Description: 将替换结果写回 IR 供现有渲染使用。
- Trigger/Input: 组内目标字号。
- Processing rules: 优先写 run.style.font_size；无 runs 时写 elem.style.font_size。
- Output/Result: 渲染阶段读取到归一化字号。
- Error/Failure behavior: 回写失败条目跳过并继续。
- Priority: Must

## 5. Acceptance Criteria (Release Gate)
- AG-001 (FR-001): 后处理仅发生在单页渲染前，且不改变页面处理主流程顺序。
- AG-002 (FR-002): 在混合来源文本（run/elem/bbox）场景下，样本提取覆盖率符合预期且无崩溃。
- AG-003 (FR-003): 任一组内满足 bold 一致且 `max/min <= 1.1`；不符合条件不应强行并组。
- AG-004 (FR-004): 组大小>=2时，组内最终字号为同一整数 pt 且等于中位数四舍五入结果。
- AG-005 (FR-005): 有 runs 的文本元素以 run 级字号生效；无 runs 文本元素以 elem 级字号生效。
- AG-006 (FR-001~005): 非文本元素渲染结果与现状一致。

## 6. Verification Plan
- Unit tests required:
  - 分组算法：bold 分桶、10% 阈值、组大小阈值、边界值（恰好10%）。
  - 中位数计算与整数化（奇偶样本数）。
  - 回写优先级（run 优先 / elem fallback）。
- Integration tests required:
  - 在 `process_page` 流程中验证插入点生效与渲染结果读取一致。
  - OCR + MinerU 混合文本页验证统一后字号分布收敛。
- Functional/smoke tests required:
  - 选取 demo 页检查视觉一致性（粗体类与非粗体类各自内部趋于一致）。
- Evidence needed for sign-off:
  - 测试通过记录 + 至少一组前后对比样例（字号统计或截图）。

## 7. Open Questions
- Q1: 分组在同一 bold 桶内采用何种“确定性策略”（例如排序后贪心）以保证结果稳定；当前 PRD 仅约束分组条件与结果约束，不强制具体算法实现细节。
