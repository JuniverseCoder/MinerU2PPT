# Feature Brief: Unified IR Refactor with Mandatory TextRunIR.bbox

## 1. Overview
本需求将当前转换链路中的通用元素字典重构为统一强类型 IR：`ImageIR` 与 `TextIR`，并引入 `TextRunIR` 作为文本片段表达单元。重构采用一次性替换策略，默认 OCR 优先合并，MinerU 的 `text_runs` 默认懒生成。该重构目标是减少历史字级切分路径复杂度、统一数据契约、提升后续渲染与合并逻辑的一致性与可维护性。

## 2. Goals
- 将页面元素统一为 `ImageIR` / `TextIR` 两类，不再在基础 IR 层保留 `list` 语义。
- 建立 `TextRunIR` 强约束：`bbox` 必填且必须是合法矩形。
- OCR 路径默认输出完整 `text_runs`；MinerU 路径默认 `text_runs=None`，按需懒生成。
- OCR 与 MinerU 重叠合并时，默认 OCR 优先，并支持 `text` 与 `text_runs` 同步合并。
- 新增/更新单元与集成测试，确保结构校验和行为回归全部通过。

## 3. User Stories

- **US-001: 统一文本与图像元素类型**
  - **As a** 开发者, **I want to** 在转换链路中只处理 `ImageIR` 和 `TextIR`, **so that I can** 降低分支复杂度并统一后续处理逻辑。
  - **Acceptance Criteria:**
    - [ ] 页面 IR 元素仅包含 `ImageIR` 与 `TextIR`。
    - [ ] 基础 IR 层不再引入/依赖 `list` 语义类型。
    - [ ] 现有渲染流程可消费新的统一 IR。

- **US-002: 强制 TextRunIR.bbox 必填**
  - **As a** 开发者, **I want to** 对 `TextRunIR` 建立强校验, **so that I can** 保证片段级排版与行归属分析可靠。
  - **Acceptance Criteria:**
    - [ ] `TextRunIR.bbox` 缺失时校验失败。
    - [ ] `bbox` 非法（长度不为4、坐标非数值、非矩形）时校验失败。
    - [ ] 相关测试覆盖必填与非法输入场景。

- **US-003: 区分 OCR 与 MinerU 的 text_runs 生成策略**
  - **As a** 开发者, **I want to** 让 OCR 与 MinerU 采用不同的 runs 生成策略, **so that I can** 在保持质量前提下降低不必要复杂度。
  - **Acceptance Criteria:**
    - [ ] OCR 产出的 `TextIR` 默认带 `text_runs`，且每个 run 都有必填 `bbox`。
    - [ ] MinerU 产出的 `TextIR` 默认 `text_runs=None`。
    - [ ] 当下游需要 run 级信息时，MinerU 可按需懒生成。

- **US-004: 合并时同时合并 text 与 text_runs**
  - **As a** 开发者, **I want to** 在 OCR/MinerU 合并阶段同步处理 `text` 与 `text_runs`, **so that I can** 保证最终文本框内容与片段结构一致。
  - **Acceptance Criteria:**
    - [ ] 默认 OCR 优先覆盖重叠文本。
    - [ ] 多 OCR 片段可在同一文本框内按行与前后关系合并。
    - [ ] 合并后 `TextIR.text` 与 `TextIR.text_runs` 语义一致（可互相还原验证）。

## 4. Functional Requirements

- **FR-1（IR 类型）**：系统必须定义并使用 `ImageIR` 与 `TextIR` 作为基础元素类型（对应 US-001）。
- **FR-2（TextIR 基础字段）**：`TextIR` 必须包含 `bbox`、完整 `text`、`source`、`order`，并允许 `text_runs` 为 `None`（对应 US-001/US-003）。
- **FR-3（TextRunIR 必填 bbox）**：`TextRunIR.bbox` 必须为必填字段，不允许缺失或空值（对应 US-002）。
- **FR-4（TextRunIR bbox 合法性）**：`TextRunIR.bbox` 必须为长度 4 的数值数组，满足 `x2 > x1` 且 `y2 > y1`（对应 US-002）。
- **FR-5（OCR runs 约束）**：OCR 适配器输出的 `TextIR` 必须默认携带 `text_runs`，且每个 run 满足 FR-3/FR-4（对应 US-003）。
- **FR-6（MinerU runs 策略）**：MinerU 适配器输出的 `TextIR` 默认 `text_runs=None`，需要 run 级信息时再触发懒生成（对应 US-003）。
- **FR-7（合并优先级）**：重叠文本合并默认 OCR 优先（对应 US-004）。
- **FR-8（runs 合并）**：合并器必须支持 `text_runs` 合并，按“行归属 + 前后顺序”生成稳定结果（对应 US-004）。
- **FR-9（text 与 runs 一致性）**：合并后 `TextIR.text` 必须与 `text_runs` 的拼接结果一致（允许定义明确的换行规则）（对应 US-004）。
- **FR-10（无 list 语义）**：基础 IR 与合并层不处理 `list` 高层语义；此类语义留给未来布局分析层（对应 US-001）。
- **FR-11（校验边界）**：在适配器输出和合并输出两个边界都必须执行 IR 校验，发现违规应显式失败（对应 US-002/US-004）。
- **FR-12（回归保障）**：系统必须新增/更新单元与集成测试覆盖上述规则（对应全部 User Stories）。

## 5. Non-Goals (Out of Scope)
- 不在本次实现 `list`、标题层级、段落结构等高层语义分析。
- 不引入新的布局理解模型（如阅读顺序学习模型、语义分块模型）。
- 不做 UI 层功能改造（CLI/GUI 仅保持兼容接入）。
- 不以本次需求为目标进行性能优化专项（仅保证不明显退化并通过既有测试）。

## 6. Success Metrics
- 结构校验通过率：`TextRunIR.bbox` 必填与合法性测试通过率 **100%**。
- 受影响测试通过率：新增与受影响单元/集成测试 **100% 通过**。
- 合并一致性：至少覆盖 1 个真实样例（如 case1）验证“多 OCR 片段合并为单 TextIR”并通过断言。
- 回归稳定性：现有 OCR 关键回归用例（case1/2/3 相关）全部通过。

## 7. Open Questions
1. `TextRunIR` 的 style 字段最小集合是否固定为 `{bold, font_size, color, align}`，还是允许可扩展键？
2. `TextIR.text` 与 `text_runs` 拼接时的换行规则是否统一为“按 `line_index` 分组并以 `\n` 连接行”？
3. MinerU 懒生成 `text_runs` 时是否需要字级 bbox（逐字）还是行级 run 即可满足当前渲染与合并需求？
