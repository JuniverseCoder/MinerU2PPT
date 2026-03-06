# Technical Blueprint: OCR-First Forced Textbox Refine

**Functional Spec:** `docs/interview/spec-ocr-first-forced-textbox-refine.md`

## 1. Technical Approach Overview

本方案将文本处理主路径切换为 **OCR-first 且强制 OCR**，并将 MinerU 的角色收敛为“重叠关系参考 + 聚合辅助”，不再用于覆盖 OCR 文本框或主导样式推导。

### 1.1 强制 OCR（不回退）
- 移除“可选 OCR”语义：执行流程中 OCR 必须运行。
- OCR 初始化/推理失败时直接抛错终止，不再 fallback 到 MinerU-only。
- 对外接口层（CLI/GUI）删除 OCR 开关，避免用户误解为可关闭。

### 1.2 OCR 文本框校正（两阶段）
对每个 OCR 文本框执行如下流程：

1) **预构造 pad 搜索窗口（左右/上下各 10%）**
- 目的：一次性完成横向字体颜色统计缓存，减少重复扫描开销。
- 注意：该 pad 窗口不直接用于第一波裁剪，仅用于后续 extend 搜索范围。

2) **第一波：原框内向裁剪（去除过多）**
- 仅在原始 OCR 框内进行上/下边界裁剪。
- 规则：
  - 从上往下，直到遇到字体色行，裁掉其上无效区域；
  - 从下往上，直到遇到字体色行，裁掉其下无效区域。
- 目标：解决 OCR 原框偏大问题。

3) **第二波：按需外向 extend（补齐过少）**
- 仅对“第一波未裁剪到边界的一侧”进行外扩。
- 在预构造的 pad 窗口内继续向外搜索，直到遇到无字体色行。
- 目标：解决 OCR 原框偏小导致的下沿截断。

### 1.3 OCR-first 样式推导
- 文本颜色、字号、行数优先基于 OCR 结果 + 图像采样：
  - 颜色：校正后框（或行框）采样；
  - 字号：按 OCR 行高/框高推导；
  - 行数：以 OCR 行结果为准。
- OCR 来源文本不再走当前 MinerU 主导的字符级投影分析主路径。

### 1.4 MinerU 仅用于框聚合参考
- 当前“重叠则丢 OCR”策略改为：
  - 重叠关系用于分组；
  - 最终框采用 **OCR 框聚合后的最小外接框**（可参考重叠 MinerU 框参与分组判定，但不直接替换最终框）。
- 文本内容与样式仍由 OCR-first 路径生成。

---

## 2. File Impact Analysis

### 2.1 `converter/ocr_merge.py`（核心改动）
- 新增/重构：
  - OCR bbox 校正管线（pad window + 第一波内向裁剪 + 第二波外向 extend）。
  - 字体色行判定工具（行级颜色匹配、上/下边界搜索）。
  - OCR 行合并/框合并工具（英文拆段场景）。
- 修改：
  - `merge_ocr_text_elements(...)` 从“重叠丢弃 OCR”改为“重叠分组并输出 OCR 主导最小外接框”。

### 2.2 `converter/generator.py`（主流程改动）
- `process_page(...)`：
  - 去除可选分支与 fallback 日志路径；OCR 失败直接异常。
  - 接入新的 OCR-first 合并输出。
- `_process_text(...)`：
  - 增加 `source=ocr` 快路径：按 OCR lines + 校正框推导 `text_runs`。
  - 保留 MinerU 路径仅用于兼容/非 OCR 来源。

### 2.3 `main.py`（接口改动）
- 删除 `--ocr-merge` 参数及相关透传。
- `convert_mineru_to_ppt(...)` 调用始终执行 OCR 路径。

### 2.4 `gui.py`（接口改动）
- 删除 OCR 复选框（单文件与批任务对话框）。
- conversion 调用不再传 OCR 开关参数。
- 继续复用 shared OCR engine 以控制初始化开销。

### 2.5 `tests/*`（回归改造）
- 更新已有测试中“可选 OCR / fallback / overlap-drop”的断言。
- 新增 bbox 两阶段校正与 case3 边界规则测试。

---

## 3. Task Breakdown by User Story

### US-001: 强制 OCR 主流程
- [ ] T1-1：移除 CLI `--ocr-merge` 参数与透传逻辑。
- [ ] T1-2：移除 GUI OCR 开关（single + batch dialog）。
- [ ] T1-3：`process_page` 改为强制 OCR，移除 fallback to MinerU-only 分支。
- [ ] T1-4：统一 OCR 失败异常信息（便于定位与测试断言）。

### US-002: OCR 文本框上下限校正
- [ ] T2-1：实现 pad 搜索窗口（10%）及缓存统计。
- [ ] T2-2：实现第一波原框内向裁剪（上/下）。
- [ ] T2-3：实现第二波按需 extend（上/下）。
- [ ] T2-4：输出调试可视化（原框、pad 窗口、校正框）。

### US-003: OCR 样式优先推导
- [ ] T3-1：为 OCR 来源新增样式构建路径（颜色/字号/行数）。
- [ ] T3-2：引入 OCR 行内合并规则，避免英文误拆导致样式错位。
- [ ] T3-3：调整 `_process_text` 路径选择逻辑（OCR-first）。

### US-004: MinerU 仅作框聚合参考
- [ ] T4-1：重写 overlap merge 策略（由 drop 改为 grouping + union）。
- [ ] T4-2：最终框使用 OCR 聚合最小外接框，不直接采用 MinerU 原框。
- [ ] T4-3：更新 merge 统计字段（groups/merged/added）。

---

## 4. Test Plan

### 4.1 单元测试
- `tests/unit/test_ocr_merge.py`（更新）
  - 重叠场景不再丢 OCR，改断言为分组聚合结果。
- 新增 `tests/unit/test_ocr_bbox_refine.py`
  - 第一波内向裁剪：能去除 pad 过多边界。
  - 第二波外向 extend：能补齐 pad 不足边界。
  - 规则约束：第一波不使用 pad 区裁剪，第二波才使用 pad 区。

### 4.2 集成测试
- `tests/integration/test_generator_ocr_merge.py`（更新）
  - 删除 fallback 成功断言，改为 OCR 失败即异常。
- `tests/integration/test_cli_ocr_option.py`（替换/重命名）
  - 改为强制 OCR 的 CLI 行为测试（无开关）。
- `tests/integration/test_case1_ocr.py`（保留并更新）
  - 目标文本必须识别成功。
- `tests/integration/test_case1_ocr_bbox_alignment.py`（保留）
  - 标题 IoU ≥ 0.90。
  - 英文行合并后与 JSON 行框 IoU ≥ 0.90。
- `tests/integration/test_case3_ocr_bbox_bottom.py`（更新判定规则）
  - **不使用固定 y2 阈值**。
  - 判定标准：校正后 bbox 的 `y2` 行应为“无字体颜色行”（即边界落在文本之外第一行）。

### 4.3 手工验证
- 生成 debug 图：对比原框、pad 窗口、校正框与最终渲染。
- 样本集人工检查 case3 目标句下沿是否仍有可见截断。

---

## 5. Technical Considerations & Risks

1. **颜色判定阈值敏感**
- 字体色距离阈值在浅色背景/反走样场景下可能波动。
- 需将阈值常量集中管理，便于调优与回归。

2. **OCR 分段与行合并风险**
- 英文常被拆成多个框；合并阈值过松会误并，过紧会漏并。
- 需使用 case1 英文行做稳定门槛回归。

3. **性能影响**
- 新增两阶段边界扫描会增加开销。
- 通过“先 pad 一次统计”缓存减少重复扫描。

4. **接口变更风险**
- 删除 CLI/GUI 开关为破坏性行为，需在文档与发布说明中明确。

5. **回归风险**
- 当前测试里有多处基于“可选 OCR / fallback / overlap-drop”的历史断言，需同步改造避免伪失败。
