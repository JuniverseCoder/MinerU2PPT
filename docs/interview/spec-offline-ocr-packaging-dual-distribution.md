# Feature Brief: Offline OCR Packaging with CPU/GPU Dual Distribution

## 1. Overview
为 MinerU2PPT 增加可离线分发能力：发布 Windows x64 的双发行包（CPU 通用版 + GPU cu118 可选版），将 OCR 模型随程序打包，运行时优先使用 GPU（开发默认 cu118），初始化失败自动回退 CPU，确保 GUI/CLI 在无网络环境下可直接完成转换。

## 2. Goals
- 发布两类安装产物：`CPU 通用版` 与 `GPU cu118 可选版`。
- 安装后首次运行不触发模型下载，离线环境可直接 OCR 转换。
- GUI 与 CLI 均支持同一离线模型加载机制。
- 运行时 GPU 优先并自动回退 CPU，失败原因可在日志中追踪。
- 打包产物中不依赖用户 Home 目录模型缓存。

## 3. User Stories

- **US-001: 离线可用的最终用户**
  - **As a** Windows 用户, **I want to** 安装后无需联网即可使用 OCR 转换, **so that I can** 在内网/无网环境稳定工作。
  - **Acceptance Criteria:**
    - [ ] 安装后断网运行 GUI，可完成 OCR 转换。
    - [ ] 安装后断网运行 CLI，可完成 OCR 转换。
    - [ ] 运行日志中无模型在线下载行为。

- **US-002: 有 NVIDIA 显卡的用户**
  - **As a** GPU 用户, **I want to** 在兼容环境下自动使用 GPU, **so that I can** 获得更高 OCR 性能。
  - **Acceptance Criteria:**
    - [ ] 在 cu118 兼容环境中默认走 GPU OCR。
    - [ ] GPU 初始化失败时自动回退 CPU 且任务不中断。
    - [ ] 日志明确记录“GPU使用/回退CPU”状态与原因。

- **US-003: 分发维护者**
  - **As a** 发布维护者, **I want to** 维护 CPU 与 GPU 两套可安装发行物, **so that I can** 面向不同硬件环境稳定分发。
  - **Acceptance Criteria:**
    - [ ] 可独立构建 CPU 安装包与 GPU cu118 安装包。
    - [ ] 两个包均内置模型目录并可离线运行。
    - [ ] 发布文档说明安装选择与适配场景。

## 4. Functional Requirements
- **FR-1**：系统必须提供 Windows x64 双发行物：CPU 通用版、GPU cu118 可选版。（US-003）
- **FR-2**：OCR 模型文件必须随发行物打包，并从程序内置路径加载，不依赖默认 Home 缓存路径。（US-001）
- **FR-3**：GUI 与 CLI 必须共享同一模型路径解析与 OCR 初始化逻辑。（US-001）
- **FR-4**：运行策略必须为“GPU 优先，失败自动回退 CPU”，并保持任务继续执行。（US-002）
- **FR-5**：系统必须在日志输出当前 OCR 后端（GPU/CPU）及回退原因。（US-002）
- **FR-6**：在无网络环境中，首次启动不得要求下载模型或在线依赖。（US-001）
- **FR-7**：构建流程必须支持安装式分发（当单文件 EXE 无法满足模型内置与稳定加载时）。（US-003）
- **FR-8**：开发环境默认 GPU 目标为 cu118，并提供可验证的本地开发/测试入口。（US-002, US-003）

## 5. Non-Goals (Out of Scope)
- 首版不覆盖 macOS / Linux 发行。
- 首版不提供“兼容所有 CUDA 版本”的单一 GPU 包。
- 首版不包含云端 OCR 服务或远程推理方案。
- 首版不引入多语言全量模型（待范围确认）。
- 首版不保证 onefile 单 EXE 形态必须成立（以可安装离线可用为优先）。

## 6. Success Metrics
- 双发行包构建成功率：100%（CPU/GPU 各至少 1 套可安装产物）。
- 离线启动成功率：100%（在测试机断网条件下 GUI/CLI 均可完成 1 个标准用例）。
- GPU 回退可用性：GPU 不可用测试机上，任务成功率 100%，且日志含回退原因。
- 模型离线命中率：100%（运行时加载路径指向程序内置模型目录）。
- 发布验收覆盖：CPU 包与 GPU 包各至少通过 1 组集成回归测试。

## 7. Open Questions
1. 首版内置模型语言范围是否固定为 `ch`（中文）？是否需要同时内置 `en`？
2. 安装器技术选型是 Inno Setup 还是 NSIS（或其他）？
3. GPU 可选版是否允许在无 NVIDIA 设备上安装（仅运行时自动回退 CPU），还是安装阶段即做硬件前置检查？
4. 发布物命名与目录规范是否有既定要求（例如 `MinerU2PPT-CPU` / `MinerU2PPT-GPU-cu118`）？