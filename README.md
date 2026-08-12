# SO-101 VLA + HITL 可复现流水线

面向无 sudo 权限 Linux 训练服务器的 SO-101 端到端工程：固定源码版本、创建隔离环境、下载公开真机数据与基础模型、构建 LeRobot 数据集、训练 ACT / Diffusion Policy / SmolVLA、训练风险模型、执行离线 checkpoint 推理，并生成可审计报告。HITL 数据按轮次只增不改，可持续回流再训练。

> 当前可核验的真机基线是公开数据包中保存的 **31/50（62%）**，不是三个 1-step smoke checkpoint 的“准确率”。本仓库已贯通训练和部署前验证，但服务器没有连接机械臂和相机，因此不能冒充完成新的本地真机评测；文档中的私有 240-episode 双相机数据也未出现在工作区。详细口径见 [结果说明](docs/RESULTS.md)。

## 已完成

| 环节 | 本机验证结果 |
|---|---|
| 环境 | Python 3.12.13、LeRobot 0.6.0、PyTorch 2.11.0+cu130、4×RTX 4090、`pip check` 通过 |
| 数据 | 6 个真实 SO-101 数据集，合并后 160 episodes / 23,169 frames / 2 个语言任务；视频、时间戳、帧号和数值审计无异常 |
| 策略链路 | ACT、Diffusion、SmolVLA 均完成真实 1-step smoke 训练；checkpoint 强制离线重载并输出有限 6 维动作 |
| 公开真机评测 | 50 trials，31 success，成功率 62% |
| 风险模型 | 50 个 rollout / 6,916 frames；10-episode 小留出集帧级 85.7%、episode 级 100%，仅证明链路可运行 |
| 安全触发 | 连续低价值、关节越界、动作振荡三类 HITL guard，5 个自动化测试通过 |
| 正式训练 | 100k-step 配置和空闲 GPU 等待器已就绪；是否完成须以 `outputs/train` checkpoint 为准 |

机器可读证据位于 [`outputs/reports`](outputs/reports)，复现步骤见 [docs/REPRODUCING.md](docs/REPRODUCING.md)，62% 的原因与改进方案见 [docs/OPTIMIZATION_ROADMAP.md](docs/OPTIMIZATION_ROADMAP.md)。

## 一键复现

要求：x86_64 Linux、Git、wget/curl、NVIDIA 驱动（GPU 训练时）。不需要 sudo，Miniforge、FFmpeg 和 Python 环境都安装在仓库内部。

```bash
git clone https://github.com/tianduing/so101-vla-hitl-pipeline.git
cd so101-vla-hitl-pipeline
./scripts/run_pipeline.sh
```

流水线依次执行：

```text
锁定源码 → 本地 Miniforge 环境 → 模型/数据下载与 SHA256 校验
→ 数据解包/规范化/审计 → 风险数据与模型 → 自动测试
→ 等待满足显存阈值的 GPU → 三策略 smoke 训练 → checkpoint 离线推理与报告
```

下载物、环境、数据、视频、权重和训练日志都被 `.gitignore` 排除。首次运行需要下载数 GB 资源；缓存保留后可断点复用。代理环境可直接设置 `HTTPS_PROXY` / `https_proxy`。

分阶段执行：

```bash
make bootstrap      # 固定源码 + 本地环境，无 sudo
make download       # 基础模型与公开数据
make extract
make dataset        # 统一任务文本、合并、审计
make risk           # 50 条 rollout 风险模型
make test
make smoke          # 等待空闲 GPU 后训练三种策略
make report         # checkpoint 推理和机器可读报告
```

需要覆盖默认参数时：

```bash
cp configs/project.env.example configs/project.env
# 编辑本机路径、显存阈值或轮询周期；该文件不会提交到 GitHub
```

## 正式训练

默认 full 配置为 100,000 steps、每 20,000 steps 保存一次，ACT → Diffusion → SmolVLA 顺序执行：

```bash
make full
```

`run_when_gpu_free.sh` 只选择空闲显存达到 `GPU_MIN_FREE_MIB`（默认 18 GiB）的 GPU，不杀进程、不抢占其他用户任务。可用 `STEPS`、`BATCH_SIZE`、`NUM_WORKERS` 覆盖训练参数。每次运行写入独立时间戳目录，并记录命令、依赖版本、GPU 信息和权重哈希。

## 接入实验室数据

```bash
source env.sh
./scripts/ingest_lab_dataset.sh /path/to/so101_vla_240eps
cp configs/project.env.example configs/project.env
# 将 DATASET_ID / DATASET_ROOT 改为导入后的数据集，再执行 make full
```

导入脚本拒绝覆盖已有数据，并生成数据审计和 SHA256 清单。公开数据和私有实验室数据始终使用不同目录与结果口径。

## 真机与 HITL 安全边界

训练服务器未发现 SO-101 串口或相机设备。控制机侧复制并填写：

```bash
cp configs/robot/so101_controller.example.env configs/robot/so101_controller.env
./scripts/controller_rollout.sh
```

使用稳定的 `/dev/serial/by-id`、`/dev/v4l/by-id` 路径。真实机械臂首次部署必须有人守在急停旁，先空载、低速、限幅验证；这一步不能安全地自动取消。

`hitl_guard.py` 支持连续 3 步 `Value < 0.35`、关节目标越界、连续动作振荡三类接管触发。纠偏片段按“接管前 2 秒 + 接管过程 + 恢复后 2 秒”保存，`build_replay_dataset.sh` 每轮创建不可覆盖的新数据版本。

## 仓库结构

```text
configs/            训练与控制机配置示例
scripts/            下载、审计、训练、验证、HITL 与报告脚本
tests/              HITL guard 自动化测试
outputs/reports/    本次已验证的轻量 JSON 结果
manifests/          锁定提交、依赖快照和环境验证
data/ models/ ...   运行时自动生成，全部不提交
```

上游版本锁定在 [`manifests/git_commits.txt`](manifests/git_commits.txt)。本工程不修改 LeRobot 上游源码。
