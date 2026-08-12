# 复现手册

## 1. 无 sudo 环境

```bash
git clone https://github.com/tianduing/so101-vla-hitl-pipeline.git
cd so101-vla-hitl-pipeline
./scripts/bootstrap.sh
source env.sh
```

`bootstrap.sh` 会检出两个固定 commit，将 Miniforge 安装到 `downloads/miniforge`，将 Python 环境安装到 `.conda_env`，并安装与校验 LeRobot。不会写系统目录。

若服务器需代理：

```bash
export HTTPS_PROXY=http://host:port
export https_proxy="$HTTPS_PROXY"
./scripts/bootstrap.sh
```

## 2. 数据和基础模型

```bash
source env.sh
./scripts/download_resources.sh
./scripts/extract_public_data.sh
./scripts/build_public_dataset.sh
python scripts/build_risk_dataset.py
```

下载脚本获取 `lerobot/smolvla_base`、SmolVLM2 processor 文件以及 `Shaibk/so101-smolvla-thesis` 数据归档。归档在解包前校验发布方 SHA256；解包脚本拒绝绝对路径和 `..` 路径穿越成员。

## 3. 自动验证

```bash
./scripts/verify_install.sh
python -m pytest -q tests
python scripts/analyze_eval_failures.py
```

预期：5 tests passed，数据审计 issues 为空，公开评测为 31/50。完整 JSON 写入 `outputs/reports`。

## 4. 策略训练

快速贯通：

```bash
STEPS=1 SAVE_FREQ=1 BATCH_SIZE=1 NUM_WORKERS=0 DEVICE=cpu \
  ./scripts/train_policies.sh smoke all
./scripts/finalize_reports.sh
```

GPU smoke：

```bash
make smoke
make report
```

正式训练：

```bash
make full
```

每个训练目录包含实际命令、`pip freeze`、上游 commits、GPU 信息、日志和 checkpoint SHA256。`finalize_reports.sh` 自动选择每种策略最新的完整 checkpoint，强制 CPU/离线重载并执行真实样本单步推理。

## 5. 一键执行

```bash
./scripts/run_pipeline.sh
```

此入口覆盖步骤 1–4 并生成最终报告。若 GPU 空闲显存不足，等待器按 `GPU_POLL_SECONDS` 轮询；它不会终止或抢占现有进程。

## 6. 真机控制机

真机需要串口、相机和现场急停，不能在纯训练服务器伪造：

```bash
cp configs/robot/so101_controller.example.env configs/robot/so101_controller.env
# 填写稳定的 by-id 设备路径、机器人 ID、任务和 checkpoint
./scripts/controller_rollout.sh
```

首次执行先断开负载、降低速度、检查关节方向和软限位。部署验收必须由现场人员确认后进行。

## 7. 完整性核验

```bash
git diff --exit-code
cat manifests/git_commits.txt
cat manifests/install_verification.json
cat outputs/reports/pipeline_status.json
```

模型、数据和视频有意不进入 GitHub。它们可由固定来源重新下载；本地权重哈希由 `scripts/capture_manifests.sh` 生成。
