# 四卡训练与抓取可视化

## 四张 RTX 4090 怎么用

本工程使用 LeRobot 原生支持的 Hugging Face Accelerate + PyTorch FSDP，而不额外引入 DeepSpeed。`FULL_SHARD` 会切分参数、梯度和优化器状态，语义对应 DeepSpeed ZeRO-3；与普通 DDP 不同，它确实能降低每卡模型状态占用。

已经在当前 4×RTX 4090 上完成 ACT、Diffusion、SmolVLA 的 2-step FSDP 实测和完整 checkpoint 离线重载。每卡 batch=1 时，训练器报告的峰值显存约为：

| 模型 | 每卡峰值 | 备注 |
|---|---:|---|
| ACT | 0.37 GB | 有效 batch=4 |
| Diffusion | 1.38 GB | 有效 batch=4 |
| SmolVLA | 1.78 GB | 有效 batch=4，冻结 VLM 复制、可训练专家 FULL_SHARD |

这些是本次训练进程的 PyTorch 峰值，不含服务器上其他进程已经占用的约 14GB/卡。实测时每卡仍有约 9.7–9.8GB 空闲，因此将正式队列门槛从“任一单卡空闲18GB”改为“四卡各空闲7GB且瞬时利用率不高于阈值”。

```bash
make fsdp-smoke
make fsdp-full
```

`make fsdp-full` 会顺序训练三种策略；全部成功后自动离线重载最终 checkpoint、刷新报告，并为每种策略导出带“预测动作/录制动作”叠字的短视频。当前工作区已有 ACT、Diffusion、SmolVLA 的 100k checkpoint；发布到 GitHub 的是复现脚本和轻量证据，不包含大体积权重。

动态选卡器会读取每张 GPU 的剩余显存和利用率，选择满足条件的卡并记录快照。FSDP 是同步训练，一张卡突然 OOM 会导致整组失败，所以仍保留约 5GB 的实测安全余量，且不会终止其他用户进程。

配置文件：

- `configs/accelerate/fsdp_4gpu_act.yaml`
- `configs/accelerate/fsdp_4gpu_diffusion.yaml`
- `configs/accelerate/fsdp_4gpu_smolvla.yaml`

SmolVLA 的自定义注意力混合 bf16/fp32 参数并直接读取层投影，原始 FSDP1 会失败。`patches/lerobot-fsdp-trainable-fp32.patch` 在 FSDP 模式下保留统一 fp32 可训练主权重（计算仍用 bf16），冻结 VLM 不分片，可训练动作专家使用根级 `FULL_SHARD`。补丁由 bootstrap 自动、幂等应用。

## 可以看到哪些动画

### 1. 已有真实机器人抓取动画

公开评测包含真实 SO-101 相机视频。本工程已导出一个成功与失败的同步对比：

```bash
make viz-animation
```

输出：

- `outputs/visualization/real_so101_success_vs_failure.mp4`
- `outputs/visualization/real_so101_success_vs_failure.gif`

这是真实机器人录像，但属于公开上游评测，不是新 checkpoint 的执行结果。

服务器本身不需要浏览器：MP4/GIF 都是普通文件，可以直接用 VS Code Remote 下载，或在自己的电脑执行：

```bash
scp USER@SERVER:/path/to/so101-vla-hitl-pipeline/outputs/visualization/real_so101_success_vs_failure.mp4 .
```

### 2. Rerun / Foxglove 交互回放

Rerun 0.33.1 和 Foxglove SDK 已安装在项目隔离环境，无需 sudo。Rerun 可以同时播放相机、6维状态和6维动作，并可拖动时间轴：

```bash
./scripts/visualize_episode.sh rrd 0
```

生成的 `.rrd` 文件可下载到有桌面的电脑后用相同版本 Rerun 打开。也可以在服务器启动 Web Viewer，并通过 SSH 端口转发访问：

```bash
./scripts/visualize_episode.sh rerun-server 0
# 本地电脑：ssh -L 9090:127.0.0.1:9090 -L 9876:127.0.0.1:9876 USER@SERVER
```

Foxglove：

```bash
./scripts/visualize_episode.sh foxglove 0
# 客户端连接 ws://SERVER:8765；公网服务器建议使用 SSH 隧道而非直接开放端口
```

### 3. 新 checkpoint 的离线推理动画

```bash
source env.sh
python scripts/export_policy_inference_video.py \
  --checkpoint outputs/train/某次训练/checkpoints/last/pretrained_model \
  --episode 0 --device cpu
```

视频会在真实录制画面上逐帧显示“数据中实际动作”和“模型预测动作”。这是离线 replay，可检查动作趋势、抖动和明显偏差，但不是闭环抓取成功率。

### 4. 新模型的真实抓取动画

要看到新模型真正抓取，必须把训练 checkpoint 放到连接 SO-101、相机和急停的控制机上执行 rollout，并录制成 LeRobotDataset。`controller_rollout.sh` 已启用实时显示；录制结果随后可直接交给 Rerun/Foxglove 和上述 MP4 导出器。

训练服务器没有机械臂和摄像头，因此不能用仿真或生成视频冒充真实抓取。真实闭环录像必须在控制机实际执行后产生。
