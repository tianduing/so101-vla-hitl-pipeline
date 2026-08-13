# ACT 低成功率纠错审计

日期：2026-08-12

## 结论

用户对“成功率这么低肯定有问题”的判断是对的。旧 ACT 100k 的 `0/10` 是在错误评测配置下产生的，现已作废，不能继续称为模型准确率。

纠错后，MuJoCo 已能让真实成功 Episode 0 在纯接触、摩擦和重力条件下抓起物体；但旧 100k checkpoint 随后未通过真实训练帧动作回归门禁。因此当前问题已经从“仿真评测错误”进一步定位到“旧训练权重的部署分支发生动作均值塌缩”，需要重新训练纠正版 ACT。

## 找到的五个问题

### 1. Policy 相机与训练数据完全不同

训练画面是近俯视木色工作区、白色机械臂和单个绿色目标。旧仿真输入却是低角度灰色棋盘背景、黄色机械臂，还出现蓝色目标区和红色方块。

修复：Policy 专用相机按发布的桌面 homography 和 Episode 0 画面校准；方块中心由约 `(241, 297)` 对齐到仿真约 `(245, 306)`；Policy 画面隐藏红块和蓝区，并将机器人渲染为白色。

### 2. 把真实伺服坐标误当成 MuJoCo 关节角

旧映射只用 Episode 0 首帧对齐官方 pickup keyframe。模型输出训练分布内的 `shoulder_lift=100–113°` 时，MuJoCo 轴持续裁到上限，平均动作饱和比例约为 `1/6 = 16.4%`。

修复：使用发布的 35 点桌面标定和 35 点方块顶部标定，通过 SO-101 正向运动学拟合 scale/offset。标定网格 XY RMSE 为 4.35 mm，相对高度 RMSE 为 2.06 mm；纠正后真实轨迹动作饱和率为 0。

### 3. 通用相机支架发生虚假自碰撞

Menagerie 的 `camera_box2` 碰撞体与 shoulder 自碰撞，使目标关节约 121°时实际只能到约 68°。这不是源真实机械臂的相机支架形状。

修复：保留相机支架视觉模型，但关闭两个通用 camera-box collider。修复后 Episode 0 的实际关节能跟踪目标，例如目标 121.4°、实际 121.4°。

### 4. 旧成功阈值高于真实成功轨迹

旧阈值要求从桌面静止中心高度抬升约 37 mm，而发布的真实成功 Episode 0 在标定运动学中的有效抬升约 35 mm。真实成功轨迹因此也会被旧判据判失败。

修复：改为相对每个 trial 重置高度抬升 20 mm，并保持 0.25 秒。它仍要求物体明显离桌，不会把推动或轻微弹跳算成功。

### 5. 旧 ACT 训练权重发生部署分支均值塌缩

旧 100k 训练使用 `use_vae=true`。训练时 VAE encoder 能读取真实未来动作；部署时没有未来动作，LeRobot 按标准实现使用全零 latent。审计发现部署输出的 100 步几乎是一条常数线。

证据：

- Episode 0 抽帧动作 MAE 约 18.2°；
- 标准化门禁的 16 个全数据集均匀抽样帧 MAE 为 21.40°；
- 门禁阈值为 8°，因此旧 checkpoint 明确 FAILED；
- 最大误差集中在 shoulder_lift 和 elbow_flex，而它们正是完成接近/抬升的主运动轴。

训练日志的低 loss 不能推翻这个结果，因为训练分支看到了未来动作，而真实部署分支没有。

## 修复前后硬验收

| 项目 | 修复前 | 修复后 |
|---|---:|---:|
| Policy 画面与真实构图 | 严重不一致 | 相机/颜色/干扰物已对齐 |
| 动作饱和比例 | 约 16.4% | 0%（真实轨迹） |
| 真实成功轨迹最大抬升 | 约 0.1 cm | 3.47 cm |
| 真实成功轨迹成功帧 | 0 | 51 |
| 旧 ACT 真实帧门禁 | 未设置 | 21.40°，FAILED |

真实轨迹校准动画：`outputs/real_trajectory_replay_calibrated.mp4`。

## 纠正版训练

纠正版配置：

- `use_vae=false`：删除训练/部署信息不对称；
- `chunk_size=50`：预测约 1.67 秒；
- `n_action_steps=10`：每约 0.33 秒重新看图规划；
- 启用图像颜色和小幅仿射增强，降低 Real-to-Sim 敏感度；
- 四卡 FSDP；
- 100k steps；
- 训练完成后必须先通过 32 个真实帧、MAE 不高于 8°的离线门禁；
- 门禁通过后才运行 10×30 秒 MuJoCo 正式闭环并生成动画。

后台单元：`so101-vla-corrected-pipeline.service`。当前服务器四张 4090 被其他用户任务各占约 21.4 GB，调度器等待每卡至少 7 GB 空闲后自动启动。纠正版 ACT 完成后，原先承诺的 Diffusion 和 SmolVLA 流程仍会继续。

## 证据路径

- `outputs/act100k_failed_real_data_gate.json`：旧权重门禁失败；
- `outputs/act100k_real_dataset_offline_audit.json`：Episode 0 逐帧抽样误差；
- `outputs/act_checkpoint_offline_comparison.json`：20k–100k checkpoints 对比；
- `outputs/real_trajectory_replay_calibrated.metadata.json`：真实轨迹校准后逐步数据；
- `outputs/calibration_sweep_gripper_object.json`：夹爪/物体位置扫描；
- `configs/joint_mapping.yaml`：纠正后的映射与拟合证据；
- `scripts/audit_checkpoint_real_data.py`：自动门禁实现；
- `scripts/run_corrected_pipeline.sh`：等待、训练、门禁、仿真和后续模型流水线。
