# SO-101 MuJoCo 仿真推理动画

这是严格限定为仿真的 SO-101 工程：不会扫描串口、打开 Feetech 电机、连接控制机或调用 `lerobot-record`。它从官方 MuJoCo Menagerie SO-101 模型出发，交付四种永久标明真实性等级的动画：

| 模式 | 含义 | 当前结果 |
|---|---|---|
| `SCENE_SMOKE` | 官方 `scene_box.xml` 真实物理推进和离屏渲染 | PASS |
| `REAL_TRAJECTORY_REPLAY` | 真实数据关节轨迹映射到虚拟机器人 | Episode 0，163 帧；不是 Policy |
| `SCRIPTED_EXPERT` | ground-truth 物体位姿 + IK，验证场景抓取物理 | PASS；不计入 Policy |
| `POLICY_CLOSED_LOOP` | 仿真新图像/状态 → checkpoint → 动作 → 下一物理状态 | ACT 三专家在开发位置 9/10；单 ACT 在未见位置 9/20 |
| `RGB_VISUAL_DEMONSTRATION_RETRIEVAL` | RGB 定位 → 最近物理成功示范 → 恢复执行 | 未见 seed 52–71 为 17/20（85%）；不是 ACT/VLA 模型成绩 |

## 最新准确率边界

| 对象 | 测试位置 | 成绩 |
|---|---|---:|
| ACT 三专家 | 开发位置 seed 42–51 | 9/10，90% |
| 单 ACT 主模型 | 未见位置 seed 52–71 | 9/20，45% |
| RGB 视觉检索恢复系统 | 未见位置 seed 52–71 | 17/20，85% |
| 抓起后严格保持 3 秒 | 开发位置 seed 42–51 | 4/10，40% |

因此只能说完整工程系统在这 20 个未见仿真位置上达到 85%，不能说 ACT 或 VLA 模型本身达到 85%。完整时间线、数学原理、失败实验、定位决策树和下一阶段路线见 [抓取学习手册](../problem_records/ROBOT-GRASP-001_从0到85准确率学习手册/README.md)。

一键复现 20 次 RGB 恢复系统评估：

```bash
bash scripts/run_rgb_retrieval_85_eval.sh
```

当前可直接观看：

- `outputs/so101_mujoco_showcase.mp4`：64.1 秒合集，1280×720、30 FPS、H.264；
- `outputs/scene_smoke.mp4`：官方模型冒烟；
- `outputs/real_trajectory_replay.mp4`：真实轨迹回放；
- `outputs/scripted_expert_reference.mp4`：成功的脚本专家物理参考；
- `outputs/scripted_expert_reference_v1_oscillatory.mp4`：保留的旧版抖动证据，用于控制器前后对比；
- `outputs/gripper_mapping_test.mp4`：夹爪 raw→MuJoCo 方向和边界动画；
- `outputs/policy_closed_loop_act60k/trial_*.mp4`：ACT 60k 十次完整失败试验；
- `outputs/policy_closed_loop_act100k/trial_*.mp4`：ACT 100k 十次完整闭环评测试验；
- `outputs/policy_closed_loop_act100k/closest_attempt.mp4`：100k 中物体抬升最高的第 4 次试验，不代表成功；
- `outputs/contact_sheet.png`：四类成果接触表。

## 重要真实性说明

实际数据资产不是需求模板预期的“240 episodes、front+wrist 双相机”，而是 160 episodes、23,169 frames、单相机 `observation.images.scene`、640×480、30 FPS，任务文本是 `grasp the green block`。场景和 Policy 输入因此按实际 checkpoint schema构建。Wrist 画面用于仿真可视化，但 ACT 当前真正接收的是 Front/scene 一路图像。

真实数据只记录了机器人 state/action 和单相机视频，没有物体 6D pose。因此轨迹回放只声称机器人关节轨迹来自真实数据；虚拟方块只按 MuJoCo 接触、摩擦和重力运动。

ACT 100k checkpoint 确实参与过闭环，但旧 0/10 数字不能继续当成模型准确率：旧配置的 Policy 相机与训练画面严重不一致，一轴长期饱和，Menagerie 通用相机支架与肩部自碰撞，且原成功高度高于真实成功轨迹能够达到的高度。校准后，真实成功 Episode 0 已能在相同 MuJoCo 场景抬升 3.47 cm，并连续 51 帧通过判定。

旧 100k 权重本身也未通过新增的真实帧回归门禁：16 个均匀抽样训练帧的动作 MAE 为 21.40°，而门限为 8°；100 步输出接近常数，属于 VAE 部署分支动作均值塌缩。旧权重不再进入正式成功率统计。完整纠错证据见 [纠错审计报告](reports/CORRECTION_AUDIT_20260812.md)。

## 从空环境复现

不需要 sudo，环境和依赖都位于本目录：

```bash
cd so101-vla-hitl-pipeline/sim_mujoco
./scripts/setup.sh
./scripts/run_tests.sh
./scripts/run_smoke.sh
./scripts/run_replay.sh
./scripts/run_expert_reference.sh
./scripts/run_closed_loop.sh --device cpu --trials 10 --seconds 30
./scripts/render_showcase.sh
```

指定 checkpoint：

```bash
./scripts/run_closed_loop.sh \
  --checkpoint /absolute/path/to/pretrained_model \
  --device cuda --trials 10 --seconds 30
```

服务器没有浏览器也不影响结果。MP4 是普通文件，可用 VS Code Remote 下载，或在本地电脑运行：

```bash
scp USER@SERVER:/path/to/so101-vla-hitl-pipeline/sim_mujoco/outputs/so101_mujoco_showcase.mp4 .
```

## 自动等待最终模型

`so101-mujoco-final-after-training.service` 正在后台执行：

1. 等待当前四卡正式训练结束；
2. 若 Diffusion 或 SmolVLA 因共享 GPU 波动未完成，使用四卡 FSDP 等待器自动补跑；
3. 确认 ACT、Diffusion、SmolVLA 都存在 100k checkpoint；
4. 等待一张卡至少 5GB 空闲；
5. 对最终 SmolVLA 运行 10×30秒真闭环 MuJoCo 试验；
6. GPU 失败时保存证据，并回退 CPU；
7. 自动刷新视频清单、contact sheet 和 showcase。

查看状态：

```bash
make status
tail -f runs/final_after_training/logs/stdout.log
```

## 映射与成功判定

数据和官方 MJCF 使用相同六维关节名称，真实值为度，MuJoCo 为弧度。仿真 offset 通过“真实 Episode 0 首帧 → 官方 `scene_box.xml` pickup keyframe”逐轴对齐，并明确记录在 `configs/joint_mapping.yaml`；它是可审计的仿真校准假设，不冒充真实电机 calibration。

Policy 成功必须满足物体被抬高并保持，不能只看机械臂动作。Place 场景还要求物体进入目标区域、高度合理且速度稳定。Scripted Expert 与 Policy 使用同一个环境和成功检测器，但其成功不计入模型成功率。

当前专家控制器采用平滑笛卡尔轨迹、阻尼最小二乘逆解和零空间姿态保持。早期版本只约束三维位置却控制五个手臂关节，曾在等价逆解间跳变；当前主视频已经修复，旧版单独保留以便审计。

## 项目结构

```text
configs/       场景、关节和动作映射
env/           Python/Conda/pip 环境清单
manifests/     source commit、下载和产物 SHA256
reports/       系统、安装、资产和结果说明
src/           MuJoCo 环境、PolicyAdapter、渲染器
scripts/       安装、冒烟、回放、闭环、专家、收尾入口
tests/         真实性、映射、渲染、成功判定和视频测试
runs/          后台任务日志和状态
outputs/       MP4、PNG、JSONL、metrics 和 manifest
problem_records/ 已知问题、根因、量化修复和防回归知识库
vendor/        固定 commit 的官方 Menagerie 稀疏检出
```

遇到抽搐、碰撞异常、黑屏、动作映射或 Policy 闭环问题时，先查看 [problem_records/README.md](problem_records/README.md)。
