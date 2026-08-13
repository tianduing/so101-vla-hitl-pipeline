# 仿真结果

## 已完成

- `SCENE_SMOKE`：PASS。官方 `scene_box.xml`，4秒真实物理 + 2秒 provenance card；帧方差最小 1385.87、平均相邻帧差 0.42；
- `REAL_TRAJECTORY_REPLAY`：Episode 0，163帧。真实机器人 state 驱动 MuJoCo，真实相机只作为标注对照；
- `SCRIPTED_EXPERT`：PASS。新版使用平滑笛卡尔轨迹、阻尼最小二乘 IK 和零空间姿态保持，840 control steps；依靠接触、摩擦、重力完成抓取放置，不属于 Policy。旧版位置约束不足导致的抖动视频保留为 `scripted_expert_reference_v1_oscillatory.mp4`；
- `ACT 60k POLICY_CLOSED_LOOP`：0/10。每个 trial 900 steps、900 `select_action`、9次网络 action-chunk 重规划，全部 timeout；
- `ACT 100k POLICY_CLOSED_LOOP（旧配置，已作废）`：曾得到 0/10，但后续审计发现相机、关节映射、自碰撞和成功阈值均不满足真实轨迹回放基准，因此不得作为模型准确率；
- `REAL TRAJECTORY CALIBRATION GATE`：PASS。真实成功 Episode 0 在纠正版 MuJoCo 中抬升 3.47 cm，连续 51 帧满足成功判据，且动作饱和率为 0；
- `ACT 100k REAL-DATA GATE`：FAILED。16 个真实训练帧动作 MAE 21.40°，高于 8°门限；该 checkpoint 不进入正式仿真成功率；
- ACT CUDA 路径：额外完成 1×5秒闭环；此前一次 CUDA OOM 被原样保留，原因是共享 GPU 瞬时只剩约 0.8GB；显存恢复后 CUDA 验证成功执行；
- 测试：10/10 通过；视频均为 1280×720、30FPS、H.264 yuv420p。

## 如何理解 0/10

这不是代码冒烟失败。日志证明每一步 observation 都发生变化，ACT checkpoint 生成的 action 也发生变化，并且动作经过限速/裁剪后真实推进 MuJoCo。失败表示这个由真实单相机数据训练的中间 checkpoint，在当前仿真相机、纹理和物理条件下没有抓起方块。

Scripted Expert 能通过同一成功判定，证明场景的接触、抓取和判定链条可行，但它读取 ground-truth 位姿，不能代替 Policy。

原 60k/100k 高度对比是在错误标定场景中产生的，只保留为历史证据，不再用于模型优劣结论。纠正版先要求真实成功轨迹通过物理门禁，再要求 checkpoint 通过真实帧动作回归门禁，最后才运行 10 次 Policy 闭环。详见 [CORRECTION_AUDIT_20260812.md](CORRECTION_AUDIT_20260812.md)。

旧控制器只约束夹爪三维位置，却让五个手臂关节参与逆解，存在两个未约束自由度；快速追赶远端目标时会在等价姿态之间跳变。新版增加平滑目标插值、阻尼伪逆、零空间姿态约束和释放后的固定位置保持，横移阶段最大关节变化从约 3.13°/帧降至 0.19°/帧。

## 尚在自动执行

当前正式四卡训练仍在运行。`so101-mujoco-final-after-training.service` 将补齐三种 100k checkpoint，并对最终 SmolVLA 跑 10 次闭环。最终成功或失败都会保存，不会只挑成功视频。
