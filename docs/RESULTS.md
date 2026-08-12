# 结果与口径

## 可以复现的结果

本机在 2026-08-12 完成以下验证：

- 环境安装和 CLI 校验通过；CUDA 可见 4 张 RTX 4090。
- 6 个公开真实 SO-101 数据集成功解包和合并，合计 160 episodes、23,169 frames、772.3 秒；单 scene 相机、6 维 state、6 维 action。
- 数据审计没有发现缺帧、非单调时间戳、错误 frame index、非有限 state/action 或不可读视频。
- ACT、Diffusion Policy、SmolVLA 均完成 1-step CPU smoke 训练并保存 checkpoint；三个 checkpoint 可在离线模式下重载，对真实数据帧输出有限 6 维动作。
- 公开发布的 50 次系统真机评测包含 31 成功、19 失败，即 62%。
- 由这 50 次 rollout 构建的轨迹风险模型，在固定 seed 的 10-episode 小型留出集上得到 85.7% 帧级准确率、100% episode 级准确率。
- HITL guard 的 5 个自动化测试通过。

原始机器可读记录：

- `outputs/reports/pipeline_status.json`
- `outputs/reports/merged_dataset_audit.json`
- `outputs/reports/public_eval_summary.json`
- `outputs/reports/failure_analysis.json`
- `outputs/reports/checkpoint_inference.json`
- `outputs/reports/risk_dataset_audit.json`

## 62% 到底代表什么

62% 是上游数据包中一组真实机器人 rollout 的任务成功率，不是分类准确率，也不是三个 smoke 模型的评测结果。Smoke training 只验证“数据能读、梯度能走、checkpoint 能保存并重新推理”，不能用于宣称策略性能。

失败并非均匀分布：`unseen_low_y` 和 `unseen_left_extreme` 两个区域均为 0/5；去掉这两个明确的分布外区域后，其余位置为 31/40（77.5%）。目标 `y < 45 mm` 的样本为 8/20（40%）。与之对应，80 条黄色干扰物训练 episode 的 placement 元数据中，`y < 22 mm` 只有 2 条、`x < -25 mm` 只有 4 条。因此最强证据指向空间覆盖不足和分布外泛化失败。

## 不能宣称的结果

- 项目技术文档中的 240-episode 双相机私有数据未提供，81.7% / 88.3% 不视为本机复现。
- 当前服务器没有连接 leader、follower 或相机，无法新增真机示教、闭环 rollout 或两轮现场 HITL 指标。
- 风险模型只有 50 个 episode，且标签来自相同任务；100% episode 留出准确率样本过小，不应解读为跨场景泛化。
- 正式 100k-step 训练可能因共享 GPU 忙碌而排队；只有实际 checkpoint、训练日志和评测记录能证明完成。

## 验收原则

后续每次模型对比固定相同的 50 个 placement、相同相机标定、相同成功判定和超时；至少报告总成功率、各空间桶成功率、接管率、安全违规次数和 95% 置信区间。不能用训练 loss 或离线单步动作代替真实闭环成功率。
