# 抓取—保持—搬运—放入大盒任务报告

更新时间：2026-08-13

## 结论

六阶段 MuJoCo 任务骨架已打通：方块从桌面静止状态开始，机械臂接近、夹取、抬升，通过严格 3 秒稳定保持门后，搬运到固定大盒上方，松爪放入，最后退出且不扰动盒内方块。

脚本专家在固定大盒、方块起点±1 mm 扰动下完成 10 次，全流程 10/10。这是任务可行性和数据生成器验证，不是 ACT、Diffusion 或 SmolVLA 的策略成功率。

为什么这次会达到 10/10、具体控制优化、数学原理和统计边界，见 [为什么六阶段脚本专家 10 次全成功](../problem_records/TASK-EVAL-003_为什么六阶段脚本专家10次全成功_从小白到原理.md)。

## 六阶段与中间质量门

| 阶段 | 通过条件 | 10 次结果 |
|---|---|---:|
| 接近 approach | 夹爪参考点到方块的最小距离 ≤ 30 mm | 10/10 |
| 夹取 grasp | 固定指和运动指同时接触方块，连续 ≥ 3 帧 | 10/10 |
| 抬升 lift | 阶段结束时相对桌面初始高度抬升 ≥ 20 mm | 10/10 |
| 3 秒稳定保持门 | 90 个连续控制帧均保持抬升 ≥ 20 mm | 10/10 |
| 搬运 transport | 全程抬升 ≥ 15 mm，结束 XY 进入盒上方容差 | 10/10 |
| 放置 place | 方块完全位于盒内、高度合理、速度 ≤ 0.025 m/s | 10/10 |
| 退出 retreat | 退出期间方块始终在盒内，最终夹爪距方块 ≥ 120 mm | 10/10 |

补充量化证据：

- 3 秒保持期间，10 次的最低抬升高度为 77.17–79.29 mm；
- 搬运期间最差试验的最低抬升高度为 77.15 mm；
- 搬运结束的最大 XY 误差为 15.12 mm，门限为 45 mm；
- 退出后最小夹爪—方块距离为 228.80 mm，门限为 120 mm；
- 10/10 的 Wilson 95% 置信区间约为 72.2%–100%，样本仍小。

## 物理场景

新场景 `scene_pick_place_box.xml` 使用程序化 MuJoCo box geom 构建，不需要额外模型或网格下载。盒子具有：

- 140×140 mm 内腔；
- 独立底板和四面碰撞墙；
- 可视蓝色材质；
- 真实接触、摩擦、重力和落体沉降。

抓取后没有 weld、equality constraint 或物体传送。方块仅在 reset 时设置初始位置，后续全部由 MuJoCo 物理推进。

## 数据集

本地已构建：

`data/lerobot/local/so101_green_block_pick_hold_place_box_scripted_v1`

| 属性 | 值 |
|---|---:|
| episodes | 10 |
| frames | 8,140 |
| 时长 | 271.33 秒 |
| FPS | 30 |
| 视觉 | 单路 640×480 AV1/YUV420p |
| state/action | 各6维 |
| 任务文本 | `pick up the green block, hold it for 3 seconds, and place it in the large blue box` |
| 磁盘占用 | 约53 MB |
| 审计问题 | 0 |

`pick_place_box_manifest.json` 为每条 episode 保存初始位置、10 个子阶段边界、3 秒保持证据和最终物体位姿。数据集审计证据为 `reports/PICK_HOLD_PLACE_BOX_DATASET_AUDIT.json`。

## 动画和结果证据

- 完整动画：`outputs/PICK_HOLD_PLACE_BOX_SUCCESS.mp4`；
- 视频规格：H.264、1280×720、30 FPS、844 帧、28.13 秒；
- 当前文件 SHA256：`ddc5f7b612de713404622cef93fd96467c74ef81760675e8759b45db212e2813`；
- 10 次机器可读评估：`reports/PICK_HOLD_PLACE_BOX_EVAL.json`。

## 复现

```bash
cd so101-vla-hitl-pipeline/sim_mujoco
./scripts/run_pick_place_box.sh
```

仅跑评估、不编码动画：

```bash
./scripts/run_pick_place_box.sh --trials 10 --no-video
```

数据集已存在时构建器会拒绝覆盖。需要新版本时应使用新目录名：

```bash
./scripts/build_pick_place_box_dataset.sh \
  --episodes 10 \
  --output ../data/lerobot/local/so101_green_block_pick_hold_place_box_scripted_v2
```

## 下一步

当前 10 条数据仅在约 2×2 mm 起点区域内，适合验证训练链路，不足以支持“随机桌面位置放盒”的泛化声明。推荐的后续顺序是：

1. 将方块起点扩展到多个 XY 网格，每个网格只保留物理全流程成功轨迹；
2. 使用阶段平衡采样训练 ACT，先验证单模型能否通过 3 秒保持门；
3. 在未参与生成和调参的新位置上分阶段评估，不用脚本专家 100% 代替策略成绩；
4. 仿真盲测稳定后，再以低速、低高、有急停和人员监护的方式进入真机。
