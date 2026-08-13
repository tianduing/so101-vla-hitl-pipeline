# 资产选择与差异审计

## Dataset

选中：`../data/lerobot/local/so101_green_block_grasp_train_all_prompt_v2`

- LeRobot v3.0，160 episodes，23,169 frames，30 FPS；
- 一路视频：`observation.images.scene`，480×640×3，H.264；
- 6D state/action，同名关节：shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper；
- 任务：`grasp the green block` 和 `grasp the green block, ignore the yellow block`；
- 大小约 302MB；Episode 0 数据完整，选作回放；
- 与模板期望的 240 episodes、front+wrist 双相机不一致，以实际资产为准，没有伪造缺失视角。

## Checkpoint

当前已评测：ACT full 60k：

`../outputs/train/20260812_155201_act_full_fsdp4/checkpoints/060000/pretrained_model`

- 类型 ACT；输入单相机 scene 3×480×640 + 6D state；输出 6D action；
- chunk_size=100，n_action_steps=100；
- 权重文件 SHA256：`26923738d7e5824f050f894e00126fa2ce3b75c99bbaee577d25de2534ada199`；
- PolicyAdapter 组合 manifest hash：`a0e18517a0fb4dda6ab8771bcc3e25a2b8fe78c76e02c10f10a5c0dada92f039`；
- 完成 CPU 10-trial 和 CUDA 1-trial 真闭环调用。

SmolVLA 2-step checkpoint 仅用于早前工程冒烟，不作为最终结果。后台任务只会在 SmolVLA 100k checkpoint 完整存在后生成最终 SmolVLA 结果。

## Source

- LeRobot commit：`30da8e687a6dfc617fcd94afc367ac7071c376ce`；
- Menagerie commit：`da76818e269b82289eba39808e2fb91d679d6994`；
- `robotstudio_so101/README.md` 明确要求 MuJoCo ≥3.1.3；当前为 3.3.7。
