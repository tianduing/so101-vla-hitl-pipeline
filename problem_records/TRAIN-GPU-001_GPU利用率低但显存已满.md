# TRAIN-GPU-001：GPU 利用率看似很低，但显存其实已经满了

## 基本信息

- 日期：2026-08-12
- 分类：`TRAIN`，共享 GPU 与显存判断
- 状态：已定位，自动等待恢复
- 影响：ACT 100k 已完成；Diffusion 启动后因共享用户占满显存而 OOM

## 小白版解释

GPU 面板中最容易混淆的是两个数字：

- **GPU 利用率**：这一瞬间计算单元有多忙；
- **显存使用量**：GPU 上已经放了多少模型和训练数据。

可以把 GPU 想成厨房：

- GPU 利用率相当于“炉子这一秒有没有开火”；
- 显存相当于“厨房台面已经摆了多少锅和食材”。

炉子暂时没开火，不代表台面是空的。一个训练程序可能正在等数据、做通信或保存 checkpoint，此时利用率会瞬间变成0%，但它的模型仍然占着20GB显存。

所以判断能不能再启动任务，优先看 `memory.free`，不能只看 `utilization.gpu`。

## 本次实际证据

2026-08-12 18:14 CST：

| GPU | 总显存 | 已用 | 空闲 | 瞬时利用率 |
|---:|---:|---:|---:|---:|
| 0 | 24,564 MiB | 23,657 MiB | 425 MiB | 0% |
| 1 | 24,564 MiB | 23,551 MiB | 531 MiB | 100% |
| 2 | 24,564 MiB | 23,657 MiB | 425 MiB | 6% |
| 3 | 24,564 MiB | 23,666 MiB | 413 MiB | 0% |

其中另一用户 `sjy` 的四个训练进程每卡约占 `21,416 MiB`；本项目刚启动的 Diffusion 每卡约占 `2,518–2,632 MiB`。两者相加后，四张卡都只剩约0.4–0.5GB。

因此 GPU 0/3 即使瞬时利用率显示0%，也没有空间继续分配几十 MiB，Diffusion 随即出现 CUDA OOM。

## 当前处理

- 没有终止或抢占其他用户进程；
- ACT 100k checkpoint 已经完整保存，不受影响；
- Diffusion 失败日志被保留；
- `so101-mujoco-final-after-training.service` 会检查缺少的完整 checkpoint；
- 四卡各恢复至少7GB空闲后，`run_distributed_when_ready.sh` 才会重新启动 Diffusion；
- Diffusion 完成后再执行 SmolVLA，最后运行10次 MuJoCo 闭环。

## 正确检查命令

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

判断规则：

1. 利用率低但空闲显存也低：不能认为 GPU 空闲；
2. 空闲显存足够但利用率高：可以装下模型，但会争抢算力、训练变慢；
3. 空闲显存和利用率都满足阈值：才适合启动同步四卡训练；
4. FSDP 是同步训练，任何一张卡 OOM，整组训练都会失败。
