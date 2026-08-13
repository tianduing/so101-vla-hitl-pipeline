# 安装报告

- 独立环境：`sim_mujoco/.sim_env`，基于既有 Python 3.12 环境的 `venv --system-site-packages`，没有修改原始 checkpoint 和数据；
- MuJoCo：3.3.7；官方模型要求 ≥3.1.3；
- Headless 依赖：PyOpenGL 3.1.10、glfw 2.10.0；
- 声明依赖：absl-py 2.3.1、etils 1.13.0；
- `pip check`：No broken requirements found；
- 服务器 `/opt/miniconda3` 是 Conda 4.5/Python 3.7，与当前用户 urllib3/OpenSSL 冲突；未修改该系统安装，`env/conda_explicit.txt` 从既有环境 `conda-meta` 的真实 package URL 重建；
- Menagerie：官方仓库稀疏检出 `robotstudio_so101`，commit `da76818e269b82289eba39808e2fb91d679d6994`；
- Menagerie SO-101 许可证：Apache-2.0；vendor 文件未修改；
- 总仿真工程约 140MB，远低于 30GB 上限。

首次 ACT 加载额外下载官方 torchvision ResNet-18 权重 `resnet18-f37072fd.pth`（44.7MB），用于构造 ACT 视觉骨干；训练 checkpoint 随后覆盖训练参数。缓存位于当前用户 PyTorch cache，不需要 sudo。
