# DeepLearning-Journey(学习笔记)
> **硬件环境**：RTX 4060 (8GB) | **驱动上限**：CUDA 12.7
> **要求**：规范化、容器化(不同类型的项目建对应的实验环境)

---

## 快速导航
1. [环境配置标准 SOP (必看)](#环境配置标准-sop)
2. [学习路线与资源](#学习路线与资源)
3. [学习日记 (Progress Log)](#学习日记-progress-log)

---

## 环境配置标准 SOP 
*记录于 2026-04-06，这是我的“实验环境建造方法论”。*
### 注
- D2L的含义就是Deep to learning，而D2L是《动手学深度学习》的作者制作的打包库,在对于本书的学习中,使用这个库是合适的。而在以后的工作当中，需要按需调整。
- 对于深度学习初学来说，Anaconda3以及Pycharm免费版够用
- 既然环境在E盘，代码也放E盘
### 1. 核心逻辑：容器化思维
- **不要** 在系统全局安装包。
- **不要** 迷信旧版书本的版本号，必须匹配 RTX 4060 硬件。
- **必须** 使用独立文件夹隔离（推荐 E 盘）。

### 2. 施工步骤 (Windows + Conda)
1. **查地基**：`nvidia-smi` 确认 CUDA 支持上限。
2. **建房间** (E 盘强行隔离)：
   ```bash 
   conda create --prefix E:\cv_env python=3.9 -y
   ```
   入驻房间：
   ```bash
   conda activate E:\cv_env
   ```
   安装引擎 (PyTorch 2.5 + CUDA 12.1)：
   本地下载大件：`torch-2.5.1+cu121-cp39-cp39-win_amd64.whl`

   补全插件：
   ```bash
   pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   配套工具箱：
   ```bash
   pip install d2l
   ```
3. **验证指令**
   ```python
   import torch
   print(torch.cuda.is_available()) # 必须返回 True
   ```
---
## 学习路线与资源
- [ ] Phase 1: 预备知识 (张量、线性代数、自动微分)
- [ ] Phase 2: 线性神经网络 (线性回归、Softmax)
- [ ] Phase 3: 多层感知机 
- [ ] Phase 4: 深度学习计算
- [ ] Phase 5: 卷积神经网络 (CNN) 
- [ ] Phase 6: 伪装场景理解 (文献阅读 + U-Net 复现)
---
## 学习日记 (Progress Log)

**Day 1:**
-  搭建 GitHub 个人主页。
- 配置笔记环境。
- 了解md文件，并学习其语法与更新个人主页方法
- **笔记**  
  >1.markdown需要注意缩进格式,其决定了文案的隶属关系。
  >
  >2.可以通过阅读Github网站的md文件学习语法。

**Day 2:**
-  理清了 CUDA 驱动与 PyTorch 版本、系统级与环境级 CUDA 的区别。
-  在 E 盘 成功搭建 cv_env 隔离环境。
-  RTX 4060 成功接入，`torch.cuda.is_available()` 返回 True。
- 建立了规范化操作体系
- **笔记**   
  >1.Pycharm里面每个工作空间(文件夹)都应该在创立初期设置对应的环境(Pycharm右下角标识)，这是应该注意到的

**Day 3:**
- **笔记** 
  >从数学的角度看，有监督学习算法就是一个寻找映射函数 $f$ 的过程。假设有输入数据 $X$（特征）和对应的标签 $Y$（答案）。有监督学习算法的目标就是利用训练数据，找到一个函数：$$Y = f(X)$$一旦这个 $f$ 被找出来了（也就是模型训练好了），当你给它一个从未见过的 $X_{new}$，它就能通过这个函数计算出预测值 $\hat{Y}$。