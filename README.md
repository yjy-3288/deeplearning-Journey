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
- 不要系统全局安装包。
- 必须匹配 RTX 4060 硬件。
- 使用独立文件夹隔离（推荐 E 盘）。

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

---
markdown需要注意缩进格式,其决定了文案的隶属关系。

可以通过阅读Github网站的md文件学习语法。

'---'加一行在笔记文案上方交即可

'>'后面的内容,不适合放tex代码(容易遇到转义及行中公式渲染错误等问题)



**Day 2:**
-  理清了 CUDA 驱动与 PyTorch 版本、系统级与环境级 CUDA 的区别。
-  在 E 盘 成功搭建 cv_env 隔离环境。
-  RTX 4060 成功接入，`torch.cuda.is_available()` 返回 True。
- 建立了规范化操作体系
 
--- 
Pycharm里面每个工作空间(文件夹)都应该在创立初期设置对应的环境(Pycharm右下角标识)，这是应该注意到的

**Day 3:**

---
从数学的角度看，有监督的学习算法就是一个寻找映射函数 $f$ 的过程(监督学习是一个大框架,可以采取多种形式的模型)。假设有输入数据 $X$（特征）和对应的标签 $Y$（答案）。有监督学习算法的目标就是利用训练数据，找到一个函数：

$$Y=f(X)$$

一旦这个 $f$ 被找出来了（也就是模型训练好了），当你给它一个从未见过的 $X_{new}$，它就能通过这个函数计算出预测值 $\hat{Y}$。

需要注意:面积 100 平米，3 个卧室，2 个卫生间，距离地铁 500 米,数值提取出来，得到特征向量(feature vector)$`\mathbf{x}_A = [100, 3, 2, 500]`$。但这和代数里的特征向量(Eigenvector)是不一样的。

**Day 4:**

---
原地更新内存省去大量“销毁旧内存、分配新内存”的时间和空间开销。

[:]在左边表示对内存的原地复写。简单来说,Z=X+Y(普通赋值),相当于买了一个新抽屉，东西放进去，把就抽屉的标签撕下来贴到新抽屉上去；Z[:]=X+Y,相当于把东西塞到旧抽屉里，原先的内容覆盖掉

不写类，全程序的计时都得用同一个“黑板”（一个全局变量），测完这段代码去测另一段，时间就乱套了。

写了类 class Timer，就等于有一整箱的“秒表”。要测 A 代码，就拿个新秒表（t1 = Timer()）；要测 B 代码，再拿个新秒表（t2 = Timer()）。它们各自记录各自的时间

$$\text{似然 (概率)} \propto \exp(- \text{误差}^2)$$

(最小化目标函数和执行极大似然估计等价)

**Day 5:**
- 学习了深度网络如何实现优化
- 可使用Pytorch的高级API更加简洁地实现模型

---
求一个关于列向量的partial，实际上就是被导数对列向量中的每个元素求偏导,再排成一列.对于矩阵也是一样的
  
numpy可以做向量运算，但偏导公式得手动推导并写代码;Pytorch不仅能做向量运算，还通过“计算图”技术，把复杂的向量偏导变成了“自动执行”的后台任务

**为什么不用全梯度下降？**

1000 个一起上（全批量）：它是极其稳健的。每一次更新的方向都绝对指向当前最完美的“下坡”方向。
缺陷：如果山坡上有一个小坑（局部最小值），全批量由于太听话，会直接滑进坑里死掉，再也出不来了。

10 个一组（小批量）：每一组 10 道题给出的建议其实是带点偏见的。这组 10 道题可能让你往左偏，下一组 10 道题可能让你往右偏。

优势：这种“摇摇晃晃”的下山路径自带抖动。这种抖动往往能让模型在滑进小坑时，被随机的噪声“踢”出来，从而有机会找到真正的山谷（全局最小值）。

在训练的数据当中,那个"label"是加过噪声的,而这个噪声服从正态分布(现实中的噪声往往是由无数个微小、独立且随机的干扰因素叠加而成的，而根据中心极限定理，这种“大规模混乱的累加”在宏观上必然会趋向于完美的钟形正态分布)

**Day 6:**

---
**用集合映射的语言来精确描述“softmax回归的输出层是一个全连接层”**

从 $\mathbb{R}^4$ 到 $\mathbb{R}^3$ 的仿射映射：$`f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}`$

**是满射**

只要这个矩阵的行向量是线性无关的（即行满秩，这在随机初始化的神经网络中发生概率是 100%），这个仿射变换就能覆盖整个 3 维输出空间 $\mathbb{R}^3$
  
**proof:不是单射**

lemma:

已知

$$\dim(\text{输入空间}) = \text{rank}(\mathbf{W}) + \dim(\ker(\mathbf{W}))$$

又因$`\text{rank}(\mathbf{W}) \leq 3`$,
故$`\dim(\ker(\mathbf{W})) \geq 1`$

现证明: 如果 $`\mathbf{x}_1 \neq \mathbf{x}_2`$，则必定 $`f(\mathbf{x}_1) \neq f\mathbf{x}_2)`$

假设有一个输入向量 $\mathbf{x}$，它映射到了输出 $\mathbf{y}$：

$$\mathbf{W}\mathbf{x}+\mathbf{b}=\mathbf{y}$$

因为我们刚才证明了 $\dim(\ker(\mathbf{W})) \geq 1$，所以一定存在一个非零向量 $`\mathbf{v}`$ 属于这个核（即 $`\mathbf{W}\mathbf{v} = \mathbf{0}`$ 且 $`\mathbf{v} \neq \mathbf{0}`$）。现在，我们构造一个新的输入点：$`\mathbf{x}_{new} = \mathbf{x} + \mathbf{v}`$。显然，$`\mathbf{x}_{new} \neq \mathbf{x}`$。但如果我们把 $`\mathbf{x}_{new}`$ 喂给全连接层：

$$\begin{aligned} f(\mathbf{x}_{new}) &= \mathbf{W}(\mathbf{x} + \mathbf{v}) + \mathbf{b} \\ 
&= \mathbf{W}\mathbf{x} + \mathbf{W}\mathbf{v} + \mathbf{b} \\
&= \mathbf{W}\mathbf{x} + \mathbf{0} + \mathbf{b} \\ 
&= \mathbf{y} \end{aligned}$$

证毕！ 

---
小数连乘会导致数值下溢出,故考虑对数似然,变成连加。极大似然估计的目标就是找到一个参数$`W^*`$使得对数似然函数达到最大,等价于最小化负对数似然(交叉熵损失),而方程没有解析解,因此需要求导算梯度找最优参数.

---
**为什么交叉熵损失先对$`o_j`$ 求导？（追责机制）**

$$ W \stackrel{\text{计算}}{\longrightarrow} o_j \stackrel{\text{Softmax}}{\longrightarrow} \hat{y}_j \stackrel{\text{交叉熵}}{\longrightarrow} l $$

$$\frac{\partial l}{\partial \mathbf{w}_j} = \frac{\partial l}{\partial o_j} \cdot \frac{\partial o_j}{\partial \mathbf{w}_j} = (\hat{y}_j - y_j) \cdot \mathbf{x}$$

偏导数衡量的是“敏感度”:

如果在反向调查时，发现稍微动一下员工甲的参数，最终的 Loss 就会产生剧烈的波动，说明员工甲就是影响全局的关键节点（根源）。就会被贴上一张数值极大的梯度条子。
如果动一下员工乙的参数，Loss 根本没什么反应，说明员工乙的错误无关紧要，梯度就会很小.

**Day 7:**

---
PyTorch 规定的标准图像张量形状是：[通道数 (Channel), 高度 (Height), 宽度 (Width)]

如果是彩色图（RGB）： 它的形状是 [3, 28, 28]。这就相当于 3 本 $28 \times 28$ 页的账本叠在一起。

如果是灰度图（Fashion-MNIST）： 虽然它本质上只是一个二维矩阵，但为了遵守规矩，PyTorch 会强制给它套上一个厚度(通道数)为 1 的“外壳”，把它变成 [1, 28, 28]。

---

```trans = [transforms.ToTensor()]```是动词的集合,其第0个元素是"转化成张量"这个动作.

```transforms.Resize(resize)```是用来设定图片的尺寸,如果要"放大",PyTorch 的 Resize 默认使用的是“双线性插值”,将四周的像素加权平均,并填满

```trans=transforms.Copsose(trans)```相当于给trans里的几个动作首尾相连(打包成一个单盒机器)

---
DataLoader是一个具有延迟加载特点的迭代器

---
```
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```
y表示,在这一批次的训练数据中,只抽了两张照片,其正确答案分别是0=狗和2=猫。而y_hat则表示样本1和样本2在三个类别(0=狗,1=猪,2=猫)中分别作预测得到的概率。而每一个训练数据中的样本都应该有其正确答案,y_hat[[0, 1], y]则是找到每个样本在此次训练之下,给"标准答案"作出的预测概率。

交叉熵公式:

$$L = - \sum_{j=1}^q y_j \log(\hat{y}_j)$$

假设有一个 3 分类问题，某张图片的真实标签是类别 2=猫。它的真实概率分布 $y$ 写成独热编码就是：$`y = [0, 0, 1]`$。模型的预测概率是：$`\hat{y} = [0.1, 0.8, 0.1]`$。现在，开始套公式算交叉熵：

$$L = - (y_0 \log\hat{y}_0 + y_1 \log\hat{y}_1 + y_2 \log\hat{y}_2)$$

代入数字：

$$L = - (0 \times \log 0.1 + 0 \times \log 0.8 + 1 \times \log 0.1)$$

所以,只需要一行代码可以实现交叉熵损失函数:
```
def cross_entropy(y_hat, y):
return- torch.log(y_hat[range(len(y_hat)), y])
cross_entropy(y_hat, y)
```
此时输出```tensor([2.306,0.6931])```,也就是根据交叉熵损失函数打的分.

当然,可以一次性抽4张照片,或者更多:```y = torch.tensor([0, 2, 0, 1])```。此时打的分数可以是(打比方)```tensor([2.306,0.6931,0.7891,4.1565])```,它里面的每一个数字，都和最初的图片有着极其严密的物理绑定关系

---
**训练**

定义一个函数来训练一个迭代周期
```
def train_epoch_ch3(net, train_iter, loss, updater): #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            # grad采用累加赋值,
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
```
训练循环实际上是这样的：

- 拿 32 张图 喂进流水线。

- 图片流过 net[0], net[1], net[2]... 最终吐出预测结果。

- 算 Loss。backward()：从后往前跑，给每一层（net[0], net[1]...）的参数口袋里都塞进梯度。

- updater.step()：优化器查看名册，发现名册里有 net[0] 的 $`w,b`$，也有 net[1] 的参数... 于是一把抓，把所有层的参数全都更新了。

- 清空所有层的梯度，迎接下 32 张图。


注: 参数的更新过程如下

$$\mathbf{w}_i^{(\text{new})} = \mathbf{w}_i^{(\text{old})} - \eta \frac{\partial L}{\partial \mathbf{w}_i}$$

$$\mathbf{w}_i^{(\text{new})} = \mathbf{w}_i^{(\text{old})} - \eta (\hat{y}_i - y_i) \mathbf{x}$$


---
**问题:在书中介绍的softmax的net只连接了两层,如果是多层,该如何反向传播(计算梯度,更新参数)？**

假设我们有一个精简版的 3 层神经网络。我们忽略偏置项，用最简单的公式表示它的前向传播（从左到右算答案）：

- 第 1 层： $`A_1 = \sigma(W_1 X)`$   (输入数据 $X$，算出第 1 层输出 $A_1$)
- 第 2 层： $`A_2 = \sigma(W_2 A_1)`$  (输入 $A_1$，算出第 2 层输出 $A_2$)
- 第 3 层： $`\hat{Y} = W_3 A_2`$       (输入 $A_2$，算出最终预测结果 $`\hat{Y}`$)最终误差： $`L = \text{Loss}(\hat{Y}, Y)`$

根据微积分链式法则，既然 $W_2$ 影响了 $A_2$，$A_2$ 影响了 $`\hat{Y}`$，$`\hat{Y}`$ 影响了 $L$，我们就把这根链条从右往左反着乘起来：

$$\frac{\partial L}{\partial W_2} = \underbrace{ \frac{\partial L}{\partial \hat{Y}} \cdot \frac{\partial \hat{Y}}{\partial A_2} }_{\text{来自第3层的误差信号}} \cdot \underbrace{ \frac{\partial A_2}{\partial W_2} }_{\text{第2层本身的局部导数}}$$

第 2 层在算梯度时，不仅需要上面传下来的 $`\delta_3`$，还需要自己前向传播时的输入数据 $A_1$。这意味着，在数据从左到右流动（做题）的时候，神经网络不能算完即扔！它必须把每一层中间产生的临时结果（激活值）全部像宝藏一样保存在显存里，一直等到反向传播时拿出来用。

这就是为什么层数越多（比如 100 层的 ResNet），训练时我们的显存（VRAM）就越容易爆掉的原因——它在为了反向推导做准备。

---
**机器求导，真正在起作用的确实是“计算图的拓扑结构（公式联系）”，而那个最终算出来的具体的 Loss 数值（比如 1.34），对于机器算梯度而言，其实并没有那么重要**

**1. 既然靠公式求导，那个具体的 Loss 数值算出来到底有啥用？**
- 让人类知道模型是否"学崩了"
- 在部分手动推导的时候,需要知道某偏导的系数

**2. 为什么梯度可以累加？**

$$\frac{\partial L_{\text{total}}}{\partial w} = \frac{1}{n} \left( \frac{\partial l_1}{\partial w} + \frac{\partial l_2}{\partial w} + \dots + \frac{\partial l_{n}}{\partial w} \right)$$

