<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#支持向量机" data-toc-modified-id="支持向量机-1">支持向量机</a></span></li><li><span><a href="#1.-线性可分支持向量机" data-toc-modified-id="1.-线性可分支持向量机-2">1. 线性可分支持向量机</a></span><ul class="toc-item"><li><span><a href="#1.1-间隔计算公式推导" data-toc-modified-id="1.1-间隔计算公式推导-2.1">1.1 间隔计算公式推导</a></span></li><li><span><a href="#1.2-硬间隔最大化" data-toc-modified-id="1.2-硬间隔最大化-2.2">1.2 硬间隔最大化</a></span><ul class="toc-item"><li><span><a href="#1.2.1-原始问题" data-toc-modified-id="1.2.1-原始问题-2.2.1">1.2.1 原始问题</a></span></li><li><span><a href="#1.2.2-对偶算法" data-toc-modified-id="1.2.2-对偶算法-2.2.2">1.2.2 对偶算法</a></span></li></ul></li><li><span><a href="#1.3-支持向量" data-toc-modified-id="1.3-支持向量-2.3">1.3 支持向量</a></span></li></ul></li><li><span><a href="#2.-线性支持向量机" data-toc-modified-id="2.-线性支持向量机-3">2. 线性支持向量机</a></span><ul class="toc-item"><li><span><a href="#2.1-软间隔最大化" data-toc-modified-id="2.1-软间隔最大化-3.1">2.1 软间隔最大化</a></span><ul class="toc-item"><li><span><a href="#2.1.1-原始问题" data-toc-modified-id="2.1.1-原始问题-3.1.1">2.1.1 原始问题</a></span></li><li><span><a href="#2.1.2-对偶算法" data-toc-modified-id="2.1.2-对偶算法-3.1.2">2.1.2 对偶算法</a></span></li></ul></li><li><span><a href="#2.2-支持向量" data-toc-modified-id="2.2-支持向量-3.2">2.2 支持向量</a></span></li><li><span><a href="#2.3-合页损失函数" data-toc-modified-id="2.3-合页损失函数-3.3">2.3 合页损失函数</a></span></li></ul></li><li><span><a href="#参考资料" data-toc-modified-id="参考资料-4">参考资料</a></span></li></ul></div>

# 支持向量机

`支持向量机（Support Vector Machines, SVM）`：是一种**二分类模型**，它的基本模型是定义在特征空间上的间隔最大化的线性分类器，**间隔最大化**使它有别于感知器；支持向量机还包括核技巧（通过非线性函数转换为线性模型），这使它成为实质上的非线性分类器。

支持向量机的学习策略就是**间隔最大化**，可以形式化为一个求解**凸二次规划**（convex quadratic programming）的问题，也等价于正则化的**合页损失函数最小化问题**，支持向量机的学习算法是求解凸二次规划的最优算法。<sup>[1]

按训练数据分线性可分程度，支持向量机的分类以及对应采取的算法如下：

1. `线性可分支持向量机`：硬间隔最大化（线性可分）

2. `线性支持向量机`：软间隔最大化（近似线性可分）

3. `非线性支持向量机`：核技巧+软间隔最大化（线性不可分）

线性可分支持向量机可以看作是线性支持向量机的一个特例；非线性支持向量机则通过**核函数**将线性不可分数据转换为线性可分数据，从而转换为线性支持向量机。

间隔最大化：按间隔中是否能出现样本点，分为硬间隔最大化和软间隔最大化。硬间隔最大化中不允许有样本点出现，因此适合于数据线性可分的情况；软间隔最大化中允许有少量样本点出现，适合于含有噪声的线性可分数据（称为近似线性可分）。

本文将按照上述思路介绍的三类支持向量机、核函数。

# 1. 线性可分支持向量机

**线性可分支持向量机**（linear support vector machine in linearly separable case）与**硬间隔最大化**（hard margin maximization）

首先我们来看一个线性可分的例子，给定训练样本$
D=\left\{\left(\boldsymbol{x}_{1}, y_{1}\right),\left(\boldsymbol{x}_{2}, y_{2}\right), \ldots,\left(\boldsymbol{x}_{m}, y_{m}\right)\right\}, y_{i} \in\{-1,+1\}
$，分类学习最基本的想法就是基于训练集 $D$ 在样本空间中找到一个**划分超平面**，将不同类别的样本分开，但能将训练样本分开的划分超平面可能有很多，如下图所示：

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/Screen Shot 2019-08-16 at 11.52.50.png" width="420" >

<center>图1 存在多个划分超平面将两类训练样本分开</center>

直观上，应该去找位于两类训练样本“正中间”的划分超平面，因为该划分超平面对训练样本局部扰动的“容忍”性最好。例如，由于训练集的局限性或噪声的影响，训练集外的样本可能比图 1 中的训练样本更接近两个类的分隔届，这将使许多划分超平面出现错误，而红色的超平面受影响最小。换而言之，这个划分超平面所产生的分类结果鲁棒性是最好的，对预测数据的泛化能力也是最强的。

在样本空间中，**划分超平面**可通过如下线性方程来描述：

$$
\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b=0 \tag{1}
$$

其中 $\boldsymbol{w}=\left(w_{1} ; w_{2} ; \dots ; w_{d}\right)$ 为**法向量**，决定了超平面的方向；$b$ 为**位移项**，决定了超平面与原点之间的距离。因此划分超平面可以唯一的被法向量 $\boldsymbol{w}$ 和位移项 $b$ 确定，我们将这个超平面记为 $(\boldsymbol{w},b)$。

样本空间中任意点 $\boldsymbol{x}$ 到超平面 $(\boldsymbol{w},b)$ 的距离可以写为：

$$
r=\frac{\left|\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b\right|}{\|\boldsymbol{w}\|}\tag{2}
$$

假设超平面 $(\boldsymbol{w},b)$ 能将训练样本正确分类，即对于 $\left(\boldsymbol{x}_{i}, y_{i}\right) \in D$，若 $y_i=+1$，则有 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b>0$；若 $y_i=-1$，则有 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b<0$，如下：

$$
\left\{\begin{array}{ll}{\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b \geqslant+1,} & {y_{i}=+1} \\ {\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b \leqslant-1,} & {y_{i}=-1}\end{array}\right. \tag{3}
$$

如图 2 所示，距离超平面**最近的**这几个训练样本点使公式 (3) 的等号成立，它们被称为“`支持向量`”（support vector），两个异类向量到超平面的距离之和为：

$$
\gamma=\frac{2}{\|\boldsymbol{w}\|}\tag{4}
$$

它被称为`间隔`（margin）。

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/Screen Shot 2019-08-16 at 13.55.30.png" width="420" >
<center> 图2  支持向量与间隔</center>

## 1.1 间隔计算公式推导

首先，设$W=\left(w_{1}, w_{2}\right), x=\left(x_{1}, x_{2}\right)$，并且 $W x=w_{1} x_{1}+w_{2} x_{2}$。

假设我们有三条直线：

$\begin{array}{l}{\bullet W x+b=1} \\ {\bullet W x+b=0} \\ {\bullet W x+b=-1}\end{array}$

其中第二条为超平面的表达式，第一条和第三条为边界的表达式。由于这三条线为等距平行线，因此想要确定间隔（两条边界的距离），只需要计算边界与超平面之间的距离，并乘 2 就得到了间隔。

下面上边界到超平面的距离：

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/Screen Shot 2019-08-16 at 14.02.00.png" width="320" >

<center> 图3  边界与超平面</center>

由于移动直线并不改边之间的距离，因此可以将超平面移动到与原点相交，此时直线方程变为：

$\begin{array}{l}{\bullet W x=1} \\ {\bullet W x=0}\end{array}$

现在，超平面的直线表达式为$W x=0$，因此其法向量为$W=\left(w_{1}, w_{2}\right)$。

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/Screen Shot 2019-08-16 at 14.09.35.png" width="320" >

<center> 图4  移动后的边界与超平面以及法向量</center>

法向量 $W$ 与边界相交于蓝点，假设该点的坐标为 $(p, q)$，由于该点位于法向量 $W=\left(w_{1}, w_{2}\right)$ 上，所以 $(p,q)$ 是 $(w_1,w_2)$ 的倍数，即：

$\bullet (p,q)= k(w_1,w_2)$

将点 $(p,q)$ 代入边界方程 $Wx=1$ 有：

$$
\begin{aligned} 
&\quad\quad W x \\
&  \Rightarrow (w_1,w_2) (x_1,x_2)^T\\
& \Rightarrow (w_1,w_2) (p,q)^T\\
& \Rightarrow (w_1,w_2) k(w_1,w_2)^T\\
& \Rightarrow k(w_1^2+w_2^2)=1\\
\end{aligned}
$$

因此$k=\frac{1}{w_{1}^{2}+w_{2}^{2}}=\frac{1}{|W|^{2}}$，所以蓝点坐标可化为：

$$
\begin{aligned} 
(p,q)&= k(w_1,w_2)\\
&=kW \\
&=\frac{1}{|W|^{2}}W\\
&=\frac{W}{|W|^{2}}
\end{aligned}
$$

现在，两条直线之间的距离是蓝色向量的范数，由于分母是一个标量，向量 $\frac{W}{|W|^{2}}$ 的范数正是 $\frac{|W|}{|W|^{2}}$，即 $\frac{1}{|W|}$，如下图所示：

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/Screen Shot 2019-08-16 at 14.54.49.png" width="320" >

<center> 图5 边界到分离超平面的距离 </center>

最后，最终的间隔为两倍距离，即：

$$
\gamma=\frac{2}{|\boldsymbol{w}|}
$$

## 1.2 硬间隔最大化

支持向量机学习的基本想法是求解能够正确划分训练数据集并且间隔最大化的分离超平面。对于线性可分的训练数据集而言，线性可分的分离超平面有无数个，但是间隔最大的分离超平面是唯一的。

这里的间隔最大化又称为`硬间隔最大化`，间隔最大化的直观解释是：对训练数据集找到间隔最大的超平面以为着以充分大的确信度对训练数据进行分类，也就是说，不仅将正负实例点分开，而且最最难分的实例点（离超平面最近的点）也有足够大的确信度将它们分开。这样的超平面应该对未知的新实例有很好的分类预测能力。

求解硬间隔最大化下的超平面可以分为一下几步：

1. 将**间隔最大化**问题化为其**倒数的最小化**问题（为了应用拉格朗日对偶性求解最小化规划问题的解），我们将这个最小化问题称为**原始问题**，相应的算法1.1称为**最大间隔法**；

2. 利用**拉格朗日对偶性**将原始问题转换为拉格朗日函数，分两步求解得到算法1.1的最优解，我们将这个算法称为**对偶算法**；最后利用最优解代入公式可以得到**分离超平面**以及**分离决策函数**的方程式。

下面我将详细的介绍这两个步骤。

### 1.2.1 原始问题

由1.1可知，间隔的计算公式为$\gamma=\frac{2}{\|\boldsymbol{w}\|}$，因此最大化间隔$\frac{2}{\|\boldsymbol{w}\|}$ 相当于最小化 $\frac{1}{2}\|w\|^2$，于是就得到下面的线性可分支持向量学习的最优化问题：

***

**算法1.1 线性可分支持向量机学习方法——最大间隔法**

输入：线性可分训练数据集$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$，其中，$x_i \in X=R^n$，$y_{i} \in \mathcal{Y}=\{-1,+1\}, \quad i=1,2, \cdots, N$；

输出：最大间隔分离超平面和分类决策函数。
    
（1） 构造并求解月时最优化问题：
    
$$
\begin{array}{ll}{\min \limits_{w, b}} & {\frac{1}{2}\|w\|^{2}} \tag{5}\\ {\text { s.t. }} & {y_{i}\left(w \cdot x_{i}+b\right)-1 \geqslant 0, \quad i=1,2, \cdots, N}\end{array}
$$

求得最优解$w^{*}, b^{*}$。

（2）由此得到分离超平面：

$$
w^{*} \cdot x+b^{*}=0\tag{6}
$$

分离决策函数：

$$
f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)\tag{7}
$$
***

**例 1.1**

已知一个如图 6 所示的训练数据集，其正例点是 $x_1=(3,3)^T, x_2=(4,3)^T$，负例点是 $x_3=(1,1)^T$ ，试求最大间隔分离超平面。

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/Screen Shot 2019-08-16 at 17.50.16.png" width="420" >

<center> 图6 间隔最大分离超平面实例</center>

按照算法1.1，根据训练数据集构造约束最优化问题：

$$
\begin{array}{cl}
{\min \limits_{x, b}} & {\frac{1}{2}\left(w_{1}^{2}+w_{2}^{2}\right)} \\ 
{\text { s.t. }} & {3 w_{1}+3 w_{2}+b \geqslant 1} \\ 
{} & {4 w_{1}+3 w_{2}+b \geqslant 1} \\
{} & {-w_{1}-w_{2}-b \geqslant 1}
\end{array}
$$

求得此最优化问题的解 $w_1=w_2=\frac{1}{2}, b=-2$，于是最大间隔分离超平面为：

$$\frac{1}{2}x^{(1)}+\frac{1}{2}x^{(2)}-2=0$$

其中，$x_1=(3,3)^T$与$x_2=(1,1)^T$ 为支持向量。
***

### 1.2.2 对偶算法

为了求得算法1.1 的最优化约束条件下的解 $w^{*}, b^{*}$，我们将它作为原始最优化问题，利用**拉格朗日对偶性**，通过求对偶问题（dual problem）得到原始问题（primal problem）的最优解，这就是线性可分支持向量机的**对偶算法**。

首先建立拉格朗日函数（Lagrange function），为此，对不等式约束条件

$$
y_{i}\left(w \cdot x_{i}+b\right)-1 \geqslant 0, \quad i=1,2, \cdots, N\tag{8}
$$

引入拉格朗日乘子（Largrange multiplier）:

$$\alpha_{i} \geqslant 0, i=1,2, \cdots, N\tag{9}$$

定义拉格朗日函数：

$$
L(w, b, \alpha)=\frac{1}{2}\|w\|^{2}-\sum_{i=1}^{N} \alpha_{i} y_{i}\left(w \cdot x_{i}+b\right)+\sum_{i=1}^{N} \alpha_{i}\tag{10}
$$

其中，$\alpha=\left(\alpha_{1}, \alpha_{2}, \cdots, \alpha_{N}\right)^{\mathrm{T}}$ 为拉格朗日乘子向量。

根据**拉格朗日对偶性**，原始问题的对偶问题是极大极小问题：

$$
\max _{\alpha} \min _{w, b} L(w, b, \alpha) \tag{11}
$$

所以为了得到对偶问题的解，需要**先对$L(w, b, \alpha) $ 求 $ w, b$ 的极小，再对 $\alpha$ 求极大。**

(1) 求 $\min \limits_{w, b} L(w, b, \alpha)$

将拉格朗日函数 $L(w, b, \alpha)$ 分别对 $ w, b$ 求偏导数，并令其等于0:

$$
\begin{array}{l}{\nabla_{w} L(w, b, \alpha)=w-\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}=0} \\ {\nabla_{b} L(w, b, \alpha)=\sum_{i=1}^{N} \alpha_{i} y_{i}=0}\end{array}\tag{12}
$$

得：

$$
\begin{array}{l}{w=\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}} \\ {\sum_{i=1}^{N} \alpha_{i} y_{i}=0}\end{array}\tag{13}
$$

将 (13) 代入拉格朗日公式 (10)，得：

$$
\begin{aligned} L(w, b, \alpha) &=\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} y_{i}\left(\left(\sum_{j=1}^{N} \alpha_{j} y_{j} x_{j}\right) \cdot x_{i}+b\right)+\sum_{i=1}^{N} \alpha_{i} \\ &=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i} \end{aligned}\tag{14}
$$

即

$$
\min _{w, b} L(w, b, \alpha)=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}\tag{15}
$$

（2）求 $\min _{w, b} L(w, b, \alpha)$ 对 $\alpha$ 的极大，即使对偶问题：

$$
\quad\quad\quad\quad\quad\max _{\alpha}\bigg[-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}\bigg]\tag{16}
$$

$$
\begin{array}{ll}{\text { s.t. }} & {\sum_{i=1}^{N} \alpha_{i} y_{i}=0} \\ {} & {\alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N}\end{array}\tag{17}
$$


将公式 (16) 的最大化转换为最小化，就得到了下面与之等价的对偶最优化问题：

$$
\quad\quad\quad\quad\min _{\alpha}\bigg[\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}\bigg]\tag{18}
$$

$$
\begin{array}{ll}{\text { s.t. }} & {\sum_{i=1}^{N} \alpha_{i} y_{i}=0} \\ {} & {\alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N}\end{array}\tag{19}
$$

设 $\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^T$ 是对偶最优化问题 (18-19) 的解，则存在下标 $j$，并可按下式求得原始最优化问题 (5) 的解 $w^*,b^*$：

$$w^*=\sum_{i=1}^Na_i^*y_ix_i \tag{20}$$

$$b^*=y_j-\sum_{i=1}^Na_i^*y_i(x_i\cdot x_j)\tag{21}$$

***

**证明** 对于原始问题和对偶问题，KKT（Karush-Kuhu-Tucker）条件成立，即得：

$$
\begin{array}{l}{\nabla_{w} L\left(w^{*}, b^{*}, \alpha^{*}\right)=w^{*}-\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} x_{i}=0} \\ {\nabla_{b} L\left(w^{*}, b^{*}, \alpha^{*}\right)=-\sum_{i=1}^{N} \alpha_{i}^{*} y_{i}=0} \\ {\alpha_{i}^{*}\left(y_{i}\left(w^{*} \cdot x_{i}+b^{*}\right)-1\right)=0, \quad i=1,2, \cdots, N} \\ {y_{i}\left(w^{*} \cdot x_{i}+b^{*}\right)-1 \geqslant 0, \quad i=1,2, \cdots, N} \\ {\alpha_{i}^{*} \geqslant 0, \quad i=1,2, \cdots, N}\end{array}\tag{22}
$$

由此得：

$$w^*=\sum_i \alpha_i^*y_ix_i\tag{23}$$

其中至少有一个 $a_j^*>0$ （反证法，假设 $a^*=0$，由 (22) 第一条公式可知 $w^*=0$，而 $w^*=0$ 不是原始最优化问题 (5) 的解，产生矛盾），对此 $j$ 有：

$$y_j(w^*\cdot x_j+b^*)-1=0\tag{24}$$

将 (20) 代入 (24)，并注意到 $y_j^2=1$，即得：

$$b^*=y_j-\sum_{i=1}^Na_i^*y_i(x_i\cdot x_j)\tag{25}$$

***

将 $w^*, b^*$ 代入公式 (6)、(7)可得分离超平面的计算公式：

$$\sum_{i=1}^Na_i^*y_i(x\cdot x_i)+b^*=0 \tag{26}$$

分类决策函数可以写为：

$$f(x)=sign\bigg[\sum_{i=1}^Na_i^*y_i(x\cdot x_i) \bigg] \tag{27}$$

这就是说，对于给定的线性可分数据集，可以首先求对偶问题 (18-19) 的解 $a^*$；再利用公式 (20-21) 求得原始问题的解 $w^*,b^*$；从而根据公式 (6-7) 得到分离超平面以及分类决策函数。

这种算法分称为**线性可分支持向量机的对偶学习算法**，是线性可分支持向量机学习的基本算法。

我们可以将上述过程总结为算法 1.2。

***

**算法1.2 线性可分支持向量机学习算法**

输入：线性可分训练数据集$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$，其中，$x_i \in X=R^n$，$y_{i} \in \mathcal{Y}=\{-1,+1\}, \quad i=1,2, \cdots, N$；

输出：最大间隔分离超平面和分类决策函数。
    
（1） 构造并求解约束时最优化问题：

$$
\quad\quad\quad\quad\min _{\alpha}\bigg[\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}\bigg]\tag{28}
$$

$$
\begin{array}{ll}{\text { s.t. }} & {\sum_{i=1}^{N} \alpha_{i} y_{i}=0} \\ {} & {\alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N}\end{array}\tag{29}
$$

求得最优解 $\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^T$ 。

（2）计算

$$w^*=\sum_{i=1}^N \alpha_i^*y_ix_i\tag{30}$$

并选择 $\alpha^*$ 的一个正分量 $\alpha_j^*>0$，计算：

$$b^*=y_j-\sum_{i=1}^Na_i^*y_i(x_i\cdot x_j)\tag{31}$$

（3）求得分离超平面：

$$
w^{*} \cdot x+b^{*}=0\tag{32}
$$

分离决策函数：

$$
f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)\tag{33}
$$
***

**例 1.2** 

与例1.1 相同，对一个如图 7 所示的训练数据集，其正例点是 $x_1=(3,3)^T, x_2=(4,3)^T$，负例点是 $x_3=(1,1)^T$ ，试用算法1.2求最大间隔分离超平面。

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/Screen Shot 2019-08-16 at 17.50.16.png" width="420" >

<center> 图7 间隔最大分离超平面实例</center>

根据所给数据，对偶问题是：

$$
\begin{array}{ll}{\min \limits_{\alpha}} & {\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}} \\ {} & {=\frac{1}{2}\left(18 \alpha_{1}^{2}+25 \alpha_{2}^{2}+2 \alpha_{3}^{2}+42 \alpha_{1} \alpha_{2}-12 \alpha_{1} \alpha_{3}-14 \alpha_{2} \alpha_{3}\right)-\alpha_{1}-\alpha_{2}-\alpha_{3}} \\ {\text { s.t. }} & {\alpha_{1}+\alpha_{2}-\alpha_{3}=0} \\ {} & {\alpha_{i} \geqslant 0, \quad i=1,2,3}\end{array}
$$

解着一最优化问题，将 $\alpha_1 = \alpha_1+\alpha_2$ 代入目标函数并记为：

$$s(\alpha_1,\alpha_2)=4\alpha_1^2+\frac{13}{2}\alpha_2^2+10\alpha_1\alpha_2-2\alpha_2-2\alpha_2$$

对 $\alpha_1, \alpha_2$ 求偏导数并令其为 0，易知 $s(\alpha_1,\alpha_2)$ 在点 $(\frac{3}{2},-1)^T$ 取极值，但该点不满足约束条件 $\alpha_2 \geq 0$，所以最小值应在边界上达到。

当 $\alpha_1=0$ 时，最小值 $s(0,\frac{2}{13})=-\frac{2}{13}$；当 $\alpha_2=0$ 时，最小值 $s(\frac{1}{4},0)=-\frac{1}{4}$，于是 $\alpha_1, \alpha_2$ 在 $\alpha_1=\frac{1}{4},\alpha_2=0$ 达到最小，此时 $\alpha_3=\alpha_1+\alpha_3=\frac{1}{4}$。

由于 $\alpha_1^*=\alpha_3^*=\frac{1}{4}>0$，所以由1.3可知其对应的实例点 $x_1,x_3$ 是支持向量。根据公式 (30-31) 计算得：

$$w_1^*=w_2^*=\frac{1}{2}$$

$$b^*=-2$$

分离超平面为：

$$\frac{1}{2}x^{(1)}+\frac{1}{2}x^{(2)}-2=0$$

分类决策函数为：

$$f(x)=sign(\frac{1}{2}x^{(1)}+\frac{1}{2}x^{(2)}-2)$$
***

##  1.3 支持向量

考虑原始最优化问题 (5) 以及对偶最优化问题 (28-29)，将训练数据集中对应与 $\alpha_i^*>0$ 的样本点 $(x_i,y_i)$的实例 $x_i\in\boldsymbol{R^n}$ 称为`支持向量`。

根据这一定义，支持向量一定在间隔边界上，由 KKT 互补条件可知：

$${\alpha_{i}^{*}\left(y_{i}\left(w^{*} \cdot x_{i}+b^{*}\right)-1\right)=0, \quad i=1,2, \cdots, N}  \tag{34}$$

对应与 $a_i^*>0$ 的实例 $x_i$，有：

$$y_{i}(w^{*} \cdot x_{i}+b^{*})-1=0 \tag{35}$$

或

$$w^{*} \cdot x_{i}+b^{*}=\pm1 \tag{36}$$

即 $x_i$ 一定在间隔边界上，这里的支持向量的定义与之前给出的支持向量的定义是一致的。

# 2. 线性支持向量机

线性可分问题的支持向量机学习方法，对线性不可分训练数据是不使用的，因为这是上述方法中的不等式约束并不能都成立。因此此时需要修改硬间隔最大化，使其称为软间隔最大化。

## 2.1 软间隔最大化

假定给定一个特征空间上的训练数据集：

$$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\} \tag{37}$$

其中$x_i \in X=R^n$，$y_{i} \in \mathcal{Y}=\{-1,+1\}, \quad i=1,2, \cdots, N, x_i$ 为第 $i$ 个特征向量，$y_1$ 为 $x_i$ 的类标记。再假设训练数据集不是线性可分的。通常情况是，训练数据中有一些特异点（outlier），将这些特异点去除后，剩下大部分的样本点组成的集合是线性可分的，我们称之为近似线性可分的。

近似线性可分意味着某些样本点 $(x_i,y_i)$ 不能满足函数间隔大于 1 的约束条件 (5)。为了解决这个问题，可以对每个样本点 $(x_i,y_i)$ 引入一个松弛变量 $\xi \geq 0$，使函数间隔加上松弛变量大于等于 1 。这样，约束条件变为：

$$y_i(w\cdot x_i+b)\geq 1-\xi_i \tag{38}$$

同时，对每个松弛变量 $\xi_i$，支付一个代价 $\xi_i$，目标函数由原来的 $\frac{1}{2}\|w\|^2$ 变成：

$$\frac{1}{2}\|w\|^2+C\sum_{i=1}^N\xi_i\tag{38}$$

这里，$C>0$ 称为惩罚参数，一般由应用问题决定，$C$ 值越大时对误分类的惩罚增大，$C$ 值小时对误分类的惩罚减小，如下图所示：

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/Screen Shot 2019-08-16 at 20.42.42.png" width="520" >

<center> 图8 C 参数对间隔的调节</center>

最小化目标函数 (38) 有两层含义：使 $\frac{1}{2}\|w\|^2$ 尽量小即间隔尽量大，同时使误分类点的个数尽量小，$C$ 是调和二者的系数。

有了上面的思路，可以和训练数据集线性可分时一样来考虑训练数据集线性近似可分时的线性支持向量机学习问题，相应于硬间隔最大化，它称为`软间隔最大化`（soft margin maximization）。

### 2.1.1 原始问题

近似线性可分的线性支持向量机的学习问题变成如下凸二次规划问题（**原始问题**）：

$$\min \limits_{w,b,\xi}\quad \frac{1}{2}\|w\|^2+C\sum_{i=1}^N\xi_i\tag{39}$$

$$
\quad\quad\quad\quad\quad\quad\quad\quad\begin{array}{ll}{\text { s.t. }} & {y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \cdots, N} \tag{40}\\ {} & {\xi_{i} \geqslant 0, \quad i=1,2, \cdots, N}\end{array}
$$

原始问题 (39-40) 是一个凸二次规划问题，因而关于 $(w,b,\xi)$ 的解时存在的。可以证明 $w$ 的解时唯一的， 但 $b$ 的解不唯一，但可以证明 $b$ 的解存在一个区间。

### 2.1.2 对偶算法

原始最优化问题 (39-40) 的拉格朗日函数是：

$$\begin{equation}
L(w, b, \xi, \alpha, \mu) \equiv \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i}-\sum_{i=1}^{N} \alpha_{i}\left(y_{i}\left(w \cdot x_{i}+b\right)-1+\xi_{i}\right)-\sum_{i=1}^{N} \mu_{i} \xi_{i}\tag{41}
\end{equation}$$

其中，$\alpha_i \geq0,\mu_i\geq0$。

与 1.2.2 相同，对偶问题是拉格朗日函数的极大极小问题，首先求 $L(w, b, \xi, \alpha, \mu)$ 对 $w, b, \xi$ 的极小并代入 (41) 得：

$$\begin{equation}
\min _{w, b, \xi} L(w, b, \xi, \alpha, \mu)=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}\tag{42}
\end{equation}$$

再对 $\min \limits_{w, b, \xi} L(w, b, \xi, \alpha, \mu)$ 求 $\alpha$ 的极大，即得对偶问题：

$$\begin{equation}
\quad\quad\quad\quad\quad\max \limits_{\alpha}\bigg[-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}\bigg]\tag{43}
\end{equation}$$

$$\begin{equation}
\begin{array}{ll}{\text { s.t. }} & {\sum_{i=1}^{N} \alpha_{i} y_{i}=0} \\ {} & {C-\alpha_{i}-\mu_{i}=0} \\ {} & {\alpha_{i} \geqslant 0} \\ {} & {\mu_{i} \geqslant 0, \quad i=1,2, \cdots, N}\end{array}\tag{44}
\end{equation}$$

将对偶最优化问题 (43-44) 进行变换：利用 (44) 第二条公式消去 $\mu_i$，从而只留下变量 $\alpha_i$，同时将目标函数求极大转换为求极小，可得原始问题  (39-40)的**对偶问题** (45-46)：

$$\begin{equation}
\quad\quad\quad\quad\quad\quad\quad\quad\min \limits_{\alpha}\bigg[\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}\bigg]\tag{45}
\end{equation}$$

$$\begin{equation}
\begin{array}{ll}{\text { s.t. }} & {\sum_{i=1}^{N} \alpha_{i} y_{i}=0} \\ 
{} & {0 \leq \alpha_i \leq C} \\
\end{array}\tag{46}
\end{equation}$$

设 $\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^T$ 是对偶问题 (45-46) 的一个解，若存在 $\alpha^*$ 的一个分量 $\alpha_j^*$，$0 \leq \alpha_i \leq C$，则原始问题 (39-40) 的解 $w^*,b^*$ 可按下式求得：

$$w^*=\sum_{i=1}^N\alpha_i^*y_ix_i\tag{47}$$

$$b^*=y_i-\sum_{i=1}^Ny_i\alpha_i^*(x_i\cdot x_i)\tag{48}$$

综上，可得线性支持向量机学习算法。

***
**算法2 线性支持向量机学习算法**


输入：线性可分训练数据集$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$，其中，$x_i \in X=R^n$，$y_{i} \in \mathcal{Y}=\{-1,+1\}, \quad i=1,2, \cdots, N$；

输出：分离超平面和分类决策函数。
    
（1） 选择惩罚参数 $C>0$，构造并求解约束时最优化问题：

$$\begin{equation}
\quad\quad\quad\quad\quad\quad\quad\quad\min \limits_{\alpha}\bigg[\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}\bigg]\tag{49}
\end{equation}$$

$$\begin{equation}
\begin{array}{ll}{\text { s.t. }} & {\sum_{i=1}^{N} \alpha_{i} y_{i}=0} \\ 
{} & {0 \leq \alpha_i \leq C} \\
\end{array}\tag{50}
\end{equation}$$

求得最优解 $\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^T$ 。

（2）计算

$$w^*=\sum_{i=1}^N\alpha_i^*y_ix_i\tag{51}$$

并选择 $\alpha^*$ 的一个分量 $0 \leq \alpha_i \leq C$，计算：

$$b^*=y_i-\sum_{i=1}^Ny_i\alpha_i^*(x_i\cdot x_i)\tag{52}$$

（3）求得分离超平面：

$$
w^{*} \cdot x+b^{*}=0\tag{53}
$$

分离决策函数：

$$
f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)\tag{54}
$$

步骤 (2) 中，对任一适合条件 $0 \leq \alpha_i \leq C$ 的 $\alpha_j^*$，按 (52) 都可以计算出 $b^*$，但是由于原始问题 (43-44) 对 $b$ 的解并不唯一，所以实际计算时可以取所有符合条件的样本点上的平均值。

***

## 2.2 支持向量

在线性不可分的情况下，将对偶问题 (49-50) 的解  $\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^T$ 中对应于 $\alpha_i^*$ 的样本点 $(x_i,y_i)$ 的实例 $x_i$ 称为支持向量（软间隔的支持向量）。

如图 9 所示，分离超平面由实现表示，间隔边界由虚线表示，正例点由“$\circ$”表示，负例点由“$\times$”表示。实例 $x_i$ 到间隔边界的距离为 $\frac{\xi_i}{\|w\|}$。

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/Screen Shot 2019-08-16 at 22.17.12.png" width="420" >

<center> 图9 软间隔的支持向量</center>

软间隔的支持向量 $x_i$ 或者在间隔边界上，或者在间隔边界与分离超平面之间，或者在分离超平面误分一侧：

- 若 $\alpha_i^*<C$ ，则 $\xi_i=0$，支持向量 $x_i$ 恰好落在`区间边界上`；

- 若 $\alpha_i^*=C$ ， $0<\xi_i<1$，则分类正确，支持向量 $x_i$ 在`区间边界与分离超平面之间`；

- 若 $\alpha_i^*=C$ ， $\xi_i=1$，则支持向量 $x_i$ 在`分离超平面上`；

- 若 $\alpha_i^*=C$ ， $\xi_i>1$，则支持向量 $x_i$ 在`分离超平面误分一侧`。

## 2.3 合页损失函数

对于线性支持向量机来说，其模型为分离超平面 $w^* \cdot x + b^*=0$ 以及决策函数 $f(x)=sign(w^* \cdot x + b^*)$，其学习策略为软间隔最大化，学习算法为凸二次规划。

线性支持向量机学习还有另外一种解释，就是最小以下目标函数：

$$\sum_{i=1}^N[1-y_i(w \cdot x_i+b)]_+ + \lambda \|w\|^2 \tag{55}$$

1. 目标函数的第一项式`经验损失`或经验风险。

函数：

$$L(y(w \cdot x+b)) = [1-y(w \cdot x+b)]_+ \tag{56}$$

称为`合页损失函数`（hinge loss function）。下标“+”表示以下取正值的函数：

\begin{equation}
[z]_{+}=\left\{\begin{array}{ll}{z,} & {z>0} \\ {0,} & {z \leqslant 0}\end{array}\right.\tag{57}
\end{equation}

这就是说，当样本点 $(x_i,y_i)$ 被正确分类且函数间隔（确信度） $y_i(w\cdot x_i+b)$ 大于 1 时，损失时0，否则损失是 $1-y(w \cdot x+b)$，注意图 9 中的实例点 $x_4$ 被正确分类但损失不是 0 。

2. 目标函数的第 2 项是系数为 $\lambda$ 的 $w$ 的 $L_2$ 范数，是正则化项。

***
**定理**

线性支持向量机原始最优化问题：

$$\min \limits_{w,b,\xi}\quad \frac{1}{2}\|w\|^2+C\sum_{i=1}^N\xi_i\tag{58}$$

$$
\quad\quad\quad\quad\quad\quad\quad\quad\begin{array}{ll}{\text { s.t. }} & {y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \cdots, N} \tag{59}\\ {} & {\xi_{i} \geqslant 0, \quad i=1,2, \cdots, N}\end{array}
$$

等价于最优化问题：

$$\min \limits_{w,b}\quad \sum_{i=1}^N[1-y_i(w \cdot x_i+b)]_+ + \lambda \|w\|^2 \tag{60}$$
***

`合页损失函数`的图形如下图所示，横轴是函数间隔 $y(w \cdot x+b)$， 纵轴是损失，由于函数形状像一个合页，故命名合页损失函数。

<img style="float:center" src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/Screen Shot 2019-08-16 at 23.57.34 copy.png" width="520" >

<center> 图10 合页损失函数</center>

图中还画出 0-1 损失函数，可以认为它是二类分类问题的真正的损失函数，而合页损失函数是 0-1 损失函数的上界，由于 0-1 损失函数不是连续可导的，直接优化由其构成的目标函数比较困难，可以认为线性支持向量机就是优化由 0-1 损失函数的上界（合页损失函数）构成的目标函数。这时的上界损失函数由称为代理损失函数（surrograte loss function）。

图中虚线显示的是感知器的损失函数 $[y_i(w \cdot x_i+b)]_+$。这时，当样本点 $(x_i,y_i)$ 被正确分类时，损失是 0，否则损失是 $-y_i(w \cdot x_i+b)$，相比之下，合页损失函数不仅要分类正确，而且确信度足够高时损失才是 0，也就是说，合页损失函数对学习由更高的要求。

# 参考资料

[1] 李航. 统计学习方法[M]. 北京: 清华大学出版社, 2012: 95-115.

[2] 周志华. 机器学习[M]. 北京: 清华大学出版社, 2016: 121-126.
