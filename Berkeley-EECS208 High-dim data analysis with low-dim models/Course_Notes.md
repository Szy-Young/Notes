# 0.

本笔记是基于UC Berkeley马毅教授的书：
**High-Dimensional Data Analysis with Low-Dimensional Models:
Principles, Computation, and Applications**
* [原书和课程slides地址](https://book-wright-ma.github.io/)


# 1. Sparse Model

很多问题可以抽象为基于线性系统 $y = Ax$ 的观察 $y$ 来还原 $x$.
然而，该系统经常是 underdetermined，即 $y$ 的数量远小于 $x$.
如果假设待还原的 $x$ 是稀疏的，能够使问题变得可解，很多时候也符合实际情况。

## 1.1 $l^0$ minimization

向量的 0-norm 表示向量中非零元素的个数，最小化 0-norm 可以推动 $x$ 变得稀疏.
因此，求解线性系统 $y = Ax (其中 A \in \mathbb{R}^{m \times n}, x \in \mathbb{R}^{n}, y \in \mathbb{R}^{m})$ 的稀疏解可以转化为 $l^0$ 优化问题：
* $min ||x||_0,\ s.t.\ Ax = y$.

定义矩阵 $A$ 的 Kruskal rank ($krank(A)$) 为：
* 最大的 r：$A$ 中的任意 r 列都线性无关。

$l^0$ 优化问题能得到唯一解的条件：
* 如果 $x_o$ 满足 $y = A x_o$，同时 $||x_o||_0 \leq krank(A) / 2$，那么 $x_o$ 是 $l^0$ 优化问题的唯一解。


## 1.2 A relaxation: $l^1$ minimization

直接求解 $l^0$ 优化十分困难。
一种方法是假设 $x$ 的 0-norm 为 k = 0, 1, ...，同时遍历 $x$ 中不同元素为 0 的所有情况，分别求解 $Ax = y$.
但这种方法太过低效。
为了简化，我们用 $l^1$ 优化来取代 $l^0$ 优化，因为 $l^1$ 是最接近 $l^0$ 的凸函数。
凸优化问题的好处在于每一步利用局部的梯度信息（最简单的梯度下降法），即可确保最后收敛到全局最优点。
因此，求解线性系统 $y = Ax$ 的稀疏解转化为了 $l^1$ 优化问题：
* $min ||x||_1,\ s.t.\ Ax = y$.

尽管 $l^1$ 是凸函数，但求解 $l^1$ 优化问题仍然存在两个困难：(i) 如何处理约束 $y = Ax$; (ii) $l^1$ 在对于稀疏性最重要的0点不可导。
针对上述问题，分别提出了解决方法：
* **利用 projected gradient 解决约束问题。** 对每一步基于无约束优化的更新 $z = x_k - t_k \nabla f(x_k)$，寻找符合约束的集合 $C = \{x | Ax = y\}$ 中与 $z$ 相距最近的 $P_C[z]$：$P_{C = \{x | Ax = y\}}[z] = z - A* (AA*)^{-1} (Az - y)$.
* **利用 subgradients 近似0点处的梯度。** 定义函数 $f(x)$ 在 $x_0$ 的 subgradients 为 $\partial f(x_0) = \{u | f(x) \geq f(x_0) + <u, x-x_0>, \forall x\}$. 当函数 $f(x)$ 在 $x_0$ 不可导时，可以取任何 $u \in \partial f(x_0)$ 来用于梯度下降。

综上，我们得到求解 $l^1$ 优化问题的 projected subgradient 算法。


## 1.3 Study of $l_1$ minimization

这里研究了 $l_1$ 优化能成功得到唯一的稀疏解 $x_o$ 的条件。通常递进地进行如下分析：
* 在...条件下，$x_o$ 是 $l_0$ 优化问题的唯一解；
* 在...条件下，$x_o$ 是 $l_1$ 优化问题的唯一解。

### 1.3.1 Condition with incoherence

定义矩阵 $A$ 的 mutual coherence ($\mu(A)$) 为：
* $A$ 中不同两列的最大 cosine similarity。$\mu(A)$ 越小，$A$ 中的列在空间中越分散（越接近正交）。
* **Kruskal rank 和 mutual coherence 的关系：** 对任意矩阵 $A$，有 $krank(A) \geq \frac{1}{\mu(A)}$。

基于mutual coherence，$l^0$ 优化问题能得到唯一解的条件：
* 如果 $x_o$ 满足 $y = A x_o$，同时 $||x_o||_0 \leq \frac{1}{2 \mu(A)}$，那么 $x_o$ 是 $l^0$ 优化问题的唯一解。

$l^1$ 优化问题能得到唯一解的条件：
* 矩阵 $A$ 的列是单位向量。如果 $x_o$ 满足 $y = A x_o$，同时 $||x_o||_0 \leq \frac{1}{2 \mu(A)}$，那么 $x_o$ 是 $l^1$ 优化问题的唯一解。

然而，incoherence的条件在实际中是比较局限的。首先考察一般情况下矩阵的 mutual coherence：
* 假设矩阵 $A$ 的列是独立地从半径为1的超球上基于均匀分布随机采样的单位向量。那么很大概率有：$\mu(A) \leq C \sqrt{\frac{log(n)}{m}}$（$C > 0$ 是数值常数）。
反过来说：
* 如果要成功求解 $k-$稀疏的 $x_o$，需要的观察值（i.e.,采样）数量 $m$ 有：$m \geq C' k^2 log(n)$。
即，要得到 $x_o$ 的 $k$ 个非0值，需要 $m = O(k^2)$ 的采样，比较低效。

### 1.3.2 Condition with Restricted Isometry Property (RIP)

Isometry 指的是保持向量norm不变的映射，而 restricted isometry 就是对于稀疏向量来说接近 isometry 对映射。定义矩阵 $A$ 的 $k$ 阶RIP（$\delta_k(A)$）为：
* 最小的 $\delta$：对任意 $k-$稀疏的向量 $x$，满足 $(1-\delta) \|x\|_2^2 \leq \|Ax\|_2^2 \leq (1+\delta) \|x\|_2^2$。

基于RIP，$l^0$ 优化问题能得到唯一解的条件：
* 如果 $k-$稀疏的 $x_o$ 满足 $y = A x_o$，同时 $\delta_{2k}(A) < 1$，那么 $x_o$ 是 $l^0$ 优化问题的唯一解。

$l^1$ 优化问题能得到唯一解的条件：
* 如果 $k-$稀疏的 $x_o$ 满足 $y = A x_o$，同时 $\delta_{2k}(A) < \sqrt{2} - 1$，那么 $x_o$ 是 $l^1$ 优化问题的唯一解。

接下来考察RIP的条件在实际中的情况：
* 假设矩阵 $A$ 的元素是独立地从 $N(0, 1/m)$ 高斯分布采样得到，那么指定上限 $\delta$，很大概率有 $\delta_k(A) < \delta$，如果 $m \geq C k log(n/k) / \delta^2$（$C > 0$ 是数值常数）。
* 如果有 $k \propto n$，则需要的观察值数量 $m = O(k log(n/k)) = O(k)$，相对于 1.3.1 中的 $m = O(k^2)$ 高效了很多。
