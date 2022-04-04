# 0.

本笔记是基于UC Berkeley马毅教授的书：
**High-Dimensional Data Analysis with Low-Dimensional Models:
Principles, Computation, and Applications**


# 1. Sparse Model

很多问题可以抽象为基于线性系统 $y = Ax$ 的观察 $y$ 来还原 $x$.
然而，该系统经常是 underdetermined，即 $y$ 的数量远小于 $x$.
如果假设待还原的 $x$ 是稀疏的，能够使问题变得可解，很多时候也符合实际情况。

## 1.1 $l^0$ minimization

向量的 0-norm 表示向量中非零元素的个数，最小化 0-norm 可以推动 $x$ 变得稀疏.
因此，求解线性系统 $y = Ax$ 的稀疏解可以转化为 $l^0$ 优化问题：
* $min ||x||_0,\ s.t.\ Ax = y$.

$l^0$ 优化问题能得到唯一解的条件：
* 定义矩阵 $A$ 的 Kruskal Rank ($krank(A)$) 为最大的 r: $A$ 中的任意 r 列都线性无关。
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