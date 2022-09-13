本课程介绍了强化学习的基本内容，原课程来自CUHK周博磊教授。
* [课程地址](https://github.com/zhoubolei/introRL)
* [code example](https://github.com/cuhkrlcourse/RLexample)
* [OpenAI gym](https://github.com/openai/gym)

# 0. Basics

强化学习(RL)定义：
* a computational approach to learning whereby **an agent** tries to maximize the total amount of **reward** it receives while interacting with a complex and uncertain **environment**. (from Sutton and Barto)

RL相比一般监督学习的特点：
* 输入是非*i.i.d*的序列数据。
* 每一步不会立刻得到监督信号，只有延迟一段时间后的reward.
* agent不会被告知在某一步是否应该采取某个action，必须通过尝试(exploration and exploitation)寻找能最大化reward的policy.

RL可以描述为一个序列决策过程。其中一个**agent**包含以下要素：
* **Policy:** 从state/observation到action的映射，形式上通常是带有随机性的$\pi(a|s) = P[A_t=a | S_t=s]$或确定性的$a^* = \mathop{\arg\max}_{a} \pi(a|s)$.
* **Value function:** *固定policy下*，当前state/action未来累积获得reward的期望值，累积时通常以速率$\gamma$随时间衰减（更看重能在近期获得的reward）。
(i) **state value (V function):** $v_{\pi}(s) = E_{\pi}[G_t | S_t=s] = E_{\pi}[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t=s]$; 
(ii) **action value (Q function):** $q_{\pi}(s, a) = E_{\pi}[G_t | S_t=s, A_t=a] = E_{\pi}[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t=s, A_t=a]$.
* **Model:** 对环境的预测。
(i) **State prediction:** $P_{ss'}^a = P[S_{t+1}=s' | S_t=s, A_t=a]$;
(ii) **Reward prediction:** $R_{s}^a = E[R_{t+1} | S_t=s, A_t=a]$.

根据学习内容，agent可以分为value-based, policy-based, 或二者的结合actor-critic; 根据agent是否带有环境model，可以分为model-based或model-free.


# 1. Markov Decision Process (MDP)

MDP是描述RL问题的标准框架。本章节中，我们首先讨论MDP模型已知的情况(model-based agent)。

## 1.1 Markov Reward Process (MRP)

* **Markov Process (MP):** 未来由当前状态完全决定，无需考虑过去状态。核心是状态转移概率$P(s_{t+1}=s' | s_t=s)$.
* **Markov Reward Process (MRP):** 相比MP多了一个奖励函数$R(s_t=s) = E[r_t | s_t=s]$，以及相应的衰减系数$\gamma$.

对于MRP中的状态，可以估计其价值函数：$V(s) = E[G_t | S_t=s] = E[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... | S_t=s]$. 此外，MRP的价值函数满足递推式的Bellman equation：$V(s) = R(s) + \sum_{s'} P(s' | s) V(s')$，其中$R(s)$代表到达状态$s$后即时得到的奖励，另一项代表未来的累积奖励。

MRP的价值函数估计有两种主要方法：
1. Monte-Carlo: 对某个状态，从它开始采样多段序列，计算累积奖励的平均值。
2. Dynamic Programming (DP) / Bootstrapping：初始化每个状态的价值，利用Bellman equation迭代计算。

## 1.2 Markov Decision Process (MDP)

MDP相比MRP多了action，状态转移概率和奖励函数都会受到action影响，变成$P(s_{t+1}=s' | s_t=s, a_t=a)$和$R(s_t=s, a_t=a) = E[r_t | s_t=s, a_t=a]$.
在固定的策略$\pi(a|s) = P(a_t=a | s_t=s)$下，MDP可以转换为MRP：状态转移概率$P^{\pi}(s'|s) = \sum_{a} \pi(a|s) P(s'|s, a)$，奖励函数$R^{\pi}(s)= \sum_{a} \pi(a|s) R(s, a)$.

MDP中的Bellman equation，价值函数$v(s)$和$q(s, a)$相互耦合：
* $v^{\pi}(s) = \sum_a \pi(a|s) q^{\pi}(s, a)$
* $q^{\pi}(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) v^{\pi}(s')$

但我们可以将二者解耦：
* $v^{\pi}(s) = \sum_a \pi(a|s) (R(s, a) + \gamma \sum_{s'} P(s'|s, a) v^{\pi}(s'))$
* $q^{\pi}(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) \sum_{a'} \pi(a'|s') q^{\pi}(s', a')$

### 1.2.1 MDP prediction
**给定策略下估计MDP价值函数的问题**，称为value prediction或policy evaluation。如前所述，给定策略下的MDP等价于MRP，因此可以沿用MRP价值函数估计的Monte-Carlo采样的方法或Bootstrapping迭代的方法。

### 1.2.1 MDP control
**寻找最大化奖励的策略的问题称为MDP control**，即$\pi^*(s) = \mathop{\arg\max}_{\pi} v^{\pi}(s)$. MDP的最优价值是固定的，但最优策略不一定唯一（可以有多个策略具有相同的价值函数）。

这里介绍两种解法：
1. **Policy Iteration:** 迭代进行以下两个步骤：(i) **policy evaluation:** （利用Boostrapping方法迭代）估计当前策略$\pi$的价值函数；(ii) **policy improvement:** 利用贪心算法改进当前策略，$\pi_{i+1}(s) = \mathop{\arg\max}_{a} q^{\pi_i}(s, a)$.
2. **Value Iteration:** 这种方法基于Bellman optimality equation: $v*(s) = \mathop{\arg\max}_{a} q*(s, a)$，以DP/Bootstrapping的方式求解。首先初始化所有状态的价值$v(s)$，随后迭代进行如下步骤：(i) $q_{k+1}(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) v_k(s')$ (ii) 基于Bellman optimality equation更新最优价值函数$v_{k+1}(s) = \mathop{\arg\max}_{a} q_{k+1}(s, a)$. 在得到最优价值函数后，可以利用$\pi*(s) = \mathop{\arg\max}_{a} R(s, a) + \gamma \sum_{s'} P(s'|s, a) v^*(s')$得到最优策略。相比Policy Iteration的方法，Value Iteration只需进行一轮迭代。


# 2. Model-free Prediction and Control

很多时候，RL问题中的MDP模型是未知的。具体来说，状态转移概率$P(s'|s, a)$和奖励函数$R(s, a)$未知。

本章节介绍如何在MDP模型未知的情况下(model-free)解决prediction和control的问题。

## 2.1 Model-free Prediction

主要有两种解法: (i) MC (Monte-Carlo) (ii) TD (Temporal Difference) Learning.

### 2.1.1 MC Policy Evaluation

MC的具体做法前面已经介绍过：对某个状态，从它开始采样多段序列，计算累积奖励的**平均值**。

**增量式更新**（后面的方法一般都会采取增量式更新）的MC: 从新采样的一段序列得到状态$s_t$的累计奖励$G_t$，更新步长（学习率）为$\alpha$，更新状态$s$的价值函数，即$v^{\pi}(s_t) \leftarrow v^{\pi}(s_t) + \alpha (G_t - v^{\pi}(s_t))$.

### 2.1.2 TD Learning

TD Learning是MC方法和Bootstrapping方法的结合：从当前状态$s_t$出发，采样一步，得到奖励$R_{t+1}$并进入状态$s_{t+1}$，此时可以利用当前的价值函数估计$v^{\pi}(s_{t+1})$构造$v^{\pi}(s_t)$的更新目标，即$v^{\pi}(s_t) \leftarrow v^{\pi}(s_t) + \alpha (R_{t+1} + \gamma v^{\pi}(s_{t+1}) - v^{\pi}(s_t))$.

类似地，TD也可以用于$q^{\pi}(s, a)$的价值函数估计，即$q^{\pi}(s_t, a_t) \leftarrow q^{\pi}(s_t, a_t) + \alpha (R_{t+1} + \gamma \mathop{\max}_{a_{t+1}}q^{\pi}(s_{t+1}, a_{t+1}) - v^{\pi}(s_t))$，此时的TD方法又称为**Q-learning**.

上述操作是最基本的TD Learning，只向前采样一步就进行更新，被称为TD(0). 也可以采样n步后再进行更新，称为n-step TD. 当采样无穷多步（直到序列终止）时，TD等同于MC.

TD和MC的对比：
* TD可以在每一步或几步采样后实时更新，MC必须等序列终止才能更新；因此，TD可以从不完整的采样序列学习，而且可以用于无限循环的MDP问题，MC则不行。
* TD得到的更新目标是对价值函数的有偏估计，但方差小；MC得到的是无偏估计，但方差大。
* MC可以用于非Markov的过程，TD则不行。

## 2.2 Model-free Control

model-free control的总体思路是generalized policy iteration: 用MC或TD（TD(0)或n-step TD）的方法代替boostrapping，对价值函数$q^{\pi}(s, a)$进行估计，然后基于$\pi_{i+1}(s) = \mathop{\arg\max}_{a} q^{\pi_i}(s, a)$来改进当前策略。

**$\epsilon$-greedy:** 在MC或TD方法进行采样时，从当前状态$s$出发，有$1-\epsilon$的概率选择最大化价值的$\mathop{\arg\max}_{a} q^{\pi}(s, a)$，有$\epsilon$的概率选择随机的action. 只是一种典型的tradeoff between exploration and exploitation.

**SARSA:** 基于TD(0)的policy iteration，向前采样一步，得到的更新目标为$q^{\pi}(s_t, a_t) \leftarrow q^{\pi}(s_t, a_t) + \alpha (R_{t+1} + \gamma q^{\pi}(s_{t+1}, a_{t+1}) - q^{\pi}(s_t, a_t))$. 由于包含t时刻的S,A和t+1时刻的R,S,A, 所以称为SARSA算法。(SARSA和Q-learning的区别在于SARSA采用$\epsilon$-greedy的策略向前采样1步，而Q-learning直接以greedy的方式向前采样一步)

**On/Off policy learning:** SARSA是一种on-policy learning的方法，直接用正在优化的策略进行采样；off-policy则在优化策略的同时使用另一个独立的策略进行采样。


# 3. Value Function Approximation

此前对于价值函数，我们保存一个lookup table，每个状态$v(s)$或$q(s, a)$的价值函数都被独立地记录。当RL问题的状态数量过多时，lookup table过大，此时可以用一个共享的参数化函数来拟合价值函数，即：$\hat{v}(s; w) = f(s; w)$或$\hat{q}(s, a; w) = f(s, a; w)$。函数$f$可以选择线性函数或非线性的deep network，从而直接利用梯度优化。

## 3.1 Value Function Approximation for Prediction & Control

1. **Prediction:** 沿用上一章节的MC或TD方法，采样得到价值函数的更新目标，然后基于least square来优化价值函数的参数，即$\mathop{\min}_{w} (\hat{v}(s; w) - v(s))$（$\hat{q}(s, a; w)$同理)。
2. **Control:** 与prediction类似的，沿用此前介绍的generalized policy iteration方法（MC policy iteration/SARSA/Q-learning，...），将其中基于查找表的更新替换为对价值函数的参数优化。

## 3.2 Linear/Non-linear value function

1. **Linear combination of features:** 首先对输入状态提取人工设计的特征，然后优化对特征进行线性组合的参数；如何设计对状态的特征表示是一个关键问题。如果对所有状态采用one-hot的特征编码，则线性组合得到的价值函数等同于lookup table.
2. **Non-linear deep network:** 利用deep network拟合价值函数，能够自动学习特征提取的步骤。

**Challenges in convergence:** 价值函数拟合并不能确保收敛，因为有如下挑战：
* *Boostrapping:* 不同于监督学习中基于groundtruth的优化目标，（基于TD的policy iteration中）价值函数拟合的优化目标本身就是对价值函数的有偏估计。
* *Off-policy learning:* 如果采用off-policy learning可以启发更多的exploration，但训练和测试时存在状态转移的分布不一致的问题（训练时的状态转移由负责采样的policy得到，测试时的状态转移由被优化的policy本身得到）。
* *Function approximation:* 用参数化函数来拟合价值函数本身也是一种近似。

**Batch RL:** 此前采用的增量式更新简单但低效。为此，我们可以参照监督学习中的做法，将多次采样的更新目标放入一个batch，同时用于优化价值函数的参数。这种做法又称为**least square prediction & control**.

### 3.2.1 Deep Q-learning (Deep Q Network, DQN)

原文*Human-level control through deep reinforcement learning*由DeepMind于2015年发表在Nature. 

DQN利用RL解决Atari Games，首次实现了直接以屏幕上的游戏图像作为输入状态，用一个deep network拟合价值函数$q(s, a)$. 其中，作者发现DQN的训练面临两个问题：
* 输入是非*i.i.d*的序列数据。
* policy不断更新，导致价值函数的更新目标不断变化，优化过程不稳定。

为此，DQN提出了相应的解决办法：
* **Experience replay:** 将采样到的experience存储起来，每次从中随机抽取experience来优化价值函数，避免了训练样本之间的关联。
* **Fixed target:** 用另一个更新频率较低的network提供优化目标，这个network在主network训练若干次后才同步更新一次参数。