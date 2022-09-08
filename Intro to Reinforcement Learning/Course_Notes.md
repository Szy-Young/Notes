本课程介绍了强化学习的基本内容，原课程来自CUHK周博磊教授。
* [课程地址](https://github.com/zhoubolei/introRL)
* [code example](https://github.com/cuhkrlcourse/RLexample)
* [OpenAI gym](https://github.com/openai/gym)

# 0. Basics

强化学习(RL)定义：
* a computational approach to learning whereby **an agent** tries to maximize the total amount of **reward** it receives while interacting with a complex and uncertain **environment**. (from Sutton and Barto)

RL相比一般监督学习的特点：
* 输入是非*i.i.d*的序列数据
* 每一步不会立刻得到监督信号，只有延迟一段时间后的reward
* agent不会被告知在某一步是否应该采取某个action，必须通过尝试(exploration and exploitation)寻找能最大化reward的policy

## 0.1 Sequential Decision Making

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

MDP是描述RL问题的标准框架。在MDP问题中，环境是完全可观测的(model-based agent).

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
* $q^{\pi}(s, a) = R_s^a + \gamma \sum_{s'} P(s'|s, a) v^{\pi}(s')$

但我们可以将二者解耦：
* $v^{\pi}(s) = \sum_a \pi(a|s) (R_s^a + \gamma \sum_{s'} P(s'|s, a) v^{\pi}(s'))$
* $q^{\pi}(s, a) = R_s^a + \gamma \sum_{s'} P(s'|s, a) \sum_{a'} \pi(a'|s') q^{\pi}(s', a')$

### 1.2.1 MDP prediction
给定策略下估计MDP价值函数的问题，称为value prediction或policy evaluation。如前所述，给定策略下的MDP等价于MRP，因此可以沿用MRP价值函数估计的Monte-Carlo采样的方法或Bootstrapping迭代的方法。

### 1.2.1 MDP control
寻找最大化奖励的策略的问题称为MDP control，即$\pi^*(s) = \mathop{\arg\max}_{\pi} v^{\pi}(s)$. MDP的最优价值是固定的，但最优策略不一定唯一（可以有多个策略具有相同的价值函数）。

这里介绍两种解法：
1. **Policy Iteration:** 迭代进行以下两个步骤：(i) **policy evaluation:** 估计当前策略$\pi$的价值函数；(ii) **policy improvement:** 利用贪心算法改进当前策略，$\pi_{i+1}(s) = \mathop{\arg\max}_{a} q^{\pi_i}(s, a)$.
2. **Value Iteration:** 这种方法基于Bellman optimality equation: $v*(s) = \mathop{\arg\max}_{a} q*(s, a)$，以DP/Bootstrapping的方式求解。首先初始化所有状态的价值$v(s)$，随后迭代进行如下步骤：(i) $q_{k+1}(s, a) = R_s^a + \gamma \sum_{s'} P(s'|s, a) v_k(s')$ (ii) 基于Bellman optimality equation更新最优价值函数$v_{k+1}(s) = \mathop{\arg\max}_{a} q_{k+1}(s, a)$. 在得到最优价值函数后，可以利用$\pi*(s) = \mathop{\arg\max}_{a} R_s^a + \gamma \sum_{s'} P(s'|s, a) v^*(s')$得到最优策略。相比Policy Iteration的方法，Value Iteration只需进行一轮迭代。
