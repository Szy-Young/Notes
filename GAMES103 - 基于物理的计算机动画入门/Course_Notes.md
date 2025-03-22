# 0. 

该课程讲述了计算机动画中代表性的物理模拟方法，包括刚体（rigid body）、衣物（cloth）、软体（soft body）和流体（fluid）的模拟。
[原课程](https://games-cn.org/games103/)来自Ohio State Univ.王华民教授。


# 1. Rigid Body 刚体

## 1.1 Rigid Body Dynamics

### 1.1.1 Translational Motion

理论上，从时刻$t_0$到$t_1$，要通过整个时间区间内的加速度积分得到速度，再通过整个时间区间内的速度积分得到位置。
实际应用中，通常用数值积分（包括显式积分和隐式积分两种方式）来近似。