# 0. 

该课程讲述了计算机动画中代表性的物理模拟方法，包括刚体（rigid body）、衣物（cloth）、软体（soft body）和流体（fluid）的模拟。
[原课程](https://games-cn.org/games103/)来自Ohio State Univ.王华民教授。


# 1. Rigid Body 刚体

## 1.1 Rigid Body Dynamics

### 1.1.1 Translational motion

理论上，从时刻$t_0$到$t_1$，要通过整个时间区间内的加速度积分得到速度，再通过整个时间区间内的速度积分得到位置。
实际应用中，通常用数值积分（包括显式积分，隐式积分，以及两者折中的leapfrog积分等方式）来近似。

<img src="figures/1_leapfrog.png" width=600>

### 1.2.1 Rotational motion

旋转运动由扭矩和转动惯量决定。不同于平移运动中只与力和物体质量的数值有关，扭矩/转动惯量与力/质量在物体上的分布有关。

<img src="figures/1_torque_inertia.png" width=600>

### 1.2.3 Summary

总体上，物体的质量和转动惯量（与物体的质量分布有关）是预先固定的。在模拟过程中设定每个时刻的力与扭矩，就可以不断更新物体的位移（以及速度）和旋转（以及角加速度）。

<img src="figures/1_rbd_summary.png" width=600>

最后总结一下刚体动力学模拟的实现步骤。

<img src="figures/1_rbd_steps.png" width=600>

## 1.2 Rigid Body Contacts

### 1.2.1 Penalty method

Penalty方法的核心思想是，检测到物体和碰撞表面overlap之后，根据物体嵌入碰撞表面的深度，施加一个向外的弹力。
为了防止模拟中物体真的嵌入碰撞表面，可以将碰撞表面适当向外扩展出一个缓冲区。

<img src="figures/1_penalty_method.png" width=600>

图中的$\phi(x)$指的是物体上的点$x$在碰撞表面的SDF中的取值。

### 1.2.2 Impulse method

Impulse方法的核心思想是，检测到碰撞后直接更新物体的位置和速度，而不是像penalty方法那样间接地先产生一个弹力然后模拟出物体的位置和速度。

<img src="figures/1_impulse_method.png" width=600>

<img src="figures/1_impulse_method_vel.png" width=600>

### 1.2.3 Rigid body collision detection and response

对于碰撞检测，简单的做法是将物体的所有顶点都放到碰撞表面的SDF里进行检测。

对于碰撞模拟，上述方法计算的都是物体上碰撞点的局部速度变化，但刚体模拟中，我们需要整个物体的速度。
常用的做法是先计算出碰撞点局部（期望发生的）速度变化，然后计算要产生这些速度变化需要施加在碰撞点局部的力和扭矩，将其施加在刚体上。

<img src="figures/1_rbd_response.png" width=600>

### 1.2.4 Shape matching

Shape matching是一种相比于1.2.3的方法更简化的刚体碰撞模拟方法，无需计算对物体的力和力矩，只对碰撞点局部的速度和位置独立模拟。
各个顶点独立模拟后，物体已经发生了形变，此时再根据各顶点新的位置计算一个rigid transformation，把物体重新约束回一个刚体。

<img src="figures/1_shape_matching.png" width=600>


# 2. Cloth 衣物

## 2.1 Mass-Spring System

<img src="figures/2_mass_spring.png" width=600>

通过组合多个质点并在它们之间连接弹簧，很适合用来模拟衣物。

<img src="figures/2_mass_spring_system.png" width=600>

对于整齐排列的矩形网格，我们可以在网格原有的边以及对角线上加入弹簧，来模拟沿各个方向的拉力。
对于三角形排列并不规整的mesh网格，我们也可以直接利用原有的边、并在相邻两个三角形的对角顶点之间连线，作为弹簧。

### 2.1.1 Explicit integration of a mass-spring system 质点弹簧系统的显式积分

<img src="figures/2_spring_explicit_int.png" width=600>

用显式积分的方式直接模拟拉力作用下弹簧的运动非常方便，但显式积分存在数值不稳定问题，特别是在弹性系数$k$或时间间隔$\delta t$取得过大时。

### 2.1.2 Implicit integration of a mass-spring system 质点弹簧系统的隐式积分

<img src="figures/2_spring_implicit_int.png" width=600>

隐式积分有更好的数值稳定，但相比于显式积分简单的前向计算，隐式积分的计算更复杂，需要求解非线性方程。
对方程中未知数$x^{[1]}$的求解，可以用Newton法来迭代计算。

<img src="figures/2_implicit_Fx.png" width=600>

特别地，我们把这个解方程问题转化为对函数$F(x)$的优化问题，因为从求解优化问题的角度，可以直观理解模拟结果中的很多现象。
例如，我们分析目标函数$F(x)$的Hessian矩阵，会发现当弹簧拉长时，Hessian矩阵正定（优化问题有全局最优解）；弹簧缩短时，Hessian矩阵不能保证正定。
直观来看，弹簧拉长时，必然是沿直线拉伸的，拉伸方向可以唯一地确定；而弹簧缩短时则可能向不同方向弯折。

## 2.2 Bending & Locking Issues

### 2.2.1 Bending issue

由于质点弹簧系统中能量（力）只与弹簧长度有关，当发生小幅度的弯折时，可能会出现对应方向上的弹簧没有拉力产生的问题。
为了更好地模拟弯折的情况，提出了其他专门针对弯折的模型，典型的有dihedral angle model(二面角模型)和quadratic bending model.

<img src="figures/2_bending_issue.png" width=600>

### 2.2.2 Locking issue

质点弹簧系统中，总是存在某些方向，沿着这些方向对衣物进行弯折会改变一些弹簧的长度从而产生拉力。这种情况至今还没有很好的解决办法。

<img src="figures/2_locking_issue.png" width=600>

## 2.3 Position Based Dynamics (PBD) & Strain Limiting

### 2.3.1 PBD

PBD方法并不显式地描述完整的物理过程(e.g.,弹簧的拉伸/压缩产生的能量/力影响弹簧的运动)。
它采用一种更简化的思路，直接基于物理约束指定相应的目标函数(e.g.,弹簧长度不变)，并在每次模拟时把变量（弹簧顶点位置）更新到能满足约束的值。
特别地，更新变量时会从满足约束的区域中寻找离变量当前值最近的值，这种方式称为projection.

<img src="figures/2_projection.png" width=600>

当系统包含多个约束时（多条弹簧），有两种常用的迭代计算的方法来求解它们联合的projection：
* **Gauss-Seidel方法:** 每次基于一条约束，用projection方法更新与之相关的变量。这种方法的问题是对遍历约束的顺序很敏感，容易导致模拟结果朝某个方向的bias.
* **Jacobi方法:** 把所有约束都遍历一遍，对每个变量从不同约束计算出的更新值取平均。这种方法避免了Gauss-Seidel潜在的bias，但可能收敛会更慢。

<img src="figures/2_pbd.png" width=250>

上图展示了典型的PBD模拟流程，在每一步中：1）先基于粒子/刚体模拟（类似第1章介绍的）更新顶点位置和速度；2）基于PBD再次更新顶点和速度，使其符合给定约束(e.g.,弹簧长度)。
**注意:** PBD并不仅限于衣物模拟，也可以用于其他场景和约束。

可以看到，PBD方法并不显式地描述完整的物理过程：PBD中无法指定弹簧的刚度系数stiffness，只能靠调整projection中迭代的次数或mesh网格的分辨率来近似不同刚度的模拟效果。
另一个重要的点：由于PBD更新了顶点位置，顶点速度也需要相应更新。

### 2.3.2 Strain limiting

PBD方法中，强行约束了弹簧长度不变，如果想模拟具备一定弹性的弹簧，只能靠调整projection中迭代次数这样的方法。
Strain limiting方法作为PBD的进阶版，改进了这一问题。它的设计思路是：
允许弹簧长度在一定阈值内变化(e.g.,不超过弹簧原长的20%)。当长度变化在阈值内时，基于质点弹簧系统做常规的受力模拟；当长度变化超过阈值时，用PBD的方法进行强制约束。
这种思路非常符合真实世界中弹簧的常见情况：具备一定程度的弹性，但形变到一定程度时很难再进一步拉伸/压缩。


# 3. Finite Element Method (FEM) / Finite Volume Method (FVM) 有限元方法

不同于质点弹簧系统以边为对象，能量和受力都与边的形变（拉伸/压缩）相关；FEM/FVM以构成物体的element（通常是三角形/四面体）为对象，顶点上的受力与三角形/四面体的形变（例如剪切，比边的拉伸/压缩更复杂）有关。

这里介绍一种简单的、以三角形面片为对象的有限元方法：
* **F (deformation gradient):** 首先定义deformation gradient，用来描述三角形从静止状态（reference）到当前状态的运动。

<img src="figures/3_deform_grad.png" width=600>

* **G (Green strain):** 从deformation gradient导出Green strain，它消除了三角形运动中的旋转部分，准确地描述了三角形形变的幅度。

<img src="figures/3_green_strain.png" width=600>

* **W (Energy density):** 定义形变量Green Strain (G)相关的能量模型。这里把单位面积上的能量密度定义为形变量G的函数W(G)，称为StVK模型。这是FEM中最简单的一种能量模型，对不同的问题还有其他复杂得多的能量模型。

<img src="figures/3_stvk_energy.png" width=600>

* **f (force):** 计算能量对顶点位移的梯度，即可得到该顶点的受力。

<img src="figures/3_stvk_force.png" width=600>

FVM与FEM在参考系和推导过程上有所不同：FEM是建立在静止状态（reference）上、从能量密度的微分推导出的，FVM则是建立在当前形变后的状态上、从traction的积分导出的。
但在三角形/四面体这类线性的element上，两者推导出的能量/受力模型是等效的。