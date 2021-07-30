# 0.

该课程来自UZH & ETH（[原课程地址](http://rpg.ifi.uzh.ch/teaching.html)），介绍了计算机视觉中感知三维世界的基本方法。


# 1. Introduction 引言

开篇介绍几个重要概念：
1. **Visual Odometry (VO):** 从连贯的图片序列中估计相机位姿变化（相机运动）的过程。整个过程一般分为如下步骤：特征点检测 $\rightarrow$ 特征点匹配 $\rightarrow$ 位姿估计 $\rightarrow$ 局部优化。
2. **visual SLAM:** 任务定义与VO相同，但VO只能保证对局部短时间内运动的准确估计；vSLAM可以保证对长时间运动轨迹的准确估计。可以认为: vSLAM = VO + loop detection & closure.
3. **Structure From Motion (SFM):** 从无序的图片集中估计每张图片对应的相机位姿，同时重建3D场景。
可以看到，SFM是更加宽泛的任务。从覆盖关系上，SFM > vSLAM > VO.


# 2. Camera 相机

## 2.1 Image Formation

### 2.1.1 Pinhole approximation
在针孔相机中，针孔越小，同一个object point发出的光越容易汇聚到一起，形成更清晰的成像；然而，针孔太小又会导致通过的光照不足。
因此，现代相机通常采用凸透镜成像。object point距离透镜的距离Z、透镜的焦距f、汇聚点到透镜的距离e符合如下规律：
$\frac{1}{f} = \frac{1}{e} + \frac{1}{Z}$.
可见，距离透镜远近不同的object point不能汇聚到同一个成像平面上。但是，当 $Z >> e, f$ 时，可近似得到 $e = f$， 成像平面到镜头的距离等于焦距，这就是pinhole approximation.

### 2.1.2 Perspective projection
在基于pinhole approximation的相机模型中，object point的成像点服从透视投影方程：
$\frac{X}{Z} = \frac{x}{f}$.
基于这一规律，会产生“近大远小”、“平行线相交于vanishing point”等现象。

### 2.1.3 Camera model
将世界坐标系中的点投影到图像上需要以下三个步骤：
1. 将世界坐标系下的坐标转换为相机坐标系下的坐标；
2. 将相机坐标系下的坐标透视投影到成像平面上；
3. 将成像平面上的物理坐标转换为像素坐标。
4. （如果有）基于像素坐标进行畸变（Radial distortion是最常见的畸变）。

## 2.2 Camera Calibration

给定若干组3D-2D对应点，估计相机（内/外）参数的过程称为相机标定。

### 2.2.1 Direct Linear Transform (DLT)

1. **Tsai's Method**
将相机投影方程改写成以相机参数为未知数的齐次线性方程组，每组3D-2D对应点能够提供2个方程。相机参数是 $3 \times 4$ 的矩阵，共计12个未知数，确定所有未知数（up to a scale）需要11个方程，即至少6组在3D空间中不共面的对应点。
当对应点数量更多时，从求解齐次线性方程组转化为求解linear least squares问题，可以通过对矩阵进行奇异值分解（SVD）来求解。
最后，可以通过QR分解的方式从完整的参数矩阵分离出内参数K和外参数R, T.

2. **Zhang's Method**
当给定的所有3D点共面时，沿用上述的方法构造齐次线性方程组。此时相机参数变为 $3 \times 3$ 的矩阵，共计9个未知数，这种两个平面上的点之间的变换关系称为homography。确定所有未知数（up to a scale）需要8个方程，即至少4组在3D空间中共面的对应点。
类似地，对应点数量更多时也可以利用SVD求解linear least squares问题。
不同于Tsai's Method，此处从参数矩阵分离出内参数和外参数，需要从不同视角下求解各个视角的homography，随后联合求解。

DLT方法能直接求解完整的参数矩阵，也能在内参数已知的情况下用于求解外参数。
此外，可以在基础DLT方法得到的结果的基础上，以优化重投影误差为目标，采用非线性优化的方式进行non-linear refinement。

### 2.2.2 Perspective N Points (PnP)

在内参数已知的情况下，还可以用PnP算法来估计相机的6DoF位姿（即外参数）。
给定1组、2组对应点时，PnP算法有无穷多解；给定3组对应点时，最多有4个解；给定4组对应点时有唯一解。
相比DLT方法，PnP方法（尤其是EPnP方法）更准确、更高效；但不能处理内、外参数都未知的情况。


# 3. Image Feature 图像特征

本章讨论从图像中检测特征点、以及在不同视角的图像之间匹配特征点的方法。

## 3.1 Filter

滤波可以按kernel的形式分为：
1. **linear filter:** 直接通过卷积运算实现，例如Gaussian filter.
2. **non-linear filter:** 例如median filter和bilateral filter. 在对图像进行模糊时，以Gaussian filter为代表的线性滤波会无差别地模糊掉所有edges；而median filter和bilateral filter可以保留图像中的strong edges.

此外，还可以按功能分为：
1. **low-pass filter:** 用于模糊信号（图像），上述的Gaussian/median/bilateral filter都属于低通滤波。这类滤波器kernel中各元素之和通常为1.
2. **high-pass filter:** 用于提取信号中的突变（图像中的边缘），常见的有derivative filter，即以卷积形式表示的差分运算。这类滤波器kernel中各元素之和通常为0.
值得注意的是，直接对含有噪声的信号应用derivative filter不能有效提取出突变，此时应先对信号做模糊处理；此处还可以调整运算顺序，先对用于模糊信号的低通滤波kernel取差分，再将得到的结果作为kernel应用于待处理信号，即：
$(I * G) * H = I * (G * H)$.
其中I为待处理信号，G为Gaussian filter，H为derivative filter.

## 3.2 Template Matching

模板匹配是指在图像中寻找与给定模板最相似的区域。常见的度量两个patch相似性的方法有以下几种：
1. **Normalized Cross Correlation (NCC):** 基于NCC度量时，在图像中匹配模板的过程可以通过filter的方式实现。NCC对affine illumination changes具有不变性；
2. **Sum of Squared/Absolute Differences (SSD/SAD):** 计算两个patch在向量空间中的距离，这两种度量都不具备对affine illumination changes的不变性。
3. **Census Transform:** 对每个patch做如下变换：根据其中每个点的像素值大于/小于patch中心位置的像素值，将该点置为0/1，从而将每个patch转换为二进制编码。随后，基于Hamming distance可以非常高效地度量两个patch的相似性，且对affine illumination changes具备不变性。
总体来说，模板匹配要求待搜索物体在图像中呈现的表观（包括orientation, scale, illumination等）和模板具有高度相似性。

## 3.3 Point Feature Detection & Matching

从不同图像中提取特征点并进行匹配是构成VO的重要步骤。整个过程可分为以下3个部分：
1. **如何检测特征点？**
检测方法要具备**复用性**：能在不同的geometric (rotation, scale, viewpoint) 和photometric (illumination) changes条件下准确检测到期望的特征点；还要考虑得到的特征点的**可辨识性**：不同特征点之间易于区分，便于后续匹配。
2. **如何描述特征点？**
最常见的描述子是取图像上以特征点为中心的patch。好的描述子要尽可能放大不同特征点的区别，提高可辨识性。
3. **如何匹配特征点？**
包括基于描述子来度量特征点相似性的方法（前面Template Matching部分所述），和根据相似性匹配特征点的规则。

接下来介绍两种经典的特征点提取和匹配方法。

### 3.3.1 Harris corner detector

角点是指图像中2条或更多条edges的交点。角点检测具备很好的复用性，但可辨识性不如后面要介绍的blob. 
**Harris角点检测的物理意义:** 对于给定pixel，取以其为中心的patch，考察该patch中横向、纵向和两条对角线方向，取四个方向中梯度变化之和最小的值，作为该pixel点上的corner measure。换句话说，认为以角点为中心的patch在各个方向上的梯度变化都应该比较大。
Harris角点检测一般分为如下步骤：计算图像上每点x和y方向的梯度 $\rightarrow$ 计算图像上每点的corner measure $\rightarrow$ 过滤掉低于给定阈值的点 $\rightarrow$ 利用NMS寻找局部corner measure最高的点作为角点。
Harris角点检测对orientation具有不变性（各个方向的梯度变化都要考虑），对affine illumination changes也具有不变性（得益于NMS）；但对scale变化不具备。

随后，可以对检测到的角点提取周围patch并基于SSD相似性度量进行特征点匹配；也可以对接其他描述子和匹配方法。

### 3.3.2 SIFT

SIFT (Scale Invariant Feature Transform) 包含一整套特征点检测、描述和匹配的方法，接下来分别介绍。
1. **SIFT detector:** 首先将图像缩放到不同尺度，堆叠成pyramid；随后采用DoG (Difference of Gaussian) 来近似LoG (Laplacian of Gaussian) 对多个尺度的图像进行滤波；最后选取局部极值点（在相邻的scale和空间区域构成的3维邻域内的极值，包含极大值点和极小值点）作为特征点。
从物理意义上来说，SIFT检测的是blob（区域内部具有相似的像素值，且与周围明显不同）。blob的检测准确性不如角点，但可辨识性更好。
此外，SIFT能同时得到特征点的位置和scale，从而实现对scale的不变性。
2. **HoG descriptor:** 首先选取以特征点为中心的patch，基于patch区域的图像梯度计算dominant orientation，并对patch进行de-rotate；随后，将patch分成$4 \times 4$的16个block，计算各block的梯度直方图并拼接；最后将拼接结果化为单位向量。
de-rotate实现了对rotation的不变性；此外，HoG特征基于梯度进行描述，并且最后将描述子化为单位向量，确保了对illumination的不变性。
3. **Ratio-test matching:** 匹配两组特征点时，对第一组中每个点，不再简单地取最优匹配（另一组中与其相似性最高的点），而要同时确保次优匹配（与其相似性第二高的点）与最优匹配的差距足够大。


# 4. Multi-View Geometry 多视图几何

本章讨论从多个视角的图像重构出场景的3D几何结构的方法。
基于给定条件的不同，本章探讨的问题可分为两类：
1. **3D Reconstruction:** 已知每张图像对应的相机内、外参数，求解场景中点的3D位置。
2. **Structure From Motion (SFM):** 每张图像对应的相机参数未知，同时求解相机参数和场景中点的3D位置。

接下来，首先基于2个视角的情况进行讨论。

## 4.1 Two-View Geometry

### 4.1.1 Two-view 3D reconstruction (Stereo vision)

#### Triangulation
已知相机参数和两张图像中点的对应关系，求解两张图像上每组对应点在3D空间中的位置，这一过程称为triangulation（三角化）.
考虑最简单的情况，两张图像对应的相机（内参数）相同，且沿x轴对齐。此时，3D空间中一点$P$在两张图像上的投影点位于同一高度、不同的水平位置上，其位置之差称为disparity（视差），两个相机原点的距离称为baseline（基线），相机焦距为f，点$P$的深度$Z_P$满足以下关系：
$Z_P = \frac{b f}{d}$, 其中$b=baseline, d=disparity$.

然而，上述情况实际中很难满足，接下来探讨一般形式的三角化。对3D空间中一点$P$，在两个相机平面上的成像点分别为$p_1$和$p_2$，相机参数矩阵分别为$M_1$和$M_2$，有如下关系：
${\lambda}_1 p_1 = M_1 P, {\lambda}_2 p_2 = M_2 P$.
经过变换，$p_1 \times M_1 P = 0, p_2 \times M_2 P = 0$.
其中，两组$p \times M$都是已知的，相当于得到以$P$为未知数的线性方程，可通过least squares approximation的方法求解。
随后，可以再基于上述结果，以优化重投影误差为目标进行non-linear refinement。

#### Correspondance
实际中两张图像上点的对应关系通常是未知的。为确定这种对应关系，引入**epipolar constraint（对极约束）**：
已知3D空间中一点$P$在其中一张图像上的投影点$p_1$，点$p_1$和两个相机原点确定的平面称为epipolar plane，这个平面和两个相机成像平面的交线称为epipolar line，基线和成像平面的交点称为epipole（可以看到，一张图像上的所有epipolar line汇聚于epipole）；$p_1$在另一张图像上的对应点$p_2$在其epipolar line上。寻找对应点的问题从二维搜索变成一维搜索。
特别地，在两张图像对应的相机（内参数）相同、且沿x轴对齐的情况下，epipolar line是两条高度相同的水平直线。为方便处理，有时会通过对两个相机的投影矩阵进行变换，以获得水平的epipolar line，这一过程称为stereo rectification.

### 4.1.2 Two-view SFM

这一节探讨的是已知两张图像中点的对应关系，求解相机参数、以及两张图像上每组对应点在3D空间中的位置的问题。
一般解法是先估计出相机参数，再通过上一节介绍的三角化重建出3D空间中的点。需要注意，由于该问题本身存在**scale ambiguity**，只能得到up to a scale的解。

首先考虑相机内参数已知的情况。给定n组对应点，存在3n+5（以其中一个相机坐标系为世界坐标系，考虑另一相机的相对位姿，包含3个旋转自由度，2个位移自由度；由于scale ambiguity，少1个位移自由度）个未知数和4n个方程；因此，需要至少5组点得到确定的解。这种方法称为**5点法**。
利用对极约束构造的**8点法**是更常用的解法：
对一组对应点$p_1, p_2$，有
${\lambda}_1 p_1 = K_1 [I | 0] P$,
${\lambda}_2 p_2 = K_2 [R | T] P$.
首先基于已知的内参数矩阵，将$p_1, p_2$转为归一化图像坐标$\overline{p} = K_{-1} p$.
根据对极约束，$\overline{p_2}, R \overline{p_1}, T$是共面的，可得
$\overline{p_2}^T \cdot (T \times R \overline{p_1}) = \overline{p_2}^T \cdot ([T_x] R) \cdot \overline{p_1} = \overline{p_2}^T E \overline{p_1} = 0$.
其中，矩阵$E = [T_x] R$称为essential matrix.
可见，每组$p_1, p_2$提供一个方程；确定$3 \times 3$的E矩阵（up to a scale），需要至少8组点。
最后，可以利用SVD分解从得到的E矩阵中提取出R, T. 这种方式会得到4组解，只有一组是合理的（三角化得到的点在两个成像平面前方）。

接下来考虑相机内参数未知的情况，由于$\overline{p} = K_{-1} p$，可得$\overline{p_2}^T E \overline{p_1} = {p_2}^T {K_2}^{-T} E {K_1}^{-1} p_1 = {p_2}^T F p_1 = 0$. 
其中，矩阵$F = {K_2}^{-T} [T_x] R {K_1}^{-1}$称为fundamental matrix.
$3 \times 3$的F矩阵同样可利用8点法求解，但从中抽取K, R, T的过程相对复杂，不做详述。
在利用8点法求解F矩阵时，由于$p_1, p_2$是像素坐标，数值较大，（匹配不准产生的）噪声对结果的影响很大。为解决这一问题，提出了**归一化8点法**，首先将$p_1, p_2$的坐标缩放到 $[-1, 1]$ 之间，再进行求解。


