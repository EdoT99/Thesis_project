# Project
This repository contains the code for developing a cross-validation method that is eager to tackle an oversampling problem in machine learning.  
KDE is exploited as an alternative oversampling tool to augment imbalanced biomedical datasets. IN this context, several machine-learning algorithms (default setup) are used to assess the efficacy of the technique using SMOTE as a gold standard for comparison. 
The method runs on several binary and multiclass datasets from the [CUMIDA repository] (https://sbcb.inf.ufrgs.br/cumida#datasets) by considering a discrete imbalance ratio between the classes. The set of selected datasets for the experiment can be downloaded from the following [link] (https://drive.google.com/drive/folders/1zB5xFM9qrurKZjgKxmVpv4QKcaXUCSJY?usp=drive_link)


## Kernel Density Estimation
Kernel Density Estimation, or KDE, is a non-parametric method for estimating the probability density function of a set of random variables. Unlike parametric methods that estimate parameters by maximizing the Maximum Likelihood of obtaining the current sample, KDE estimates the density distribution directly from the data. 
<p align="center">
   $$f(x) = \frac{1}{n}\sum_{i=1}^{n}K_h(x-x_i)$$
</p>

Where, $\mathit{K}$  is the kernel function, $\mathit{h}$  is the bandwidth parameter, and $\mathit{n}$  is the number of observations. Intuitively, the true value of $\mathit{f(x)}$ is estimated as the average distance from $\mathit{x}$  to the sample data points $x_i$.  Given a continuous random variable, KDE produces a curve, which is an estimate of the underlying distribution of this data.

<p align="center">
  <img src=https://github.com/user-attachments/assets/77c6285b-f25d-4ae9-97a4-e795ce9995d5\>
</p>


Moreover, the kernel estimator depends on two parameters, i.e. the kernel function $\mathit{K}$  and the bandwidth $\mathit{h}$.
The first refers to how the data points are weighted depending on the type of kernel function. There are plenty of available kernel functions: epanechikov, biweight, triangular, gaussian, and rectangular kernels

<p align="center">
  <img src=https://github.com/user-attachments/assets/7a29f9bf-2a3f-49af-a738-f3d34c6f833f\>
</p>

## 2-Dimensional KDE visualization 


![KDE_ORI20240914_1851](https://github.com/user-attachments/assets/2cbfdb55-a1d8-454c-bf16-81d5d944fd6c)

![KDE_OVSAP20240914_1748](https://github.com/user-attachments/assets/69d0549a-3e32-465c-a618-09cf2345d46d)

## Approach

## Validation
The approach is tested on a stratified 10-fold cross-validation (CV) comparing algorithms classification performances trained on the original, KDE-oversampled, and SMOTE-oversampled partitions.

<p align="center">
  <img src=https://github.com/user-attachments/assets/a5e577fe-6a5e-4670-a34a-2479622d3ffc\>
</p>




