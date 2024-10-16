# Project
This repository contains the code for developing a cross-validation method eager to tackle an oversampling problem in Machine Learning. The method implements KDE or Kernel Density Estimation to oversample minority class examples
from a biological dataset.
The method runs on several binary and multiclass datasets from the [CUMIDA repository] (https://sbcb.inf.ufrgs.br/cumida#datasets) by considering a discrete imbalance ratio between the classes. The set of selected datasets for the experiment can be downloaded from the following [link] (https://drive.google.com/drive/folders/1zB5xFM9qrurKZjgKxmVpv4QKcaXUCSJY?usp=drive_link)

## imbalance problem

## Kernel Density Estimation
Kernel Density Estimation, or KDE, is a non-parametric method for estimating the probability density function of a set of random variables. Unlike parametric methods that estimate parameters by maximizing the Maximum Likelihood of obtaining the current sample, KDE estimates the density distribution directly from the data. Intuitively, KDE can be simplified to a histogram representation, with the main difference being that data drives the block’s location. Instead of having a sum of boxes, a Kernel density estimator is composed of a sum of 'bumps', whose width is controlled by a smoothing parameter usually called 'bandwidth'\cite{SilvermanDENSITYANALYSIS}. These bumps are centered on the point they represent, leading to a better representation of the underlying data. Moreover, the more the points are in a given location, the more the contribution of those points is for the resulting density estimation. Thus, indicating the probability of seeing a point in that location.

\begin{equation}\label{KDE}
    f(x) = \frac{1}{n}\sum_{i=1}^{n}K_h(x-x_i)
\end{equation}

## Approach



