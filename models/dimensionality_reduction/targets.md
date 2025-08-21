# 降维算法

|      简称      |                             全称                             |                             链接                             |                             参考                             |
| :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      PCA       |                 Principal component analysis                 | [[link]](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA) |                          [[link]]()                          |
|   KernelPCA    |             Kernel Principal component analysis              | [[link]](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA) |                          [[link]]()                          |
| IncrementalPCA |           Incremental Principal component analysis           | [[link]](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA) |                          [[link]]()                          |
|   SparsePCA    |             Sparse Principal Components Analysis             | [[link]](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA) |                          [[link]]()                          |
| FactorAnalysis |                        FactorAnalysis                        | [[link]](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html#sklearn.decomposition.FactorAnalysis) |                          [[link]]()                          |
|      ICA       |                Independent component analysis                | [[link]](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA) |                          [[link]]()                          |
|      MDS       |                  Multi-dimensional Scaling                   | [[link]](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS) | [[link]](https://nbviewer.org/github/drewwilimitis/Manifold-Learning/blob/master/Multidimensional_Scaling.ipynb) |
|     Isomap     | Non-linear dimensionality reduction through Isometric Mapping | [[link]](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#sklearn.manifold.Isomap) |                          [[link]]()                          |
|      LLE       |                    Local Linear Embedding                    | [[link]](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn.manifold.LocallyLinearEmbedding) | [[link]](https://nbviewer.org/github/drewwilimitis/Manifold-Learning/blob/master/Locally_Linear_Embedding.ipynb) |
|      TSNE      |         T-distributed Stochastic Neighbor Embedding          | [[link]](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE) |      [[link]](https://github.com/mxl1990/tsne-pytorch)       |

## 基本介绍

复现的时候统一使用Torch，并参考`scikit-learn`的接口形式，对于降维算法留出`fit_transform`作为算法执行的通用接口即可。

上述算法中主要包括了三种：

1. 主成分分析及其变体：PCA, KernelPCA, IncrementalPCA, SparsePCA。
2. 因子分析算法：FactorAnalysis, ICA。
3. 流形学习算法：MDS, Isomap, LLM, TSNE。

对于上述算法，接口的形式可以参考`scikit-learn`，但由于该库的封装度过高，很多情况下很难直接从中摘取代码，所以可以在网上搜：

> 算法名字 torch github

等关键词进行检索相关的库，流形学习方法我给出了参考网页，里面有一些公式可以参考。或是直接参考机器学习西瓜书里面的公式。

## 基本要求

1. PCA及其变体都算是比较有代表性的，因此最好全部都能复现出来；
2. 因子分析法的话ICA比较有代表性；
3. 流形学习也算是降维算法中不可或缺的一部分，因此最好能复现出MDS和Isomap。