# CS 131 Lecture 10 Notes

###### Authors: Bryan Gopal, Juan Sosa, Parker Killion, Arafat Mohammed, Nathan James Gugel, Robert Ross, Vy Thai
###### Date: 10/22/2020

# 1 K-Means Clustering

## 1.1 Clustering

We have motivated the desire to segment images in previous lectures. Now we will look at a simple image and its intensity histogram:

 ![](https://i.imgur.com/fINB0vg.png)
 
Observe that the histogram groups pixels in the image into sections that are effectively the true segmentations in the image. As such, we can perform a clean segmentation based on the per-pixel intensities of the image.

However, this strategy quickly degenerates when even the tiniest bit of noise is added into the image, as seen below:

![](https://i.imgur.com/Uxa3q6P.png)

In this noisy version of the same image, it becomes much harder to distinguish pixel group membership due to the overlapping gray and white intensity groups. As such, we will need a more robust segmentation method.

By separating our histogram into a 1-Dimensional array, we are able to see the distribution of points as represented by their intensity levels.

![](https://i.imgur.com/cUwY5Hx.png)
![](https://i.imgur.com/9JWAEEM.png)

Our goal with this representation is to label every point on the histogram according to the center it is nearest to.

The best cluster centers are those that minimize the Sum of Square Distances (SSD) between all points nearest their cluster center $c_i$ such that:

$$SSD = \min \sum_{cluster_i} \sum_{x\in cluster_i} (x-c_i)^2$$

Symbolically, our goal is to find $c^*,\delta^*$ where:

$$c^*,\delta^*= argmin_{c,\delta}\frac{1}{N}\sum^K_j\sum^K_i\delta_{ij}(c_i-x_j)^2$$

Where $c_i$ represents the cluster center $i$, $x_j$ is the relevant point $j$ and $\delta_{ij}$ is whether $x_j$ belongs to $c_i$.

This objective can be achieved in two ways: by assuming we already have the cluster centers and want to find the allocations and by assuming we have the allocations and want to find the cluster centers.

If we have the former, we can calculate the distance of a given pixel $x_j$ to each center $c_i$ and assign it to the corresponding smallest distance.

![](https://i.imgur.com/wYIBXUN.png)

Conversely, if we have the allocations, we can calculate the cluster center by calculating the mean position of all the members in each allocation.

![](https://i.imgur.com/b5Rye1n.png)

## 1.2 K-Means Algorithm
1. Initialize cluster centers $c_1, ...., c_k$ according to an initialization metric
  - Commonly used: random initialization to points
2. compute: $\delta^t$ by assigning each point to the nearest center
$$\delta^t = argmin_\delta\frac{1}{N}\sum^N_j \sum^K_i \delta^{t-1}_{ij}(c^{t-1}_i-x_j)^2$$
3. Compute $\delta^t$ by assigning each point to the center of minimal distance to that point
$$ \delta^t = argmin_\delta\frac{1}{N}\sum^N_j \sum^K_i \delta^{t-1}_{ij}(c^{t-1}_i-x_j)^2$$
4. compute $c^t$: update cluster centers as the means of the points
$$c^t = argmin_c \frac{1}{N}\sum^N_j \sum^K_i \delta^t_{ij}(c^{t-1}_i-x_j)^2$$
5. Repeat steps 2-4 until convergence or stop
Step 2 and 3 can also use different distance measures: 
- Euclidean distance measure: $\text{sim}(x, x') = x^\intercal x'$
- Cosine similarity measure: $\frac{\text{sim}(x, x')}{||x|| \cdot ||x'||}$

### 1.2.1 Limitations of K-means algorithm

-  Convergence to a local minimum}. Although K-means algorithm will eventually converge to a point, that point is not guarantee to be global minimum that we are looking for. This is because the problem is non-convex while K-mean is a heuristic. An example is shown below where K-means gets stuck in a local minimum and gives an in correct result:

![](https://i.imgur.com/lHOVWFk.png)

One solution for this problem is increase the iterations different initial clusters and select the best performance.

- **Better fit for spherical data**.  K-means works best on separable spherical clusters because the mean converges towards the cluster center. 
 
- **Clusters are expected to be around the same size**. The following figures show examples of K-means gives an inaccurate result to a non-spherical cluster.

![](https://i.imgur.com/WpzpFXH.png)
 
![](https://i.imgur.com/tGyWejH.png)

## 1.3 Feature Space

We can group pixels in different ways depending on what we choose as *feature space*. Here are some examples:

- **Grouping pixels based on intensity similarity**.  The *feature space*  for this way of grouping is just the intensity value (1D).

![](https://i.imgur.com/rH99DFv.png)

- **Grouping pixels based on color similarity**. In this case, the *feature space* will be a 3D color value.

![](https://i.imgur.com/3bORxmc.png)

- **Grouping pixels based on texture similarity**.  In this case, the *feature space* will be filter bank response (e.g 24D for the following figure).

![](https://i.imgur.com/g79KwkH.png)

**Smoothing Out Cluster Assignments**

When we assign a cluster for each pixel based on different feature space (intensity or filters), we need to consider their spatial location and proximity to prevent unwanted outliers (shown in following figure).

![](https://i.imgur.com/UJc8HMB.png)

One possible solution is changing how we group the pixels. Now, we can group the pixels based on intensity and location similarity.

## 1.4 K-Means Clustering Results

1. If we are clustering based on intensity or colors then we are essentially doing a vector quantization of the image attributes, hence coherent spatiality. 

2. If we cluster based on the (r, g, b, x, y) values then we enforce more spartial coherency.  

## 1.5 Cluster Evaluation

There are two main ways of evaluating clusters: 

1. Generative way: how well does the algorithm construct points from the clusters. For example, is the center of the cluster a good representation of the data?

2. Discriminative way: how well do clusters compare to labels. In orders words, assuming we do know the labels of the data, we can compute how well the clusters compare to the labels. For example, consider running the algorithm on an image with multiple objects. For this evaluation method, we will compute how well each cluster compare to the labels assigned to the objects. This method can only works with supervised learning where we do know the labels.

## 1.6 Choosing the Correct Number of Clusters

Here we try various values of $k$ (number of clusters) and find the corresponding performance using a validation set. To determine the performance, we find $k$ values corresponding to abrupt changes on the plot which suggests the optimal number of clusters in the data. This technique is known as "knee finding" or "elbow finding". For example, from the graph below, we plot the objective functions for $k$ values from 1 to 6 and we can see that at point $k = 2$ we have an abrupt change which suggests the existence of two clusters in the data.

![](https://i.imgur.com/eRYtTmN.png)

## 1.7 Pros & Cons
**Pros**
1. It is simple to implement
2. It is Efficient
3. Ensures good representation of data by finding clusters that minimize conditional variance.

**Cons**
1. Must experiment with different values of $k$ to find the best one
2. Sensitive to outliers
3. Always finds a local maximum but may not find absolute maximum
4. Assumes that clusters will be spherical
5. All clusters have the same parameters (e.g. distance measure is not adaptive)

![](https://i.imgur.com/HXUyeR3.png)
![](https://i.imgur.com/iAHTdMs.png)

**Usage**
1. Usually used for unsupervised clustering
2. Rarely used for pixel segmentation

# 2 Mean-Shift Clustering
**Mean-Shift Segmentation**
- Mean-Shift Segmentation was one of the most popular segmentation algorithms used before the neural networks
- Still widely used today
- An advanced and versatile technique for clustering-based segmentation
![](https://i.imgur.com/zBlQ0A8.png)
![](https://i.imgur.com/0jzvKU9.png)

**Mean-Shift Algorithm**

- We iteratively search for the mode as follows:
  1.Initialize random seed and window W
  2. Calculate center of gravity (the "mean") of W with $\sum_{x \in W}x H(x)$
  3. Shift the search window to the mean
  4. Repeat from step 2 until convergence
Visually this may look like:
![](https://i.imgur.com/G6uCTxg.png)
![](https://i.imgur.com/qj70Rsa.png)
In practice, we can run this at many different start states.
![](https://i.imgur.com/oyC4AWD.png)
By observing where these different initializations converge, we can get a sense of how to segment the data.
![](https://i.imgur.com/cbPdGM3.png)
A Cluster is the set of points within an Attraction Basin of a mode and the Attraction Basin is the region where all trajectories lead to the same mode.
![](https://i.imgur.com/9dCErYQ.png)
Therefore, in order to do Mean-Shift Clustering/Segmentation we must:
- Find our relevant features (color,gradients, etc.)
- Initialize windows at pixel locations
- Do Mean-Shift until convergence on each window
- Merge each group of windows that have a similar 'peak' or mode
![](https://i.imgur.com/4ocVHov.png)
We can see from the following examples that this process is successful in both segmenting and merging colors, but may fail to segment perfectly with smaller segments.
![](https://i.imgur.com/KM9olWb.jpg)
This algorithm also has a noticeable Computational Complexity where several window shifting may become redundant resulting in several unnecessary computations.
![](https://i.imgur.com/IyISJm5.png)
We can speed up the runtime of this algorithm in a few ways. The first would be a Basin of Attention speedup, where we assign all points within a certain radius of the endpoint to the mode cluster, as seen below:
![](https://i.imgur.com/jJSh70R.png)
We can push this method even further by incorporating a heuristic that assigns all points within a smaller radius that lie on the trajectory of the search path to the mode as well. As seen below, this can drastically reduce the number of points to search and therefore the runtime.
![](https://i.imgur.com/s6MjCQQ.png)
## 2.1 Technical Details
This can be accomplished technically with:
$$\hat{f}_K = \frac{1}{nh^d}\sum^n_{i=1}K(\frac{x-x_i}{h})$$
Where $n$ is the number of data points $x_i \in \R^d$, $h$ is the radius of the kernel (also called the bandwidth) and K is defined as
$$K(x)=c_k*k(||x||^2)$$
Where $c_k$ represents the normalization constant.
Symbolically this algorithm takes the neighborhood $K(x)$ and normalizes it.

## 2.2 Other Kernels
A kernel is a function that satisfies the following requirements:
$$\int_{R^d} \phi(x) = 1$$
$$\phi(x) > 0$$
A Gaussian kernel is extremely common, and is defined as follows: 
$$\phi(x) = e^{-\frac{x^2}{2 \sigma^2}}$$

## 2.3 More Technical Details
We can compute the gradient of $\hat{f}_K$ as follows:
$$\hat{f}_K = \frac{1}{nh^d}\sum^n_{i=1}K(\frac{x-x_i}{h})$$
$$\nabla\hat{f}(x) = \frac{2c_{k,d}}{nh^{d+2}}(\sum_{i = 1}^{n}g(\mid \mid\frac{x - x_i}{h}\mid\mid)^2)(\frac{\sum_{i = 1}^{n}x_i g (\mid \mid\frac{x - x_i}{h}\mid\mid)^2}{\sum_{i = 1}^{n}g(\mid \mid\frac{x - x_i}{h}\mid\mid)^2} - x),$$
where $g(x) = -k'(x)$ denotes the derivative of the selected kernel profile.

The first term in the equation is proportional to the density estimate at $x$ (similar to equation from two slides ago). 

The second term is the mean-shift vector that points towards the direction of maximum density.

Finally, the mean shift procedure from a given point $x_t$ is: 
1. Compute the mean shift vector $m$:
$$(\frac{\sum_{i = 1}^{n}x_i g (\mid \mid\frac{x - x_i}{h}\mid\mid)^2}{\sum_{i = 1}^{n}g(\mid \mid\frac{x - x_i}{h}\mid\mid)^2} - x)$$
2. Translate the density window:
$$x_{i}^{t + 1} = x_{i}^{t} + m(x_{i}^{t})$$
3. Iterate steps 1 and 2 until convergence.
$$\nabla f(x_i) = 0$$

## 2.4 Summary Mean-Shift
- **Pros**
  - General, application-independent tool
  - Model-free, does not assume any prior shape (spherical, elliptical, etc.) on data clusters
  - Just a single parameter (window size h)
    - h has a physical meaning (unlike k-means)
  - Find variable number of modes
  - Robust to outliers
  
- **Cons**
  - Output depends on window size
  - Window size (bandwidth) selection is not trivial
  - Computationally (relatively) expensive (~2s/image)
  - Does not scale well with dimension of feature space
