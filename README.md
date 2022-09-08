# K-means and Fuzzy C-means

The program takes the data file as a command line argument. Choice of k-means or c-means is
made interactively, as well as r value and number of clusters (k). The chosen algorithm is run r
times, and for each a solution is returned. The solution with the minimum within-cluster sum of squares (WCSS) value is chosen
and displayed at the end. The other solutions are displayed in a plot for each iteration, when
that iteration value is divisible by the ‘interval’ value (in order to decrease memory usage when
the algorithms approach 100 iterations).

The following plots show the progress of the algorithm through different iterations. However,
since these were all run with r = 10, they are not guaranteed to be from the same sequence
(e.g. iteration 7 may not be contiguous with iteration 4 as they are from different r-runs). In any
case, it shows the progress of the minimization of the WCSS over the iterations.

## K-Means 
For r = 10:
K = 3
![K 3](https://github.com/IntoTheVortex/ML-Prog3-k-c-means/blob/master/images/kmeans_k3.PNG?raw=true)

K = 4
![K 4](https://github.com/IntoTheVortex/ML-Prog3-k-c-means/blob/master/images/kmeans_k4.PNG?raw=true)

K = 5
![K 5](https://github.com/IntoTheVortex/ML-Prog3-k-c-means/blob/master/images/kmeans_k5.PNG?raw=true)

K = 6
![K 6](https://github.com/IntoTheVortex/ML-Prog3-k-c-means/blob/master/images/kmeans_k6.PNG?raw=true)

K = 7
![K 7](https://github.com/IntoTheVortex/ML-Prog3-k-c-means/blob/master/images/kmeans_k7.PNG?raw=true)

As shown in the plots, the number of clusters was inversely correlated with the best WCSS
values, so that the largest number of clusters attempted, 7, had the best WCSS value at ~550.

## Fuzzy C-Means 
For r = 10:
K = 3
![K 3](https://github.com/IntoTheVortex/ML-Prog3-k-c-means/blob/master/images/cmeans_k3.PNG?raw=true)

K = 4
![K 4](https://github.com/IntoTheVortex/ML-Prog3-k-c-means/blob/master/images/cmeans_k4.PNG?raw=true)

K = 5
![K 5](https://github.com/IntoTheVortex/ML-Prog3-k-c-means/blob/master/images/cmeans_k5.PNG?raw=true)

K = 6
![K 6](https://github.com/IntoTheVortex/ML-Prog3-k-c-means/blob/master/images/cmeans_k6.PNG?raw=true)

As shown in the plots, the number of clusters was inversely correlated with the best WCSS
values, so that the largest number of clusters attempted, 7, had the best WCSS value at ~640.
Interestingly, the fuzzy c-means algorithm had consistently higher/worse WCSS values for the
same number of clusters compared to the k-means algorithm.
