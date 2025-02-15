# RajaRavi

This is an image processing project inspired by the idea that paintings simplify the reference material: reducing colors and refining shapes. To do this, I used a hierarchical agglomerative clustering algorithm. In this algorithm, you start with one cluster for every data point and then progressively combine the two closest clusters (based on some metric) until you're left with a single cluster that contains all of the data. The big upside of this approach is that you don't have a predetermined number of clusters, and you can stop the agglomeration at any point when you think the clusters look reasonable. This algorithm is also able to represent some cluster shapes that aren't possible with algorithms like k-means or k-mediods. Roughly these are shapes where data points are close to some other point in the cluster but not necessarily close to the average (either mean or medioid) of the cluster. A good example is an S shape. I thought that this would make the agglomerative algorithm a good fit for paintings.

In the image case, our data points are pixels, and our clusters are groups of pixels (which I call **patches**). Additionally, we only consider combining neighboring patches.

The big hurdle with the agglomerative algorithm is that it's resistant to parallelization because you choose the closest matches one at a time and the inter-cluster distances change as you're combining clusters. This makes it difficult to scale the algorithm to a large number of points like the number of pixels in an image. It certainly helps that you only need to consider neighbors cause the number of neighbors is roughly constant instead of scaling with the size of the dataset, but it's still prohibitively slow past a certain number of pixels. Although you can get decent results by just reducing the resolution before running the algorithm.

To get around this problem, I did bulk merges i.e. instead of finding the single closest pair at each iteration, find the say 10% closest pairs and combine all of them. This significantly reduces the number of iterations as the number of clusters decrease exponentially over time instead of linearly. After a reasonable number of patches are left, we switch back to the single merge at a time regime and run the algorithm as normal.

The algorithm does run in around a minute, and it does produce nice looking images, but what's clear is that there is definitely more to painting than just simplifying colors. My next step was attempting to introduce some concept of soft vs hard edges, and I wrote some post processing functions for this, but they don't help a ton.

After this phase, I explored some much more parallelizable and therefore faster approaches in another project linked here: 
## [Clyde](https://github.com/christyjestin/clyde/tree/main)


Anyway, here are some examples:
### Original Image:
![image](https://github.com/user-attachments/assets/9d84b37e-fc82-492e-9e17-a6b4ceaa9eca)
### About 2400 Patches:
![image](https://github.com/user-attachments/assets/5935b256-9f99-4d77-98c0-c8ac78045040)
### About 700 Patches:
![image](https://github.com/user-attachments/assets/cc97a137-8fba-4051-9c5d-a89ecfb6de69)
### 80 Patches:
![image](https://github.com/user-attachments/assets/a2becaae-3922-4c76-8ad6-698ff7b18dc3)

### Original Image
![image](https://github.com/user-attachments/assets/23fc8041-30e9-4771-9982-8921e06efe3c)
### 1000 Patches:
![image](https://github.com/user-attachments/assets/aafb3c42-a231-4ca3-8249-23d4cdeb9e76)
### 400 Patches:
![image](https://github.com/user-attachments/assets/ab5576e9-5034-403e-8481-291a2f5a1e9a)
### 120 Patches:
![image](https://github.com/user-attachments/assets/8cb12964-539e-4e8a-a6dd-798beba762df)


### Randomized Post Processing of 1000 Patches
![image](https://github.com/user-attachments/assets/80b717c7-3337-46ec-9817-b406867201ed)
