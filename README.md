# Cell-classification-based-on-mRNA-features
## Classifying different point patterns
We used a Morpho Dilation layer, first introduced by Santiago Velasco, to classify different point patterns. This layer works like a convolutional layer but uses max plus algebra. Instead of multiplication, it adds each weight to its corresponding pixel value, and then the maximum result is passed to the next layer.
 
<img src="https://github.com/mehranmo93/Cell-classification-based-on-mRNA-features/blob/master/IMG/Dilation.png" width="300">


But the logic behind this model is mathematical concept, Choquet capacity.
Choquet capacity T<sub>X</sub>(K), or denoted simply by T(K), is the probability that the compact set K ("closed" and "bounded") hits the set X, Which is equal to 1 minus the probability of k being subset of X

<img src="https://github.com/mehranmo93/Cell-classification-based-on-mRNA-features/blob/master/IMG/Choquet_Capacity.png" width="400">

The function T has the following properties: 

* T is bounded with 0 ≤ T ≤ 1 and T(∅) = 0
* T is increasing in the sense T(K) ≤ T(K ∪ K<sub>0</sub>)

We view the kernel as playing the role of set K, and the images as playing the role of set X in the dilation layer. Additionally, we consider that the kernel values can either be 0 or -1, while the image pixels can take the values 0 if it was in X<sub>c</sub> and 1 if it was in X. When the kernel touches X, it yields 1; otherwise, it yields 0. By sliding the kernel through the image and calculating the average, we can determine T(K) for that set. Here you can see the schematics of morpho dilation layer we used.

<img src="https://github.com/mehranmo93/Cell-classification-based-on-mRNA-features/blob/master/IMG/Schematic_Dilation.png" width="400">

This is a model with just one Dilation layer you can add more layer if you want. This model has a good result on classifying Poisson point pattern and Neyman Scotte point pattern but on cell data the accuracy is not good enough. So, we have to use some trick!

![Poisson point pattern and Naman-Scott point pattern](https://github.com/mehranmo93/Cell-classification-based-on-mRNA-features/blob/master/IMG/point_pattern.jpg)

## Problem definition for genetic data :
Gene expression control is a crucial mechanism that regulates various cellular processes, and it heavily relies on the abundance and localization of RNA within cells. The dataset obtained from numerous experiments provides valuable information on cell populations and can be described as point patterns, where each point is associated with various features. There are 8 distinct classes in this dataset, each characterized by different mRNA distributions. However, distribution is not the sole factor influencing classification; other crucial factors are at play. Factors such as the distance of each mRNA point to the cell membrane, the distance to the nuclear membrane, the location within the nucleus, or being outside the nucleus are significant. The dilation model is unable to capture these pieces of information. Here are the definitions for the various classes:

* **Cell-edge**: RNA molecules prominently attached to the cellular membrane. 
* **Extranuclear**: This class is marked by the absence of RNA within the nucleus. 
* **Foci**: Within this class, RNA molecules tend to form small clusters or foci. 
* **Intranuclear**: Here, RNA resides exclusively within the nucleus. 
* **Nuclear-edge**: Cells of this class exhibit RNA molecules adhering to the nucleus membrane. 
* **Pericellular**: In these cells, RNA molecules tend to be closer to the cell membrane.
* **Perinuclear**: RNA in this class tends to congregate close to the nucleus membrane.  
* **Random**: RNA molecules scattered without a discernible pattern.

<img src="https://github.com/mehranmo93/Cell-classification-based-on-mRNA-features/blob/master/IMG/cell_images.png" width="400">

To capture this information, we employed a distance function in Python. This function enabled us to effectively gather the necessary details. Below, you can observe images generated by the distance function. These masks are applied to the images, assigning greater weights to points in proximity to the nuclear membrane or the cell membrane.

![](https://github.com/mehranmo93/Cell-classification-based-on-mRNA-features/blob/master/IMG/mask.jpg)

A) Cell image with cell membrane, cell nucleus membrane, and mRNA spots. B) Distance mask to capture points outside the nucleus and close to the cell membrane or nucleus membrane. C) Distance mask to capture points inside the nucleus and close to the nucleus membrane. D & E) Distance mask to capture points close to the cell membrane.

After computing the weighted average of the points, we can concatenate it with the results from the dilation layer and then pass it through the Softmax function following a linear transition. Therefore, the final model incorporating the distance function will appear as follows:


<img src="https://github.com/mehranmo93/Cell-classification-based-on-mRNA-features/blob/master/IMG/schematic_dil_distance.png" width="400">

The final accuracy dilation layer without the distance function is 64%  and you can see the confusion matrix for that. 

![](https://github.com/mehranmo93/Cell-classification-based-on-mRNA-features/blob/master/IMG/confusion_matrix.png)

As it is clear there are some difficulties in classifying:
*	Cell_edge, extranuclear & random
*	Intranuclear & nuclear_edge, Perinuclear
*	Pericellular & perinuclear
After adding distance function, the accuracy increases to 92% and you can see the confusion matrix for that:

![](https://github.com/mehranmo93/Cell-classification-based-on-mRNA-features/blob/master/IMG/final_confusion_matrix.png)

The only remaining issue lies in the classification of the cell_edge and extranuclear classes, stemming from the conversion of the original 3D images to 2D. Since the morpho dilation layer is designed for 2D and adapting it to 3D would involve increased computational costs and slower processing, we opted to convert the 3D images to 2D. As a result, these converted images now exhibit striking similarities, as illustrated in the images below:

![cell_edge_extranuclear](https://github.com/mehranmo93/Cell-classification-based-on-mRNA-features/blob/master/IMG/cell_edge_extranuclear.png)

The next work could be converting 2D dilation to 3D dilation to address the problem.

Code files contain:

400_points_ppm_i_scale : Convert 3D point pattern to fixed size 2D images.
Mask_function : Create a mask images with the distance function.
Dil_400_conf_dist4 : The main code to train the network.

