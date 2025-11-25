# What?
A tool to download images of a city from Mapillary and stitch them together into a 3D model of that city using [Gaussian Splatting](https://www.youtube.com/watch?v=VkIJbpdTujE)

# Why?
To create realistic city replicas that can be used to simulate traffic, pedestrians, and down the line more complex social interactions within the model city. 

# How?
As of now: upload N images to a Lambda Cloud instance. There, use COLMAP with CUDA support to feature match images and solve for camera positions  (assumes mapillary does not provide camera positions in advance). The binaries with camera models, poses, and 3D points are passed through Brush on Mac using Gaussian Splatting. A .ply file with the final positions and covariances is made that can be vizualized in 3D with Brush also. See below for output examples for a 15k iterations run on images of Chinatown, San Francisco.

![v.01 - 15k iterations - Image 1] (../assets/v_0.1_15k_image_1.png)

![v.01 - 15k iterations - Image 2] (../assets/v_0.1_15k_image_2.png)      
