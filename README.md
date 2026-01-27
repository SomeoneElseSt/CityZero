# What?
A tool to download images of a city from Mapillary and stitch them together into a 3D model of that city using [Gaussian Splatting](https://www.youtube.com/watch?v=VkIJbpdTujE)

# Why?
To create realistic city replicas that can be used to simulate traffic, pedestrians, and down the line more complex social interactions within the model city. 

# How?
As of now: upload N images to a Lambda Cloud instance. There, use COLMAP with CUDA support to feature match images and solve for camera positions  (assumes mapillary does not provide camera positions in advance). The binaries with camera models, poses, and 3D points are passed through Brush on Mac using Gaussian Splatting. A .ply file with the final positions and covariances is made that can be vizualized in 3D with Brush also. See below some output examples from a 15k iterations run on images of Chinatown, San Francisco.

![v.01 - 15k iterations - Image 1](./assets/sf_findi_run/v_0.1_15k_image_1.png)

![v.01 - 15k iterations - Image 2](./assets/sf_findi_run/v_0.1_15k_image_2.png)      

Later I trained a 30k iteration version too that made some interesting internal mini-representations of parts of the training images, like the Transamerica pyramid shown below in a small set of clustered gaussians.

![v.01 - 30k iterations - Image 1](./assets/sf_findi_run/v_0.1_30k_image_1.png)


---

# Logs

**January 17, 2026**

After waiting for almost a month for the ~600k+ images or so to download, I realized a new type of algorithm is needed to make this work.

Logically, it's rebudant to, for example, compare an image in the South of a city with one in the North. If [as they most likely will] are not related, it's wasted computation, and if they are, it's a false positive.

So I'm working on using geofencing and clustering to make this work. I had to wait another week for Mapillary's API to fetch the latitude/longitude of each image I had downloaded before, and while running some tests, I ran into something interesting (if not catastrophic):

![Scatter Plot showing the coordinates for each downloaded image in the dataset](./assets/scatter_plots/sf_scatter.png)

For whatever reason, my initial downloading algorithm didn't get all the images of the city. It missed a huge chunk of the Financial District and only caught mini-clusters within. 

The good news is that it did download Sunset district OK, which should be enough for what I wanted to test. 

Contextualized, this means there is so much more potential once all the images are queried from Mapillary. There, each road (and in particular the FinDi) is sprawling with images. If my initial approach works, it'll be tougher to do the raw data work than the reconstruction. 

Now I'll have to change some things so the experiments work on the Sunset, but asides that, I'm very happy I ended up going for the harder approach and pulling the coordinates. This explains why a lot of my earlier experiments focused on the FinDi failed. 

**January 18, 2026**

I've now made a new set of boxes of the sunset district based on what I found yesterday. They seem promising!

![Scatter Plot showing images within the Sunset district segmentation boxes](./assets/scatter_plots/sunset_scatter.png)

**January 19, 2026**

Mostly worked on infra -- I injected the GPS coordinates of each image I had gotten from Mapillary's API into the database I got after feature extraction. It should help COLMAP match images in box edges using a spatial matcher (although it does make me wonder if using a spatial matcher standalone could work as well).

I also realized COLMAP does custom matches using a .txt list of <img1> <img2>, so there was no need to copy images into a folder per box. It would've been enough with creating the .txt files (I'll modify the script for box-making to reflect this.).

In either case, both of the above are done. Now I just have to wire up the script to put it all together. 

**January 27, 2026**

After being away for a sabbatical I've come back with some new ideas. 

First, I'm still going to use boxes for localized reconstructions. I think this is good because if I ever want to update a box, I won't have to re-feature match and reconstruct the entire dataset. There are ways to register images into an existing reconstruction in COLMAP.

Second, I'm not going to use an exhaustive matcher within the boxes. I'll use an spatial matcher. It's [way] less computationally expensive. Empirically, a box with 11000 images only uses 300k+ or so comparisons. An exhaustive matcher would've needed 121 Million. Not great.

Third, I'll only use an exhaustive matcher with query expansion to connect images at the fringes of each box. This should, on paper, provide enough conectivity during reconstruction, and it aligns with how the Rome paper succeeded connecting diseparate clusters. 

I ran spatial feature matching on each box. I now have to work on matching at the fringes. Once that's done I'll finally be able to start reconstructing. I currently have got 1.1M matches or so, so I'm aiming to keep it under 2M, just to see how long it takes on Lambda. 

I'm not 100% sure of wether this approach is optimal (although I am confident it'll work), because, given that I do know each image's GPS coordinates, I could theoretically just use a spatial matcher always. Both for in-box matching and at the fringes.

However, given that COLMAP spatial_matcher doesn't ingest custom lists, and reconnecting boxes isn't just about connecting close images (imagine for instance, reconnecting a valley. If only close images were feature matched, images at the top of the valley with new information might be ignored. Query expansion solves this for sure.), it more likely than not is better to use some version of exhaustive matching at the fringes. 

Down the line, I might only use query expansion with a Vocab tree or Deep Clustering. We'll see.

A fun note: I asked ChatGPT, based on this log, what the likely profile of the person writing it may be. It said it might be of someone "slightly masochistic (in a good way)", which I found rather hilarious. 

