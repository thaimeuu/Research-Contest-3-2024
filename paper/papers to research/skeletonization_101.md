## SKELETONIZATION 101

**Skeletonization** is a **computer vision** and **image processing** algorithm used to **extract** a simplified representation of an object or shape in an image while **preserving** its essential structural features

## Why skeletonization

1. **Feature extraction**: Skeletonization helps in extracting the essential features of an object, reducing complex shapes into simplified forms
2. **Pattern Recognition (PR)**: Skeletonization is often used as a preprocessing step for **PR**, making it easier to analyze and compare shapes
3. **Shape analysis**: Skeleton images are useful in tasks such as **shape analysis** and **classification**
4. **Image compression**: Skeletons provide a concise and compressed representation of the object which helps in the task of **Compression**

## How skeletonization works

1. **Thinning algorithms**: Skeletons are often achieved through **thinning algorithm**, which iteratively remove pixels from the boundary of an object until a one-pixel-wide skeleton remains
2. **Distance transform**: Some methods use **distance transform** to assign each pixel a value based on its distance to the nearest background pixel. Skeleton points are identified based on local maxima in the distance map

## What algorithm in Skeletonization is like K-Means in Clustering i.e. classic approach

**Zhang-Suen thinning algorithm** in 1984