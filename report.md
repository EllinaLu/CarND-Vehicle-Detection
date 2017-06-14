
# Vehicle Detection Project #
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./output_images/test6.jpg_beforeheat.png
[image4]: ./output_images/test6.jpg_output.png
[image5]: ./output_images/test6.jpg_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function extract_features in file images.py (lines 90~141 of images.py).  

I started by reading in all the `vehicle` and `non-vehicle` images in the function train_model in images.py.

I used the function skimage.feature.hog to extract HOG features from the images read.

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that models trained in the RGB colorspace are affected by shades and brightness of the image.
So I converted the images to YUV instead, where the brightness is already specified by the Y channel.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM model using the vehicle and non-vehicle images in function train_model (lines 407~461 of images.py)
First, HOG, histogram, and spatial features are extracted from both the vehicle and non-vehicle images.
Then the features are stacked and normalized.
The images are then shuffled and split into training set and test set using the train_test_split function with a ratio of 0.2.
Then I used the obtained the training set to fit a SVC model.
Lastly, I used the test set to verify the accuracy of the trained SVM model.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the function find_cars (lines 296 ~ 374 in images.py).
First the region of interest is extracted, which usually locates in the lower half of the image.
Then I converted the region into YUV colorspace.
Then I resized the region according to the input scale to accomodate different size of search windows.
Then HOG features are calculated for the region.
Then I loop over the windows (sliding the window through the interested region). In each loop, the HOG features for the current window
is extracted and concatenated along with spatial features and histogram features.
Then normalization is applied before using the trained model to predict whether a vehicle exists.
I used 2 different sizes of windows, and each window has a different searching range in the input image.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  
To increase the performance of my classifier, I implemented the HOG subsampling approach to prevent the HOG features from being calculated repeatedly for each window searched.
I also implemented a heat map and find a window from all the windows that gave a positive result.
Here are some example images:

![alt text][image4]
![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

