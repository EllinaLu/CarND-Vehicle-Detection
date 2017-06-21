
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
[image1]: ./output_images/image0003.png_hog.png
[image2]: ./output_images/test6.jpg_beforeheat.png
[image3]: ./output_images/test6.jpg_output.png
[image4]: ./output_images/test6.jpg_heat.png
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

For each of the input files, I converted the image into YUV colorspace.

Then I used the function skimage.feature.hog to extract HOG features from the converted images for all three channels.

Below is an example from the training set. The left most image is the original training image. The other three images
are HOG visualizations for all three channels. 
You can roughly make out the shape of the car from the first channel.

![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that models trained in the RGB colorspace are affected by shades and brightness of the image.
So I converted the images to YUV instead, where the brightness is separated from the color.
However, through experimentation, I found that using all three channels of the YUV colorspace gave better results than just using a single channel.
Also, making the number of orientation bins for HOG bigger will make the classifier more prone to errors because a little bit of angle difference might result in different features, and possibly different prediction.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM model using the vehicle and non-vehicle images in function train_model (lines 437~493 of images.py)
First, HOG, histogram, and spatial features are extracted from both the vehicle and non-vehicle images.
Then the features are stacked and normalized.
The images are then shuffled and split into training set and test set using the train_test_split function with a ratio of 0.2.
Then I used the obtained training set to fit a SVC model.
Lastly, I used the test set to verify the accuracy of the trained SVM model, which achieved > 99% accuracy.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the function find_cars (lines 318 ~ 404 in images.py).
First the region of interest is extracted, which usually locates in the lower half of the image.
Then I converted the region into YUV colorspace.
Then I resized the region according to the input scale to accomodate different size of search windows.
Then HOG features are calculated for the region.
Then I loop over the windows (sliding the window through the interested region). In each loop, the HOG features for the current window
is extracted and concatenated along with spatial features and histogram features calculated for the current window.
Then normalization is applied before using the trained model to predict whether a vehicle exists.
I used 2 different sizes of windows, and each window has a different searching range in the input image.

![alt text][image2]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  
To increase the performance of my classifier, I implemented the HOG subsampling approach to prevent the HOG features from being calculated repeatedly for each window searched.
I also implemented a heat map and find a window from all the windows that gave a positive result.
Here are some example images:

![alt text][image3]
![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I implemented the filter for false positives in the function processImage, around lines 485~519 of video.py.

First, I used a heat map to find bounding boxes of cars for one frame, and applied a first threshold on the minimum number of windows needed to be determined as a car.
This is to combine the overlapping boxes to find where a real car is , and where the potential real boundary of the car is.
This first threshold is different depending on the number of overlapping boxes predicted in that frame.

Then I extracted the resulting position of the above bounding boxes and append them to found_box_history.
This list is used to save the history position of found cars over a few frames.

Then I flatten the heat map (containing the heat of cars for the current window from the first step) by setting all pixels > 0 to 1.
In this way, the contribution of potential cars from each frame will be 0~1.

Then I accumulated the results from the previous few frames, and applied a threshold on this new heat map.
In this way, I am hoping to filter out false positives that only exist for two or three frames.
The resulting bounding boxes are what is drawn on top of the output image.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

When I tried my pipeline on the test video for the first time, there were a lot of false positives, even though the test images result OK.
Then I found that I was using the heat map in a wrong way.
I was accumulating all the positive sliding windows in the heat map for some consecutive few frames.
But if each frame contributed a varying number of heats on the pixels, it would be very hard for me to determine a fixed threshold to apply to the heat map.
So instead I used a heat map to find bounding boxes of cars for one frame, and set all pixels > 0 to 1, thus flattening all contributions of a frame to 1 for each pixel.
Then I labeled and saved the result compare with the next few frames.
In this way, the threshold is easier to determine, since it will mean the number of frames that pixel was detected as car.

My pipeline might fail when two cars are too close to each other, the heat map approach will consider them together as one bounding box.
Maybe a deep learning network can help to distinguish the two cars, or tracking the path of individual cars can help detect that the cars are driving on their own paths.

