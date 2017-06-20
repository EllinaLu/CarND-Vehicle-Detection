import argparse
import numpy as np
import cv2
import glob
import time
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pylab as pylab
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pathlib import Path
import pickle
import collections
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


#parameters
#window_size_1 = (64, 64)
scale1 = 1
y_start_stop_1 = [350, 500] # Min and max in y to search in slide_window()
#window_size_2 = (128, 128)
scale2 = 2
#scale2 = 2
y_start_stop_2 = [350, 650] # Min and max in y to search in slide_window()
#window_size_3 = (180, 180)
#y_start_stop_3 = [350, 720] # Min and max in y to search in slide_window()


hog_feat = True # HOG features on or off
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient_bins = 9 # HOG orientations
pix_per_cell = 8
cell_per_block = 2
hog_use_channel = "ALL" # Can be 0, 1, 2, or "ALL"

spatial_feat = True # Spatial features on or off
spatial_size = (16, 16) # Spatial binning dimensions

hist_feat = True # Histogram features on or off
hist_bins = 16        # Number of histogram bins

pickleVar = {}
modelFilePath = "model.p"
path = Path(modelFilePath)
#Load the model file if it exists
if path.is_file() == True :
    print ('Loading trained model from file...')
    modelFile = open( modelFilePath, "rb" )
    pickleVar = pickle.load( modelFile )
    clf = pickleVar["clf"]
    X_scaler = pickleVar["scaler"]
    modelFile.close()
else:
    print ('Error!  No existing model to use!!!')
    exit

#global variables
frameCount = 0
found_box_history = collections.deque(maxlen = 7)
heat_threshold = 6

#parameters = {'kernel'=['linear', 'rbf'], 'C'=[1, 10]}

#####################################################################################
# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :,2], bins=nbins, range=bins_range)
    # Generating bin centers
    #bin_centers = (rhist[1][0:len(rhist)-1] + rhist[1][1:])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    #return rhist, ghist, bhist, bin_centers, hist_features
    return hist_features
    
    
# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space='BGR', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: feature_image = np.copy(img) 
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features
    
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image  = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
        return features
        
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='BGR', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        #image = mpimg.imread(file)
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'BGR':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='BGR', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'BGR'
    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)
       
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    
    # Compute the number of pixels per step in x/y
    x_step_pix = np.int(xy_window[0]*(1 - xy_overlap[0]))
    y_step_pix = np.int(xy_window[1]*(1 - xy_overlap[1]))
    
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    x_num_win = np.int((xspan-nx_buffer)/x_step_pix)
    y_num_win = np.int((yspan-ny_buffer)/y_step_pix)
    
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    for ys in range(y_num_win):
        for xs in range(x_num_win):
            # Calculate window position
            startx = xs*x_step_pix + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*y_step_pix + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='BGR', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    #8) Return windows for positive detections
    return on_windows

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def convert_color(img, conv='BGR2YUV'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'BGR2YUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cells_per_step = 2, xy_window=(64, 64), spatial_feat=True):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='BGR2YUV')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    nxblocks_per_window = (xy_window[0] // pix_per_cell) - cell_per_block + 1
    nyblocks_per_window = (xy_window[1] // pix_per_cell) - cell_per_block + 1
    #cells_per_step = 1.5  # Instead of overlap, define how many cells to step
    nxsteps = np.int((nxblocks - nxblocks_per_window) // cells_per_step)
    nysteps = np.int((nyblocks - nyblocks_per_window) // cells_per_step)
    #print ("window=", window, "nblocks_per_window=", nblocks_per_window)
    #print ("nxblocks=", nxblocks, " nyblocks=", nyblocks)
    #print ("nxstep=", nxsteps, ", ", nysteps)    
    
    # Compute individual channel HOG features for the entire image
    if hog_use_channel == 'ALL':
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        hog = get_hog_features(ctrans_tosearch[:,:,hog_use_channel], orient, pix_per_cell, cell_per_block, feature_vec=False)

    print ("Windows=", nxsteps*nysteps)
    for xb in range(nxsteps):
        for yb in range(nysteps):
            img_features = []
            
            ypos = np.int(yb*cells_per_step)
            xpos = np.int(xb*cells_per_step)
            
            # Extract HOG for this patch
            if hog_use_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos+nyblocks_per_window, xpos:xpos+nxblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nyblocks_per_window, xpos:xpos+nxblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nyblocks_per_window, xpos:xpos+nxblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog[ypos:ypos+nyblocks_per_window, xpos:xpos+nxblocks_per_window].ravel() 

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+xy_window[1], xleft:xleft+xy_window[0]], (64,64))

            # Get color features
            if spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            img_features.append(spatial_features)
            img_features.append(hist_features)
            img_features.append(hog_features)
            #print("spat=", spatial_features.shape)
            #print ("l1=", np.concatenate(img_features).shape)
            #print ("l2=", np.array(np.concatenate(img_features)).reshape(1, -1).shape)
            test_features = X_scaler.transform(np.concatenate(img_features).reshape(1, -1))
            #test_features = X_scaler.transform(np.array(np.concatenate(img_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_xdraw = np.int(xy_window[0]*scale)
                win_ydraw = np.int(xy_window[1]*scale)
                on_windows.append([(xbox_left, ytop_draw+ystart),(xbox_left+win_xdraw,ytop_draw+win_ydraw+ystart)])
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 

    return on_windows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap

def flatten_heat_map(heatmap):
    #Set all pixels whose value > 0 to 1
    heatmap[heatmap > 0] = 1
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 3)
    # Return the image
    return img

#Process each image in the input video clip
def processImage(image):
    global frameCount
    frameCount = frameCount + 1

    draw_image = np.copy(image)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    #draw_image = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)
    hot_windows = []
    
    scale = scale1
    hot_windows = np.array(find_cars(image, y_start_stop_1[0], y_start_stop_1[1], scale, clf, X_scaler, orient_bins, pix_per_cell, cell_per_block, 
                spatial_size, hist_bins, cells_per_step = 1.5, spatial_feat=spatial_feat))
    print ("Got [", frameCount, "] (64x64) ", hot_windows.size)
    scale = scale2
    win2 = np.array(find_cars(image, y_start_stop_2[0], y_start_stop_2[1], scale, clf, X_scaler, orient_bins, pix_per_cell, cell_per_block, 
                spatial_size, hist_bins, cells_per_step = 1.5, spatial_feat=spatial_feat))
    print ("        (128x128) ", win2.size)  
    if win2.size > 0 and hot_windows.size > 0:
        hot_windows = np.concatenate((hot_windows, win2), axis=0)
    elif win2.size > 0 and hot_windows.size == 0:
        hot_windows = win2

    if 0:#frameCount % 5 == 1:
        box_count = 0
        for found_box in hot_windows:        
            start = found_box[0]
            end = found_box[1]
            
            if start[0] < 700:
                #print ("found=", found_box)            
        
                out = cv2.resize(draw_image[start[0]:end[0], start[1]:end[1]], (64, 64))
                mpimg.imsave('output_images/{0}_box{1}.png'.format(frameCount, box_count), out)

                out = draw_image
                mpimg.imsave('output_images/{0}_frame.png'.format(frameCount), out)

                box_count = box_count + 1

    #Infer bounding box of cars for current frame
    add_heat(heat, hot_windows)
    if 0: #hot_windows.size > 20:
        heat = apply_threshold(heat, 2)
    else:
        heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    current_list = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        current_list.append(bbox)
    
    found_box_history.append(current_list)
         
    flatten_heat_map(heat)
        
    # Add heat to each box in box list, along with historical boxes
    #found_box_history.append(hot_windows)
    for bbox in found_box_history:
        add_heat(heat, bbox)  
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, heat_threshold)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    labels = label(heatmap)
    window_img = draw_labeled_bboxes(draw_image, labels)
    #window_img = draw_boxes(draw_image, hot_windows, color=(0, 255, 0), thick=3)
        
    return window_img

def main():
    parser = argparse.ArgumentParser(description='Detect vehicles in video')
    parser.add_argument(
        'inputFile',
        type=str,
        default='',
        help='Path to input video file.'
    )
    parser.add_argument(
        'outputFile',
        type=str,
        default='',
        help='Filename of output video')
    args = parser.parse_args()

    
    clip = VideoFileClip(args.inputFile)
    new_clip = clip.fl_image( processImage )
    new_clip.write_videofile(args.outputFile, audio=False)
    
    
    
if __name__ == '__main__':
    main()