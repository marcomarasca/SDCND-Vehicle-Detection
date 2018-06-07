# **Vehicle Detection Project**

[![Vehicle Detection](https://img.youtube.com/vi/1km0QDbck40/0.jpg)](https://www.youtube.com/watch?v=1km0QDbck40 "Vehicle Detection")

In this project we built a pipeline to detect vehicles in images and video frames. The project consisted in the following steps: 

* Using a labeled set of images feature extraction that included Histogram of Oriented Gradients (HOG), spatial binning and histogram of colors
* Train a Liner SVM classifier using the extracted features
* Implementation a multi-scale sliding-window technique to search for vehicles in images and video frames using
the trained classifier on the extracted windows
* Heatmap from the detected windows with thresholding on the confidence of the detection as well as on the heatmap
value itself
* Average smoothing of historic heatmaps to reduce false positives with thresholding applied on the average
* Bounding box estimation from the resulting heatmap to output the detected vehicles

[//]: # (Image References)
[cars_notcars]: ./output_images/cars_notcars.png
[cars_notcars_udacity]: ./output_images/cars_notcars_udacity.png
[train_features]: ./output_images/training_features.png
[normalized_features]: ./output_images/normalized_features.png
[windows_search]: ./output_images/test5_window_search.jpg
[windows_search_2]: ./output_images/test6_window_search.jpg
[windows_search_3]: ./output_images/test3_window_search.jpg
[windows_search_gif]: ./output_images/window_search.gif
[heatmap]: ./output_images/test5_heatmap.jpg
[heatmap_gif]: ./output_images/heatmap.gif
[heatmap_gif_threshold]: ./output_images/heatmap_threshold.gif
[labeled]: ./output_images/test5_labeled.jpg
[lane_detection_gif]: ./output_images/lane_detection.gif

[video_1]: ./output_videos/project_video_processed.mp4
[video_2]: ./output_videos/project_video_processed_lanes.mp4
[video_2]: ./output_videos/project_video_pipeline.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### In the following I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I implemented the feature extraction process for the images in the [FeatureExtractor](./feature_extractor.py) class that can be found in the [feature_extractor.py](./feature_extractor.py) file. In addition I also used spatial binning of color as well as histogram of colors for each channel of the image. 

The feature extractor is initialized with various parameters to customize the feature extraction process including the color space to used for feature extraction; later on invoking the extract_features method on a set of images returns an array with the features for each image.

The feature extractor is used by the [model.py](./model.py) script for training a linear SVM classifier. The script starts off loading the dataset from a given input folder and looking for `vehicle` and `non-vehicle` images:

![alt text][cars_notcars]

I then started exploring various color spaces and parameters for the HOG features quickly checking which ones would give a me distinctive shape in the various images:

![alt text][train_features]

In the image above I was using the following parameters:

* Color space: **YCrCb**
* Orientations: **16**
* Pixel per cell: **8**
* Cell per block: **2**

I didn't spend much time manually visualizing the result, but I rather took an automated approach as explained in the next section.

#### 2. Explain how you settled on your final choice of HOG parameters.

I soon realized that the combination of parameters to choose from to train my classifier would quickly explode, so I started out implementing directly a Linear SVM classifier (using the LinearSVC from sklearn) and then I added a grid search over color spaces, HOG parameters and the dimensions for spatial binning as well as the number of bins for the histograms of colors using a range that would make sense (and that I experimented a little already in the classroom) and using the accuracy of the model to discriminate. I also experimented using different combinations of features. The code can be found in the [model.py](./model.py) script in the search_parameters function. 

After the search I settled for the following parameters that would retain the best accuracy (around **99.1%**):

* Color space: **YCrCb**
* Spatial binning dimension: **16**
* Histogram of colors bins: **32**
* Orientations: **16**
* Pixel per cell: **8**
* Cell per block: **2**

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

As explained above I quickly implemented the classifier and I started directly training to get a sense of what features would work best for training. The code is in the [model.py](./model.py) script.

In particular the training goes as follows:

* Load the dataset: ([model.py](./model.py), line 277)
* Extract the features: ([model.py](./model.py), line 127 and function at 20)
* Shuffle and split in training and test sets: ([model.py](./model.py), line 134)
* Normalize the features vector: ([model.py](./model.py), line 137 and function at 77)
* Train the **LinearSVC** on the training set: ([model.py](./model.py), line 140 and function at 86)
* Test the model on the testing set: ([model.py](./model.py), line 143 and function at 108)
* Save the resulting model along with the scaler and other parameters that are needed later on: ([model.py](./model.py), line 311)

An important step was **normalizing** the features vector of both the training set and testing set fitting the StandardScaler() supplied by sklearn on the training set:

![alt text][normalized_features]

I initally also performed parameter tuning using the GridSearchCV from sklearn on various parameters of the LinearSVC (value of C and dual) but found that it would not make much difference so I dropped it from the code.

The final training was performed with the following parameters:

* Color space: **YCrCb**
* Spatial binning dimension: **16**
* Histogram of colors bins: **32**
* Orientations: **16**
* Pixel per cell: **8**
* Cell per block: **2**

With a final feature vector length of 10272. I implemented some optimizations to speed up training, in particular in the feature extraction the HOG features that are computed from each channel of the image are processed in parallel. The same for the histogram of colors, since I can process each channel in parallel. This provided a speed up of around 2 times for extracting features on **17760** images (8792 cars and 8968 not-car images).

The final accuracy was of over **99.1%**.

I also experimented with the annotated [Udacity Dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to extract additional images (both of cars and not-cars). The code can be found in the [udacity_parser.py](./udacity_parser.py) script. Unfortunately training using this dataset didn't yield very good performance using the LinearSVC reaching around 95% of accuracy which turned out to be not enough for the sliding window search approach implemented later on.

In the following some examples of images extracted from the Udacity dataset:

![alt text][cars_notcars_udacity]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a slighlty modified version of the hog-subsampling windows search method introduced in the classroom. The code can be found in the WindowSearch class in the [vehicle_detector.py](./vehicle_detector.py) file. In particular if was clear that using only one window would not yield good results so I implemented a multi-scale search using hog-subsampling with various scale factors to detect vehicles of different dimensions (e.g. because of depth and distance from the camera) in the image.

I initially experimented with different scales to see how they would adapt on the test images and after some trial and error (mostly due to false positive) I settled on using the following windows:

![alt text][windows_search]

Note how I use different scales with different overlaps (cells_per_step) in each window, this allows to be more precise in various situations (e.g. distant cars) and to allow a better thresholding in presence of false positives.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline simply performs a search on various layers with different scales, extracting the features from the layer (and subsampling the HOG features in the different poisitions of the sub-windows), scaling them with the scaler used for training and computing a confidence level of the prediction. Note that the images are treated the sames as during training (e.g. converting to YCrCb, extracting the features with the same parameters etc.)

In order to increase the accuracy of the window search I put a threshold on the **confidence** of the prediction output from the classifier (e.g. the distance from the hyperplane), this allowed to discard poorly classified windows as a first cleaning step.

To speed up the windows search I implemented **paralell processing** for each layer (as they can be computed independently), on 4 cores this increased substantially the performance (around 2 times faster).

In the following some additional examples from the test images:

![alt text][windows_search_2]

The windows in red are those thresholded by the minimum confidence for the detection.

![alt text][windows_search_3]

In the latest image we can see that increasing the amount of overlap for the "distant" window allows to have more "hits" on the detected cars (while using the same scale).

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The final processed video can be found [here][video_1]. The script used to generate the video is in the [video_gen.py](./video_gen.py) that also allows to debug the various steps of the pipeline outputting a video for each step (sliding window for each layer, combined, heatmaps etc.), in the following an extract of the multi-scale sliding windows search:

![alt text][windows_search_gif]

I later on integrated the **lane detection** project, the video that integrates boths the vehicle detection and the lanes detection can be found [here][video_2]. Interestingly the processed result contains more false positives due to the camera undistorsion. Since the image is distorted the choosen sliding windows are a little offsets and "out of scale" and the pipeline get "fooled" by the roofs of the cars passing in the opposite lane. Adjusting the windows and the threshold should fix the problem.

![alt text][lane_detection_gif]

Finally a video montage with the various stages is uploaded on youtube and can be found in [here][video_3]:

[![Vehicle Detection](https://img.youtube.com/vi/1km0QDbck40/0.jpg)](https://www.youtube.com/watch?v=1km0QDbck40 "Vehicle Detection")

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Despite the high accuracy obtained by the classifier searching this amount of windows (more than 1500) per image yields several false positives. Luckily false positive are hardly present in subsequent frames, therefore I build an heatmap from the positive detected windows (See line 254 in the [VehicleDetector](./vehicle_detector.py)) and threshold over the average of the last X frames (See [VehicleDetector](./vehicle_detector.py) lines 313-320).

![alt text][heatmap]

In the following an extract of the un-thresholded heatmap along with its thresholded version (average over the last 8 frames):

![alt text][heatmap_gif] ![alt text][heatmap_gif_threshold]

For the final video I used the following parameters:

* Min confidence: **0.5**
* Number of frames: **8**
* Threshold: **3.5**

I then proceed in computing the vehicles bounding boxes using the `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap (See [VehicleDetector](./vehicle_detector.py) lines 265-294). I then assumed each blob corresponded to a vehicle (same approach as shown in the classroom). I finally drop bounding boxes that are too small to further reduce false positives.

![alt text][labeled]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The project implementation was quite smooth in the beginning and considering that I obtained a relatively high accuracy with the LinearSVC I was quite happy with how it was going :) When I integrated the pipeline on the video the issues started, several false positives started popping up and it took me quite a while to find a decent configuration of sliding windows and thresholding. I also went back to the model trying to see if there was a problem there but could not improve much. I experimented with the annotated dataset provided by udacity, extracting an additional 9000 car and 9000 not-car images but got worse results (with the LinearSVC) and more false positives. This probably shows the limitation in the selected features for discriminating between cars and not-cars or more simply a sign of overfitting.

I soon realized that unfortunately the accuracy of the model was not a good indicator of the performance on the final video: due to the amount of sliding windows for each frame even with a very high accuracy on the test data the false positive are naturaly accentuated.

Having the time and resources I would probably try working with better features or a different classifier, but I think the best bet would be to go towards a deep learning approach. My intuition is that even using the same approach (e.g. car vs not-car classification) a simple CNN would yield slighly better results as the features would be learned by the model and would output less false positive (assuming a large enough dataset). Better yet would be to try a YOLO architecture so that we could apply an end-to-end approach so that the model could spit out directly the bounding boxes with better performance (assuming we are on a GPU). This was another issue working with the project, it's not exactly feasible to use it in real world as I can barely get 2 frames per second even after implementing multiprocessing due to the high amount of searched windows (surely there is a lot of space for improvement there).

