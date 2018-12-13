## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/original_undistort.png "Undistorted"
[image2]: ./output_images/original_test_img.png "Original image"
[image3]: ./output_images/undistort_test_img.png "Road Transformed"
[image4]: ./output_images/combined_thresholded.png "Combined Thresholded Example"
[image5]: ./output_images/perspective_transformed_image.png "Warp Example"
[image6]: ./output_images/test1.jpg "Final Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./Advanced_Lane_Lines.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

The distortion coefficient and associated matrix were obtained from the Camera class. Subsequently, the __call__ class function was used to undistort the image.

Below is the result of the undistorted image:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

After applying gradient and color thresholding independently, I used their combination to generate a binary image (combined thresholding steps can be seen within the `Combined Thresholding` markdown header on the Advanced_Lane_Lines.ipynb).  Here's an example of output images from the combined step:

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

We identified four trapezoidal shaped source points where the lane lines are straight, and found four lines that after applying `perspective_transform()` function made the lines parallel from a bird's eye view perspective. With the source points defined, we assumed the road was flat, the camera perspective remain unchanged and applied it to new images. Below is the code block showing how we defined the source and destination points and how the images were warped:

```python
height = image.shape[0]
width = image.shape[1]
# Rectangular coordinates in the source image
c1 = [width // 2 - 76, height * 0.625]
c2 = [width // 2 + 76, height * 0.625]
c3 = [-100, height]
c4 = [width + 100, height]
src = np.float32([c1, c2, c3, c4])
# Rectangular coordinates in the destination image
off1 = [100, 0]
off2 = [width - 100, 0]
off3 = [100, height]
off4 = [width - 100, height]
dst = np.float32([off1, off2, off3, off4])
# Given src and dst points we calculate the perspective transform matrix
t_Mtx = cv2.getPerspectiveTransform(src, dst)
# Warp the image
warped = cv2.warpPerspective(image, t_Mtx, (width, height))
# We also calculate the oposite transform
unwarp_Mtx = cv2.getPerspectiveTransform(dst, src)
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 564, 450      | 100, 0        |
| 716, 450      | 1180, 0       |
| -100, 720     | 100, 720      |
| 1380, 720     | 1180, 720     |


To verify transformation, the lane lines appeared parallel in warped image example below:

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After applying calibration, thresholding, and perspective transformation to the image, I used the lane finding method of Peaks in a Histogram to determine the start of the left and right lane lines. These were implemented within the `SlidingWindow() and LaneLine()` classes. In our thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram were good indicators of the x-position of the base of the lane lines. These were used as the starting point for where to search for the lines. From that point, I implemented the `SlidingWindow()` class with windows and window hyper parameters. Next, I iterated through the windows to ensure it found the mean position of activated pixels within the window.

With the hot pixels tracked to the top of the frame, the `LaneLine()` class is first implemented to fit the lane lines. Subsequently, if the lane line is still within the center of the minpix  of 50 defined, the quadratic coefficient that fits the original line is used to fit new lane lines.  Below is a snapshop of detected lane lines from an image.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Firstly, I derived a conversion from pixel to real world space. I assumed in real world that the lane is about 30 meters long and 3.7 meters wide. I estimated that the y-dimension of our image is 720 pixels while the x-dimension between the left and right lane lines is 700 pixels.

```python
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

Eventually, the `radius_of_curvature() and camera_distance()` functions were implemented within the `LaneLine()` class to estimate the radius of curvature and position of the vehicle.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step within the `Pipeline()` class. Here is an example of my output from a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./videos/project_video.mp4)

[Link to youtube](https://youtu.be/05ds0wKQpUk)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Our pipeline was designed to handled normal road conditions which is representative of the "project_video" video file. Therefore, no issues or problems were faced when feeding it to our pipeline. However, feeding the "challenge_video" video file, we experienced some challenges due to the shadows being cast from an overhead bridge and the non-uniformity of lane lines in some parts. Next step, would be to implement a more robust pipeline which can handle all possible road and weather conditions / scenarios.  
