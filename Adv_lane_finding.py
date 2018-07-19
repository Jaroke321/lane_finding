
# coding: utf-8

# # Calibrate the camera
# 
# 1). Get camera calibration coefficients from chess board images in camera_cal folder <br>
# 2). Use cal. coeffiecients to apply distortion correction to raw images
# 
# >  __Inputs__: <br>
# -  distorted chessboard images
# -  found in camera_cal folder, stored as .jpg
# 
# > __Outputs__: <br>
# -  calibration coefficients for camera

# In[1]:


import glob
import numpy as np
import cv2
#get chessboard images

images = glob.glob("camera_cal/calibration*.jpg")

objpoints = []
imgpoints = []

objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

for filename in images:
    
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        
test = cv2.imread('camera_cal/calibration3.jpg')
shape = (test.shape[1], test.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#use cv2.undistort to undistort test image
undist = cv2.undistort(test, mtx, dist, None, mtx)

#plot original image with undistorted image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(test)
ax1.set_title('original image')
ax2.imshow(undist)
ax2.set_title('undistorted image')

#save image to output_images file
cv2.imwrite('output_images/undistorted.jpg', undist)
print('images saved')


# #  Create threshold binary images
# 
# 1. Using gradients, color transform, etc. make binary images from lane images taken by car
# 2. Given calibration coeffiecients, undistort image first and then use method to get final binary image
# 
# > __Inputs__: <br>
# -  calibration coeffiecients for the camera
# -  images taken by car camera
# 
# > __Outputs__: <br>
# -  undistorted binary image
# -  undistorted image

# In[3]:


import matplotlib.image as mpimg
#load test images
image_files = glob.glob('test_images/test*.jpg')

test_images = []
for image in image_files:
    a = mpimg.imread(image)
    test_images.append(a)


#write and implement function(s) to get binary images
def to_binary(img, matrix = mtx, distortion = dist):
    
    #undistort incoming image
    undist = cv2.undistort(img, matrix, distortion, None, matrix)
    
    #convert to HLS
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    #grab s channel
    S = hls[:,:,2]
              
    s_thresh = (170, 255)                                  #threshold values for s channel
    binary = np.zeros_like(S)                              
    binary[(S > s_thresh[0]) & (S <= s_thresh[1])] = 1     #create binary image from s channel using threshold
              
    
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)        #create gray image
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)             #apply sobel function 
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    sobel_thresh = (20, 100)                               #threshold values for sobel binary image
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1  #sobel binary
              
    #combine sxbinary and binary for a combined binary image 
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(binary == 1) | (sxbinary == 1)] = 1
    
    #return undistorted image and binary image of road
    return combined_binary, undist


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
#test to_binary function on test_images

combined1, undist1 = to_binary(test_images[1])
combined2, undist2 = to_binary(test_images[3])
combined3, undist3 = to_binary(test_images[5])

fig = plt.figure(figsize=(20,20))

ax1 = fig.add_subplot(3,2,1)
ax2 = fig.add_subplot(3,2,2)
ax1.imshow(undist1)
ax2.imshow(combined1, cmap = 'gray')

ax3 = fig.add_subplot(3,2,3)
ax4 = fig.add_subplot(3,2,4)
ax3.imshow(undist2)
ax4.imshow(combined2, cmap = 'gray')

ax5 = fig.add_subplot(3,2,5)
ax6 = fig.add_subplot(3,2,6)
ax5.imshow(undist3)
ax6.imshow(combined3, cmap = 'gray')

#save images to out_images file
mpimg.imsave('output_images/binary1.jpg', combined1, cmap = 'gray')
mpimg.imsave('output_images/binary2.jpg', combined2, cmap = 'gray')
mpimg.imsave('output_images/binary3.jpg', combined3, cmap = 'gray')
print('images saved')


# # Perspective Transform
# 
# 1. Apply a perspective transform on binary images
# 2. Should turn calibrated binary images into a birds-eye view of the road
# 
# > __Inputs__: <br>
# -  binary image from to_binary(): 
# 
# > __Outputs__: <br>
# -  birds-eye view of the binary image
# 

# In[5]:


straight_image = mpimg.imread('test_images/straight_lines2.jpg')
src = np.float32([[230, 700], [575, 460], [700, 460], [1050,700]])
dst = np.float32([[260, 720], [260, 0], [1020, 0], [1020, 720]])

M = cv2.getPerspectiveTransform(src, dst)

img_size = (straight_image.shape[1], straight_image.shape[0])
warped = cv2.warpPerspective(straight_image, M, img_size)


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(straight_image, cmap = 'gray')
ax2.imshow(warped, cmap = 'gray')


# In[7]:


#write and implement function to apply perspective transform on binary image
def perspective_transform(bin_image):
    
    #source points
    src = np.float32([[230, 700], [575, 460], [700, 460], [1050, 700]])
    #destination points
    dst = np.float32([[300, 720], [300, 0], [1000, 0], [1000, 720]])
    
    img_size = (bin_image.shape[1], bin_image.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(bin_image, M, img_size)
    #return transformed image
    return warped, M, Minv


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
#test transform function and save images to output_images folder
warp_test1, _, minv1 = perspective_transform(combined1)
warp_test2, _, minv2 = perspective_transform(combined3)

fig = plt.figure(figsize = (20, 10))

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

mpimg.imsave('output_images/transformed_binary1.jpg', warp_test1, cmap = 'gray')
mpimg.imsave('output_images/transformed_binary2.jpg', warp_test2, cmap = 'gray')

ax1.imshow(combined1, cmap = 'gray')
ax1.set_title('original binary 1')
ax2.imshow(warp_test1, cmap = 'gray')
ax2.set_title('tranformed binary 1')
ax3.imshow(combined3, cmap = 'gray')
ax3.set_title('original binary 2')
ax4.imshow(warp_test2, cmap = 'gray')
ax4.set_title('transformed binary 2')


# # Detect lane pixels and fit line
# 
# 1. Using the image from perspective transform find the lane pixels on each side of lane <br>
# 2. Fit these points with a line representing each side of the lane(Histogram)
# 
# > __Inputs__: <br>
# -  transformed images from perspective_transform()
# 
# > __Outputs__: <br>
# -  left x points: x points of the fitted line for the left side of lane
# -  right x points: x points of the fitted line for the right side of lane
# -  y points: the y coordinate points for each line

# In[9]:


#write and implement function to find polynomial line for each side of lane using binary transformed images
def detect_lanes(transformed_image):
    
    #create histogram of binary image
    histogram = np.sum(transformed_image[transformed_image.shape[0]//2:,:], axis = 0)
    #image to test result
    out_image = np.dstack((transformed_image, transformed_image, transformed_image))*255
    
    #base of left and right lanes
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    #number of sliding windows to use
    nwindows = 9
    #height of windows
    window_height = np.int(transformed_image.shape[0]//nwindows)
    #get x and y positions of all pixels in image
    nonzero = transformed_image.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]
    
    #current position to be updated for windows
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    #set windows margins
    margin = 100
    minpix = 50
    
    #create lists to store left and right pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    #iterate through windows
    for window in range(nwindows):
        #identify window boundaries
        windowy_low = transformed_image.shape[0]- (window+1)*window_height
        windowy_high = transformed_image.shape[0] - window*window_height
        leftx_low = leftx_current - margin
        leftx_high = leftx_current + margin
        rightx_low = rightx_current - margin
        rightx_high = rightx_current + margin
        
        #draw rectangles on out_img
        cv2.rectangle(out_image, (leftx_low, windowy_low), (leftx_high, windowy_high), (0, 255, 0), 2)
        cv2.rectangle(out_image, (rightx_low, windowy_low), (rightx_high, windowy_high), (0, 255, 0), 2)
        
        #grab nonzero inds for window
        good_left_inds = ((nonzeroy >= windowy_low) & (nonzeroy < windowy_high) & 
                         (nonzerox >= leftx_low) & (nonzerox < leftx_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= windowy_low) & (nonzeroy < windowy_high) & 
                          (nonzerox >= rightx_low) & (nonzerox < rightx_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        #recenter next window on the mean of the previous good indices
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    #concatenate arrays of inds
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    #get left and right lane line pixels
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    #fit polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    #get x and y values for plotting
    ploty = np.linspace(0, transformed_image.shape[0]-1, transformed_image.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #return points for fitted lines of each side of lane
    return ploty, left_fitx, right_fitx, leftx_base, rightx_base, out_image


# In[10]:


#test and save images to output_images folder
ploty, left_fitx, right_fitx, leftx_base, rightx_base, out_img = detect_lanes(warp_test1)



plt.imshow(out_img)
plt.plot(left_fitx, ploty, color = 'yellow')
plt.plot(right_fitx, ploty, color = 'yellow')
plt.xlim(0,1280)
plt.ylim(720,0)
plt.savefig('output_images/detected_lane.jpg')


# # Find Curvature of Lane Lines and Distance from Center
# 
# 1. Using the fitted lines find the curvature of the lane <br>
# 2. Also figure out where the cars center is in respect to the center of the lane
# 
# > __Inputs__: <br>
# -  ploty: y points along graph
# -  left_fitx: fitted left lane line
# -  right_fitx: fitted right lane line
# 
# > __Outputs__: 
# -  calculated curvature of lane
# -  distance the car is from center of lane

# In[11]:


#write and implement function that finds curvature of lane and position of car
def curvature(ploty, left_fitx, right_fitx, leftx_base, rightx_base):
    
    #convert from pixel space to real world space
    pixel_width = rightx_base - leftx_base
    ym_per_pix = 30/720
    xm_per_pix = 3.7/pixel_width
    y_eval = np.max(ploty)
    
    #fit new lines in real world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    #find curvature of the lanes
    left_curve = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curve = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    avg_curve = (left_curve + right_curve) / 2
    
    #find cars distance from center of the detected lane
    image_center = 640
    pix_from_center = image_center - ((pixel_width/2) + leftx_base)
    dist_from_center = pix_from_center * xm_per_pix
    
    #return curvature of each lane line and the distance that the car is from the center of the lane
    return avg_curve, dist_from_center
    


# In[12]:


#test on images and save to output_images folder
curve, distance = curvature(ploty, left_fitx, right_fitx, leftx_base, rightx_base)

print(curve, 'm.....', distance)


# # Output
# 
# 1). Now that you have the lane boundaries(best fit lines for each side of lane), curvature of lane and the cars distance from the canter of the lane... use these to warp the lane boundaries back onto the original image. This should be a visual representation of the lanes boundaries and the lane itself (color the drivable portion of the road, not just the lane boundaries themselves, see "tips and tricks for project" in classroom for example). There should also be a readout for ever images that calculates the curvature of the lane and the position of the car within the lane. 
# 
# > __Inputs__: <br>
# -  original undistorted image: 
# -  warped binary image: 
# -  leftx, rightx, and y points: for lane lines
# -  curvature value:
# -  center distance value
# 
# > __Outputs__:
# -  final image: original undistorted image with lane lines drawn on

# In[13]:


from PIL import ImageDraw, Image
#write function that takes lane lines and draws them to original image(undistort) to see lane
def draw_lanes(undist_img, transformed_img, left_fitx, right_fitx, ploty, Minv, curve, center_dist):
    
    #create image to draw on
    warped_img = np.zeros_like(transformed_img).astype(np.uint8)
    color_warp = np.dstack((warped_img, warped_img, warped_img))
    
    #make x and y point usable for polyfill
    left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_points = np.array ([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((left_points, right_points))
    
    #draw lanes onto warped image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    #warp back onto original image to see lanes
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist_img.shape[1], undist_img.shape[0]))
    result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)
    
    #convert rgb image with lanes into rgba
    result_rgba = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
    #make rgba image usable for ImageDraw
    result_rgba = Image.fromarray(result_rgba, mode = 'RGBA')
    #make new image to draw on
    draw_image = Image.new('RGBA', result_rgba.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(draw_image)
    d.text((10, 10), 'Lane Curvature: ' + str(curve) + 'm', fill = (255, 255, 255, 255))
    d.text((10, 40), 'Distance from Center: ' + str(center_dist) + 'm', fill = (255, 255, 255, 255))
    
    #overlap two images to put curve and distance values in image
    final_output = Image.alpha_composite(result_rgba, draw_image)
    #output lane curvature and cars position in lane 
    return final_output


# In[14]:


#test and save to output_images folder before using on video data
final_test = draw_lanes(undist1, warp_test1, left_fitx, right_fitx, ploty, minv1, curve, distance)

mpimg.imsave('output_images/result.jpg', final_test)
final_test


# # Pipeline
# 
# 1. Once the above functions are ready and worked. And images from each of those steps is loaded into the output_images file. This pipeline will be implemented and tested using those same functions <br>
# 2. Pipeline should take in distorted images from camera(video file in this case). And then using the above functions...
# > -  undistort images
# > -  binarize those images
# > -  transform binary images
# > -  detect and fit lane pixels
# > -  find curvature of lane and distance from center
# > -  finally draw lane on image in warped space
# > -  unwarp this image and map it to original undistorted image
# > -  print curvature and distance from center of lane on final image too
# 
# > __Inputs__: <br>
# -  image: distorted image taken from car
# -  calibration coef: found using calibration function on chessboard images
# 
# > __Outputs__:
# -  final image: undistorted original image with lanes highlighted

# In[15]:


def pipeline(dist_image, matrix, distortion):
    
    #undistort image and create binary image
    binary_image, undist = to_binary(dist_image, matrix, distortion)
    #transform image to see from birds-eye-view
    transformed, M, Minv = perspective_transform(binary_image)
    #detect line pixels and get points for lines of best fit
    ploty, left_fitx, right_fitx, leftx_base, rightx_base, _outim = detect_lanes(transformed)
    #find curvature of lane and distance from center
    curve, center = curvature(ploty, left_fitx, right_fitx, leftx_base, rightx_base)
    #draw lane lines on original undistorted image
    final_image = draw_lanes(undist, transformed, left_fitx, right_fitx, ploty, Minv, curve, center)
    
    return final_image
    


# In[16]:


pipeline_test = pipeline(straight_image, mtx, dist)

mpimg.imsave('output_images/pipeline_test.jpg', pipeline_test)
pipeline_test


# In[17]:


#run each frame of video through pipeline
cap = cv2.VideoCapture('project_video.mp4')
count = 0
#go through video until is done
while(cap.isOpened()):
    
    #get individual frames and return value
    ret, frame = cap.read()
    #make sure cap returnedd a frame
    if ret == True:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_image = pipeline(rgb_frame, mtx, dist)
        #save pipeline output images to file as jpg
        mpimg.imsave('data/frame{}.jpg'.format(count), detected_image)
        count += 1
    else:
        break
        
#release the capture object
cap.release()


# In[18]:


#grab each modified image and make a video from it
files = glob.glob('data/frame*.jpg')
frame_array = []

#sort files to get frames into the correct order
files.sort(key = lambda x: int(x[10:-4]))
path = 'data/'
#go through filenames and write them to array
for i in range(len(files)):
    #get accurate file path
    frame = files[i].split('\\')[1]
    filename = path + frame
    #read in image
    img = cv2.imread(filename)
    #get height, width, and size for videowriter
    height, width, channels = img.shape
    size = (width, height)
    #append img to frame_array
    frame_array.append(img)
    
    
#create videowriter to make video from individual frames
pathOut = 'video2.avi'
fps = 25
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

#cycle through frames and write to video file
for i in range(len(frame_array)):
    out.write(frame_array[i])
    
out.release  

