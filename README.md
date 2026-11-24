
# Lane Detection using Hough Transform

## Ex. No: 07

## Aim

To implement lane detection using Hough Transform by detecting and highlighting road lane boundaries using OpenCV.

## Software Used

- Anaconda – Python 3.7
- Jupyter Notebook / VS Code
- OpenCV (cv2)
- NumPy
- Matplotlib

## Algorithm

**Step 1: Import Libraries**

Import the required libraries: OpenCV, NumPy, and Matplotlib.

**Step 2: Read the Image**

Read the input road image named `road.jpg` using OpenCV.

**Step 3: Convert to Grayscale**

Convert the original color image into a grayscale image to simplify image processing.

**Step 4: Display Images**

Display the original input image and grayscale image using Matplotlib.

**Step 5: Thresholding**

Apply global thresholding based on grayscale intensity to separate bright lane markings from darker regions.

**Step 6: Region of Interest (ROI)**

Define a polygonal region of interest and create a mask. Apply the mask to isolate the required road region.

**Step 7: Edge Detection**

Apply Canny edge detection to identify the edges of lane markings.

**Step 8: Gaussian Blur**

Apply Gaussian Blur to smooth the detected edges and reduce noise.

**Step 9: Hough Transform**

Apply the Probabilistic Hough Transform using `cv2.HoughLinesP()` to detect line segments. Reshape the detected lines into the required format for drawing.

**Step 10: Lane Detection and Extrapolation**

Separate the detected lines into left and right lane lines based on their slopes. Calculate their average slopes and intercepts to extrapolate the lane boundaries.

## Program

```python
# Import
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image using OpenCV
img = cv2.imread("road.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display the image using Matplotlib
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Input Image')

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')

plt.show()

# Use global threshold based on grayscale intensity
threshold = cv2.threshold(
    gray, 150, 255, cv2.THRESH_BINARY
)[1]

# Display threshold image
plt.figure(figsize=(20, 10))
plt.subplot(1, 1, 1)
plt.imshow(threshold, cmap='gray')
plt.title('Threshold')
plt.show()

# Region masking
roi_vertices = np.array([[[100, 540],
                          [900, 540],
                          [515, 320],
                          [450, 320]]])

# Defining a blank mask
mask = np.zeros_like(threshold)

# Defining mask color
if len(threshold.shape) > 2:
    channel_count = threshold.shape[2]
    ignore_mask_color = (255,) * channel_count
else:
    ignore_mask_color = 255

# Filling pixels inside the polygon
cv2.fillPoly(mask, roi_vertices, ignore_mask_color)

# Constructing the region of interest
roi = cv2.bitwise_and(threshold, mask)

# Display ROI images
plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.imshow(threshold, cmap='gray')
plt.title('Initial threshold')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Polyfill mask')

plt.subplot(1, 3, 3)
plt.imshow(roi, cmap='gray')
plt.title('Isolated roi')

plt.show()

# Perform Edge Detection using Canny
edges = cv2.Canny(roi, 50, 150)

# Smooth with Gaussian Blur
canny_blur = cv2.GaussianBlur(edges, (5, 5), 0)

# Display edge detection and blurred edges
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title('Edge detection')

plt.subplot(1, 2, 2)
plt.imshow(canny_blur, cmap='gray')
plt.title('Blurred edges')

plt.show()


# Function for drawing lines
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# Hough Transform
lines = cv2.HoughLinesP(
    canny_blur,
    rho=1,
    theta=np.pi / 180,
    threshold=20,
    minLineLength=20,
    maxLineGap=300
)

# Reshape detected lines to the required format
if lines is not None:
    lines = lines.reshape(-1, 1, 4)

# Draw all detected lines
hough = np.zeros(
    (img.shape[0], img.shape[1], 3),
    dtype=np.uint8
)

draw_lines(hough, lines)

if lines is not None:
    print("Found {} lines, including: {}".format(len(lines), lines[0]))
else:
    print("No lines detected.")

plt.figure(figsize=(15, 10))
plt.imshow(hough)
plt.title('Detected Hough Lines')
plt.show()


# Separate left and right lane lines
def separate_left_right_lines(lines):
    """Separate left and right lines depending on the slope."""
    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 > y2:
                    left_lines.append([x1, y1, x2, y2])
                elif y1 < y2:
                    right_lines.append([x1, y1, x2, y2])

    return left_lines, right_lines


# Calculate average
def cal_avg(values):
    """Calculate average value."""
    if not (type(values) == 'NoneType'):
        if len(values) > 0:
            n = len(values)
        else:
            n = 1

        return sum(values) / n


# Extrapolate lane lines
def extrapolate_lines(lines, upper_border, lower_border):
    """Extrapolate lines using upper and lower border intersections."""
    slopes = []
    consts = []

    if lines is not None:
        for x1, y1, x2, y2 in lines:
            slope = (y1 - y2) / (x1 - x2)
            slopes.append(slope)

            c = y1 - slope * x1
            consts.append(c)

    avg_slope = cal_avg(slopes)
    avg_consts = cal_avg(consts)

    # Calculate lower border intersection
    x_lane_lower_point = int(
        (lower_border - avg_consts) / avg_slope
    )

    # Calculate upper border intersection
    x_lane_upper_point = int(
        (upper_border - avg_consts) / avg_slope
    )

    return [
        x_lane_lower_point,
        lower_border,
        x_lane_upper_point,
        upper_border
    ]


# Define ROI boundaries
roi_upper_border = 340
roi_lower_border = 540

# Create blank image for lane results
lanes_img = np.zeros(
    (img.shape[0], img.shape[1], 3),
    dtype=np.uint8
)

# Separate left and right lane lines
lines_left, lines_right = separate_left_right_lines(lines)

# Extrapolate lane lines
lane_left = extrapolate_lines(
    lines_left, roi_upper_border, roi_lower_border
)

lane_right = extrapolate_lines(
    lines_right, roi_upper_border, roi_lower_border
)

# Draw extrapolated lane lines
draw_lines(lanes_img, [[lane_left]], thickness=10)
draw_lines(lanes_img, [[lane_right]], thickness=10)

# Display final results
fig = plt.figure(figsize=(20, 20))

ax = fig.add_subplot(1, 2, 1)
plt.imshow(hough)
ax.set_title('Before extrapolation')

ax = fig.add_subplot(1, 2, 2)
plt.imshow(lanes_img)
ax.set_title('After extrapolation')

plt.show()
```

## Developed By

**Name:** Varun A

**Register No:** 212224240178

**Slot No:** ___________________

## Output

### 1. Input Image and Grayscale

<img width="1327" height="746" alt="Screenshot 2026-11-24 085939" src="https://github.com/user-attachments/assets/c6c0a094-cb1b-4631-ad21-7c442bba7ec4" />

### 2. Threshold Image

<img width="706" height="712" alt="Screenshot 2026-11-24 090015" src="https://github.com/user-attachments/assets/0dafa804-a0f8-41c8-9ee3-e47d76d36899" />


### 3. Region of Interest

<img width="1411" height="627" alt="Screenshot 2026-11-24 090027" src="https://github.com/user-attachments/assets/5f8f95d3-bb46-4041-a0e6-9f6bc796dcda" />

### 4. Edge Detection

<img width="1307" height="708" alt="Screenshot 2026-11-24 090036" src="https://github.com/user-attachments/assets/ed426c4e-701c-4b84-854e-727162310cbc" />


### 5. Hough Transform

<img width="788" height="712" alt="Screenshot 2026-11-24 090109" src="https://github.com/user-attachments/assets/f146dde2-c3d4-4ee0-a3ed-d68ac67840c7" />


### 6. Lane Detection and Extrapolation

<img width="1551" height="753" alt="Screenshot 2026-11-24 090122" src="https://github.com/user-attachments/assets/285362ce-a89a-48c1-9d84-a4df53bfe967" />


## Result

Thus, lane detection using Hough Transform is successfully implemented using OpenCV. The system detects and highlights the left and right lane boundaries through image processing and line extrapolation.


