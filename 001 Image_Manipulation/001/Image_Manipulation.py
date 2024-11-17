import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image file
image = cv2.imread("monalisa.jpg")

# Convert the image color from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image using Matplotlib
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Get image properties
height, width, channels = image.shape
color_depth = image.dtype

# Print the properties of the image
print("Image Properties:")
print(f"Dimensions: {width} x {height}")
print(f"Number of channels: {channels}")
print(f"Color depth: {color_depth}")

# Define coordinates and dimensions for the rectangle
x, y, w, h = 100, 100, 32, 32

# Draw a rectangle on a copy of the image
image_with_rectangle = cv2.rectangle(
    image.copy(),
    (y, x),  # Top-left corner
    (y + h, x + w),  # Bottom-right corner
    (0, 0, 255),  # Red color in BGR
    3  # Thickness of the rectangle
)

# Display the image with the rectangle
plt.imshow(cv2.cvtColor(image_with_rectangle, cv2.COLOR_BGR2RGB))
plt.title("Image with Rectangle")
plt.axis("off")
plt.show()

# Crop the rectangle region from the original image
cropped_image = image[y:y + h, x:x + w]

# Display the cropped region
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title("Cropped Image")
plt.axis("off")
plt.show()

# Split the image into its blue, green, and red channels
b, g, r = cv2.split(image)

# Display each channel separately
"""
plt.subplots 函数详细说明：
--------------------------------
1. 功能：
   - 创建一个包含多个子图（subplots）的图形区域。
   - 子图按行和列排列，形成网格布局。
   - 每个子图可以独立绘制内容。

2. 参数解释：
   - nrows=1: 子图的行数，这里设置为 1 行。
   - ncols=3: 子图的列数，这里设置为 3 列。
   - figsize=(10, 4): 整个图形的宽度为 10 英寸，高度为 4 英寸。

3. 返回值：
   - fig: 整个绘图区域的 Figure 对象（画布）。
   - axes: 子图的 Axes 对象数组，每个子图一个 Axes。
"""
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

axes[0].imshow(b, cmap="Blues")
axes[0].set_title("Blue Channel")
axes[0].axis("off")

axes[1].imshow(g, cmap="Greens")
axes[1].set_title("Green Channel")
axes[1].axis("off")

axes[2].imshow(r, cmap="Reds")
axes[2].set_title("Red Channel")
axes[2].axis("off")

plt.show()

# Create an image highlighting the blue channel
"""
cv2.merge 函数详细说明：
--------------------------------
1. 功能：
   - 将多个单通道图像合并为一个多通道图像。
   - 在图像处理中，用于组合分离的通道，生成 RGB 或其他格式的彩色图像。

2. 参数解释：
   - cv2.merge((b, g, r)): 将三个通道（b, g, r）按顺序合并成一个三通道图像。
     - b: 蓝色通道。
     - g: 绿色通道。
     - r: 红色通道。
"""
blue_image = cv2.merge((b, np.zeros_like(g), np.zeros_like(r)))

# Display the blue effect image
plt.imshow(cv2.cvtColor(blue_image, cv2.COLOR_BGR2RGB))
plt.title("Blue Effect")
plt.axis("off")
plt.show()
