import random
from winreg import EnumValue
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("monalisa.jpg")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")
plt.show()

# Print image attributes
print("Image Attributes:")
print(f"Dimensions: {image.shape}")
print(f"Data Type: {image.dtype}")

image_size = image_rgb.shape
resolution = 16

indices_x = np.arange(0, image_size[1], resolution)
indices_y = np.arange(0, image_size[0], resolution)
points_x, points_y = np.meshgrid(indices_x, indices_y)

plt.imshow(image_rgb, cmap="gray")
plt.scatter(points_x, points_y, color="red", s=1, label="Grid Points")

"""
xticks 和 yticks 的作用：

xticks：设置或获取 x 轴的刻度值和标签。如果不传入参数，则只会显示当前的 x 轴刻度。
        如果提供参数，则可以自定义 x 轴刻度的位置和标签。

yticks：设置或获取 y 轴的刻度值和标签。如果不传入参数，则只会显示当前的 y 轴刻度。
        如果提供参数，则可以自定义 y 轴刻度的位置和标签。
"""
plt.xticks()
plt.yticks()
plt.show()

resolutions = [1, 2, 4, 8, 16]

num_rows = 1
num_cols = len(resolutions)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))
"""
enumerate(resolutions)：
   - 将列表 resolutions 转换为一个枚举对象。
   - 它在每次迭代时返回一个由两个元素组成的元组：索引 (index) 和列表中的对应值 (value)。
"""
for i, res in enumerate(resolutions):
    """
    NumPy 切片语法：

    1. 基本格式：
       - [start:stop:step] 表示切片的范围和步长。
         * start: 起始索引（默认为 0）。
         * stop: 结束索引（不包括 stop 本身，默认为数组的末尾）。
         * step: 步长（默认为 1）。
    """
    downsampled_image = image_rgb[::res, ::res, :]

    axes[i].imshow(downsampled_image)
    axes[i].set_title(f"{res}x")

axes[0].set_ylabel("Downsampled Image")

if len(resolutions) < num_cols:
    for j in range(len(resolutions), num_cols):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


from PIL import Image

def downsample_image(image: np.ndarray, scale_factor: int) -> np.ndarray:
    height, width = image.shape[:2]
    new_width, new_height = int(width / scale_factor), int(height / scale_factor)
    downsampled_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            """
            axis=(0, 1)` 表示对第 0 维（行）和第 1 维（列）同时求平均值，即所有元素的均值。
            """
            downsampled_image[y, x] = np.mean(
                image[y*scale_factor:(y+1)*scale_factor, x*scale_factor:(x+1)*scale_factor],
                axis=(0, 1)
            )

    return downsampled_image

scale_factors = [2, 4, 8, 16]

fig, axes = plt.subplots(1, len(scale_factors) + 1, figsize=(12, 4))

axes[0].imshow(image_rgb)
axes[0].set_title("Original")

for i, scale in enumerate(scale_factors):
    downsampled_image = downsample_image(image_rgb, scale)
    axes[i+1].imshow(downsampled_image)
    axes[i+1].set_title(f'Downsampled (1/{scale})')

plt.tight_layout()
plt.show()


image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
plt.imshow(image_gray,cmap='gray')
plt.axis("off")
plt.show()

quantization_levels = [2, 4, 8, 16, 32]

num_rows = 2
num_cols = len(quantization_levels)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

for i, levels in enumerate(quantization_levels):
    """
    quantized_image = (image_gray // (256 // levels)) * (256 // levels) 的含义：

    1. 图像量化的背景：
       - 灰度图像的像素值范围是 [0, 255]。
       - 量化（Quantization）是将连续的像素值分为有限的区间，用较少的级别表示像素值。
       - `levels` 表示量化级别（例如：2、4、8 等），即将灰度值划分为多少个区间。

    2. 表达式分解：
       - `256 // levels`：
         * 计算每个量化区间的宽度（步长）。
         * 例如，如果 `levels=4`，则每个区间的步长为 `256 // 4 = 64`。
         * 区间划分为：[0, 63], [64, 127], [128, 191], [192, 255]。

       - `image_gray // (256 // levels)`：
         * 将灰度值按区间宽度进行整数除法。
         * 结果是每个像素属于哪个区间的编号（从 0 开始编号）。
         * 例如，像素值 50 属于区间 [0, 63]，编号为 0；像素值 120 属于区间 [64, 127]，编号为 1。

       - `* (256 // levels)`：
         * 将区间编号还原为区间的起始值，完成量化。
         * 例如，编号为 0 的像素值映射到区间起始值 0，编号为 1 的像素值映射到区间起始值 64。

    3. 量化后的结果：
       - 量化后，所有像素值被替换为对应区间的起始值。
       - 例如，对于 `levels=4` 的量化：
         * 原始像素值 `[50, 120, 200]` 被量化为 `[0, 64, 192]`。
       - 量化后的图像只包含指定级别的灰度值。

    4. 整体逻辑：
       - 通过这条表达式，将灰度图像的像素值按照 `levels` 指定的量化级别进行量化处理。
       - 量化后的图像像素值更少，适用于减少数据复杂度或模拟低精度显示效果。
    """
    quantized_image = (image_gray // (256 // levels)) * (256 // levels)

    axes[0, i].imshow(quantized_image, cmap="gray")
    axes[0, i].set_title(f"{levels} Levels")

    axes[1, i].bar(range(levels), range(levels), color="gray", align="center", width=1)
    axes[1, i].set_title(f"{levels} Levels")

axes[0, 0].set_ylabel("Quantized Image")
axes[1, 0].set_ylabel("Mapping Table")

if len(quantization_levels) < num_cols:
    for j in range(len(quantization_levels), num_cols):
        fig.delaxes(axes[0, j])
        fig.delaxes(axes[1, j])

plt.tight_layout()
plt.show()