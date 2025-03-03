
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

def rgb_to_ycbcr(image):
    """ Convert RGB image to YCbCr color space """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

def ycbcr_to_rgb(image):
    """ Convert YCbCr image back to RGB color space """
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)

def dct2(block):
    """ Apply 2D Discrete Cosine Transform (DCT) """
    return scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """ Apply 2D Inverse Discrete Cosine Transform (IDCT) """
    return scipy.fftpack.idct(scipy.fftpack.idct(block.T, norm='ortho').T, norm='ortho')

def quantize(block, quant_matrix):
    """ Quantize the DCT coefficients using a quantization matrix """
    return np.round(block / quant_matrix)

def dequantize(block, quant_matrix):
    """ Dequantize the DCT coefficients using a quantization matrix """
    return block * quant_matrix

# Standard JPEG Quantization Matrix (Luminance)
quant_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

# Load an image
image = cv2.imread('example.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ycbcr_image = rgb_to_ycbcr(image)

# Get Y (Luminance) Channel
y_channel = ycbcr_image[:, :, 0]

# Divide into 8x8 blocks and apply DCT, Quantization
compressed_blocks = np.zeros_like(y_channel, dtype=np.float32)
rows, cols = y_channel.shape

for i in range(0, rows, 8):
    for j in range(0, cols, 8):
        block = y_channel[i:i+8, j:j+8]
        dct_block = dct2(block)
        quantized_block = quantize(dct_block, quant_matrix)
        compressed_blocks[i:i+8, j:j+8] = quantized_block

# Decompress: Dequantize, Apply IDCT
reconstructed_blocks = np.zeros_like(compressed_blocks)

for i in range(0, rows, 8):
    for j in range(0, cols, 8):
        quantized_block = compressed_blocks[i:i+8, j:j+8]
        dequantized_block = dequantize(quantized_block, quant_matrix)
        reconstructed_block = idct2(dequantized_block)
        reconstructed_blocks[i:i+8, j:j+8] = reconstructed_block

# Clip values to valid range
reconstructed_blocks = np.clip(reconstructed_blocks, 0, 255).astype(np.uint8)

# Replace Y channel with decompressed Y and convert back to RGB
ycbcr_image[:, :, 0] = reconstructed_blocks
reconstructed_image = ycbcr_to_rgb(ycbcr_image)

# Display original and compressed image
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image)
axs[0].set_title("Original Image")
axs[0].axis('off')

axs[1].imshow(reconstructed_image)
axs[1].set_title("JPEG Compressed Image")
axs[1].axis('off')

plt.show()
