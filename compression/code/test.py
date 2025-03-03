import cv2
import numpy as np
import heapq
import matplotlib.pyplot as plt
from collections import Counter, namedtuple

# Define Node class for Huffman Tree
class Node(namedtuple("Node", ["value", "freq", "left", "right"])):
    def __lt__(self, other):
        return self.freq < other.freq

# Build Huffman Tree
def build_huffman_tree(frequencies):
    heap = [Node(value, freq, None, None) for value, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)
    
    return heap[0]

# Generate Huffman Codes
def generate_huffman_codes(node, prefix="", codebook={}):
    if node is not None:
        if node.value is not None:
            codebook[node.value] = prefix
        generate_huffman_codes(node.left, prefix + "0", codebook)
        generate_huffman_codes(node.right, prefix + "1", codebook)
    return codebook

# Huffman Encode
def huffman_encode(image):
    """ Huffman encode a grayscale image """
    flat_pixels = image.flatten()
    frequencies = Counter(flat_pixels)
    
    huffman_tree = build_huffman_tree(frequencies)
    huffman_codes = generate_huffman_codes(huffman_tree)
    
    encoded_image = "".join(huffman_codes[pixel] for pixel in flat_pixels)
    return huffman_tree, huffman_codes, encoded_image

# Huffman Decode
def huffman_decode(encoded_data, huffman_tree, shape):
    """ Decode Huffman encoded image back into grayscale """
    decoded_pixels = []
    node = huffman_tree
    
    for bit in encoded_data:
        node = node.left if bit == "0" else node.right
        if node.value is not None:
            decoded_pixels.append(node.value)
            node = huffman_tree
    
    return np.array(decoded_pixels, dtype=np.uint8).reshape(shape)

# Load a grayscale image
image = cv2.imread('example_grayscale.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found! Please use a valid grayscale image.")

# Encode using Huffman Coding
huffman_tree, huffman_codes, encoded_image = huffman_encode(image)

# Decode to reconstruct the image
decoded_image = huffman_decode(encoded_image, huffman_tree, image.shape)

# Display the original and reconstructed image
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(decoded_image, cmap='gray')
axs[1].set_title('Reconstructed Image (Huffman Decoded)')
axs[1].axis('off')

plt.show()
