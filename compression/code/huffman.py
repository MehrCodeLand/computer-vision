import cv2 
import numpy as np 
import heapq
from collections import Counter, namedtuple



class Node(namedtuple("Node", ["value","freq","left","right"])):
    def __lt__(self,other):
        return self.freq < other.freq
    
def build_huffman_tree(frequencies):
    heap = [Node(value,freq,None , None ) for value , freq in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None,left.freq + right.freq, left , right)
        heapq.heappush(heap,merged)

    return heap[0]