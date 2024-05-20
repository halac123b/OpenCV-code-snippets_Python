import cv2
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread("../images/sky.jpg")

# Define the coordinates and the dot size
x, y = 256, 226
dot_size = 10
color = (0, 0, 255)  # Red color in BGR

# Draw the dot
cv2.circle(src, (x, y), dot_size, color, -1)

rgb_image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image)
plt.title("Loaded Image")
plt.show()
