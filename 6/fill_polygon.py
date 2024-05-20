import cv2
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread("../images/sky.jpg")

# Create a mask fill with black
src_mask = np.zeros(src.shape, src.dtype)
# Define the polygon
poly = np.array(
    [[4, 80], [30, 54], [151, 63], [254, 37], [298, 90], [272, 134], [43, 122]],
    np.int32,
)
# Fill the polygon with white color
cv2.fillPoly(src_mask, [poly], (255, 255, 255))

rgb_image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image)
plt.title("Loaded Image")
plt.show()
