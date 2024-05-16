import cv2
import matplotlib.pyplot as plt

img = cv2.imread("zelda.jpg")

# OpenCV read image dùng BGR, còn Matplotlib dùng RGB, nên cần convert qua RGB
rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(rgb_image)
# Title của Plot
plt.title("Loaded Image")
# Show plot window
plt.show()
