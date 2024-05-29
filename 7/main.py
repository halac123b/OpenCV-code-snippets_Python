import cv2
import numpy as np

# Open và read image, default uint8
img = cv2.imread("path_to_img")
# Convert sang int16, vì range lớn hơn nên k cần lo
new_img = img.astype(np.int16)

### Thực hiện các tác vụ khác (..) ###

# Chuyển đổi về uint8, lúc này giá trị có thể đã nằm ngoài range nên cần xử lí
# Normalize the image to 0-255 range
# Assuming the image has some known range, e.g., -32768 to 32767 for int16
img_normalized = cv2.normalize(new_img, None, 0, 255, cv2.NORM_MINMAX)
# Clip the values to ensure they are within 0-255
img_clipped = np.clip(img_normalized, 0, 255)
# Convert to uint8
new_img = img_clipped.astype(np.uint8)
