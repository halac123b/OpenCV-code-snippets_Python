import cv2
import numpy as np
import skimage.exposure
import matplotlib.pyplot as plt

# specify desired RGB color for new face and make into Numpy array
desired_color = (131, 158, 215)
desired_color = np.asarray(desired_color, dtype=np.float64)

# Tạo array full màu desired_color (bản chất các image trong OpenCV cũng là các NumpyArray)
swatch = np.full(shape=(200, 200, 3), fill_value=desired_color, dtype=np.uint8)

# read image
img = cv2.imread("zelda.jpg")
# read face mask as grayscale and threshold to binary
facemask = cv2.imread("zelda_facemask.png", cv2.IMREAD_GRAYSCALE)
# Giới hạn giá trị của các màu trong ảnh
## Flag: cv2.THRESH_BINARY: Lấy giá trị 0 hoặc 255
## 128: threshold, nếu dưới là 0, trên là 255
## Return: [0]: ret, [1]: ảnh sau khi threshold
facemask = cv2.threshold(facemask, 128, 255, cv2.THRESH_BINARY)[1]

# get average bgr color of face
avg_color = cv2.mean(img, mask=facemask)[:3]
print(f"Average color: {avg_color}")

# compute difference colors and make into an image the same size as input
diff_color = desired_color - avg_color
diff_color = np.full_like(img, diff_color, dtype=np.uint8)

# shift input image color
new_img = cv2.add(img, diff_color)

# antialias mask, convert to float in range 0 to 1 and make 3-channels
facemask = cv2.GaussianBlur(
    facemask, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT
)

facemask = skimage.exposure.rescale_intensity(
    facemask, in_range=(100, 150), out_range=(0, 1)
).astype(np.float32)
facemask = cv2.merge([facemask, facemask, facemask])

# combine img and new_img using mask
result = img * (1 - facemask) + new_img * facemask
result = result.clip(0, 255).astype(np.uint8)

# Display the image using Matplotlib
rgb_image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image)
plt.title("Loaded Image")
plt.show()
