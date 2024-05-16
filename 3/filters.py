import cv2

# Read 1 image như binh thường
img = cv2.imread("zelda.jpg")

# GaussianBlur Filter: hiệu ứng làm mờ ảnh
## SigmaX, SigmaY: độ lớn của Gaussian kernel, càng lớn càng mờ
## BorderType: cách xử lý các pixel ở biên
img = cv2.GaussianBlur(img, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
