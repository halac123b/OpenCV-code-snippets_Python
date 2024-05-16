import cv2

# Read 1 image như binh thường
img = cv2.imread("zelda.jpg")
# Thêm flag:
## cv2.IMREAD_COLOR: Đọc ảnh màu (default)
## cv2.IMREAD_GRAYSCALE: Đọc ảnh đen trắng
img = cv2.imread("zelda.jpg", flags=cv2.IMREAD_GRAYSCALE)

# Giới hạn giá trị của các màu trong ảnh
## Flag: cv2.THRESH_BINARY: Lấy giá trị 0 hoặc 255
## 128: threshold, nếu dưới là 0, trên là 255
## Return: [0]: ret, [1]: ảnh sau khi threshold
grayscale_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]

# Chuyển qua lại giữa các hệ màu
## Flag: BGR2GRAY, BGR2RGB, ...
## OpenCV read image và lưu trữ dạng BGR
rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get average color của 1 vùng trong ảnh
## mask: ảnh đen trắng, chỉ các ảnh trắng mới được tính trung bình
## return: tupe BGRA, ở đây chỉ lấy 3 giá trị đầu
avg_color = cv2.mean(img, mask=grayscale_img)[:3]

# Add each pixel of img to an average color
# Nếu tổng vượt quá data type, sẽ tự threshold tại max
new_img = cv2.add(img, avg_color)

# Merge 3 single-channel images thành 1 3-channel image
## Mục đích: dùng để tương tác với các ảnh 3-channel khác (các hàm thường yêu cầu input cùng số channel)
new_img = cv2.merge([grayscale_img, grayscale_img, grayscale_img])
