import cv2

# Load the high-res image
img = cv2.imread("test_upload.jpg")

# Force it into a 128x128 tensor shape (ignoring aspect ratio)
squished_img = cv2.resize(img, (128, 128))

# Save the ruined result!
cv2.imwrite("ruined_128.jpg", squished_img)