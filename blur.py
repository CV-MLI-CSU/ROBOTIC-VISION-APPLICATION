# Python program to explain cv2.blur() method

# importing cv2
import cv2

# path
path = '../../Downloads/WX20210209-150422@2x.png'

# Reading an image in default mode
image = cv2.imread(path)
print(image)
print('oooo')
# Window name in which image is displayed
window_name = 'Image'

cv2.imshow(window_name, image)

# ksize
ksize = (15, 15)

# Using cv2.blur() method
image = cv2.blur(image, ksize)

# Displaying the image
cv2.imshow(window_name, image)
cv2.imwrite('../../Downloads/WX20210209-150422@2x.jpg',image)
