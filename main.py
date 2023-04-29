from PIL import Image
import numpy as np

# Open image file and convert it to grayscale
image = Image.open('stadium.bmp')
image.show()
# Convert image to NumPy array
img_array = np.array(image)

# Display shape of array
target_size = (67,200)

# Resize the array to the target size
resized_array = np.resize(img_array, target_size)

print('Array shape:', img_array.shape)

# Display array values
print('Array values:\n', resized_array)


img = Image.fromarray(resized_array)
# img.show()