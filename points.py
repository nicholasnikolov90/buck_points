import cv2
import numpy as np

# Load the image
image = cv2.imread('assets/antlers.jpeg')

# Preprocess the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Extract the trunk
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=2)
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

trunk_contour = max(contours, key=cv2.contourArea)
trunk_mask = np.zeros_like(edges)
cv2.drawContours(trunk_mask, [trunk_contour], 0, 255, thickness=cv2.FILLED)

# Skeletonize the trunk
skeleton = cv2.ximgproc.thinning(trunk_mask)

# Detect branches
branch_contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the branches on the original image
result = np.copy(image)
cv2.drawContours(result, branch_contours, -1, (0, 255, 0), 2)

# Display the result
cv2.imshow('Branches', result)
cv2.waitKey(0)
cv2.destroyAllWindows()