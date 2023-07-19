import numpy as np
import cv2
import matplotlib.pyplot as plt
from fil_finder import FilFinder2D
import astropy.units as u

# Load the image
image = cv2.imread('assets/antlerscropped.jpeg')

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

#Detect the branches
fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
fil.preprocess_image(flatten_percent=85)
fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
fil.medskel(verbose=False)
fil.analyze_skeletons(branch_thresh=10* u.pix, skel_thresh=20 * u.pix, prune_criteria='length')

#Output number of branches and branch lengths
print(f""" The number of branches are: {(int) (fil.branch_properties['number'][0] - 1 )/2} """)
print(f"""The length of all branches are: {fil.branch_lengths(u.pix)[0]} """)
plt.imshow(fil.skeleton, cmap='gray')
plt.contour(fil.skeleton_longpath, colors='r')
plt.axis('off')
plt.show()