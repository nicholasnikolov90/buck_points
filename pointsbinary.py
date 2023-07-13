import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fil_finder import FilFinder2D
import astropy.units as u

# Load the image
image = cv2.imread('assets/antlers.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fil = FilFinder2D(gray, distance=250 * u.pc)
fil.preprocess_image(flatten_percent=85)
#fil.create_mask(border_masking=True, smooth_size=3 * u.pix, verbose=False, use_existing_mask=True)
#fil.medskel(verbose=False)
#fil.analyze_skeletons(branch_thresh=40 * u.pix, skel_thresh=10 * u.pix, prune_criteria='length')

# Access the skeleton data
# Skeletonize the trunk
skeleton = cv2.ximgproc.thinning(fil)
cv2.imshow('skeleton', skeleton)


# Show the skeleton
plt.imshow(skeleton, cmap='gray')
plt.axis('off')
plt.show()
