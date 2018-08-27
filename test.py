import pandas as pd
import numpy as np
from skimage import io
from skimage.transform import rotate
import cv2
import os
# import os.path
import time

for path,dirs,files in os.walk('/home/osho/Desktop/test/data/'):
	for file in files:
		print (path,file)
	