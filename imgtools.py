import os
import cv2
import matplotlib.pyplot as plt
import textwrap
import numpy as np
from typing import Union


version = 2


def load_id(path: str, imageids: Union[str, list]):
	"""Searches and loads images identified their 'imageid' in path.

	Parameters
	----------

	path : str
		Path to folder containing images.
			
	imageids : str, list of str
		List of 'imageid' strings of images.


	Returns
	-------

	images : list of numpy.ndarray
		Images encoded as described in the docs of cv2.imread().
	"""
	
	filenames = os.listdir(path)
	
	if isinstance(imageids, str): # single image
	
		img_id = imageids
		
		# Search pattern in filename
		pattern = 'image_{}_'.format(img_id)
		matches = [name for name in filenames if pattern in name]
		
		# Check results
		if len(matches) < 1:
			raise FileNotFoundError("No image matching {} was found.".format(pattern))
		elif len(matches) > 1:
			raise FileNotFoundError("Multiple images matching {} were found: \n{}".format(pattern, matches))
		else:
			img_name = matches[0]
		
		# Load & save
		img = cv2.imread(os.path.join(path, img_name))
		
		return img
	
	else: # multiple images
	
		images = list()
		
		# Iterate through IDs
		for img_id in imageids:
			
			# Search pattern in filename
			pattern = 'image_{}_'.format(img_id)
			matches = [name for name in filenames if pattern in name]
			
			# Check results
			if len(matches) < 1:
				raise FileNotFoundError("No image matching {} was found.".format(pattern))
			elif len(matches) > 1:
				raise FileNotFoundError("Multiple images matching {} were found: \n{}".format(pattern, matches))
			else:
				img_name = matches[0]
			
			# Load & save
			img = cv2.imread(os.path.join(path, img_name))
			images.append(img)
		
	return images



def disp_grid(images: list, rows: int, cols: int, titles: list=None, suptitle: str=None):
	"""Displays images in a grid fashion.

	Parameters
	----------

	images : list of numpy.ndarray
		Images encoded as described in the docs of cv2.imread().
			
	rows : int
		Number of rows. Number of rows times number columns must be higher than length of images list.

	cols : int
		Number of rows. Number of rows times number columns must be higher than length of images list.
			
	titles : list of str, optional
		Titles for images. Length must match images list length.
		
	suptitle : str, optional
		Figure title.


	Returns
	-------

	None
	"""
	
	fig = plt.figure(figsize=(16,9))
	
	if rows * cols < len(images):
		raise ValueError("Grid size is too small to plot all images.")
	
	if titles is not None and len(titles) != len(images):
		raise ValueError("Length of titles and images do not match.")

	for i in range(len(images)):
		ax = fig.add_subplot(rows, cols, i+1)		
		ax.imshow(images[i])
		ax.axis('off')
		if titles is not None:
			title = textwrap.fill(titles[i], width=40, break_long_words=False)
			ax.set_title(title, fontsize=10)
		if suptitle:
			fig.suptitle(suptitle, fontsize=16)



def mean_img(images: list):
	"""Computes mean of images.

	Parameters
	----------

	images : list of numpy.ndarray
		Images encoded as described in the docs of cv2.imread().
			
	Returns
	-------

	images_mean : numpy.ndarray
		Image encoded as described in the docs of cv2.imread().
	"""
	
	images_mean = np.asarray(images).mean(axis=0).astype('int')
	
	return images_mean