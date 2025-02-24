
import numpy as np
import cv2

from scipy.optimize import curve_fit
from scipy import ndimage

from skimage import filters, measure, morphology
from skimage.segmentation import clear_border
from skimage.util import img_as_ubyte, invert
from skimage.draw import polygon
from skimage import measure, morphology, filters
from skimage.measure import label, regionprops
try:
    from skimage.morphology.footprint import rectangle  # Newer versions (0.22+)
except ImportError:
    from skimage.morphology import rectangle  # Older versions (pre-0.22)

from tqdm import tqdm

import os

# import pandas as pd

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from tqdm import tqdm
import tabulate  # only used for displaying Aggs structure

# Import custom modules.
import tools


# To access a field, e.g.: [d['da'] for d in Aggs] OR:
def get(Aggs, key):
    p = np.array([])
    for Agg in Aggs:
        p = np.append(p, Agg[key])
    return p

# Function to show structure.
def show(Aggs):
    header = Aggs[0].keys()
    rows = [Agg.values() for Agg in Aggs]
    print(tabulate.tabulate(rows, header))

# Write a simplified version of .
def write(Aggs):
    pass
    # remove fields that are not scalar or small number of values (rect okay, don't need it though)


def seg_kmeans(imgs, pixsizes, opts0='default'):
    """
     Compiling these different feature layers results in a three 
     layer image (see FEATURE_SET output) that will be used for segmentation. 
     This is roughly equivalent to segmenting colour images, if each of 
     the layers was assigned a colour. For example, compilation of these 
     feature layers for the sample images results in the following feature 
     layers and compiled RGB image. More details of the method are 
     given in the associated paper (Sipkens and Rogak, 2021). Finally,  
     applying Matlab's `imsegkmeans(...)` function, one achieved a 
     classified image.
     
     While this will likely be adequate for many users, the 
     technique still occasionally fails, particularly if the 
     function does not adequately remove the background. The 
     method also has some notable limitations when images are 
     (i) *zoomed in* on a single aggregate while (ii) also slightly 
     overexposed. 
     
     The k-means method is associated with configuration files 
     (cf., +agg/config/), which include different versions and allow 
     for tweaking of the options associated with the method. 
     See VERSIONS below for more information. 
     
     ------------------------------------------------------------------------
     
     VERSIONS: 
      Previous versions, deprecated, used different feature layers and weights.
       <strong>6+</strong>:  Three, equally-weighted feature layers as  
            described by Sipkens and Rogak (J. Aerosol Sci., 2021). 
       <strong>6.1</strong>: Improves the adjusted feature layer 
            for clumpy aggregates.
       <strong>6.2</strong>: Switches to s-curve fitting for  
            computing the adjusted threhold layer. Not preferred.
     
     ------------------------------------------------------------------------
    
     IMG_BINARY = agg.seg_kmeans(IMGS) requires an IMGS data structure, with 
     a cropped version of the images and the pixel sizes. The output is a 
     binary mask. 
    
     IMG_BINARY = agg.seg_kmeans(IMGS,PIXSIZES) uses a cell array of cropped
     images, IMGS, and an array of pixel sizes, PIXSIZES. The cell array of
     images can be replaced by a single image. The pixel size is given in
     nm/pixel. If not given, 1 nm/pixel is assumed, with implications for the
     rolling ball transform. As before, the output is a binary mask. 
    
     IMG_BINARY = agg.seg_kmeans(IMGS,PIXSIZES,OPTS) adds a options data 
     structure that controls the minimum size of aggregates (in pixels) 
     allowed by the program. 
    
     [IMG_BINARY,IMG_KMEANS] = agg.seg_kmeans(...) adds an output for the raw
     k-means clustered results, prior to the rolling ball transform. 
    
     [IMG_BINARY,IMG_KMEANS,FEATURE_SET] = agg.seg_kmeans(...) adds an 
     additional output for false RGB images with one colour per feature layer 
     used by the k-means clustering. 
     
     ------------------------------------------------------------------------
     
     AUTHOR: Timothy Sipkens, 2020-08-13
    """
    
    opts = tools.load_config(os.path.join(os.path.dirname(__file__), f'config\\km.{opts0}.yaml'))
    
    n = len(imgs)
    img_binary = [None] * n
    img_kmeans = [None] * n
    feature_set = [None] * n

    print(f'Performing k-means segmentation ({opts0}).')
    for ii in tqdm(range(n)):
        img = imgs[ii]
        pixsize = pixsizes[ii]
        morph_param = 0.8 / pixsize  # morphological scale parameter used several places below

        img = bg_subtract(img)

        # FEATURE 3: Denoise image.
        img_denoise = cv2.bilateralFilter(img, d=15, sigmaColor=650, sigmaSpace=1)
        
        # FEATURE 1: Entropy.
        se = morphology.disk(round(5 * opts['morphsc']))
        # se = morphology.disk(round(25 * morph_param * opts['morphsc']))  # updated in Python
        i10 = cv2.morphologyEx(img_denoise, cv2.MORPH_BLACKHAT, se)

        i11 = filters.rank.entropy(i10, rectangle(15, 15))
        i11 = i11 / np.max(i11) * 255  # scale before converting to int
        i11 = i11.astype(np.uint8)

        se12 = morphology.disk(max(round(5 * morph_param), 1))
        # se12 = morphology.disk(max(round(25 * morph_param * opts['morphsc']), 1))  # updated in Python
        i12 = cv2.morphologyEx(i11, cv2.MORPH_CLOSE, se12)

        # Considered in the Python version.
        # Scale image and enhance contrast.
        # i12 = i12 - np.min(i12)  # scale image
        # i12 = 255 * (i12 / np.max(i12))
        # i12 = tools.enhance_contrast([i12.astype(np.uint8)], 1.5)[0].astype(np.float32)

        # FEATURE 2: Adjusted Otsu.
        i1 = img_as_ubyte(img_denoise)
        i1 = cv2.GaussianBlur(i1, (0,0), round(5 * morph_param), round(5 * morph_param))

        lvl2 = filters.threshold_otsu(i1)
        i2a = i1 < lvl2

        lvl3 = eval(opts['lvl3'])
        n_in = np.ones_like(lvl3)
        for ll in range(len(lvl3)):
            n_in[ll] = np.sum(i1 < lvl2 * lvl3[ll])
        for ll in range(len(lvl3) - 10):
            n_in[ll] = np.sum(n_in[np.arange(10) + ll]) / 10
        n_in[-10:-1] = max(n_in)
        n_in[-1] = max(n_in)

        if opts['lvlfun'] == 'lin':
            p = np.polyfit(lvl3[:10], n_in[:10], 1)
            n_in_pred = np.polyval(p, lvl3)
            lvlfun = (n_in - n_in_pred) / (n_in_pred + np.finfo(float).eps)
        else:
            fun = lambda x: max(n_in) / (1 + np.exp(-x[0] * (lvl3 - x[1])))
            from scipy.optimize import least_squares
            x1 = least_squares(lambda x: n_in - fun(x), [60, 1.2])
            lvlfun = fun(x1.x) / max(n_in)

        lvl4 = np.where(lvlfun > opts['lvl5'])[0]
        if lvl4.size == 0:
            lvl4 = [1]
        lvl4 = lvl3[lvl4[0]]
        i2b = i1 < (lvl2 * lvl4)

        se3 = morphology.disk(max(round(5 * morph_param), 1))
        i3 = cv2.morphologyEx(i2b.astype(np.uint8), cv2.MORPH_CLOSE, se3)

        i5 = np.zeros_like(i2b)
        bw1 = measure.label(i3)
        for jj in range(1, bw1.max() + 1):
            if np.any(i2a[bw1 == jj]):
                i5[bw1 == jj] = 1
        i5 = cv2.GaussianBlur(img_as_ubyte(i5), (0,0), int(3.75 * opts['morphsc']), int(3.75 * opts['morphsc']))

        # COMPILE FEATURES.
        feature_set[ii] = np.stack([i12, i5, img_denoise], axis=2).astype(np.float32)
        fs = feature_set[ii]

        # Reshape for k-means (each pixel as a feature vector)
        h, w, c = fs.shape
        fs = fs.reshape(-1, c)  # Convert (H, W, 3) → (H*W, 3)

        # Scale feature set removed in Python version.
        fs = fs - np.mean(np.mean(fs, 0), 0)
        fs = fs / np.std(fs.reshape(-1, 3), 0)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(fs)

        # Reshape labels back to image size
        # and convert to binary mask.
        bw = labels.reshape(h, w) == 1

        # Identify the segment with the maximum mean intensity.
        mean_values = [img_denoise[bw].mean(), img_denoise[~bw].mean()]
        ind_max = np.argmax(mean_values)

        # Segmented image.
        img_kmeans[ii] = (bw == (ind_max == 0))

        # Reverse BG and labeled data. 
        if np.mean(img[img_kmeans[ii] == 1]) > np.mean(img[img_kmeans[ii] == 0]):
            img_kmeans[ii] = 1 - img_kmeans[ii]

        ds = round(opts['morphsc'] * morph_param)
        se6 = morphology.disk(max(ds, 1))
        i7 = cv2.morphologyEx(img_kmeans[ii].astype(np.uint8), cv2.MORPH_CLOSE, se6)

        se7 = morphology.disk(max(ds - 1, 0))
        img_rb = cv2.morphologyEx(i7, cv2.MORPH_OPEN, se7)

        img_binary[ii] = morphology.remove_small_objects(img_rb.astype(bool), opts['minsize'])

    if n == 1:
        img_binary = img_binary[0]
        img_kmeans = img_kmeans[0]
        feature_set = feature_set[0]

    tools.textdone()

    return img_binary, img_kmeans, feature_set


def seg_otsu(imgs, pixsizes=None, *args):
    """
    Performs Otsu thresholding and a rolling ball transformation.

    Parameters:
        imgs (list or np.ndarray): A list of cropped images or a single image.
        pixsizes (list or float, optional): Pixel sizes in nm/pixel. Default is 1.
        *args: Arguments to be passed to the rolling_ball operation.

    Returns:
        list or np.ndarray: Binary mask(s).
    """
    # Handle inputs
    if pixsizes is None:
        raise ValueError("PIXSIZES is a required argument unless images structure is given.")
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(pixsizes, list):
        pixsizes = [pixsizes] * len(imgs)

    n = len(imgs)
    img_binary = []

    print("Performing Otsu segmentation:")
    for ii in tqdm(range(n)):
        img = imgs[ii]
        pixsize = pixsizes[ii]

        # Step 0a: Remove the background
        img = bg_subtract(img)

        # Step 0b: Perform denoising
        img = cv2.bilateralFilter(img, d=15, sigmaColor=650, sigmaSpace=1)

        # Step 1: Apply intensity threshold (Otsu)
        level = filters.threshold_otsu(img)
        bw = img > level

        # Step 2: Rolling Ball Transformation
        binary = rolling_ball(bw, pixsize, *args)
        img_binary.append(invert(binary))
    tools.textdone()

    # If a single image, return the binary mask directly
    return img_binary[0] if n == 1 else img_binary


def bg_subtract(img):
    """
    Subtracts the background from the image using a rolling ball transformation and polynomial fitting.
    
    Parameters:
        img (np.ndarray): 2D numpy array representing the image.
    
    Returns:
        img_out (np.ndarray): Image with background subtracted.
        bg (np.ndarray): Background image (fit surface).
    """

    def poly22(xy, a, b, c, d, e, f):
        x, y = xy
        return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f
    
    # -- Rolling ball transformation to determine the background --------------
    se_bg = morphology.disk(80)
    pre_bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se_bg)
    
    # -- Fit surface ----------------------------------------------------------
    X, Y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    X = X.flatten()
    Y = Y.flatten()
    Z = pre_bg.flatten()

    # Fit polynomial surface of degree 2
    params, _ = curve_fit(poly22, (X, Y), Z, p0=[1, 1, 1, 1, 1, 1])
    
    # Evaluate the fit surface
    bg = poly22((X, Y), *params).reshape(img.shape)
    
    # Convert to uint8 and normalize
    t0 = np.max(bg) - bg
    t1 = img + t0
    t2 = t1 - np.min(t1)
    img_out = np.round(255 * t2 / np.max(t2)).astype(np.uint8)
    
    return img_out


def rolling_ball(img_binary, pixsize, minparticlesize=4.9, coeffs=None):
    """
    Perform a rolling ball transformation on a binary image.
    
    Parameters:
        img_binary (np.ndarray): Binary image.
        pixsize (float): Pixel size in nm/pixel.
        minparticlesize (float, optional): Minimum particle size. Default is 4.9.
        coeffs (list, optional): Coefficient matrix defining element sizes at different stages.
    
    Returns:
        np.ndarray: Transformed binary image.
    """
    # Default coefficient matrix
    coeff_matrix = np.array([
        [0.2, 0.8, 0.4, 1.1, 0.4],
        [0.2, 0.3, 0.7, 1.1, 1.8],
        [0.3, 0.8, 0.5, 2.2, 3.5],
        [0.1, 0.8, 0.4, 1.1, 0.5]
    ])
    
    # Select coefficients based on pixsize
    if coeffs is None:
        if pixsize <= 0.181:
            coeffs = coeff_matrix[0]
        elif pixsize <= 0.361:
            coeffs = coeff_matrix[1]
        else:
            coeffs = coeff_matrix[2]
    
    a, b, c, d, e = coeffs
    
    # Rolling ball transformations (morphological operations)
    se = morphology.disk(round(a * minparticlesize / pixsize))
    img_binary = morphology.closing(img_binary, se)
    
    se = morphology.disk(round(b * minparticlesize / pixsize))
    img_binary = morphology.opening(img_binary, se)
    
    se = morphology.disk(round(c * minparticlesize / pixsize))
    img_binary = morphology.closing(img_binary, se)
    
    se = morphology.disk(round(d * minparticlesize / pixsize))
    img_binary = morphology.opening(img_binary, se)
    
    # Delete small blobs below threshold area size
    labeled_img = label(np.abs(img_binary - 1))
    regions = regionprops(labeled_img)
    
    nparts = len(regions)
    mod = 10 if nparts > 50 else 1
    
    for region in regions:
        area = region.area * pixsize ** 2
        if area <= (mod * e * minparticlesize / pixsize) ** 2:
            img_binary[labeled_img == region.label] = 1
    
    return img_binary


#== ANALYSIS FUNCTIONS ===============================================================#
def analyze_binary(imgs_binary, pixsize=None, imgs=None, fname=None, f_edges=1, f_plot=0, maxagg=50):
    # Parse inputs
    if isinstance(imgs_binary, dict):  # consider case that structure is given as input
        Imgs = imgs
        imgs = [Imgs['cropped']]
        pixsize = [Imgs['pixsize']]
        fname = [Imgs['fname']]
    else:
        if not isinstance(imgs_binary, list):
            imgs_binary = [imgs_binary]

    if pixsize is None:
        pixsize = [1] * len(imgs_binary)

    if imgs is None:
        imgs = imgs_binary
        for ii in range(len(imgs)):
            imgs[ii] = np.uint8(155 * (~imgs[ii]) + 100)

    if not isinstance(imgs, list):
        imgs = [imgs]

    if fname is None:
        fname = [None] * len(imgs_binary)

    # Initialize figure for plot
    if f_plot == 1:
        f0 = plt.figure()

    Aggs = []  # Initialize Aggs structure
    id = 0

    print("Analyzing binaries")
    print("Progress:")
    for ii in tqdm(range(len(imgs_binary))):  # loop through provided images
        img_binary = imgs_binary[ii]
        img = imgs[ii]

        # Skip images with more than 25% boundary aggregates
        bwborder = np.logical_and(img_binary, clear_border(img_binary))
        if (np.count_nonzero(bwborder) / img_binary.size) > 0.25:
            continue

        # Check if any of the borders are >20% aggregate
        nn = [
            np.count_nonzero(img_binary[:, 0]) / img_binary.shape[0],
            np.count_nonzero(img_binary[:, -1]) / img_binary.shape[0],
            np.count_nonzero(img_binary[0, :]) / img_binary.shape[1],
            np.count_nonzero(img_binary[-1, :]) / img_binary.shape[1]
        ]
        if any(x > 0.2 for x in nn):
            ia = img_binary.copy()
            if nn[0] <= 0.2: ia[:, 0] = 0
            if nn[1] <= 0.2: ia[:, -1] = 0
            if nn[2] <= 0.2: ia[0, :] = 0
            if nn[3] <= 0.2: ia[-1, :] = 0

            img_edges = np.logical_xor(ia, morphology.remove_small_objects(ia))
            img_edm = ndimage.distance_transform_edt(~img_edges)
            m1, m2 = np.unravel_index(np.argmax(img_edm), img_edm.shape)
            fun = lambda x: np.sqrt((np.indices(img_binary.shape) - x[0])**2 + (np.indices(img_binary.shape) - x[1])**2) > x[2]
            img_binary = np.logical_or(fun([m1, m2, img_edm[m1, m2] - 100]), img_binary)

        if f_edges:
            img_binary = clear_border(img_binary)

        img_binary = morphology.remove_small_objects(img_binary, min_size=10)

        # Detect distinct aggregates
        labeled_img, naggs = ndimage.label(img_binary)
        if naggs > maxagg: continue

        if naggs == 0: continue

        Aggs0 = []
        for jj in range(1, naggs + 1):  # loop through number of found aggregates
            id += 1
            agg_jj = {
                'id': id,
                'img_id': ii,
                'fname': fname[ii],
                'pixsize': pixsize[ii]
            }
            if jj == 1:
                agg_jj['image'] = img
            else:
                agg_jj['image'] = None

            mask = (labeled_img == jj).astype(np.uint8)
            agg_jj['binary'] = mask

            _,_,rect = autocrop(img, mask)
            agg_jj['rect'] = rect

            row, col = np.where(mask)
            agg_jj['length'] = max(np.ptp(row), np.ptp(col)) * pixsize[ii]
            agg_jj['width'] = min(np.ptp(row), np.ptp(col)) * pixsize[ii]
            agg_jj['aspect_ratio0'] = agg_jj['length'] / agg_jj['width']

            # d = np.sqrt((row[:, None] - row[None, :]) ** 2 + (col[:, None] - col[None, :]) ** 2)
            # dmax = np.max(d)
            # idx1, idx2 = np.unravel_index(np.argmax(d), d.shape)
            # agg_jj['lmax'] = dmax * pixsize[ii]

            # dy = col[idx2] - col[idx1]
            # dx = row[idx2] - row[idx1]
            # d_line = np.abs(dy * row - dx * col) / np.sqrt(dx ** 2 + dy ** 2)
            # dd = np.ptp(d_line)
            # agg_jj['lmin'] = dd * pixsize[ii]
            # agg_jj['aspect_ratio'] = agg_jj['lmax'] / agg_jj['lmin']

            agg_jj['num_pixels'] = np.count_nonzero(mask)
            agg_jj['da'] = 2 * np.sqrt(agg_jj['num_pixels'] / np.pi) * pixsize[ii]
            agg_jj['area'] = agg_jj['num_pixels'] * (pixsize[ii] ** 2)
            agg_jj['Rg'] = gyration(mask, pixsize[ii])

            perimeter1 = np.sum(ndimage.binary_dilation(mask) - mask)
            perimeter3 = get_perimeter2(mask)
            agg_jj['perimeter'] = pixsize[ii] * max(perimeter1, perimeter3)

            agg_jj['circularity'] = 4 * np.pi * agg_jj['area'] / (agg_jj['perimeter'] ** 2)
            # agg_jj['compact_b'] = agg_jj['area'] / (np.pi * (agg_jj['lmax'] / 2) ** 2)

            # n = np.count_nonzero(mask)
            # p = np.sum(np.abs(mask[:, 1:] - mask[:, :-1])) + np.sum(np.abs(mask[1:, :] - mask[:-1, :]))
            # Ld = (4 * n - p) / 2
            # Ldmin = n - 1
            # Ldmax = 2 * (n - np.sqrt(n))
            # agg_jj['compact'] = (Ld - Ldmin) / (Ldmax - Ldmin)

            # entropy_img = filters.rank.entropy(img, morphology.disk(25))
            # agg_jj['entropy'] = np.mean(entropy_img[mask])

            # gx, gy = np.gradient(filters.gaussian(img, 3))
            # grad = np.sqrt(gx ** 2 + gy ** 2)
            # grad_vals, ds = binner(mask, grad)
            # agg_jj['sharp'] = np.log10(np.mean(grad_vals[ds <= 5])) - np.log10(np.mean(grad_vals[ds > 5]))

            # agg_grayscale = img[mask]
            # gray_extent = np.ptp(img)
            # agg_jj['depth'] = -(np.mean(agg_grayscale) - bg_level) / 255

            agg_jj['center_mass'] = [np.mean(row), np.mean(col)]

            Aggs0.append(agg_jj)

        Aggs.extend(Aggs0)
        if f_plot == 1:
            tools.imshow_agg(Aggs0, list([jj - 1]))
            plt.show()

    print('Complete.\n')

    return Aggs



def gyration(img_binary, pixsize):
    total_area = np.count_nonzero(img_binary)  # total area [px^2]

    xpos, ypos = np.where(img_binary)  # position of each pixel
    n_pix = xpos.size

    # Compute centroid.
    centroid_x = np.mean(xpos)
    centroid_y = np.mean(ypos)

    Ar2 = (xpos - centroid_x) ** 2 + (ypos - centroid_y) ** 2  # distance to centroid
    Rg = np.sqrt(np.sum(Ar2) / total_area) * pixsize

    return Rg


def get_perimeter2(img_binary):
    # Find the contours of the binary image
    contours = measure.find_contours(img_binary, 0.5)
    
    if len(contours) == 0:
        return 0  # No perimeter if no contours found

    # Assume the first contour is the perimeter we want
    contour = contours[0]
    x_mb, y_mb = contour[:, 1], contour[:, 0]

    # Group edges
    edges_mb = np.cumsum(np.concatenate(([1], (x_mb[1:] != x_mb[:-1]) & (y_mb[1:] != y_mb[:-1]))))
    edges_mb = edges_mb - 1  # adjust so that 0 is the first edge
    
    # Connect last pixel back to the first pixel
    if (x_mb[0] == x_mb[-1]) or (y_mb[0] == y_mb[-1]):
        edges_mb[edges_mb == edges_mb[-1]] = 1

    # Get midpoints of lines
    xx_mb = np.bincount(edges_mb, weights=x_mb) / np.bincount(edges_mb)
    yy_mb = np.bincount(edges_mb, weights=y_mb) / np.bincount(edges_mb)

    # Calculate perimeter by connecting midpoints
    p_circ = np.sum(np.sqrt((xx_mb - np.roll(xx_mb, -1))**2 + (yy_mb - np.roll(yy_mb, -1))**2))
    
    return p_circ


def box_counting(img_binary):
    # Find the contours of the binary image
    i, _, _ = autocrop(img_binary, img_binary)

    eps = np.arange(2, 6)
    eps = eps[1::2]

    N = np.zeros(len(eps))
    for ii in range(len(eps)):
        n = np.int16(eps[ii])
        i2 = morphology.dilation(i, np.ones((n,n)))
        i2 = i2[np.int16((n-1)/2)::n, np.int16((n-1)/2)::n]
        N[ii] = np.sum(i2)

    d = (np.log(N[1]) - np.log(N[0])) / (np.log(1/eps[1]) - np.log(1/eps[0]))
    return d
    


def autocrop(img_orig, img_binary):
    x, y = np.where(img_binary)

    space = 3
    size_img = img_orig.shape

    x_top = min(max(x) + space, size_img[0])
    x_bottom = max(min(x) - space, 0)
    y_top = min(max(y) + space, size_img[1])
    y_bottom = max(min(y) - space, 0)

    img_binary_cropped = img_binary[x_bottom:x_top, y_bottom:y_top]
    img_cropped = img_orig[x_bottom:x_top, y_bottom:y_top]
    rect = [y_bottom, x_bottom, y_top - y_bottom, x_top - x_bottom]

    return img_cropped, img_binary_cropped, rect
