import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt, convolve
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from skimage.morphology import skeletonize, disk
from skimage.measure import label, regionprops
from skimage.filters import median, unsharp_mask
from skimage.feature import peak_local_max, canny
from skimage.transform import hough_circle, hough_circle_peaks

import cv2

from tqdm import tqdm

# Import custom modules.
import tools


def smooth(d, width):
    """
    Replicate MATLAB smooth function. 
    """
    ds = np.convolve(d, np.ones(width, dtype=int), 'valid') / width    
    r = np.arange(1, width - 1,2)

    # Replicate MATLAB at edges.
    initial = np.cumsum(d[:width-1])[::2]/r
    end = (np.cumsum(d[:-width:-1])[::2]/r)[::-1]

    return np.concatenate((initial, ds, end))


def pcm(Aggs, f_plot=False, f_backup=False, opts=None):
    """
    Performs the pair correlation method (PCM) of aggregate characterization.
    
    Developed at the University of British Columbia by Ramin Dastanpour and
    Steven N. Rogak. Modified by Timothy Sipkens.

    Parameters:
    Aggs (list of dict): List of dictionaries, each representing an aggregate.
    f_plot (bool): Flag for plotting pair correlation functions.
    f_backup (bool): Flag for saving backup copies during evaluation.
    opts (dict or str): Configuration options or path to config file.

    Returns:
    list of dict: Updated list of aggregates with characterization data.
    """

    # Default options
    default_opts = 'config/pcm.v1.s.json'
    
    # Handle options
    if opts is None:
        opts = default_opts
    elif isinstance(opts, str):
        if not opts.startswith('+pp'):
            opts = f'config/pcm.{opts}.json'

    opts = tools.load_config('config/pcm.v1.s.json')

    # Check if the data folder exists, create if not
    if not os.path.exists('data'):
        os.mkdir('data')

    if f_plot:
        plt.figure()  # create figure for visualizing current aggregate

    # Main image processing loop
    n_aggs = len(Aggs)
    print('Performing PCM loop:')

    for aa in tqdm(range(len(Aggs))):
        agg_aa = Aggs[aa]

        pixsize = agg_aa.get('pixsize')
        img_binary = tools.imcrop(agg_aa.get('binary'), agg_aa.get('rect'))  # crop the binarized image

        if np.isnan(pixsize):
            agg_aa['dp_pcm'] = np.nan
            agg_aa['dp'] = agg_aa['dp_pcm']
            continue

        # Step 3-3: Development of the pair correlation function (PCF)
        skel = skeletonize(img_binary)  # get aggregate skeleton
        skel_y, skel_x = np.where(skel)  # find skeleton pixels

        row, col = np.where(img_binary)
        thin = max(round(agg_aa.get('num_pixels') / 6e3), 1)
        X = col[::thin]
        Y = row[::thin]

        d_vec = np.sqrt((X[:, None] - skel_x) ** 2 + (Y[:, None] - skel_y) ** 2)
        d_vec = d_vec[d_vec > 0].flatten()  # vectorize the output and remove zeros

        # Construct the pair correlation
        d_max = int(max(d_vec))
        d_vec = d_vec * pixsize  # vector of distances in nm
        r = np.arange(1, (d_max * pixsize) + 1)  # radius vector in nm

        bins = r - 0.5
        bins = np.append(bins, r[-1] + 0.5)
        pcf = np.histogram(d_vec, bins=bins)[0]
        idx_p = np.nonzero(pcf)[0]
        pcf = pcf[idx_p]
        r1 = r[idx_p]

        # Smoothing the pair correlation function (PCF)
        d = 5 + 2 * d_max
        bw = np.zeros((d, d))
        bw[d_max + 3, d_max + 3] = 1
        bw = distance_transform_edt(bw == 0)
        bw = bw / d_max
        bw = bw < 1

        row, col = np.where(bw)
        d_denominator = np.sqrt((row - d_max + 3) ** 2 + (col - d_max + 3) ** 2)
        d_denominator = d_denominator[d_denominator > 0] * pixsize

        denominator = np.histogram(d_denominator, bins=bins)[0]
        denominator = denominator[idx_p] * len(skel_x) / thin
        denominator[denominator == 0] = 1  # bug fix, overcomes division by zero
        pcf = pcf / denominator
        pcf_smooth = smooth(pcf, 5)  # simple smoothing
        pcf_smooth[0] = pcf_smooth[1]  # as first point is often erroneous

        # Adjust PCF to be monotonically decreasing
        if opts['smooth'] == 1:
            for kk in range(len(pcf_smooth) - 1):
                if pcf_smooth[kk] <= pcf_smooth[kk + 1]:
                    pcf_smooth[kk + 1] = pcf_smooth[kk] - 1e-12
        else:
            pcf_smooth, r1 = tools.pcf(img_binary)
            r1 = r1 * pixsize

            # Remove zero entries
            nonzero_mask = pcf_smooth > 0
            pcf_smooth = pcf_smooth[nonzero_mask]
            r1 = r1[nonzero_mask]

            pcf_max = np.max(pcf_smooth)
            pcf_smooth[:np.argmax(pcf_smooth) + 1] = pcf_max + np.arange(np.argmax(pcf_smooth) + 1, 0, -1) * 1e-12

            for kk in range(len(pcf_smooth) - 1):
                if pcf_smooth[kk] <= pcf_smooth[kk + 1]:
                    pcf_smooth[kk + 1] = pcf_smooth[kk] - 1e-12

        # Normalize by maximum, depending on options
        if opts['norm'] == 'max':
            pcf_smooth /= np.max(pcf_smooth)

        if len(pcf_smooth) == 1:
            agg_aa['dp_pcm1'] = agg_aa['da']
            agg_aa['dp'] = agg_aa['dp_pcm1']
            continue

        # Step 3-5: Primary particle sizing
        if opts['type'] == 'simple':
            pcf_0 = 0.913
        else:
            Rg_u = 1.1 * agg_aa['Rg']
            Rg_l = 0.9 * agg_aa['Rg']
            pcf_Rg = np.interp(agg_aa['Rg'], r1, pcf_smooth)
            pcf_Rg_u = np.interp(Rg_u, r1, pcf_smooth)
            pcf_Rg_l = np.interp(Rg_l, r1, pcf_smooth)
            Rg_slope = (pcf_Rg_u + pcf_Rg_l - pcf_Rg) / (Rg_u - Rg_l)
            pcf_0 = (0.913 / 0.84) * (0.7 + 0.003 * Rg_slope ** -0.24 + 0.2 * agg_aa['aspect_ratio'] ** -1.13)

        # Get PCM diameter
        agg_aa['dp_pcm'] = 2 * np.interp(pcf_0, pcf_smooth[::-1], r1[::-1])

        # Catch case where particle is small and nearly spherical
        if np.isnan(agg_aa['dp_pcm']) and agg_aa['num_pixels'] < 500 and agg_aa['aspect_ratio'] < 1.4:
            agg_aa['dp_pcm'] = agg_aa['da']

        agg_aa['dp'] = agg_aa['dp_pcm']

    return Aggs


def edm_sbs(imgs_Aggs, pixsizes=None):
    """
    EDM_SBS Performs Euclidean distance mapping-scale based analysis. 
        Based on the work of Bescond et al., Aerosol Sci. Technol. (2014).
    Author: Timothy Sipkens, 2019-11-23 but adopted from CORIA
    
    INPUTS: 
      imgs_Aggs    Could be one of three options: 
                   (1) An Aggs structure, produced by other parts of this program
                   (2) A single binary image, where 1s indicate aggregate.
                   (3) A cellular arrays of the above images.
      pixsizes     A scalar or vector contain the pixel size for each image.
                   (Not used if an Aggs structure is provided.)
    
    OUTPUTS: 
      Aggs         A structure containing information for each aggregate.
      dp_bin       The vector of particle sizes used in S curve.
      S            The S curve as defined by Bescond et al.
      S_fit        The fit S curve used to quantify the particle size.
    """

    #-- Parse inputs ---------------------------------------------------------%
    # OPTION 1: Consider case that Aggs is given as input.
    if isinstance(imgs_Aggs, list) and all(isinstance(item, dict) for item in imgs_Aggs):
        Aggs0 = imgs_Aggs
        pixsizes = np.array([agg['pixsize'] for agg in Aggs0])
        imgs_binary0 = [agg['binary'] for agg in Aggs0]

    # OPTION 2: A single binary image is given.
    elif not isinstance(imgs_Aggs, list):
        imgs_binary0 = [imgs_Aggs]
        Aggs0 = []

    # OPTION 3: A list of images is given.
    else:
        imgs_binary0 = imgs_Aggs
        Aggs0 = []

    # Extract or assign the pixel size for each aggregate
    if pixsizes is None:
        pixsizes = 1
    if np.isscalar(pixsizes):
        pixsizes = pixsizes * np.ones(len(imgs_binary0))

    #-- Discretization for accumulated S curve -------------------------------%
    d_max = 100
    nb_classes = 250
    dp_bin = np.logspace(np.log10(1), np.log10(d_max), nb_classes)
    S = np.zeros_like(dp_bin)  # initialize S curve

    #-- Main loop over binary images -----------------------------------------%
    for aa in tqdm(range(len(imgs_binary0))):

        img_binary = Aggs0[aa].get('binary')
        pixsize = Aggs0[aa].get('pixsize')

        #== STEP 1: Morphological opening of the binary image ================%
        se_max = 150
        se_vec = np.arange(se_max + 1)
        
        counts = np.zeros_like(se_vec)  # initialize counts
        img_dist = distance_transform_edt(img_binary)  # Euclidean distance to outside of aggregate
        
        for ii in se_vec:
            counts[ii] = np.sum(img_dist > ii)
            if counts[ii] == 0:  # if all of the pixels are gone, exit loop
                counts[ii:] = 0
                break

        counts = counts / counts[0]
        dp_count = se_vec * pixsize

        #== STEP 2: Interpolate data to a common set of sizes ================%
        #   Accommodates images with different pixel size onto a common scale
        gi = interp1d(dp_count, counts, bounds_error=False, fill_value=0)
        Sa = gi(dp_bin)

        #== STEP 3: Fit a sigmoid function to the data =======================%
        #   This constitutes aggregate-level fitting.
        bet = 1.9658  # beta parameter in sigmoid function
        ome = -0.8515  # Omega parameter in sigmoid function
        a = 0.9966
        sigmoid = lambda x: a / (1 + np.exp(((np.log(x[0]) - np.log(dp_bin)) / x[1] - bet) / ome))

        x0 = [dp_bin[np.argmax(Sa < 0.5)], np.log(1.5)]
        x1 = least_squares(lambda x: (sigmoid(x) - Sa) / 100, x0, bounds=(0, [100, 1.1])).x
        x1 = np.real(x1)
        x1[x1 < 0] = np.nan
        Sa_fit = sigmoid(x1)  # for diagnostic purposes only

        # Update Aggs dictionary
        Aggs0[aa]['dp_edm'] = x1[0],  # geometric mean diameter for output
        Aggs0[aa]['sg_edm'] = x1[1],  # geometric standard deviation for output
        Aggs0[aa]['dp'] = x1[0]  # assign primary particle diameter based on dp_edm
        Aggs0[aa]['dp_edm_tot'] = None
        Aggs0[aa]['sg_edm_tot'] = None

        S += Sa  # add to accumulated S curve

    S = S / len(imgs_binary0)  # normalize S curve

    #== Fit a sigmoid function to all of the data ============================%
    sigmoid = lambda x: a / (1 + np.exp(((np.log(x[0]) - np.log(dp_bin)) / x[1] - bet) / ome))
    x0 = [25, 1.5]
    x1 = least_squares(lambda x: (sigmoid(x) - S) / 100, x0).x
    S_fit = sigmoid(x1)

    # Store average dp and sg over the entire set of samples in the first entry of Aggs
    Aggs0[0]['dp_edm_tot'] = x1[0]
    Aggs0[0]['sg_edm_tot'] = x1[1]

    return Aggs0, dp_bin, S, S_fit


def hough_simple(Aggs, f_plot=1, opts=None):

    # Parse options
    if opts is None:
        opts = {
            'rmax': 50,
            'rmin': 8,
            'sens_val': 0.75
        }

    centers0 = np.array([None, None])
    
    for aa in tqdm(range(len(Aggs))):
        pixsize = Aggs[aa]['pixsize']
        img_binary = Aggs[aa]['binary']

        img_canny = canny(img_binary * 255).astype(np.uint8)

        # Find and draw circles within aggregates
        hough_radii = np.array(range(opts['rmin'] * 2, opts['rmax']))
        hough_res = hough_circle(img_canny, hough_radii)

        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, 
                                                   min_xdistance=40, min_ydistance=40, normalize=True)
        centers = np.column_stack((cy, cx))

        fl = img_binary[centers[:,0], centers[:,1]] == 1
        radii = radii[fl]
        centers = centers[fl, :]

        if f_plot == 1 and centers.size > 0:
            tools.imshow_agg(Aggs, [0])
            plt.scatter(centers[:, 1], centers[:, 0], edgecolor='red', facecolor='none', s=radii*2)
            plt.show()

        Aggs[aa]['dp_hough_simple'] = pixsize * np.mean(2 * radii)

    return Aggs


def hough_kook2(Aggs, f_plot=1, opts=None):
    """
    HOUGH_KOOK2  Performs modified Kook algorithm that considers aggregate binaries.
    
     [AGGS] = pp.hough_kook2(AGGS) repackages the original code by Kook et al.
     to work with the common data structures in this larger code. Uses the
     individual aggregate information in AGGS to assign Hough transform
     circles to individual aggregates and filters out background circles. 
     
     [AGGS] = pp.hough_kook2(AGGS, F_PLOT) adds a flag for whether to produce a
     plot of the primary particle sizes overlaid on the image for each
     aggregate. By default, F_PLOT = 1 and the images will be plotted. Set
     F_PLOT = 0 to speed execution. 
    
     [AGGS] = pp.hough_kook2(AGGS, F_PLOT, OPTS) adds an options stucture to 
     control the algorithm. For defaults, see the "OPTIONS" section of the 
     code below. Note that if this argument is supplied, one must provide all 
     seven of the fields in the OPTS structure. Again, see the "OPTIONS"
     section in the code below.
     
     ------------------------------------------------------------------------
     
     This code was modified to provide more accurate primary particle size  
     data with fuzzier and lower quality backgrounds. The code now saves the  
     data as a MATLAB file. 
    
     Pre-processing steps are as follows:
      1. Select individual particle to be processed.
      2. Use a binary mask to assign primary particles to aggregates.
      3. Bottom hat filter to fix background illumination and 
         particle illumination. 
      4. Enhance contrast between individual particles and betweent the 
         agglomerate and background.
      5. Median filter to remove salt and pepper noise from the particles 
         and background.
    
     Canny edge detection sensitivity can be adjusted with the 
     OPTS.edge_threshold parameter.
     
     Circular hough transform sensitivity can be adjusted with OPTS.sens_val 
     of the code. In addition, the boundaries for the size of the circles 
     detected can be adjusted to filter out outliers with OPTS.rmax 
     and OPTS.rmin.
     
     Original code written by Ben Gigone and Emre Karatas, PhD.
     Adapted from Kook et al. 2016, SAE.
     Worked on Matlab 2012a or higher + Image RawImage Toolbox. 
     
     This code was modified by Yiling Kang and Timothy Sipkens at the
     University of British Columbia. 
    """

    # Parse options
    if opts is None:
        opts = {
            'max_img_count': 255,
            'self_subt': 0.8,
            'mf': 1,
            'alpha': 0.1,
            'rmax': 50,
            'rmin': 8,
            'sens_val': 0.75
        }

    # Main image processing loop
    unique_img_ids = np.unique([agg['img_id'] for agg in Aggs])
    n_imgs = len(unique_img_ids)
    n_aggs = len(Aggs)

    for ii in tqdm(range(n_imgs)):
        img_id = unique_img_ids[ii]
        idx_agg = [i for i, agg in enumerate(Aggs) if agg['img_id'] == img_id]
        a1 = idx_agg[0]

        pixsize = Aggs[a1]['pixsize']
        img = Aggs[a1]['image']

        # Image preprocessing
        bg = opts['self_subt'] * img
        img_bgs = opts['max_img_count'] - img
        img_bgs = np.maximum(img_bgs - bg, 0).astype(np.uint8)

        img_medfilter = median(img_bgs, disk(opts['mf']))

        # img_unsharp = cv2.GaussianBlur(img_medfilter, (0, 0), opts['alpha'])
        # img_unsharp = cv2.addWeighted(img_medfilter, 1.5, img_unsharp, -0.5, 0)
        alp = opts['alpha']
        lap_kernel = np.array([[-alp, alp - 1, -alp], [alp - 1, alp + 5, alp - 1], [-alp, alp - 1, -alp]]) / (1 + alp)
        # img_unsharp = cv2.filter2D(img_medfilter.astype(np.uint8()), -1, lap_kernel)  # calculate Laplacian-filtered image
        # img_unsharp = unsharp_mask(img_medfilter.astype(np.uint8), radius=0.5, amount=10)
        
        img_unsharp = (convolve(img_medfilter, lap_kernel, mode='nearest') > 0).astype(np.uint8) * 255

        img_canny = canny(img_unsharp).astype(np.uint8)

        # Find and draw circles within aggregates
        hough_radii = np.array(range(opts['rmin'] * 2, opts['rmax']))
        hough_res = hough_circle(img_canny, hough_radii)

        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=60, 
                                                   min_xdistance=25, min_ydistance=25, normalize=True)
        centers = np.column_stack((cy, cx))

        if f_plot == 1 and centers.size > 0:
            tools.imshow(Aggs[a1]['image'])
            plt.scatter(centers[:, 1], centers[:, 0], edgecolor='red', facecolor='none', s=radii*2)
            plt.show()

        for aa in idx_agg:
            img_binary = Aggs[aa]['binary']
            idx_s = np.ravel_multi_index((cy, cx), img_binary.shape)
            in_aggregate = img_binary.ravel()[idx_s]

            Pp_centers = centers[in_aggregate]
            Pp_radii = radii[in_aggregate]

            dp = Pp_radii * pixsize * 2
            Pp = {
                'centers': Pp_centers,
                'radii': Pp_radii,
                'dp': dp,
                'dpm': np.mean(dp),
                'dpg': np.exp(np.mean(np.log(dp))),
                'sg': np.log(np.std(dp)),
                'Np': len(dp)
            }

            Aggs[aa]['Pp_kook'] = Pp
            Aggs[aa]['dp_kook'] = Pp['dpg']
            Aggs[aa]['dp'] = Pp['dpg']

    return Aggs, centers, radii


# bg = opts['self_subt'] * img
# img_bgs = opts['max_img_count'] - img
# img_bgs = img_bgs - bg
# img_medfilter = median(img_bgs, disk(opts['mf']))