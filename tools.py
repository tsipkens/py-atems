
import numpy as np

import scipy.stats as stats
import scipy.optimize as op

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

import cv2
import pytesseract
from pytesseract import Output

from tkinter import Tk
from tkinter.filedialog import askopenfilenames, askdirectory

from skimage.segmentation import mark_boundaries
from skimage.filters import sobel
from skimage.morphology import dilation, disk
from skimage.measure import label, regionprops
from skimage.color import label2rgb

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from PIL import Image

from tqdm import tqdm

import json
import os
import pickle

import dm3_lib as dm3


def load_config(fn):
    """Loads JSON configuration files.

    First loads default configuration before modifying those inputs.
    """
    with open(fn) as f:
        opts = json.load(f)
        
    return opts


def imcrop(img, rect):
    return img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]


def imshow(img, cmap=None, pixsize=None):
    """
    A modified version of imshow that formats images for this program.
    Timothy Sipkens, 2020-08-25
    
    Parameters:
    img (ndarray): The image to be displayed.
    cmap (str or Colormap, optional): The colormap to be applied. Defaults to grayscale.
    pixsize (float, optional): The pixel size for overlaying a scale bar. If not provided, no scale bar is added.
    
    Returns:
    h (AxesImage): The image handle.
    """

    if cmap is None:
        cmap = 'gray'

    h = plt.imshow(img, cmap=cmap)  # Show image with colormap
    plt.axis('image')  # Adjust the axis to proper dimensions
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks

    if pixsize is not None:
        overlay_scale(pixsize)

    return h


def imshow2(imgs:list, cmap=None, n=None, pixsizes=None):
    """
    A wrapper for displaying multiple images using matplotlib.

    Parameters:
    imgs : list of arrays
        List of images to display. Must be a list (not a structure).
    cmap : str or None, optional
        Colormap to use for displaying the images.
    n : list of int or None, optional
        Indices of images to plot. If not specified, all images are considered.
    pixsizes : list or None, optional
        List of pixel sizes for each image.

    Returns:
    h : matplotlib Axes object
        The current Axes instance.
    f : matplotlib Figure object
        The current Figure instance.
    """

    # Parse inputs
    if cmap is None:
        cmap = 'gray'  # default colormap
    if pixsizes is None:
        pixsizes = []
    if not isinstance(imgs, list):
        imgs = [imgs]

    # Incorporate indices of images to plot, if specified
    if n is None:
        n = list(range(len(imgs)))
    imgs = [imgs[i] for i in n]

    # Limit plotting to first 24 images
    if len(imgs) > 24:
        imgs = imgs[:24]

    n_imgs = len(imgs)

    # If more than one image, prepare to tile and maximize figure
    if n_imgs > 1:
        plt.clf()  # clear current figure contents
        N1 = int(np.floor(np.sqrt(n_imgs)))
        N2 = int(np.ceil(n_imgs / N1))
    else:
        N1, N2 = 1, 1

    for ii in range(n_imgs):  # loop over images
        if n_imgs > 1:
            plt.subplot(N1, N2, ii + 1)
            plt.title(str(n[ii]))
        imshow(imgs[ii], cmap=cmap)


def overlay_scale(pixsize, frac=0.2):
    # Get the current image
    ax = plt.gca()
    I = ax.get_images()[0].get_array()

    # Calculate bar length in pixels and nm
    bar_length0 = int(np.floor(I.shape[1] * frac))  # in pixels
    bar_length1 = round(pixsize * bar_length0)  # in nm

    # Round up bar length if necessary
    s1 = str(bar_length1)
    b1 = int(s1[0])  # first digit
    l1 = len(s1)  # length of number
    if b1 > 5:
        if b1 > 7:
            bar_length1 = 10 ** l1
        else:
            bar_length1 = 5 * 10 ** (l1 - 1)
    
    bar_length1 = round(bar_length1, 1)  # round in nm
    bar_length = bar_length1 / pixsize  # in pixels

    # Properties for scale bar
    margin = np.floor(np.array(I.shape[::-1]) * 0.05).astype(int)
    bar_height = margin[1] // 5
    font_props = {
        'ha': 'right',
        'va': 'bottom',
        'fontsize': 11,
        'weight': 'bold'
    }

    # Draw the scale bar
    ax.add_patch(FancyBboxPatch(
        (I.shape[1] - margin[1] - bar_length, I.shape[0] - margin[0]),
        bar_length, bar_height,
        boxstyle="round,pad=0.1",
        edgecolor='none',
        facecolor='black'
    ))

    # Add text label
    if bar_length1 > 1e3:  # then use microns
        ax.text(
            I.shape[1] - margin[1],
            I.shape[0] - margin[0] - bar_height / 5,
            f'{bar_length1 / 1e3:.1f} µm',
            **font_props
        )
    else:
        ax.text(
            I.shape[1] - margin[1],
            I.shape[0] - margin[0] - bar_height / 5,
            f'{bar_length1} nm',
            **font_props
        )

    plt.draw()


def imshow_binary(img, img_binary, pixsize=None, opts=None):
    # Parse inputs
    if isinstance(img, list):
        img = img[0]
    if isinstance(img_binary, list):
        img_binary = img_binary[0]

    if opts is None:
        opts = {}

    # Set default options
    cmap = opts.get('cmap', np.ones((int(np.max(img_binary)), 3)) * [1, 0, 0.5])
    f_outline = opts.get('f_outline', True)
    label_alpha = opts.get('label_alpha', 0.25)

    # Overlay the binary mask on the image
    t0 = label(img_binary)
    t0 = label2rgb(t0, image=img, alpha=label_alpha, bg_label=0, image_alpha=0.7)  # cmap=[cmap]

    if not f_outline:
        i1 = t0
    else:
        # Calculate edges using Sobel filter
        img_edge = sobel(img_binary)

        # Dilate edges to strengthen the outline
        se = disk(1)
        img_dilated = np.logical_or(np.logical_and(dilation(img_edge, se), ~img_binary), img_edge)

        # Add borders to labeled regions
        i1 = (~img_dilated[..., np.newaxis]) * t0

    # Display the image
    plt.imshow(i1)
    plt.axis('off')


def imshow_binary2(imgs:list, imgs_binary:list, pixsizes:list=None, idx:list=None, *args):
    
    if not idx == None:
        imgs = [imgs[ii] for ii in idx]
        imgs_binary = [imgs_binary[ii] for ii in idx]
        if not pixsizes == None:
            pixsizes = [pixsizes[ii] for ii in idx]

    if len(imgs) > 24:  # only plot up to 24 images
        imgs = imgs[:24]
        imgs_binary = imgs_binary[:24]
    n_imgs = len(imgs)  # number of images

    # Prepare to tile and maximize figure if more than one image
    if n_imgs > 1:
        plt.clf()  # clear current figure contents
        N1 = int(np.floor(np.sqrt(n_imgs)))
        N2 = int(np.ceil(n_imgs / N1))
    
    for ii in range(n_imgs):
        if n_imgs > 1:
            plt.subplot(N1, N2, ii + 1)
            plt.title(str(ii + 1))
        
        if pixsizes is None:
            i1 = imshow_binary(imgs[ii], imgs_binary[ii], *args)
        else:
            i1 = imshow_binary(imgs[ii], imgs_binary[ii], pixsizes[ii], *args)


def imshow_beside(img, img_binary, *args):
    
    plt.clf()
    
    # Plot without overlay.
    plt.subplot(1, 2, 1)
    imshow(img)

    # Plot with binary overlay.
    plt.subplot(1, 2, 2)
    imshow_binary(img, img_binary, *args)


def imshow_agg(Aggs, idx=None, f_img=True, opts=None):
    # Parse inputs
    if idx is None:
        idx = np.unique([d['img_id'] for d in Aggs])
    else:
        idx = list(set(Aggs[ii]['img_id'] for ii in idx))
    
    if len(idx) > 24 and not isinstance(idx, list):
        idx = idx[:24]
    n_img = len(idx)

    if opts is None:
        opts = {
            'cmap': [1, 0, 0.5],
            'f_text': True,
            'f_show': False,
            'f_dp': True,
            'f_scale': False,
            'f_diam': True,
            'f_all': True
        }

    if n_img > 1 and not opts['f_show']:
        plt.figure()
    else:
        plt.gcf()

    if n_img > 1 and not opts['f_show']:
        N1 = int(np.floor(np.sqrt(n_img)))
        N2 = int(np.ceil(n_img / N1))
        plt.subplot(N1, N2, 1)
    
    frames = []

    for ii in range(n_img):
        if n_img > 1 and not opts['f_show']:
            plt.subplot(N1, N2, ii + 1)

        # Determine aggregates to plot for this image
        img_idx = [i for i, agg in enumerate(Aggs) if agg['img_id'] == idx[ii]]
        if not img_idx:
            print(f'Warning: No aggregates for image no. {idx[ii]}.')
            continue

        if f_img:
            img_binary = np.zeros_like(Aggs[img_idx[0]]['image'])
            for agg_idx in img_idx:
                img_binary = np.logical_or(img_binary, Aggs[agg_idx]['binary'])
            
            pixsize = Aggs[img_idx[0]]['pixsize'] if opts['f_scale'] else None

            # Display the image with binary overlay
            imshow_binary(Aggs[img_idx[0]]['image'], img_binary, pixsize, opts)
            plt.title(str(idx[ii]))
        
        for agg_idx in img_idx:
            agg = Aggs[agg_idx]

            # Plot an 'x' at the CoM. 
            plt.plot(agg['center_mass'][1], agg['center_mass'][0], 'xk', linewidth=0.75)

            # Plot ID of the aggregate at CoM. 
            if opts['f_text']:
                plt.text(agg['center_mass'][1] + 20, agg['center_mass'][0], str(agg['id']), color='black', size='small')
            
            # Plot Rg and da.
            if opts['f_diam']:
                plt.gca().add_patch(Circle((agg['center_mass'][1], agg['center_mass'][0]), 
                                           agg['Rg'] / agg['pixsize'], color=opts['cmap'], fill=False))
                plt.gca().add_patch(Circle((agg['center_mass'][1], agg['center_mass'][0]), 
                                           agg['da'] / 2 / agg['pixsize'], color=np.array(opts['cmap']) * 0.25, fill=False, linewidth=1))
            
            # Plot primary particle diameter if present. 
            if opts['f_dp'] and hasattr(agg, 'dp') and not np.isnan(agg.dp):
                plt.gca().add_patch(Circle((agg['center_mass'][1], agg['center_mass'][0]), 
                                           agg['dp'] / 2 / agg['pixsize'], color=[0.92, 0.16, 0.49], fill=False, linewidth=0.75))

    plt.draw()
    if n_img > 1:
        plt.show()


#=========================================================================#
#== UTILITIES TO LOAD IMAGES =============================================#
#=========================================================================#
def load_imgs(fd=None, n=None):
    """
    LOAD_IMGS  Loads images from files.
     
     IMGS = load_imgs() uses a file explorer to select files, loads the
     images, and attempts to detect the footer and scale of the image (using
     the detect_footer_scale subfunction). Information is output in the form
     of a data struture, with one entry per image. 
     
     IMGS = load_imgs(FD) loads all of the images in the folder specified by
     the input string, FD. For example, the sample images can be loaded using
     IMGS = load_imgs('images'). 
     
     IMGS = load_imgs(FD, N) loads the images specified by array N. By
     default, N spans 1 to the number of images in the given folder. For
     example, the 2nd and 3rd images can be loaded using N = [2,3]. This
     allows for partial loading of larger data sets for batch processing. 
     
     [~,IMGS,PIXSIZE] = load_imgs(...) loads images and outputs the imported
     images after the detector footer has been remvoed as a cell array, IMGS,
     and an array of pixel sizes in nm/pixel, PIXSIZE. 
    
     AUTHOR: Timothy Sipkens, 2019-07-04
    """
    print('Loading images:')

    if fd is None:
        fd = askopenfilenames(filetypes=[('Image files', '*.tif *.jpg *.png')])
        if not fd:
            raise ValueError('No image selected.')
        fd = list(fd)
    elif isinstance(fd, str) and os.path.isdir(fd):
        fd = [os.path.join(fd, f) for f in os.listdir(fd) if f.lower().endswith(('.tif', '.jpg', '.png'))]
    elif isinstance(fd, str) and fd.lower().startswith('http'):
        fd = [fd]

    if np.any(n is None):
        n = np.arange(len(fd))

    Imgs = [{'fname': fd[i]} for i in n]

    for img in tqdm(Imgs):
        img['raw'] = cv2.imread(img['fname'], cv2.IMREAD_GRAYSCALE)

    print('Images loaded.\n')

    f_replace = 1
    Imgs = detect_footer_scale(Imgs, f_replace)

    imgs = [img['cropped'] for img in Imgs]
    pixsize = [img.get('pixsize', np.nan) for img in Imgs]

    print('Image import complete.\n')

    return Imgs, imgs, pixsize


def detect_footer_scale(Imgs, f_replace):
    print('Looking for footers/scale:')

    for img in tqdm(Imgs):
        raw = img['raw']
        white = 255
        footer_found = False
        fl_nm = True

        # Search for row that is 90 % white.
        f_footrow = np.sum(raw, axis=1) > 0.9 * raw.shape[1] * white
        row_idx = np.where(f_footrow)[0][0]
        
        # If failed, instead look for black (e.g., NRC footer).
        if np.size(row_idx) == 0:
            f_footrow = np.sum(raw, axis=1) == 0
            row_idx = np.where(f_footrow)[0][0]
        
        if np.size(row_idx) > 0:  # if found footer satisyfing above
            footer_found = True  # flag that footer was found
            ii = row_idx
            img['cropped'] = raw[:ii, :]
            footer = raw[ii:, :]

            #-- Detecting magnification and/or pixel size ----------------#
            if pytesseract.pytesseract.get_tesseract_version():
                ocr_data = pytesseract.image_to_data(footer, output_type=Output.DICT)
                text = " ".join([ocr_data['text'][i] for i in range(len(ocr_data['text'])) if ocr_data['text'][i].strip() != ""])

                #-- Interpret OCR text ---------------------------------------#
                pixsize_end = None

                # Loop through options. Flag if nm/um.
                for keyword in ['nm/pix', 'nmlpix', 'nm/plx', 'nm/101x', 'um/pix', 'umlpix', 'um/plx', 'um/101x', 'pm/pix', 'pmlpix', 'pm/plx', 'pm/101x', 'nm', 'um', 'pm']:
                    if keyword in text:
                        pixsize_end = text.find(keyword)
                        break

                #-- Interpret scale/number in footer -------------------------#
                # Check if one can find 'Cal', the size per pixel directly.
                fl_per_pixel = text.rfind('Cal')
                
                if pixsize_end is not None:
                    pixsize_start = text.rfind(' ', 0, pixsize_end - 1) + 1
                    pixsize_str = text[pixsize_start:pixsize_end].strip()
                    try:
                        pixsize = float(pixsize_str)
                        if 'um' in text:
                            pixsize *= 1e3
                    except ValueError:
                        pixsize = np.nan
                else:
                    pixsize = np.nan

                img['pixsize'] = pixsize

            else:
                img['pixsize'] = np.nan

        if not footer_found:
            img['cropped'] = raw
            img['pixsize'] = np.nan

    pixsizes = [img['pixsize'] for img in Imgs]

    if any(np.isnan(pixsizes)):
        print("\033[93m" + "Warning: One or more footers or scales not found. The cropped image is the raw image. Assign pixel size manually if needed." + "\033[0m")
    else:
        print("Footer found for all images.\n")

    return Imgs


def bbox2mask(bboxs, img_size):
    mask = np.zeros(img_size, dtype=np.uint8)
    for bbox in bboxs:
        x, y, w, h = bbox
        mask[y:y+h, x:x+w] = 1
    return mask


def load_dm3(fd, n=None):

    print('Loading DM3 files:')

    fns = os.listdir(fd)
    fns = [fn for fn in fns if os.path.splitext(fn)[1] == '.dm3']

    if np.any(n == None):
        n = np.arange(len(fns))

    # Initialize variables.
    pixsizes = np.zeros(len(n))
    imgs = [np.array([])] * len(n)

    # Loop through dm3 files.
    for ii in tqdm(range(len(n))):
        dm3f = dm3.DM3(fd + "\\" + fns[n[ii]])
        pixsizes[ii] = dm3f.pxsize[0]
        if dm3f.pxsize[1] == 'micron':
            pixsizes[ii] = pixsizes[ii] * 1000

        img = dm3f.imagedata

        # Convert to uint8 image.
        img = img / np.max(img)
        img = 255 * img # Now scale by 255
        img = img.astype(np.uint8)
        
        imgs[ii] = img

    print('Import complete.\n')

    return imgs, pixsizes


#=========================================================================#
#== OTHER UTILITIES ======================================================#
#=========================================================================#
def pcf(img_binary, v=None, ns=1e5):
    """
    Compute the pair correlation function (PCF) for a binary image.

    AUTHOR: Timothy Sipkens, 2023-12-13
    """

    # Initialize parameters if not provided
    if v is None:
        v = []

    if isinstance(ns, int) or isinstance(ns, float):
        ns = int(ns)

    # Vector of distances
    if len(v) == 0 or np.isscalar(v):
        if np.isscalar(v):  # then radius of gyration (in px) or similar, use to generate v
            R = v
            maxd = R * 2 * 2
        else:  # otherwise, use size of the image
            maxd = min(img_binary.shape) / 4

        v = np.logspace(0, np.log10(maxd), 50)

    # Get row and column indices of binary pixels
    row, col = np.where(img_binary)
    g = np.zeros_like(v)

    for ii in range(len(v)):
        ri = np.random.randint(0, len(row), size=ns)  # get random entries

        rthe = 2 * np.pi * np.random.rand(ns)  # random angle
        rx = np.round(v[ii] * np.sin(rthe)).astype(int)  # random x dir.
        ry = np.round(v[ii] * np.cos(rthe)).astype(int)  # random y dir.

        row_new = row[ri] + ry  # new row
        col_new = col[ri] + rx  # new col

        # Catch out-of-bounds cases
        out_of_bounds = np.logical_or(
            np.logical_or(row_new < 1, row_new >= img_binary.shape[0]),
            np.logical_or(col_new < 1, col_new >= img_binary.shape[1])
        )
        nout = np.sum(out_of_bounds)
        row_new = row_new[~out_of_bounds]
        col_new = col_new[~out_of_bounds]

        # Get new pixels
        in_pixels = img_binary[row_new, col_new]

        # Pad with removed cases
        g[ii] = np.sum(in_pixels) / (len(in_pixels) + nout)

    return g, v


def enhance_contrast(imgs, contrast:float=1.0, brightness:int=0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255 * (1 - contrast) / 2))
    for ii in range(len(imgs)):
        imgs[ii] = cv2.addWeighted(imgs[ii], contrast, imgs[ii], 0, brightness)
    return imgs


def loghist(y, n=20):
    x = np.logspace(np.log10(np.min(y)), np.log10(np.max(y)), n)
    dens, _ = np.histogram(y, bins=x)

    dx = np.log(x[1]) - np.log(x[0])
    dens = dens / dx / len(y)  # normalize counts
    
    plt.stairs(dens, x)
    plt.xscale('log')

    print('Adding lognormal fits:')

    # Get first guess for GMD and GSD.
    mu, sg = stats.norm.fit(np.log(y))
    print(f'y(stats) ~ logn(mu={np.exp(mu)}, sg={np.exp(sg)})')

    # Use optimization to find GMD and GSD.
    min_fun = lambda t: np.linalg.norm(stats.norm.pdf(np.log(x[0:-1]), t[0], t[1]) - dens) ** 2
    x1 = op.fmin(min_fun, x0=[mu, sg, 1.], disp=None)
    mu = np.exp(x1[0])
    sg = np.exp(x1[1])
    print(f'y(fit) ~ logn(mu={mu}, sg={sg})\n\n')

    # Add lognormal fit.
    xmin, xmax = plt.xlim()
    x = np.logspace(np.log10(xmin), np.log10(xmax), 100)
    p = stats.norm.pdf(np.log(x), np.log(mu), np.log(sg))
    plt.plot(x, p, 'k', linewidth=2)

    # Return GMD and GSD.
    return np.exp(mu), np.exp(sg)


#== SAVING AND LOAING DATA AND IMAGES ===================#
def save_data(fname, data):
    """
    Save dat files using pickle (e.g., Aggs structures).
    """
    with open(fname, "wb") as file:
        pickle.dump(data, file)


def load_data(fname):
    """
    Load data files using pickle.
    """
    with open(fname, "rb") as file:
        out = pickle.load(file)
    return out


def write_images(fd, imgs):
    """
    Load files using pickle.
    """
    if not os.path.exists(fd):
        os.makedirs(fd)

    for ii in range(len(imgs)):
        img = Image.fromarray(imgs[ii])
        img.save(f"{fd}\\{str(ii).zfill(3)}.png")