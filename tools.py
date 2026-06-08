
import numpy as np

import scipy.stats as stats
import scipy.optimize as op

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.colors import to_rgba

from operator import itemgetter

import cv2

from tkinter.filedialog import askopenfilenames

from skimage.measure import label, find_contours

from PIL import Image

from tqdm import tqdm

import json
import yaml
import os, csv
import pickle
import pandas as pd
from pathlib import Path as PathlibPath

import dm3_lib as dm3

# ANSI color codes
GREEN = "\033[92m"
BLUE = "\033[96m"
GRAY = "\033[30m"  # alt. 90m
RESET = "\033[0m"

# custom_format = f"{{percentage:3.0f}}%|{GREEN}{{bar:25}}{RESET}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}]"
# def tqdm2(*args, **kwargs):
#     return tqdm(*args, **kwargs, ascii=' ▌█', bar_format=custom_format)

class tqdm2(tqdm):
    def format_meter(self, n, total, elapsed, **kwargs):
        if total:
            frac = n / total
            bar_length = 25

            filled_len = int(bar_length * frac)
            empty_len = bar_length - filled_len

            if frac >= 1:
                bar = (
                    GREEN + "█" * filled_len + RESET +
                    GRAY + "█" * empty_len + RESET
                )
            else:
                bar = (
                    BLUE + "█" * filled_len + RESET +
                    GRAY + "█" * empty_len + RESET
                )

            # Percentage
            percentage = f"{100 * frac:3.0f}%"

            # Timing
            rate = n / elapsed if elapsed > 0 else 0
            remaining = (total - n) / rate if rate > 0 else 0

            elapsed_str = self.format_interval(elapsed)
            remaining_str = self.format_interval(remaining) if rate > 0 else "??:??"

            return (
                f"{percentage}|{bar}| "
                f"[{n}/{total}] "
                f"[{elapsed_str}<{remaining_str}]"
            )
        else:
            return super().format_meter(n, total, elapsed, **kwargs)

class Indexer:
    def __init__(self, idx):
        self.idx = idx
    def __call__(self, arr):
        return [arr[ii] for ii in self.idx]

def load_config(fn):
    """
    Loads configuration files (either JSON or YAML).
    """
    if fn[-4:].lower() == 'json':
        with open(fn) as f:
            opts = json.load(f)

    else:
        with open(fn) as f:
            opts = yaml.safe_load(f)
        
    return opts


def imshow(img, pixsize=None, cmap=None):
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

    if pixsize is not None:
        img = overlay_scale(img, pixsize)

    h = plt.imshow(img, cmap=cmap)  # Show image with colormap
    plt.axis('image')  # Adjust the axis to proper dimensions
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks

    return h


def imshow2(imgs:list, n=None, pixsizes=None, **kwargs):
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
    if not isinstance(imgs, list):
        imgs = [imgs]

    # Incorporate indices of images to plot, if specified
    if n is None:
        n = list(range(len(imgs)))
    imgs = [imgs[ii] for ii in n]

    # Limit plotting to first 24 images
    if len(imgs) > 24:
        imgs = imgs[:24]

    n_imgs = len(imgs)  # number of images after above processing

    # Create None list of pixsizes, if not given, to avoid error below.
    if pixsizes is None:
        pixsizes = list(None for _ in range(n_imgs))

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
        imshow(imgs[ii], pixsize=pixsizes[ii], **kwargs)


def overlay_scale(img, pixsize, frac=0.3):

    img = img.copy()  # don't overwrite

    # Calculate bar length in pixels and nm
    bar_length0 = int(np.floor(img.shape[1] * frac))  # in pixels, based on fraction (`frac`) of image size 
    bar_length1 = round(pixsize * bar_length0)  # in nm, rounded for string operation below

    # Round up bar length if necessary
    s1 = str(bar_length1)  # convert to string for manipulation
    b1 = int(s1[0])  # first digit
    if b1 > 5:  # do some rounding (closest 1, 2, or 5 up)
        s1 = '0' + s1
        b1 = 1
    elif b1 > 2:
        b1 = 5

    l1 = len(s1)  # length of number
    bar_length = b1 * 10 ** (l1 - 1)  # use only first digit (rounded above) and order-of-magnitude
    bar_length_px = int(bar_length / pixsize)  # in pixels

    # Properties for scale bar
    margin = np.floor(np.array(img.shape[0:2]) * 0.05).astype(int)  # margin away from edge of the image
    bar_height = margin[1] // 5  # height of the bar
    start_y, end_x = img.shape[0] - margin[1], img.shape[1] - margin[0]  # start positions for bar

    # Draw scale bar.
    if img.ndim == 3:  # first, assign black color
        color = [0, 0, 0]
    else:
        color = 0
    img[start_y - bar_height:start_y, end_x - bar_length_px:end_x] = color  # bar

    # Add text label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = img.shape[0] / 650  # scale font and thickness as fraction of image size
    thickness = max(1, int(font_scale * 2.5))
    if bar_length >= 1e3:  # then use microns
        cv2.putText(img, f'{int(bar_length / 1e3)} um', (end_x - bar_length_px, int(start_y - 2.5 * bar_height)), 
                    font, font_scale, color, thickness, cv2.LINE_AA)
    else:
        cv2.putText(img, f'{bar_length} nm', (end_x - bar_length_px, int(start_y - 2.5 * bar_height)), 
                    font, font_scale, color, thickness, cv2.LINE_AA)

    return img


def imshow_binary(img, img_binary, pixsize=None, alpha=0.2, outline=True, colors=[(1, 0, 0.5)], image_alpha=0.7):
    # Parse inputs
    if isinstance(img, list):
        img = img[0]
    if isinstance(img_binary, list):
        img_binary = img_binary[0]

    if pixsize is not None:
        img = overlay_scale(img, pixsize)

    # Display the image
    plt.imshow(img, cmap='gray')
    
    # Get labels for plotting.
    labels = label(img_binary)
    mask = img_binary
    image = Image.fromarray(img)

    if np.any(mask):  # check if mask to plot (if no particles, would error)

        _, num_objects = label(mask, return_num=True)
        if num_objects > 50:
            plt.axis('off')
            return  # too many to plot, so meaningless, exit function

        # Step 1: Find all contours
        contours = find_contours(mask, level=0.5)
            
        # Step 2: Separate outer and inner contours
        outer_contours = []
        hole_contours = []

        for contour in contours:
            y, x = np.mean(contour, axis=0)
            if mask[int(y), int(x)] == 1:
                outer_contours.append(contour)
            else:
                hole_contours.append(contour)

        # Step 3: Create a compound polygon using matplotlib Path
        def contour_to_path(contour, code_type):
            verts = [(x, y) for y, x in contour]
            codes = [Path.MOVETO] + [code_type] * (len(verts) - 1)
            return verts, codes

        vertices = []
        codes = []

        # Add outer boundary
        for outer in outer_contours:
            verts, cs = contour_to_path(outer, Path.LINETO)
            vertices.extend(verts + [verts[0]])  # close path
            codes.extend(cs + [Path.CLOSEPOLY])

        # Add holes
        for hole in hole_contours:
            verts, cs = contour_to_path(hole, Path.LINETO)
            vertices.extend(verts + [verts[0]])
            codes.extend(cs + [Path.CLOSEPOLY])

        # Create final compound path
        compound_path = Path(vertices, codes)
        patch = PathPatch(compound_path, 
                        facecolor=to_rgba(colors[0], alpha=alpha), 
                        edgecolor=colors[0], lw=0.5)

        plt.gca().add_patch(patch)

    plt.axis('off')


def imshow_binary2(imgs:list, imgs_binary:list, pixsizes:list=None, idx:list=None, **kwargs):
    
    if not idx is None:
        imgs = [imgs[ii] for ii in idx]
        imgs_binary = [imgs_binary[ii] for ii in idx]
        if not pixsizes == None:
            pixsizes = [pixsizes[ii] for ii in idx]

    else:
        idx = np.arange(len(imgs))

    if len(imgs) > 24:  # only plot up to 24 images
        imgs = imgs[:24]
        imgs_binary = imgs_binary[:24]

    n_imgs = len(imgs)  # number of images

    # Create None list of pixsizes, if not given, to avoid error below.
    if pixsizes is None:
        pixsizes = list(None for _ in range(n_imgs))

    # Prepare to tile and maximize figure if more than one image
    if n_imgs > 1:
        plt.clf()  # clear current figure contents
        N1 = int(np.floor(np.sqrt(n_imgs)))
        N2 = int(np.ceil(n_imgs / N1))
    
    plt.figure(figsize=(12, 12*N1/N2*1.1))
    for ii in range(n_imgs):
        if n_imgs > 1:
            plt.subplot(N1, N2, ii + 1)
            plt.title(str(idx[ii]))
        
        _ = imshow_binary(imgs[ii], imgs_binary[ii], pixsize=pixsizes[ii], **kwargs)


def imshow_beside(img, img_binary, *args):
    
    plt.clf()
    
    # Plot without overlay.
    plt.subplot(1, 2, 1)
    imshow(img)

    # Plot with binary overlay.
    plt.subplot(1, 2, 2)
    imshow_binary(img, img_binary, *args)


def imshow_agg(Aggs, imgs, imgs_binary, idx=None, 
               f_img=True, f_show=False, f_scale=False, f_text=True, f_diam=True, f_dp=True, f_encl=False,
               color=[1, 0, 0.5], **kwargs):
    
    # Parse inputs
    if np.any(idx == None):
        idx = np.unique(Aggs['img_id'])
    else:
        idx = np.unique([Aggs.loc[ii]['img_id'] for ii in idx])
    
    if len(idx) > 24 and not isinstance(idx, list):
        idx = idx[:24]
    n_img = len(idx)

    if n_img > 1 and not f_show:
        N1 = int(np.floor(np.sqrt(n_img)))
        N2 = int(np.ceil(n_img / N1))
        plt.subplot(N1, N2, 1)

    print('Collecting images for plotting:')
    for ii in tqdm2(range(n_img)):
        if n_img > 1 and not f_show:
            plt.subplot(N1, N2, ii + 1)

        # Determine aggregates to plot for this image
        img_idx = Aggs.index[Aggs['img_id'] == idx[ii]].tolist()
        if not img_idx:
            print(f'Warning: No aggregates for image no. {idx[ii]}.')
            continue

        if f_img:
            img_binary = np.zeros_like(imgs[idx[ii]])
            for agg_idx in img_idx:
                img_binary = np.logical_or(img_binary, imgs_binary[idx[ii]])
            
            pixsize = Aggs.iloc[img_idx[0]]['pixsize'] if f_scale else None

            # Display the image with binary overlay
            imshow_binary(imgs[idx[ii]], img_binary, pixsize=pixsize, colors=[color], **kwargs)
            plt.title(str(idx[ii]))
        
        for agg_idx in img_idx:
            agg = Aggs.loc[agg_idx]

            # Plot an 'x' at the CoM. 
            plt.plot(agg['center_mass'][1], agg['center_mass'][0], 'xk', linewidth=0.75)

            # Plot ID of the aggregate at CoM. 
            if f_text:
                plt.text(agg['center_mass'][1] + 20, agg['center_mass'][0], str(agg['id']), color='black', size='small')
            
            # Plot Rg and da.
            if f_diam:
                plt.gca().add_patch(Circle((agg['center_mass'][1], agg['center_mass'][0]), agg['Rg'] / agg['pixsize'], 
                                           color=color, fill=False, linewidth=0.5))
                plt.gca().add_patch(Circle((agg['center_mass'][1], agg['center_mass'][0]), agg['da'] / 2 / agg['pixsize'], 
                                           color=np.array(color) * 0.25, fill=False, linewidth=0.5))
                
            if f_encl:
                # Add enclosing circle.
                plt.gca().add_patch(Circle(agg['encl_c'], agg['encl_r'], 
                                        color=np.array(color) * 0.25, fill=False, linewidth=0.5))
            
            # Plot primary particle diameter if present. 
            if f_dp and hasattr(agg, 'dp') and not np.isnan(agg.dp):
                plt.gca().add_patch(Circle((agg['center_mass'][1], agg['center_mass'][0]), 
                                           agg['dp'] / 2 / agg['pixsize'], color=[0.92, 0.16, 0.49], fill=False, linewidth=0.5))


# Also, see agg.imshow(), which shows a cropped version of the aggregate.




#=========================================================================#
#== UTILITIES TO LOAD IMAGES =============================================#
#=========================================================================#
def load_imgs(fd=None, n=None, detect=False):
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

    if fd is None:  # load using a window
        fns = askopenfilenames(filetypes=[('Image files', '*.tif *.jpg *.png')])
        if not fns:
            raise ValueError('No image selected.')
        fns = list(fns)

    elif isinstance(fd, str) and os.path.isdir(fd):  # load all images in folder
        fns = [os.path.join(fd, f) for f in os.listdir(fd) if f.lower().endswith(('.tif', '.jpg', '.png'))]

    elif isinstance(fd, str) and fd.lower().startswith('http'):  # load from the web
        fns = [fd]

    if np.any(n is None):
        n = np.arange(len(fns))

    Imgs = [{'fname': fns[i]} for i in n]

    for img in tqdm2(Imgs):
        img['raw'] = cv2.imread(img['fname'], cv2.IMREAD_GRAYSCALE)

    print('Images loaded.\n')

    if not detect:
        return [img['raw'] for img in Imgs]

    # ------ Extra processing to detect footer and scale, if desired ------ #
    # Consider using supplied pixsizes.csv in folder.
    # Now default for test images. 
    if os.path.exists(fd + '\\' + 'pixsizes.csv'):
        pixsizes = load_pixsizes(fd)
        for ii, img in enumerate(Imgs):
            img['cropped'] = img['raw']
            img['pixsize'] = pixsizes[ii]

    # Otherwise, go searching for footer using dedicated function and OCR.
    else:
        try:
            import pytesseract
        except:
            print('pytesseract not found.')

        try:
            Imgs = detect_footer_scale(Imgs)
        except:
            print('Could not get pixel size.')
            for img in Imgs:
                img['cropped'] = img['raw']
                img['pixsize'] = np.nan
    
    imgs = [img['cropped'] for img in Imgs]
    pixsize = [img.get('pixsize', np.nan) for img in Imgs]

    print('Image import complete.\n')

    return imgs, pixsize, fns


def detect_footer_scale(Imgs):
    print('Looking for footers/scale:')

    for img in tqdm2(Imgs):
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

            # Import pytesseract if required to use OCR.
            import pytesseract
            from pytesseract import Output
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

def extract_scale_bar(image_path):
    # Step 1: Load the image
    image = cv2.imread(image_path)
    
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Apply thresholding to isolate bright/dark regions (adjust values if needed)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # Adjust 200 if needed
    
    # Step 4: Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 5: Identify the largest rectangular contour (assumed to be the scale bar)
    scale_bar = None
    max_area = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0  # To filter long, thin objects
        
        # Adjust conditions based on your specific scale bar properties
        if area > max_area and aspect_ratio > 2:  
            max_area = area
            scale_bar = (x, y, w, h)

    if scale_bar:
        x, y, w, h = scale_bar
        scale_bar_region = image[y:y+h, x:x+w]
        width = np.shape(scale_bar_region)[1]
        
        # Step 6: Save the extracted scale bar
        print(f"Scale bar extracted. Length: {width}")
        
        return width
    else:
        print("No scale bar detected.")
        return None


def bbox2mask(bboxs, img_size):
    mask = np.zeros(img_size, dtype=np.uint8)
    for bbox in bboxs:
        x, y, w, h = bbox
        mask[y:y+h, x:x+w] = 1
    return mask


# ---------- Utilities to load and save pixel sizes ------------ #
def load_pixsizes(fd, file='pixsizes.csv', filenames=None):
    df = pd.read_csv(os.path.join(fd, file), header=0)
    if filenames is not None:
        df = df[df['filenames'].isin([os.path.basename(path) for path in filenames])]
    pixsizes = df['pixsizes'].values.flatten()
    return pixsizes

def write_pixsizes(fd, pixsizes, file='pixsizes.csv', filenames=None):
    if filenames is None:
        df = {'pixsizes': pixsizes}
    else:
        filenames = [os.path.basename(path) for path in filenames]
        df = {'pixsizes': pixsizes, 'filenames': filenames}
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(fd, file), index=False)
# -------------------------------------------------------------- #


def load_dm3(fd, n=None, to_scale=True):

    print('Loading DM3 files:')

    fns = os.listdir(fd)
    fns = [fn for fn in fns if os.path.splitext(fn)[1] == '.dm3']

    if np.any(n == None):
        n = np.arange(len(fns))

    # Initialize variables.
    pixsizes = np.zeros(len(n))
    imgs = [np.array([])] * len(n)

    # Loop through dm3 files.
    for ii in tqdm2(range(len(n))):
        try:
            dm3f = dm3.DM3(fd + "\\" + fns[n[ii]])
        except:
            pass  # skip this file
        pixsizes[ii] = dm3f.pxsize[0]
        if dm3f.pxsize[1] == 'micron':
            pixsizes[ii] = np.asarray(pixsizes[ii]) * 1000

        img = np.asarray(dm3f.Image)  # convert to numpy array

        # Convert to uint8 image.
        img = img - np.min(img)  # adjust minimum to start at 0
        if to_scale: img = 255 * (img / np.max(img))  # scale based on max. and cover 0 > 255
        img = img.astype(np.uint8)  # convert to integer
        
        imgs[ii] = img

    print('\n')

    return imgs, pixsizes, fns


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
    min_fun = lambda t: np.linalg.norm(stats.norm.pdf(np.log(x[1:-1]), t[0], t[1]) - dens[1:]) ** 2
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
    return mu, sg


def textdone():
    print('\r' + GREEN + 'DONE!' + RESET + '\n')


def textblue(txt):
    print('\r' + BLUE + str(txt) + RESET + '\n')


#== SAVING AND LOAING DATA AND IMAGES ===================#
def save_data(fname, data):
    """
    Save dat files using pickle (e.g., Aggs structures).
    """
    fd, _ = os.path.split(fname)
    if not os.path.exists(fd):  # create folder if necessary
        os.makedirs(fd)

    print('Saving data ...')
    with open(fname, "wb") as file:
        pickle.dump(data, file)
    textdone()


def load_data(fname):
    """
    Load data files using pickle.
    Outputs the same number of variables as was originally saved.
    """
    print('Loading data ...')
    with open(fname, "rb") as file:
        out = pickle.load(file)
    print(f'Loaded {str(len(out))} variables.')
    textdone()
    return out


def write_aggs(fname, Aggs):
    """
    Save Aggs structure to Excel.
    """
    fd, _ = os.path.split(fname)
    if not os.path.exists(fd):  # create folder if necessary
        os.makedirs(fd)

    print('Writing Aggs ...')
    if not isinstance(Aggs, pd.DataFrame):
        Aggs = pd.DataFrame(Aggs)
    Aggs = Aggs.drop(['image', 'binary'], axis=1)
    Aggs.to_excel(fname, index = False)

    textdone()


def write_images(fd, imgs, pixsizes=None, fnames=None, prefix=''):
    """
    Write images in imgs to folder.
    """
    if not os.path.exists(fd):  # create folder if necessary
        os.makedirs(fd)

    if not prefix == '':
        prefix = prefix + '_'

    if fnames == None:
        fnames = ['' for _ in range(len(imgs))]
        for ii in range(len(imgs)):
            fnames[ii] = f"{fd}\\{prefix}{str(ii).zfill(3)}.png"

    # Add scale bar. 
    if not pixsizes is None:
        for ii in range(len(imgs)):
            imgs[ii] = overlay_scale(imgs[ii], pixsizes[ii])

    print('Writing images:')
    for ii in tqdm2(range(len(imgs))):
        img = Image.fromarray(imgs[ii])
        img.save(fnames[ii])
    print('\n')

def read_images(fd):
    """
    Reads all images from a folder, resizes/converts them, 
    and stacks them into a single NumPy array.
    """
    # 1. Get all image paths (filtering for common extensions)
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    path_list = [
        p for p in PathlibPath(fd).iterdir() 
        if p.suffix.lower() in extensions
    ]
    
    # Sort paths to maintain consistency with your 'image_paths' list
    path_list.sort()
    
    images = []
    for p in path_list:
        # Open and convert to RGB
        # Pillow handles common formats (PNG, JPG, TIF)
        img = Image.open(p).convert("RGB")
        
        # Append as NumPy array
        images.append(np.asarray(img))

    # Stack in an array.
    return np.stack(images)[:,:,:,0]


def write_binary(fd, imgs, imgs_binary, pixsizes=None, ext='svg', **kwargs):
    """
    Write binary masked images to folder.
    """
    if not os.path.exists(fd):  # create folder if necessary
        os.makedirs(fd)

    n_imgs = len(imgs)  # number of images after above processing

    # Create None list of pixsizes, if not given, to avoid error below.
    if pixsizes is None:
        pixsizes = list(None for _ in range(n_imgs))

    print('Writing figures (w/ binary mask):')
    for ii in tqdm2(range(n_imgs)):
        imshow_binary(imgs[ii], imgs_binary[ii], pixsize=pixsizes[ii], **kwargs)
        plt.savefig(f"{fd}\\{str(ii).zfill(3)}.{ext}", bbox_inches='tight')
        plt.clf()


def dm32img(fd, n=None, ext='png'):
    '''
    Convert DM3 files to images.
    '''

    imgs, pixsizes, fns = load_dm3(fd, n)

    print('Writing images:')
    for ii in tqdm2(range(len(imgs))):
        cv2.imwrite(f'{fd}\\{fns[ii]}.{ext}', imgs[ii])



def iou(imgs_binary1, imgs_binary2):
    """Compute the intersection over union (Iou)"""
    intersections = []
    unions = []
    ious = []
    for img1, img2 in zip(imgs_binary1, imgs_binary2):
        intersections.append(np.logical_and(img1, img2).sum())
        unions.append(np.logical_or(img1, img2).sum())

        iou = intersections[-1] / unions[-1] if unions[-1] > 0 else 0
        ious.append(iou)
    ious = np.asarray(ious)

    intersection = np.sum(np.array(intersections))
    union = np.sum(np.array(unions))
    iou = intersection / union if union > 0 else 0
    
    return iou, ious

def compare_count(imgs_binary1, imgs_binary2):
    n1, n2 = 0, 0
    for img1, img2 in zip(imgs_binary1, imgs_binary2):
        n1 += cv2.connectedComponents(np.ascontiguousarray(img1, dtype=np.uint8))[0]
        n2 += cv2.connectedComponents(np.ascontiguousarray(img2, dtype=np.uint8))[0]
    diff = n1 - n2
    return diff, n1, n2
