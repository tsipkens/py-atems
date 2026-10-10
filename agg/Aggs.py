import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
from numpy.lib.stride_tricks import as_strided
from skimage import measure, morphology
from skimage.measure import regionprops_table
from skimage.segmentation import clear_border, flood

import tools
from tools import tqdm2 as tqdm

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_perimeter2(img_binary):
    """Computes perimeter by midpoint-connecting contour line segments."""
    contours = measure.find_contours(img_binary, 0.5)
    if len(contours) == 0:
        return 0.0

    contour = contours[0]
    x_mb, y_mb = contour[:, 1], contour[:, 0]

    edges_mb = np.cumsum(
        np.concatenate(([1], (x_mb[1:] != x_mb[:-1]) & (y_mb[1:] != y_mb[:-1])))
    ) - 1

    if (x_mb[0] == x_mb[-1]) or (y_mb[0] == y_mb[-1]):
        edges_mb[edges_mb == edges_mb[-1]] = 1

    xx_mb = np.bincount(edges_mb, weights=x_mb) / np.bincount(edges_mb)
    yy_mb = np.bincount(edges_mb, weights=y_mb) / np.bincount(edges_mb)

    p_circ = np.sum(np.sqrt((xx_mb - np.roll(xx_mb, -1)) ** 2 + (yy_mb - np.roll(yy_mb, -1)) ** 2))
    return float(p_circ)

def box_counting(img_binary):
    """Estimates the fractal dimension (Df) of a binary mask using box counting."""
    box_sizes = [2, 4, 6, 8]
    height, width = img_binary.shape

    if min(height, width) < 10:
        return np.nan

    counts = np.zeros(len(box_sizes), dtype=int)
    for idx, box_size in enumerate(box_sizes):
        h_crop = (height // box_size) * box_size
        w_crop = (width // box_size) * box_size
        cropped_img = img_binary[:h_crop, :w_crop]

        stride_view = as_strided(
            cropped_img,
            shape=(h_crop // box_size, w_crop // box_size, box_size, box_size),
            strides=(
                box_size * cropped_img.strides[0],
                box_size * cropped_img.strides[1],
                *cropped_img.strides,
            ),
        )
        counts[idx] = np.sum(np.any(stride_view, axis=(2, 3)))

    if np.any(counts == 0):
        return np.nan

    coeffs = np.polyfit(np.log(np.array(box_sizes)), np.log(counts), 1)
    return float(-coeffs[0])


def compute_iou_matrix(boxes_a, boxes_b):
    """
    Computes pairwise Intersection over Union (IoU) matrix.
    boxes_a: (N, 4) array
    boxes_b: (M, 4) array
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    # Ensure 2D shape (N, 4) and (M, 4)
    if boxes_a.ndim == 1:
        boxes_a = boxes_a.reshape(1, -1)
    if boxes_b.ndim == 1:
        boxes_b = boxes_b.reshape(1, -1)

    # ------------------------------------------------------------
    # FIX: Transpose boxes_b (M, 4) -> (4, M) BEFORE indexing or splitting
    # ------------------------------------------------------------
    x1_a, y1_a, x2_a, y2_a = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    x1_b, y1_b, x2_b, y2_b = boxes_b.T[0:1], boxes_b.T[1:2], boxes_b.T[2:3], boxes_b.T[3:4]

    # Compute intersection coordinates via broadcasting
    inter_x1 = np.maximum(x1_a, x1_b)
    inter_y1 = np.maximum(y1_a, y1_b)
    inter_x2 = np.minimum(x2_a, x2_b)
    inter_y2 = np.minimum(y2_a, y2_b)

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    # Compute individual areas
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)

    # Compute IoU
    union_area = area_a + area_b - inter_area
    iou = inter_area / np.maximum(union_area, 1e-6)
    return iou

def match_detects_to_aggs(imgs_detect, aggs, iou_threshold=0.5):
    """
    Matches detected boxes in `imgs_detect` with bounding boxes stored in `aggs.df`
    filtered by the 'img_id' column.

    Parameters:
    - imgs_detect: list of dicts (each entry has 'boxes', 'confidences', and optionally 'img_id').
    - aggs: custom class instance containing `aggs.df` with columns ['img_id', 'bbox', ...].
    - iou_threshold: float, threshold for matching.

    Returns:
    - matched_results: list of dicts with matching details per image entry.
    """
    df = aggs.df
    aggs.df['class_name'] = None  # initialize

    if 'img_id' not in df.columns:
        raise ValueError("`aggs.df` does not contain an 'img_id' column.")

    matched_results = []

    for img_idx, entry in enumerate(imgs_detect):

        # 1. Filter DataFrame rows corresponding to this img_id
        df_subset = df[df['img_id'] == img_idx]

        if not df_subset.empty:
            raw_bboxes = np.array(df_subset['bbox'].tolist(), dtype=np.float32)
            # Reorder columns: (r1, c1, r2, c2) -> (c1, r1, c2, r2)
            target_bbox = raw_bboxes[:, [1, 0, 3, 2]]
            df_indices = df_subset.index.to_numpy()
        else:
            target_bbox = np.empty((0, 4), dtype=np.float32)
            df_indices = np.array([])

        pred_boxes = entry.get('boxes', np.array([]))
        confidences = entry.get('confidences', np.array([]))

        # 2. Compute IoU matrix
        iou_matrix = compute_iou_matrix(pred_boxes, target_bbox)

        # 3. Optimal 1-to-1 Hungarian Matching
        one_to_one_matches = []
        if iou_matrix.size > 0:
            cost_matrix = 1.0 - iou_matrix
            pred_indices, target_indices = linear_sum_assignment(cost_matrix)

            for p_idx, t_idx in zip(pred_indices, target_indices):
                if iou_matrix[p_idx, t_idx] >= iou_threshold:
                    one_to_one_matches.append({
                        'pred_index': int(p_idx),
                        'agg_df_index': int(df_indices[t_idx]),  # Exact index in aggs.df
                        'iou': float(iou_matrix[p_idx, t_idx]),
                        'confidence': float(confidences[p_idx]) if len(confidences) > p_idx else None
                    })

        # 4. Overlapping / Multi-Match Associations
        overlap_matches = []
        if iou_matrix.size > 0:
            p_indices, t_indices = np.where(iou_matrix >= iou_threshold)
            for p_idx, t_idx in zip(p_indices, t_indices):
                overlap_matches.append({
                    'pred_index': int(p_idx),
                    'agg_df_index': int(df_indices[t_idx]),  # Exact index in aggs.df
                    'iou': float(iou_matrix[p_idx, t_idx]),
                    'confidence': float(confidences[p_idx]) if len(confidences) > p_idx else None
                })

        matched_results.append({
            'img_id': img_idx,
            'image_shape': entry.get('image_shape'),
            'iou_matrix': iou_matrix,
            'one_to_one_matches': one_to_one_matches,
            'overlapping_matches': overlap_matches
        })
        
        for match in one_to_one_matches:
            aggs.df.loc[match['agg_df_index'], 'class_name'] = entry['class_names'][match['pred_index']]

    return matched_results


# =============================================================================
# AGGS CLASS DEFINITION
# =============================================================================

class Aggs:
    def __init__(self,
        imgs_binary, pixsizes, imgs=None, fnames=None,
        remove_edge_aggs=False, maxagg=50, min_size=10,
        imgs_labeled=None
    ):
        """
        Parameters
        ----------
        imgs_binary : list, dict, or np.ndarray
            Binary mask or stack/list of 2D binary masks.
        pixsizes : float or list of float
            Physical pixel size (e.g., µm/pixel or nm/pixel).
        imgs : list or np.ndarray, optional
            Original grayscale/color images matching `imgs_binary`.
        fnames : list of str, optional
            Filenames or identifiers corresponding to each image.
        remove_edge_aggs : bool, optional
            Whether to strip aggregates touching frame boundaries. Default False.
        maxagg : int, optional
            Maximum allowed aggregates per image (skips image if exceeded). Default 50.
        min_size : int, optional
            Minimum size in pixels to keep an object. Default 10.
        """
        if isinstance(imgs_binary, dict):
            Imgs = imgs_binary
            imgs_binary = [Imgs["cropped"]]
            pixsizes = [Imgs["pixsize"]]
            fnames = [Imgs.get("fname", None)]

        self.imgs_binary = (
            [imgs_binary]
            if isinstance(imgs_binary, np.ndarray) and imgs_binary.ndim == 2
            else list(imgs_binary)
        )

        n_imgs = len(self.imgs_binary)

        if imgs is None:
            self.imgs = [np.uint8(155 * (~b) + 100) for b in self.imgs_binary]
        elif isinstance(imgs, np.ndarray) and imgs.ndim == 2:
            self.imgs = [imgs]
        else:
            self.imgs = list(imgs)

        if np.isscalar(pixsizes):
            self.pixsizes = [float(pixsizes)] * n_imgs
        else:
            self.pixsizes = [float(p) for p in pixsizes]

        if fnames is not None:
            self.fnames = list(fnames)
        else:
            self.fnames = [None] * n_imgs

        self.imgs_labeled = imgs_labeled

        self.remove_edge_aggs = remove_edge_aggs
        self.maxagg = maxagg
        self.min_size = min_size

        self.objects = []
        self.df = pd.DataFrame()

        self._extract_objects()

    @property
    def index(self):
        """Returns the Index (row labels) of the underlying dataframe."""
        return self.df.index

    @property
    def loc(self):
        """Access a group of rows and columns by label(s) or a boolean array."""
        return self.df.loc

    @property
    def iloc(self):
        """Purely integer-location based indexing for selection by position."""
        return self.df.iloc

    def __getitem__(self, item):
        """Supports direct bracket indexing, e.g., aggs['da']."""
        return self.df[item]

    def __len__(self):
        """Returns the number of detected aggregate objects in the dataset."""
        return len(self.df)

    def _extract_objects(self):
        """Extracts connected component aggregates and builds the dataset."""
        all_records = []
        global_id = 0

        print('Processing images to extract aggregates:')
        for img_idx, (bin_img, orig_img, pixsize, fname) in enumerate(tqdm(
            zip(self.imgs_binary, self.imgs, self.pixsizes, self.fnames),
            total=len(self.imgs)
        )):
            img_binary = bin_img.copy()

            bwborder = np.logical_and(img_binary, clear_border(img_binary))
            if (np.count_nonzero(bwborder) / img_binary.size) > 0.25:
                continue

            border_ratios = [
                np.count_nonzero(img_binary[:, 0]) / img_binary.shape[0],
                np.count_nonzero(img_binary[:, -1]) / img_binary.shape[0],
                np.count_nonzero(img_binary[0, :]) / img_binary.shape[1],
                np.count_nonzero(img_binary[-1, :]) / img_binary.shape[1],
            ]
            if any(ratio > 0.2 for ratio in border_ratios):
                continue

            if self.remove_edge_aggs:
                img_binary = clear_border(img_binary)

            img_binary = morphology.remove_small_objects(
                img_binary, min_size=self.min_size
            )

            if self.imgs_labeled is None:
                structure = np.ones((3, 3), dtype=int)
                labeled_img, naggs = ndimage.label(img_binary, structure=structure)
                if naggs == 0 or naggs > self.maxagg:
                    continue
            else:
                raw_img = self.imgs_labeled[img_idx]
                unique_labels, contiguous_img = np.unique(raw_img, return_inverse=True)
                
                # Check for background (0)
                has_bg = 0 in unique_labels
                naggs = len(unique_labels) - (1 if has_bg else 0)
                
                if naggs == 0 or naggs > self.maxagg:
                    continue
                
                labeled_img = contiguous_img.reshape(raw_img.shape)

            props_table = regionprops_table(
                labeled_img,
                intensity_image=orig_img,
                properties=("label", "centroid", "bbox", "eccentricity", 
                            "solidity", "area", "equivalent_diameter", 
                            "moments_central", "perimeter", 
                            "feret_diameter_max", "major_axis_length", "minor_axis_length"),
            )

            for jj in range(1, naggs + 1):
                idx_prop = jj - 1
                min_r = int(props_table["bbox-0"][idx_prop])
                min_c = props_table["bbox-1"][idx_prop]
                max_r = props_table["bbox-2"][idx_prop]
                max_c = props_table["bbox-3"][idx_prop]
                bbox = (min_r, min_c, max_r, max_c)

                # Localized slice extraction (avoids full-image memory boolean allocations)
                cropped_mask = labeled_img[min_r:max_r, min_c:max_c] == jj
                if not np.any(cropped_mask):
                    continue

                centroid = (
                    props_table["centroid-0"][idx_prop],
                    props_table["centroid-1"][idx_prop],
                )
                
                eccentricity = props_table["eccentricity"][idx_prop]
                solidity = props_table["solidity"][idx_prop]
                area = int(props_table["area"][idx_prop])
                area_scaled = area * (pixsize ** 2)
                da = props_table["equivalent_diameter"][idx_prop] * pixsize
                perimeter_p = props_table["perimeter"][idx_prop] * pixsize

                feret_diameter_max = props_table["feret_diameter_max"][idx_prop] * pixsize
                major_axis_length = props_table["major_axis_length"][idx_prop] * pixsize
                minor_axis_length = props_table["minor_axis_length"][idx_prop] * pixsize

                # Fast first-pixel seed point extraction
                first_pixel = np.argwhere(cropped_mask)[0]
                seed_local = (int(first_pixel[0]), int(first_pixel[1]))

                height = float((max_r - min_r) * pixsize)
                width = float((max_c - min_c) * pixsize)
                aspect_ratio = major_axis_length / minor_axis_length

                # Radius of gyration from central moments
                mu20 = props_table["moments_central-2-0"][idx_prop]
                mu02 = props_table["moments_central-0-2"][idx_prop]
                Rg = np.sqrt((mu20 + mu02) / area) * pixsize

                contours = measure.find_contours(cropped_mask.astype(float), level=0.5)
                if len(contours) > 0:
                    contour = contours[0]
                    if len(contour) >= 3:
                        try:
                            hull = ConvexHull(contour)
                            hull_points = contour[hull.vertices]
                        except Exception:
                            hull_points = contour
                    else:
                        hull_points = contour

                    x_max, y_max = np.max(hull_points, axis=0)
                    x_min, y_min = np.min(hull_points, axis=0)
                    encl_c = ((x_max + x_min) / 2.0, (y_max + y_min) / 2.0)

                    encl_c_full = (float(encl_c[1] + min_r), float(encl_c[0] + min_c))
                    encl_r = float(np.max(np.linalg.norm(hull_points - encl_c, axis=1)))
                else:
                    encl_c_full = centroid
                    encl_r = max(max_r - min_r, max_c - min_c) / 2.0

                encl_d = 2 * encl_r * pixsize
                sphericity = (da / encl_d) if encl_d > 0 else np.nan

                dilated_mask = ndimage.binary_dilation(cropped_mask)
                border_pixels = np.logical_and(dilated_mask, np.logical_not(cropped_mask))
                perimeter1 = np.sum(border_pixels)

                perimeter3 = get_perimeter2(cropped_mask)
                perimeter = pixsize * max(perimeter1, perimeter3)

                circularity = (
                    (4 * np.pi * area_scaled) / (perimeter ** 2)
                    if perimeter > 0
                    else np.nan
                )

                Df = box_counting(cropped_mask)

                obj_dict = {
                    "id": global_id,
                    "img_id": img_idx,
                    "object_id": jj,
                    "pixsize": pixsize,
                    "num_pixels": area,
                    "area_scaled": area_scaled,
                    "height": height,
                    "width": width,
                    "feret_diameter_max": feret_diameter_max,
                    "major_axis_length": major_axis_length,
                    "minor_axis_length": minor_axis_length,
                    "aspect_ratio": aspect_ratio,
                    "eccentricity": eccentricity,
                    "solidity": solidity,
                    "centroid": centroid,
                    "bbox": bbox,
                    "da": da,
                    "Rg": Rg,
                    "encl_c": encl_c_full,
                    "encl_r": encl_r,
                    "encl_d": encl_d,
                    "sphericity": sphericity,
                    "perimeter": perimeter,
                    "perimeter_p": perimeter_p,
                    "circularity": circularity,
                    "Df": Df,
                    "seed_local": seed_local,
                    "image_binary_ref": bin_img,
                    "image_orig_ref": orig_img,
                }

                if fname is not None:
                    obj_dict["fname"] = fname

                self.objects.append(obj_dict)
                all_records.append(obj_dict)
                global_id += 1

        if all_records:
            full_df = pd.DataFrame(all_records)
            drop_cols = ["image_binary_ref", "image_orig_ref"]
            self.df = full_df.drop(
                columns=[c for c in drop_cols if c in full_df.columns]
            )

    def to_df(self, sig_figs=None):
        if self.df.empty or sig_figs is None:
            return self.df

        fmt_str = f"{{:.{sig_figs}g}}"

        def format_cell(val):
            if isinstance(val, (float, np.floating)):
                return fmt_str.format(val)
            elif isinstance(val, (tuple, list)):
                return type(val)(
                    fmt_str.format(x) if isinstance(x, (float, np.floating)) else x
                    for x in val
                )
            return val

        return self.df.style.format(format_cell)

    def get_objects_by_image(self, image_index):
        return [obj for obj in self.objects if obj["img_id"] == image_index]

    def get_crop(self, object_index, padding=10):
        obj = self.objects[object_index]
        source_img = (
            obj["image_orig_ref"]
            if obj["image_orig_ref"] is not None
            else obj["image_binary_ref"]
        )

        min_r, min_c, max_r, max_c = obj["bbox"]

        min_r = max(0, min_r - padding)
        min_c = max(0, min_c - padding)
        max_r = min(source_img.shape[0], max_r + padding)
        max_c = min(source_img.shape[1], max_c + padding)

        return source_img[min_r:max_r, min_c:max_c]

    def get_binary(self, idx=None, cropped=False, padding=0):
        """
        Extracts the isolated binary mask for one, multiple, or all objects
        using localized seed flood-fill (bypassing full-image re-labeling).

        Parameters
        ----------
        idx : int, list of int, slice, or None, optional
            Target index or indices. If None, retrieves masks for all objects.
        cropped : bool, optional
            If True, returns the mask cropped to bounding box (+ padding).
            If False, returns full-sized mask matching original image dimensions.
        padding : int, optional
            Pixel padding around bounding box when `cropped=True`.

        Returns
        -------
        np.ndarray or list of np.ndarray
            Single boolean array or list of boolean arrays.
        """
        if self.df.empty:
            return [] if idx is None or not np.isscalar(idx) else None

        # Resolve target indices
        if idx is None:
            target_indices = self.df.index.tolist()
        elif isinstance(idx, slice):
            target_indices = self.df.index[idx].tolist()
        elif np.isscalar(idx):
            target_indices = [int(idx)]
        else:
            target_indices = list(idx)

        is_single = np.isscalar(idx)
        masks = []

        for i in target_indices:
            if i not in self.df.index:
                raise KeyError(f"Object index {i} not found in dataset.")

            obj = self.objects[i]
            bin_img = self.imgs_binary[obj["img_id"]]

            min_r, min_c, max_r, max_c = obj["bbox"]
            seed_r, seed_c = obj["seed_local"]

            # Crop binary region around object bounding box
            crop_bin = bin_img[min_r:max_r, min_c:max_c]

            # Flood fill starting from seed point to extract only this connected component
            isolated_crop = flood(crop_bin, seed_point=(seed_r, seed_c))

            if cropped and padding == 0:
                masks.append(isolated_crop)
                continue

            # Handle padding or full-frame mask reconstruction
            img_h, img_w = bin_img.shape
            r0 = max(0, min_r - padding)
            c0 = max(0, min_c - padding)
            r1 = min(img_h, max_r + padding)
            c1 = min(img_w, max_c + padding)

            if cropped:
                # Padded crop reconstruction
                padded_mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
                offset_r = min_r - r0
                offset_c = min_c - c0
                padded_mask[offset_r:offset_r + (max_r - min_r), offset_c:offset_c + (max_c - min_c)] = isolated_crop
                masks.append(padded_mask)
            else:
                # Full image reconstruction
                full_mask = np.zeros((img_h, img_w), dtype=bool)
                full_mask[min_r:max_r, min_c:max_c] = isolated_crop
                masks.append(full_mask)

        return masks[0] if is_single else masks


    @staticmethod
    def _to_list(idx):
        """Resolve target indices from scalar, list, or slice"""
        if isinstance(idx, slice):
            target_indices = list(range(idx.start, idx.stop, idx.step if idx.step else 1))
        elif np.isscalar(idx):
            target_indices = [int(idx)]
        else:
            target_indices = list(idx)
        return target_indices


    def imshow(self, idx=None, 
               f_img=True, f_show=False, f_scale=False, f_text=True, f_diam=True, f_dp=True, f_encl=False,
               color=[1, 0, 0.5], **kwargs):

        # Resolve target indices from scalar, list, or slice
        idx = self._to_list(idx)
        
        # Parse inputs
        if np.any(idx == None):
            idx = np.unique(self.df['img_id'])
        else:
            idx = np.unique([self.df.loc[ii]['img_id'] for ii in idx])
        
        if len(idx) > 24 and not isinstance(idx, list):
            idx = idx[:24]
        n_img = len(idx)

        if n_img > 1 and not f_show:
            N1 = int(np.floor(np.sqrt(n_img)))
            N2 = int(np.ceil(n_img / N1))
            plt.subplot(N1, N2, 1)

        print('Collecting images for plotting:')
        for ii in tqdm(range(n_img)):
            if n_img > 1 and not f_show:
                plt.subplot(N1, N2, ii + 1)

            # Determine aggregates to plot for this image
            img_idx = self.df.index[self.df['img_id'] == idx[ii]].tolist()
            if not img_idx:
                print(f'Warning: No aggregates for image no. {idx[ii]}.')
                continue

            if f_img:
                img_binary = np.zeros_like(self.imgs[idx[ii]])
                for agg_idx in img_idx:
                    img_binary = np.logical_or(img_binary, self.imgs_binary[idx[ii]])
                
                pixsize = self.df.iloc[img_idx[0]]['pixsize'] if f_scale else None

                # Display the image with binary overlay
                tools.imshow_binary0(img=self.imgs[idx[ii]], img_binary=img_binary, pixsize=pixsize, **kwargs)
                plt.title(str(idx[ii]))
            
            for agg_idx in img_idx:
                agg = self.df.loc[agg_idx]

                # Plot an 'x' at the CoM. 
                plt.plot(agg['centroid'][1], agg['centroid'][0], 'xk', linewidth=0.75)

                # Plot Rg and da.
                if f_diam:
                    plt.gca().add_patch(Circle((agg['centroid'][1], agg['centroid'][0]), agg['Rg'] / agg['pixsize'], 
                                            color=color, fill=False, linewidth=0.5))
                    plt.gca().add_patch(Circle((agg['centroid'][1], agg['centroid'][0]), agg['da'] / 2 / agg['pixsize'], 
                                            color=np.array(color) * 0.25, fill=False, linewidth=0.5))
                    
                if f_encl:
                    # Add enclosing circle.
                    plt.gca().add_patch(Circle(agg['encl_c'], agg['encl_r'], 
                                            color=np.array(color) * 0.25, fill=False, linewidth=0.5))
                
                # Plot primary particle diameter if present. 
                if f_dp and hasattr(agg, 'dp') and not np.isnan(agg.dp):
                    plt.gca().add_patch(Circle((agg['centroid'][1], agg['centroid'][0]), 
                                            agg['dp'] / 2 / agg['pixsize'], color=[0.92, 0.16, 0.49], fill=False, linewidth=0.5))

                if 'class_name' in self.df.columns:
                    plt.text(agg['centroid'][1] + 20, agg['centroid'][0], str(agg['id']) + "." + str(agg['class_name'])[0:3], color='black', size='x-small')
                else:
                    plt.text(agg['centroid'][1] + 20, agg['centroid'][0], str(agg['id']), color='black', size='small')

    def imshow1(self, idx=0, padding=50, dp_type=""):
        """Show a cropped and masked visualization of aggregate(s) with geometric annotations."""
        if self.df.empty:
            print("Dataset is empty.")
            return

        # Resolve target indices from scalar, list, or slice
        idx = self._to_list(idx)

        col_name = f"dp_{dp_type}" if dp_type else "dp"

        for i in idx:
            if i not in self.df.index:
                print(f"Error: Aggregate index {i} not found in dataset. Skipping.")
                continue

            obj = self.objects[i]
            agg = self.df.loc[i]

            # Slicing bounding box bounds with padding
            min_r, min_c, max_r, max_c = agg["bbox"]
            source_orig = obj["image_orig_ref"]
            source_bin = self.get_binary(i, cropped=False)

            r0 = max(0, min_r - padding)
            c0 = max(0, min_c - padding)
            r1 = min(source_orig.shape[0], max_r + padding)
            c1 = min(source_orig.shape[1], max_c + padding)

            crop_orig = source_orig[r0:r1, c0:c1]
            crop_bin = source_bin[r0:r1, c0:c1]

            # Display image with binary overlay using custom tool function
            tools.imshow(crop_orig, np.ma.masked_where(~crop_bin, crop_bin), show=False)

            # Convert global image centroid coordinates to local crop coordinates
            cy_global, cx_global = agg["centroid"]
            center = (cx_global - c0, cy_global - r0)

            pixsize = agg["pixsize"]
            Rg_pix = agg["Rg"] / pixsize if "Rg" in agg and not np.isnan(agg["Rg"]) else np.nan
            ra_pix = (agg["da"] / 2.0) / pixsize if "da" in agg and not np.isnan(agg["da"]) else np.nan

            theta = np.linspace(0, 2 * np.pi, 100)

            # Plot Radius of Gyration circle
            if not np.isnan(Rg_pix):
                x_rg = center[0] + Rg_pix * np.cos(theta)
                y_rg = center[1] + Rg_pix * np.sin(theta)
                plt.plot(x_rg, y_rg, "--", linewidth=1, color="k", label="Rg")

            # Plot Area-Equivalent Radius circle
            if not np.isnan(ra_pix):
                x_ra = center[0] + ra_pix * np.cos(theta)
                y_ra = center[1] + ra_pix * np.sin(theta)
                plt.plot(x_ra, y_ra, "-", linewidth=3, color=[0.92, 0.16, 0.49], label="da / 2")

            # Plot Primary Particle Diameter circle if present
            if col_name in agg and not np.isnan(agg[col_name]):
                dp_pix = (agg[col_name] / 2.0) / pixsize
                x_dp = center[0] + dp_pix * np.cos(theta)
                y_dp = center[1] + dp_pix * np.sin(theta)
                plt.plot(x_dp, y_dp, "-", linewidth=3, color="k", label=col_name)

            plt.plot(center[0], center[1], "xk", markeredgewidth=1.5)
            plt.title(f"Agg={i}")
            plt.show()