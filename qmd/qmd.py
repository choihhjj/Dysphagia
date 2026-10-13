import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def column_top_percentage_threshold(img, percentage):
    """Binarize image by selecting top N% intensity pixels per column."""
    height, width = img.shape[:2]
    n_pixels = max(1, int(height * percentage / 100))
    
    # Get sorted indices along the vertical axis
    sorted_indices = np.argsort(img, axis=0)
    
    # Select top n_pixels rows for each column
    top_indices = sorted_indices[-n_pixels:, :]
    
    # Apply thresholding using advanced indexing
    binary_img = np.zeros_like(img)
    col_indices = np.arange(width)
    binary_img[top_indices, col_indices] = 255
    
    return binary_img

def analyze_components(img, thresh_percentage): 
    """Binarize, filter noise, and extract connected components statistics."""
    # 1. Binarization
    binary_img = column_top_percentage_threshold(img, thresh_percentage)

    # 2. Morphological opening (Noise removal)
    kernel = np.ones((3, 3), np.uint8)
    morphol_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # 3. Labeling
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(morphol_img)
        
    # Exception: Return None if only background exists
    if stats.shape[0] <= 1:  
        return None, None, None, None, None, morphol_img
    
    # 4. Sort by area (descending) and exclude background
    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_idx = np.argsort(areas)[::-1]
    top_20_idx = sorted_idx[:min(20, len(sorted_idx))] + 1 
        
    # 5. Identify component with maximum height
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    max_height_idx = np.argmax(heights) + 1  
        
    return cnt, labels, stats, top_20_idx, max_height_idx, morphol_img

def determine_location(upper_third, middle_third, y_min, y_max, complex_mode=True):
    """Identify vertical position (Upper/Middle/Lower) based on y-range."""
    total_height = y_max - y_min
    if total_height <= 0:
        return "Unknown"

    # Calculate pixel overlap for each vertical section
    upper_overlap  = max(0, min(upper_third, y_max) - y_min)
    middle_overlap = max(0, min(middle_third, y_max) - max(y_min, upper_third))
    lower_overlap  = max(0, y_max - max(y_min, middle_third))
        
    # Calculate overlap ratios
    u_ratio, m_ratio, l_ratio = upper_overlap/total_height, middle_overlap/total_height, lower_overlap/total_height
        
    # Spanning detection
    if complex_mode and u_ratio > 0.15 and m_ratio > 0.15:
        return "Upper_Middle_Spanning"
        
    # Dominant section detection
    if u_ratio > max(m_ratio, l_ratio): return "Upper"
    if m_ratio > l_ratio: return "Middle"
    return "Lower"

def process_components_in_range(img, filtered_img, height, width, top_k_idx, cnt, stats, labels,
                               top1_largest_idx, ref_idx, ref_y_min, ref_y_max, margin=10, max_y=334):
    """Filter components by y-range and generate color-coded visualization."""
    result = np.zeros_like(img)
    colored = np.zeros((height, width, 3), dtype=np.uint8)

    # Define visualization colors
    C_RED, C_GREEN, C_BLUE, C_YELLOW = [255,0,0], [0,255,0], [0,0,255], [255,255,0]

    # Process Top-K components
    top_k_set = set(top_k_idx)
    for i in top_k_idx:
        y_min, y_max = stats[i, 1], stats[i, 1] + stats[i, 3]
        mask = (labels == i)
        
        # Keep if within vertical range
        if (y_min >= ref_y_min - margin) and (y_max <= max_y):
            result[mask] = filtered_img[mask]
            colored[mask] = C_BLUE if i == ref_idx else (C_YELLOW if i == top1_largest_idx else C_GREEN)
        else:
            colored[mask] = C_YELLOW if (i == top1_largest_idx and i != ref_idx) else C_RED

    # Mark non-Top-K components as noise (Red)
    for i in range(1, cnt):
        if i not in top_k_set: colored[labels == i] = C_RED

    # Pixel analysis for margin evaluation
    y_slice = colored[max(0, int(ref_y_min-margin)) : min(height, int(ref_y_max+margin)), :]
    red_px = np.sum(np.all(y_slice == C_RED, axis=-1))
    grn_px = np.sum(np.all(y_slice == C_GREEN, axis=-1))
        
    return result, colored, (red_px, grn_px)

def get_roi_simple(filename, percentage=65, max_y=None, enable_rebinarization=True):
    """
    Extract ROI from ultrasound images using connected component analysis
    
    Parameters:
    filename (str): Input image
    percentage (float): Top pixel percentage per column (default: 65%)
    max_y (int): Y-axis upper limit (None = no limit, default: None)
    enable_rebinarization (bool): Enable Upper_Middle_Spanning rebinarization (default: True)
    
    Returns:
    tuple: (original image, ROI result image, component visualization image, filtered image)
    """
    
    # Load image
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image from {filename}")
    
    height, width = img.shape[:2]
    
    # Divide image into 3 vertical sections
    upper_third = height // 3
    middle_third = 2 * height // 3
    
    # Use full height if max_y not specified
    if max_y is None:
        max_y = height
    
    # Initial component analysis
    cnt, labels, stats, top_k_idx, max_height_idx, morphol_img = analyze_components(img, percentage)
    filtered_img = np.zeros_like(img)
    filtered_img = np.where(morphol_img != 0, img, 0)
    
    # Initialize result and visualization images
    result_img = np.zeros_like(img)
    colored_components = np.zeros((height, width, 3), dtype=np.uint8)
    
    # No components found
    if top_k_idx is None:
        print(f"No components found in image {filename}")
        return img, np.zeros_like(img), np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8), filtered_img

    # Update component info
    top1_largest_idx = top_k_idx[0]
    top1_tallest_idx = max_height_idx

    # Top1-largest info
    top1_largest_x = stats[top1_largest_idx, 0]
    top1_largest_y = stats[top1_largest_idx, 1]
    top1_largest_w = stats[top1_largest_idx, 2]
    top1_largest_h = stats[top1_largest_idx, 3]
    top1_largest_y_min = top1_largest_y
    top1_largest_y_max = top1_largest_y + top1_largest_h

    # Top1-tallest info
    top1_tallest_x = stats[top1_tallest_idx, 0]
    top1_tallest_y = stats[top1_tallest_idx, 1]
    top1_tallest_w = stats[top1_tallest_idx, 2]
    top1_tallest_h = stats[top1_tallest_idx, 3]
    top1_tallest_y_min = top1_tallest_y
    top1_tallest_y_max = top1_tallest_y + top1_tallest_h
    
    # Determine location (complex_mode based on rebinarization flag)
    complex_mode = enable_rebinarization
    
    # Determine Top1-largest and Top1-tallest locations
    top1_largest_location = determine_location(upper_third, middle_third, 
                                               top1_largest_y_min, top1_largest_y_max, 
                                               complex_mode=complex_mode)
    top1_tallest_location = determine_location(upper_third, middle_third, 
                                               top1_tallest_y_min, top1_tallest_y_max, 
                                               complex_mode=complex_mode)

    print(f"Top1-largest location: {top1_largest_location}")
    print(f"Top1-tallest location: {top1_tallest_location}")

    # Top2-largest info (if exists)
    if len(top_k_idx) > 1:
        top2_largest_idx = top_k_idx[1]
        top2_largest_x = stats[top2_largest_idx, 0]
        top2_largest_y = stats[top2_largest_idx, 1]
        top2_largest_w = stats[top2_largest_idx, 2]
        top2_largest_h = stats[top2_largest_idx, 3]
        top2_largest_y_min = top2_largest_y
        top2_largest_y_max = top2_largest_y + top2_largest_h
        top2_largest_location = determine_location(upper_third, middle_third,
                                                   top2_largest_y_min, top2_largest_y_max,
                                                   complex_mode=complex_mode)
    else:
        top2_largest_idx = None
        top2_largest_location = None

    if top2_largest_idx is not None:
        print(f"Top2 location: {top2_largest_location}")
    
    # Calculate horizontal spread
    is_top1_largest_spread = top1_largest_w / width >= 0.9
    is_top1_tallest_spread = top1_tallest_w / width >= 0.9
    if top2_largest_idx is not None:
        is_top2_largest_spread = top2_largest_w / width >= 0.9
    else:
        is_top2_largest_spread = False

    print(f"Top1-largest spread: {is_top1_largest_spread}")
    print(f"Top1-tallest spread: {is_top1_tallest_spread}")
    if top2_largest_idx is not None:
        print(f"Top2-largest spread: {is_top2_largest_spread}")
    
    # Handle Upper_Middle_Spanning rebinarization (only when enable_rebinarization=True)
    if enable_rebinarization and top1_tallest_location == "Upper_Middle_Spanning":
        print(f"Top1-tallest spans Upper_Middle, reset percentage to 40% and recursive call")
        return get_roi_simple(filename, percentage=40, max_y=max_y, enable_rebinarization=False)
    
    # Logic diagram processing starts here
    # Case: Top1-largest == Top1-tallest
    if top1_largest_idx == top1_tallest_idx:
        print("Top1-largest = Top1-tallest match")
        
        # Top1 component in Middle or Lower
        if top1_largest_location in ["Middle", "Lower"]:
            if is_top1_largest_spread:
                print("Top1 component sufficiently spread (>=0.9) -> keep only top1 component from original")
                
                # Keep only Top1 component
                top1_mask = (labels == top1_largest_idx)
                result_img[top1_mask] = filtered_img[top1_mask]
                
                # Visualize components
                for i in range(1, cnt):
                    comp_mask = (labels == i)
                    if i == top1_largest_idx:
                        colored_components[comp_mask] = [0, 0, 255]  # Keep (blue - Top1-tallest)
                    else:
                        colored_components[comp_mask] = [255, 0, 0]  # Remove (red)
            else:
                print("Top1 component not sufficiently spread -> keep only values within top1 component bounds from original")
                
                # Process components in range
                result_img, colored_components, (red_pixels, green_pixels) = process_components_in_range(
                    img, filtered_img, height, width, top_k_idx, cnt, stats, labels,
                    top1_largest_idx, top1_largest_idx, top1_largest_y_min, top1_largest_y_max, margin=20, max_y=max_y
                )
                
                # Retry with wider range (if more red pixels)
                if red_pixels >= green_pixels:
                    print("Expand range and retry (margin: 20 -> 50)")
                    result_img, colored_components, _ = process_components_in_range(
                        img, filtered_img, height, width, top_k_idx, cnt, stats, labels, 
                        top1_largest_idx, top1_largest_idx, top1_largest_y_min, top1_largest_y_max, margin=50, max_y=max_y
                    )
        

        # Top1 component in Upper
        elif top1_largest_location == "Upper":
            if top2_largest_idx is not None:
                if (is_top2_largest_spread) and (top2_largest_location != "Upper"):
                    print("Top2 component sufficiently spread (>=0.9) -> keep only top2 component from original")

                    # Keep only Top2 component
                    top2_mask = (labels == top2_largest_idx)
                    result_img[top2_mask] = filtered_img[top2_mask]

                    # Visualize components
                    for i in range(1, cnt):
                        comp_mask = (labels == i)
                        if i == top2_largest_idx:
                            colored_components[comp_mask] = [0, 255, 0]  # Keep (green)
                        elif i == top1_largest_idx:
                            colored_components[comp_mask] = [0, 0, 255]  # Reference component (blue)
                        else:
                            colored_components[comp_mask] = [255, 0, 0]  # Remove (red)
                            
                elif (is_top2_largest_spread) and (top2_largest_location == "Upper"):
                    # New condition: when top2_largest_location is Upper
                    print("Top2 also in Upper -> keep only components below lower component's bottom bound")

                    # Find lower component between top1 and top2
                    if top1_largest_y_max > top2_largest_y_max:
                        lower_comp_idx = top1_largest_idx
                        lower_comp_y_max = top1_largest_y_max
                        print(f"Top1 is lower, bottom bound: {lower_comp_y_max}")
                    else:
                        lower_comp_idx = top2_largest_idx
                        lower_comp_y_max = top2_largest_y_max
                        print(f"Top2 is lower, bottom bound: {lower_comp_y_max}")

                    # Keep only components below lower component's bottom bound
                    for i in top_k_idx:
                        comp_y_min = stats[i, 1]
                        comp_y_max = comp_y_min + stats[i, 3]

                        # Keep if below or overlapping with bottom bound
                        if comp_y_min >= lower_comp_y_max or (comp_y_min < lower_comp_y_max and comp_y_max > lower_comp_y_max):
                            comp_mask = (labels == i)
                            result_img[comp_mask] = filtered_img[comp_mask]

                            if i == top1_largest_idx:
                                colored_components[comp_mask] = [0, 0, 255]  # Top1 (blue)
                            elif i == top2_largest_idx:
                                colored_components[comp_mask] = [255, 255, 0]  # Top2 (yellow)
                            else:
                                colored_components[comp_mask] = [0, 255, 0]  # Keep (green)
                        else:
                            # Remove if above bottom bound
                            comp_mask = (labels == i)
                            if i == top1_largest_idx:
                                colored_components[comp_mask] = [0, 0, 255]  # Top1 (blue)
                            elif i == top2_largest_idx:
                                colored_components[comp_mask] = [255, 255, 0]  # Top2 (yellow)
                            else:
                                colored_components[comp_mask] = [255, 0, 0]  # Remove (red)
                else:
                    # If Top2 not spread, use process_components_in_range if percentage <= 30
                    if percentage <= 30:
                        print(f"Keep from Top2 top y={top2_largest_y_min} with margin to image bottom")
                        # Process components in range
                        result_img, colored_components, (red_pixels, green_pixels) = process_components_in_range(
                            img, filtered_img, height, width, top_k_idx, cnt, stats, labels, 
                            top1_largest_idx, top2_largest_idx, top2_largest_y_min, top2_largest_y_max, margin=20, max_y=max_y
                        )

                        # Retry with wider range (if more red pixels)
                        if red_pixels >= green_pixels:
                            print("Expand range and retry (margin: 20 -> 50)")
                            result_img, colored_components, _ = process_components_in_range(
                                img, filtered_img, height, width, top_k_idx, cnt, stats, labels, 
                                top1_largest_idx, top2_largest_idx, top2_largest_y_min, top2_largest_y_max, margin=50, max_y=max_y
                            )
                    else:
                        # Recursive call if percentage > 30
                        print(f"Top2 component not sufficiently spread -> recursive call with percentage=30")
                        return get_roi_simple(filename, percentage=30, max_y=334, enable_rebinarization=False)
            else:
                # No Top2
                if percentage <= 30:
                    print("No Top2 & percentage<=30 -> return empty result")
                    # Return empty result
                    return img, result_img, colored_components, filtered_img
                else:
                    print("No Top2 -> recursive call with percentage=30")
                    return get_roi_simple(filename, percentage=30, max_y=334, enable_rebinarization=False)
    
    # Case: Top1-largest != Top1-tallest
    else:
        print("Top1-largest != Top1-tallest -> reference component = Top1-tallest")
        
        # Top1-tallest in Middle or Lower
        if top1_tallest_location in ["Middle", "Lower"]:
            if is_top1_tallest_spread:
                print("Top1-tallest component sufficiently spread (>=0.9) -> keep only top1-tallest component from original")
                
                # Keep only Top1-tallest component
                top1_tallest_mask = (labels == top1_tallest_idx)
                result_img[top1_tallest_mask] = filtered_img[top1_tallest_mask]
                
                # Visualize components
                for i in range(1, cnt):
                    comp_mask = (labels == i)
                    if i == top1_tallest_idx:
                        colored_components[comp_mask] = [0, 0, 255]  # Reference component (blue)
                    elif i == top1_largest_idx:
                        colored_components[comp_mask] = [255, 255, 0]  # Top1-largest (yellow)
                    else:
                        colored_components[comp_mask] = [255, 0, 0]  # Remove (red)
            else:
                print("Top1-tallest component not sufficiently spread -> keep only values within top1-tallest component bounds from original")
                
                # Process components in range
                result_img, colored_components, (red_pixels, green_pixels) = process_components_in_range(
                    img, filtered_img, height, width, top_k_idx, cnt, stats, labels, 
                    top1_largest_idx, top1_tallest_idx, top1_tallest_y_min, top1_tallest_y_max, margin=20, max_y=max_y
                )
                
                # Retry with wider range (if more red pixels)
                if red_pixels >= green_pixels:
                    print("Expand range and retry (margin: 20 -> 50)")
                    result_img, colored_components, _ = process_components_in_range(
                        img, filtered_img, height, width, top_k_idx, cnt, stats, labels, 
                        top1_largest_idx, top1_tallest_idx, top1_tallest_y_min, top1_tallest_y_max, margin=50, max_y=max_y
                    )

        # Top1-tallest in Upper
        elif top1_tallest_location == "Upper":
            if is_top1_largest_spread:
                print("Top1-largest component sufficiently spread (>=0.9) -> keep only Top1-largest component from original")
                
                # Keep only Top1-largest component
                top1_largest_mask = (labels == top1_largest_idx)
                result_img[top1_largest_mask] = filtered_img[top1_largest_mask]
                
                # Visualize components
                for i in range(1, cnt):
                    comp_mask = (labels == i)
                    if i == top1_largest_idx:
                        colored_components[comp_mask] = [255, 255, 0]  # Top1-largest (yellow)
                    elif i == top1_tallest_idx:
                        colored_components[comp_mask] = [0, 0, 255]  # Reference component (blue)
                    else:
                        colored_components[comp_mask] = [255, 0, 0]  # Remove (red)
            else:
                # Use process_components_in_range if percentage <= 30
                if percentage <= 30:
                    print(f"Top1-largest component not sufficiently spread -> keep from Top1-tallest top with margin to image bottom")
                    # Process components in range
                    result_img, colored_components, (red_pixels, green_pixels) = process_components_in_range(
                        img, filtered_img, height, width, top_k_idx, cnt, stats, labels, 
                        top1_largest_idx, top1_tallest_idx, top1_tallest_y_min, top1_tallest_y_max, margin=20, max_y=max_y
                    )

                    # Retry with wider range (if more red pixels)
                    if red_pixels >= green_pixels:
                        print("Expand range and retry (margin: 20 -> 50)")
                        result_img, colored_components, _ = process_components_in_range(
                            img, filtered_img, height, width, top_k_idx, cnt, stats, labels, 
                            top1_largest_idx, top1_tallest_idx, top1_tallest_y_min, top1_tallest_y_max, margin=50, max_y=max_y
                        )
                else:
                    print("Top1-largest component not sufficiently spread -> recursive call with percentage=30")
                    return get_roi_simple(filename, percentage=30, max_y=334, enable_rebinarization=False)

    return img, result_img, colored_components, filtered_img


def create_weight_matrix(region_height, x_width=5):
    """Dynamically generate weight matrix based on search region size"""
    x = np.arange(x_width)
    y = np.arange(region_height)
    X, Y = np.meshgrid(x, y)
        
    # Weight enhancement and directionality
    y_weight = np.exp(-0.05 * ((Y - region_height//2) / (region_height/4)) ** 2)
    x_weight = np.exp(-0.8 * ((X - x_width//2) / (x_width/3)) ** 2)
        
    # Directional weight
    direction_weight = np.exp(-0.1 * (Y - region_height//2))
    direction_weight = direction_weight / np.max(direction_weight)
        
    weight_matrix = y_weight * x_weight * direction_weight
    # Normalize weights
    return weight_matrix / np.max(weight_matrix)


def smooth_and_interpolate(tracked_points, width):
    """Smooth and interpolate tracked points"""
    if len(tracked_points) == 0:
        return np.column_stack((np.zeros(width), np.arange(width)))
    
    # Constants
    WINDOW_SIZE = 5
    MAX_DIFF_DEFAULT = 30
    MAX_DIFF_RIGHT = 10
    RIGHT_THRESHOLD = 0.7
    
    # Separate x, y coordinates
    y = tracked_points[:, 0]
    x = tracked_points[:, 1]
    
    # Smoothing using moving average
    if len(y) >= WINDOW_SIZE:
        y_smoothed = np.convolve(y, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')
        x_valid = x[WINDOW_SIZE-1:]
    else:
        y_smoothed = y
        x_valid = x
    
    # Interpolate over full range
    full_x = np.arange(width)
    x_min = np.min(x_valid)
    x_max = np.max(x_valid)
    valid_indices = (full_x >= x_min) & (full_x <= x_max)
    
    full_y = np.zeros(width)
    if np.any(valid_indices):
        full_y[valid_indices] = np.interp(full_x[valid_indices], x_valid, y_smoothed)
    
    # Limit rate of change
    for i in range(1, len(full_y)):
        # Skip boundaries of data regions
        if (full_y[i] == 0 and full_y[i-1] > 0) or (full_y[i] > 0 and full_y[i-1] == 0):
            continue
        
        # Apply constraint within data regions
        if full_y[i] > 0 and full_y[i-1] > 0:
            x_position = i / width
            max_diff = MAX_DIFF_RIGHT if x_position >= RIGHT_THRESHOLD else MAX_DIFF_DEFAULT
            
            diff = full_y[i] - full_y[i-1]
            if abs(diff) > max_diff:
                full_y[i] = full_y[i-1] + np.sign(diff) * max_diff
    
    return np.column_stack((full_y, full_x))

def track_from_left_with_vertical_drop(gradient_image, start_y):
    """
    Track muscle layer from left to right with immediate vertical drop when gradient disappears
    
    Parameters:
    gradient_image (numpy.ndarray): Gradient image
    start_y (int): Starting y coordinate
    
    Returns:
    numpy.ndarray: Tracked coordinates in (y, x) format
    """
    # Constants
    WINDOW_HEIGHT = 25
    SEARCH_WIDTH = 5
    PREV_ROWS_LIMIT = 5
    DROP_SPEED = 1.2
    DROP_ACCELERATION = 1.1
    GRADIENT_THRESHOLD = 0.005
    MIN_WEIGHTED_REGION = 0.1
    
    height, width = gradient_image.shape
    tracked_points = []
    prev_rows = []
    
    # Variables for vertical drop detection and handling
    vertical_drop_active = False
    current_drop_speed = DROP_SPEED
    
    # Set starting point
    current_col = 0
    current_row = start_y
    tracked_points.append((current_row, current_col))
    prev_rows.append(current_row)
    
    def has_gradient(column_data):
        """Check if gradient exists in column"""
        return np.max(column_data) > GRADIENT_THRESHOLD
    
    def update_prev_rows(row):
        """Update previous rows list with size limit"""
        prev_rows.append(row)
        if len(prev_rows) > PREV_ROWS_LIMIT:
            prev_rows.pop(0)
    
    while current_col < width - 1:
        search_width = min(SEARCH_WIDTH, width - current_col - 1)
        if search_width < 1:
            break
        
        # Check gradient presence
        current_column = gradient_image[:, current_col]
        next_column = gradient_image[:, current_col+1]
        current_has_gradient = has_gradient(current_column)
        gradient_exists = has_gradient(next_column)
        
        # Detect transition from gradient to no-gradient - start vertical drop immediately
        if current_has_gradient and not gradient_exists and not vertical_drop_active:
            vertical_drop_active = True
            current_drop_speed = DROP_SPEED
            
            # Move to next column immediately and start dropping
            current_col += 1
            current_row += current_drop_speed
            current_drop_speed *= DROP_ACCELERATION
            current_row = min(current_row, height - 1)
            
            tracked_points.append((current_row, current_col))
            update_prev_rows(current_row)
            continue
        
        # Vertical drop mode
        if vertical_drop_active:
            current_col += 1
            
            # Calculate vertical drop with acceleration
            current_row = int(current_row + current_drop_speed)
            current_drop_speed *= DROP_ACCELERATION
            current_row = min(current_row, height - 1)
            
            # Check for gradient to end vertical drop
            next_column = gradient_image[:, current_col] if current_col < width else np.zeros(height)
            if has_gradient(next_column) or current_col >= width - 1:
                vertical_drop_active = False
                
                # Move to exact gradient location if found
                if has_gradient(next_column):
                    current_row = np.argmax(next_column)
            
            tracked_points.append((current_row, current_col))
            update_prev_rows(current_row)
            continue
            
        # Normal tracking mode (when gradient exists)
        # Limit vertical search range
        if len(prev_rows) >= 3:
            predicted_row = int(np.mean(prev_rows[-3:]))
        else:
            predicted_row = current_row

        half_window = WINDOW_HEIGHT // 2
        start_row = max(0, predicted_row - half_window)
        end_row = min(height, predicted_row + half_window)
        region_height = end_row - start_row

        # Create weight matrix
        weight = create_weight_matrix(region_height, search_width)

        # Extract search region and apply weights
        region = gradient_image[start_row:end_row, 
                             current_col+1:current_col+1+search_width]
        weighted_region = region * weight

        if np.max(weighted_region) < MIN_WEIGHTED_REGION:
            # Expand search range
            prev_step = current_row - prev_rows[-1] if len(prev_rows) > 1 else 0
            predicted_row = current_row + prev_step
            start_row = max(0, predicted_row - WINDOW_HEIGHT)
            end_row = min(height, predicted_row + WINDOW_HEIGHT)
            region_height = end_row - start_row

            weight = create_weight_matrix(region_height, search_width)
            region = gradient_image[start_row:end_row, current_col+1:current_col+1+search_width]
            weighted_region = region * weight

        if np.max(weighted_region) > 0:
            max_row = np.unravel_index(np.argmax(weighted_region), weighted_region.shape)[0]
            current_row = start_row + max_row
        else:
            # Maintain previous direction
            if len(prev_rows) >= 2:
                current_row = current_row + (current_row - prev_rows[-2])
            current_row = np.clip(current_row, 0, height-1)

        current_col += 1
        tracked_points.append((current_row, current_col))
        update_prev_rows(current_row)
    
    # Interpolation and smoothing
    tracked_points = np.array(tracked_points)
    smoothed_points = smooth_and_interpolate(tracked_points, width)
    
    return smoothed_points

def track_from_right(gradient_image, start_y):
    """Track muscle layer from right to left"""
    # Constants
    WINDOW_HEIGHT = 20
    SEARCH_WIDTH = 5
    PREV_ROWS_LIMIT = 5
    MIN_WEIGHTED_REGION = 0.1
    
    height, width = gradient_image.shape
    tracked_points = []
    prev_rows = []
    
    # Set starting point (right edge)
    current_col = width - 1
    current_row = start_y
    tracked_points.append((current_row, current_col))
    prev_rows.append(current_row)
    
    def update_prev_rows(row):
        """Update previous rows list with size limit"""
        prev_rows.append(row)
        if len(prev_rows) > PREV_ROWS_LIMIT:
            prev_rows.pop(0)
    
    def get_predicted_row():
        """Get predicted row based on previous rows"""
        if len(prev_rows) >= 3:
            return int(np.mean(prev_rows[-3:]))
        return current_row
    
    def create_search_region(predicted_row, half_window):
        """Create search region boundaries"""
        start_row = max(0, predicted_row - half_window)
        end_row = min(height, predicted_row + half_window)
        return start_row, end_row
    
    while current_col > 0:
        search_width = min(SEARCH_WIDTH, current_col)
        if search_width < 1:
            break
        
        # Limit vertical search range
        predicted_row = get_predicted_row()
        half_window = WINDOW_HEIGHT // 2
        start_row, end_row = create_search_region(predicted_row, half_window)
        region_height = end_row - start_row
        
        # Create weight matrix (flip horizontally for left direction)
        weight = create_weight_matrix(region_height, search_width)
        weight = np.fliplr(weight)
        
        # Extract search region and apply weights
        region = gradient_image[start_row:end_row, 
                             current_col-search_width:current_col]
        weighted_region = region * weight
        
        if np.max(weighted_region) < MIN_WEIGHTED_REGION:
            # Expand search range
            prev_step = current_row - prev_rows[-1] if len(prev_rows) > 1 else 0
            predicted_row = current_row + prev_step
            start_row, end_row = create_search_region(predicted_row, WINDOW_HEIGHT)
            region_height = end_row - start_row
            
            weight = create_weight_matrix(region_height, search_width)
            weight = np.fliplr(weight)
            region = gradient_image[start_row:end_row, current_col-search_width:current_col]
            weighted_region = region * weight
        
        if np.max(weighted_region) > 0:
            max_indices = np.unravel_index(np.argmax(weighted_region), weighted_region.shape)
            max_row, max_col = max_indices
            current_row = start_row + max_row
            current_col = current_col - search_width + max_col
        else:
            # Maintain previous direction
            if len(prev_rows) >= 2:
                current_row = current_row + (current_row - prev_rows[-2])
            current_row = np.clip(current_row, 0, height-1)
            current_col -= 1
        
        tracked_points.append((current_row, current_col))
        update_prev_rows(current_row)
    
    # Interpolation and smoothing
    tracked_points = np.array(tracked_points)
    
    # Sort by x coordinate as results may not be sorted
    if len(tracked_points) > 0:
        sort_idx = np.argsort(tracked_points[:, 1])
        tracked_points = tracked_points[sort_idx]
    
    smoothed_points = smooth_and_interpolate(tracked_points, width)
    
    return smoothed_points

def track_muscle_layer_with_vertical_drop(gradient_image, start_y_left=None, start_y_right=None):
    """
    Bidirectional muscle layer tracking with vertical drop feature
    
    Parameters:
    gradient_image (numpy.ndarray): Gradient image
    start_y_left (int, optional): Y coordinate to start from left
    start_y_right (int, optional): Y coordinate to start from right
    
    Returns:
    tuple: (left-to-right tracking result, right-to-left tracking result)
    """
    height, width = gradient_image.shape
    
    def find_start_point(column_data, image):
        """Find starting point for tracking"""
        if np.all(column_data == 0):
            # If column has no data, use row with most data
            return np.argmax(np.sum(image > 0, axis=1))
        else:
            # Use position of maximum gradient in column
            return np.argmax(column_data)
    
    # Auto-set starting points if not provided
    if start_y_left is None:
        start_y_left = find_start_point(gradient_image[:, 0], gradient_image)
    
    if start_y_right is None:
        start_y_right = find_start_point(gradient_image[:, -1], gradient_image)
    
    # Track from left to right (red line) with vertical drop feature
    left_to_right = track_from_left_with_vertical_drop(gradient_image, start_y_left)
    
    # Track from right to left (blue line) using existing function
    right_to_left = track_from_right(gradient_image, start_y_right)
    
    return left_to_right, right_to_left


def combine_tracks_based_on_last_intersection(left_to_right, right_to_left, gradient_image, threshold=0.05):
    """
    Combine tracks by selecting based on last intersection point
    
    Parameters:
    left_to_right: Left-to-right tracking result (red line)
    right_to_left: Right-to-left tracking result (blue line)
    gradient_image: Gradient image
    threshold: Gradient detection threshold (default: 0.05)
    
    Returns:
    tuple: (final combined result, processing method description, last intersection point)
    """
    # Constants
    RELATION_ABOVE = "above"
    RELATION_BELOW = "below"
    RELATION_SAME = "same"
    
    width = min(left_to_right.shape[0], right_to_left.shape[0])
    combined = np.zeros((width, 2))
    
    # Check for gradient gaps
    max_gradients = np.max(gradient_image, axis=0)
    has_gradient_gaps = any(
        max_gradients[i-1] > threshold and max_gradients[i] <= threshold
        for i in range(1, gradient_image.shape[1])
    )
    
    def find_intersections():
        """Find all intersection points between two tracks"""
        intersections = []
        prev_relation = None
        
        for i in range(width):
            left_y = left_to_right[i, 0]
            right_y = right_to_left[i, 0]
            
            # Only check intersections when both lines are valid
            if left_y > 0 and right_y > 0:
                # Determine current position relationship
                if left_y < right_y:
                    current_relation = RELATION_ABOVE  # Red line above
                elif left_y > right_y:
                    current_relation = RELATION_BELOW  # Red line below
                else:
                    current_relation = RELATION_SAME
                
                # Record intersection when relationship changes
                if (prev_relation is not None and 
                    prev_relation != current_relation and 
                    prev_relation != RELATION_SAME and 
                    current_relation != RELATION_SAME):
                    intersections.append((i, (left_y + right_y) / 2))
                
                prev_relation = current_relation
        
        return intersections
    
    def fill_combined_with_min():
        """Fill combined array with minimum values"""
        for i in range(width):
            left_y = left_to_right[i, 0]
            right_y = right_to_left[i, 0]
            
            if left_y > 0 and right_y > 0:
                combined[i, 0] = min(left_y, right_y)
            elif left_y > 0:
                combined[i, 0] = left_y
            elif right_y > 0:
                combined[i, 0] = right_y
            
            combined[i, 1] = i
    
    def fill_combined_with_split(split_x):
        """Fill combined array split at intersection point"""
        for i in range(width):
            left_y = left_to_right[i, 0]
            right_y = right_to_left[i, 0]
            
            if i <= split_x:
                # Left of intersection: use left-to-right line (red)
                combined[i, 0] = left_y if left_y > 0 else right_y
            else:
                # Right of intersection: use right-to-left line (blue)
                combined[i, 0] = right_y if right_y > 0 else left_y
            
            combined[i, 1] = i
    
    # Find intersections
    intersection_points = find_intersections()
    last_intersection = intersection_points[-1] if intersection_points else None
    
    # Determine processing method
    if has_gradient_gaps:
        method_description = "Using Min Value at Intersections (Gaps Found)"
        fill_combined_with_min()
    elif last_intersection is not None:
        method_description = "Using Last Intersection Split Method"
        last_intersection_x = int(last_intersection[0])
        fill_combined_with_split(last_intersection_x)
    else:
        method_description = "Using Red Line (No Intersections or Gaps)"
        combined = left_to_right.copy()
    
    return combined, method_description, last_intersection

def process_muscle_layer(filename, quantile=50, data_dir='ex_data'):
    """
    Muscle layer tracking function with vertical drop feature

    Parameters:
    filename (str): Image name without extension; reads <data_dir>/<filename>.jpg
    quantile (int): Unused (kept for backward compatibility)
    data_dir (str): Directory containing input images (default: 'ex_data')

    Output:
    Writes the tracked muscle-layer mask (upper 1/5 cropped) to
    <data_dir>/<filename>_muscle.jpg, which is the input of get_qmd().
    """
    img, roi_img, component, filtered = get_roi_simple(os.path.join(data_dir, f'{filename}.jpg'))
    
    # Constants
    KERNEL_SIZE = (5, 5)
    MEDIAN_BLUR_SIZE = 11
    SOBEL_KSIZE = 3
    
    def preprocess_image(img):
        """Preprocess ROI image to extract gradient"""
        # Binarization
        roi_bin = np.where(img > 0, 255, 0).astype(np.uint8)
        
        # Morphological opening
        kernel = np.ones(KERNEL_SIZE, np.uint8)
        ex_bin_opened = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, kernel)
        
        # Reconstruct with original values
        roi_com_recon = np.where(ex_bin_opened != 0, img, 0)
        
        # Median blur
        roi_blur = cv2.medianBlur(roi_com_recon.astype(np.uint8), MEDIAN_BLUR_SIZE)
        
        # Calculate gradient
        sobel_x = cv2.Sobel(roi_blur, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
        sobel_y = cv2.Sobel(roi_blur, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return roi_bin, gradient
    
    def find_start_point(binary_img, column_idx):
        """Find starting point in specified column"""
        column = binary_img[:, column_idx]
        
        if np.all(column == 0):
            # If column has no data, use row with most data
            return np.argmax(np.sum(binary_img > 0, axis=1))
        else:
            # Use first non-zero position in column
            return np.where(column > 0)[0][0]
    
    # Preprocess image
    roi_bin, gradient = preprocess_image(roi_img)
    
    # Find starting points
    start_y_left = find_start_point(roi_bin, 0)
    start_y_right = find_start_point(roi_bin, -1)
    
    # Bidirectional tracking with vertical drop feature
    left_to_right, right_to_left = track_muscle_layer_with_vertical_drop(
        gradient, start_y_left, start_y_right)
    
    # Combine based on last intersection
    combined, method_description, last_intersection = combine_tracks_based_on_last_intersection(
        left_to_right, right_to_left, gradient)
    
    mask = np.zeros_like(img)
    valid_indices = np.where(combined[:, 0] > 0)[0]
    
    for j in range(1, len(valid_indices)):
        pt1 = (int(combined[valid_indices[j-1], 1]), int(combined[valid_indices[j-1], 0]))
        pt2 = (int(combined[valid_indices[j], 1]), int(combined[valid_indices[j], 0]))
        cv2.line(mask, pt1, pt2, 255, thickness=2)
    upper_cut = int(mask.shape[0] * (1/5))
    mask_cropped = mask[upper_cut:, :]
    cv2.imwrite(os.path.join(data_dir, f'{filename}_muscle.jpg'), mask_cropped)

   # return mask_cropped

def find_alpha1(image):
    alpha1 = np.where(np.mean(image, axis = 1) == np.mean(image, axis = 1).max())
    return alpha1[0][0]

def find_range_y_mean(image, which_y = 0):
    min_y = find_alpha1(image)
    if np.mean(image) < 3:
        m_range, x_max, x_min = [0] * 3
        
    elif (min_y + which_y >= 265) or (min_y + which_y <= 0):
        m_range, x_max, x_min = [0] * 3
    
    else:
        idx_255 = np.where(image[min_y + which_y, :] == 255)[0]
        
        if len(idx_255) == 1:
            m_range = 1        
            x_min = idx_255.max()
            x_max = idx_255.max()
        elif len(idx_255) < 1:
            m_range = 0
            x_min = 0
            x_max = 0
        else:
            x_max = idx_255.max() 
            x_min = idx_255.min()
            m_range = x_max - x_min
            
    return min_y+which_y, m_range, x_max, x_min


def get_qmd(filename, image_dura, data_dir='ex_data', show=True):
    """
    Compute the 10 QMD values from the muscle-layer mask produced by process_muscle_layer().

    Parameters:
    filename (str): Image name without extension; reads <data_dir>/<filename>_muscle.jpg
    image_dura (int): Swallowing duration in pixels; the analysis window is
                      columns [60, 60 + image_dura) of the mask
    data_dir (str): Directory containing the mask (default: 'ex_data')
    show (bool): Display the QMD visualization (default: True)

    Returns:
    dict: {QMD_sup2, QMD_sup1, QMD_ref, QMD_inf1, ..., QMD_inf7} in pixels.
          The visualization is saved to <data_dir>/<filename>_muscle_QMD.jpg.
    """
    image = cv2.imread(os.path.join(data_dir, f'{filename}_muscle.jpg'))
    swallowing = image[:, 60:60+int(image_dura)]
    
    which_y_list = []
    m_range_list = []
    x_max_list = []
    x_min_list = []

    for i in [-20, -10, 0, 10, 20, 30, 40, 50, 60, 70]:
        which_y, m_range, x_max, x_min = find_range_y_mean(swallowing, i)
        which_y_list.append(which_y)
        m_range_list.append(m_range)
        x_max_list.append(x_max)
        x_min_list.append(x_min)

    df_range = pd.DataFrame({
        'filename': filename,
        'which_y': which_y_list,
        'm_range': m_range_list,
        'x_max': x_max_list,
        'x_min': x_min_list
    })
    
    qmd_names = ['QMD_sup2', 'QMD_sup1', 'QMD_ref'] + [f'QMD_inf{i}' for i in range(1, 8)]
    qmd_dict = dict(zip(qmd_names, df_range.m_range.values))
    
    plt.title(m_range_list)
    plt.axis('off')
    plt.imshow(image)

    # Draw vertical boundaries
    plt.axvline(x=60, color='blue')
    plt.axvline(x=60+image_dura, color='blue')

    # Draw horizontal lines and scatter points for each offset
    for i in range(len(df_range)):
        # Reference line (index 2) in yellow, others in green
        line_color = 'yellow' if i == 2 else 'green'

        plt.axhline(y=df_range.which_y[i], color=line_color)
        plt.scatter(x=60+df_range.x_max[i], y=df_range.which_y[i], color='red')
        plt.scatter(x=60+df_range.x_min[i], y=df_range.which_y[i], color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f'{filename}_muscle_QMD.jpg'))
    if show:
        plt.show()
    plt.close()  
    
    return qmd_dict

