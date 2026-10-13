# QMD Pipeline Structure

## Main Workflow

1. `get_roi_simple()` - Extract the suprahyoid muscle region from the 600 × 334 image
2. `process_muscle_layer()` - Track the muscle boundary and save the mask with the upper 1/5 removed
   (600 × 268) as `<filename>_muscle.jpg`
3. `get_qmd()` - Calculate the 10 QMD (quantitative muscle movement duration) values from the mask,
   save the visualization as `<filename>_muscle_QMD.jpg`, and return them as a dict

## Function Hierarchy

### get_roi_simple
- `[analyze_components]`
  - `[column_top_percentage_threshold]`
- `[determine_location]`
- `[process_components_in_range]`
- `[get_roi_simple]` (recursive call)

### process_muscle_layer
- `[get_roi_simple]`
- `preprocess_image`
- `find_start_point`
- `[track_muscle_layer_with_vertical_drop]`
  - `find_start_point`
  - `[track_from_left_with_vertical_drop]`
    - `has_gradient`
    - `update_prev_rows`
    - `[create_weight_matrix]`
    - `[smooth_and_interpolate]`
  - `[track_from_right]`
    - `update_prev_rows`
    - `get_predicted_row`
    - `create_search_region`
    - `[create_weight_matrix]`
    - `[smooth_and_interpolate]`
- `[combine_tracks_based_on_last_intersection]`
  - `find_intersections`
  - `fill_combined_with_min`
  - `fill_combined_with_split`

### get_qmd
- `[find_range_y_mean]`
  - `[find_alpha1]`

**Legend:**
- `[function_name]`: module-level function
- `function_name`: nested helper function