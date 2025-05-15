"""
Stage D: Identifying "Areas of Movement" (Potential Joints/Flexible Regions)

Objective: To pinpoint and visualize regions on the animal's surface that 
           exhibit consistent, significant, and patterned deformation.
Methods: Thresholding, filtering, temporal consistency analysis, clustering.
Output: A visual map indicating identified "areas of movement" and structured data.
"""
import numpy as np
import os
import cv2 # For visualization later
from sklearn.cluster import DBSCAN # For clustering later

# --- Configuration ---
DEFAULT_DILATATION_THRESHOLD = 0.1 # Example: 10% area change
DEFAULT_MAX_SHEAR_THRESHOLD = 0.1  # Example shear strain threshold
DEFAULT_TEMPORAL_WINDOW = 5      # Min frames for consistent high deformation
DEFAULT_MIN_REGION_POINTS = 10     # Min points to form a significant region
DEFAULT_DBSCAN_EPS = 15.0          # Max distance for DBSCAN neighbors (pixels)
# dbscan_min_samples will default to min_region_points

def identify_movement_areas(
    stage_c_analysis_path: str,    # .npz file from Stage C
    # original_video_path: str,    # Needed for visualization, can get from Stage C metadata
    output_visualization_path: str | None = None, # Path for saving visualizations (e.g., video, images)
    output_movement_data_path: str | None = None, # .npz file to store identified movement areas data
    dilatation_threshold: float = DEFAULT_DILATATION_THRESHOLD,
    max_shear_threshold: float = DEFAULT_MAX_SHEAR_THRESHOLD,
    temporal_consistency_window: int = DEFAULT_TEMPORAL_WINDOW,
    min_region_points: int = DEFAULT_MIN_REGION_POINTS,
    dbscan_eps: float = DEFAULT_DBSCAN_EPS,
    dbscan_min_samples: int | None = None # Defaults to min_region_points if None
):
    """
    Identifies and visualizes areas of significant movement/deformation based on Stage C analysis.
    """
    print(f"Starting Stage D: Identifying Movement Areas")
    print(f"  Loading Stage C analysis from: {stage_c_analysis_path}")
    try:
        stage_c_data_loaded = np.load(stage_c_analysis_path, allow_pickle=True)
        stage_c_data = {key: stage_c_data_loaded[key] for key in stage_c_data_loaded.files}
    except Exception as e:
        print(f"Error loading Stage C analysis from {stage_c_analysis_path}: {e}. Aborting Stage D.")
        return

    print(f"  Found Stage C data for keys: {list(stage_c_data.keys())}")

    # Retrieve necessary metadata and original tracks/points if needed for visualization
    # original_video_path_from_meta = stage_c_data.get('source_multipart_tracks_path', np.array(None)).item()
    # if original_video_path_from_meta:
        # stage_b_data = np.load(original_video_path_from_meta, allow_pickle=True) # might need original points
        # original_video_path = stage_b_data.get('original_video_path', np.array(None)).item()

    all_movement_regions_data = {
        "source_stage_c_analysis_path": stage_c_analysis_path,
        "params": {
            "dilatation_threshold": dilatation_threshold,
            "max_shear_threshold": max_shear_threshold,
            "temporal_consistency_window": temporal_consistency_window,
            "min_region_points": min_region_points,
            "dbscan_eps": dbscan_eps,
            "dbscan_min_samples": dbscan_min_samples if dbscan_min_samples is not None else min_region_points
        },
        "per_frame_regions": [], # List of dicts, one per frame
        "inter_part_motion": {} # New: To store inter-part motion analysis
    }

    part_labels = sorted(list(set([k.split('_intra_deformation')[0] for k in stage_c_data if k.endswith('_intra_deformation')])))
    if not part_labels:
        print("No intra-deformation data found in Stage C results. Cannot identify movement areas.")
        # Save empty results if path provided
        if output_movement_data_path:
            np.savez_compressed(output_movement_data_path, **all_movement_regions_data)
            print(f"Saved empty movement data to {output_movement_data_path}")
        return

    num_frames = 0
    # Determine num_frames from the first valid deformation metric found
    for label in part_labels:
        deformation_data = stage_c_data.get(f"{label}_intra_deformation").item() # .item() because it's a dict
        if deformation_data and 'dilatation_fp' in deformation_data and deformation_data['dilatation_fp'].shape[0] > 0:
            num_frames = deformation_data['dilatation_fp'].shape[0]
            break
    
    if num_frames == 0:
        print("Could not determine number of frames from Stage C data.")
        if output_movement_data_path:
            np.savez_compressed(output_movement_data_path, **all_movement_regions_data)
            print(f"Saved empty movement data to {output_movement_data_path}")
        return

    # This will store, for each part, a (T, N_points) boolean array indicating high deformation
    candidate_high_deformation_points_all_parts = {}

    print(f"\nProcessing {num_frames} frames for {len(part_labels)} parts...")
    for label in part_labels:
        print(f"  Identifying candidate high deformation points for part: '{label}'")
        deformation_data_dict = stage_c_data.get(f"{label}_intra_deformation").item() # dict
        if not deformation_data_dict:
            print(f"    No deformation data for part '{label}'. Skipping.")
            continue

        dilatation_fp = deformation_data_dict.get("dilatation_fp")
        max_shear_fp = deformation_data_dict.get("max_shear_strains_fp")
        # principal_strains_fpt2 = deformation_data_dict.get("principal_strains_fpt2")
        
        if dilatation_fp is None or max_shear_fp is None:
            print(f"    Missing dilatation or max_shear data for '{label}'. Skipping thresholding.")
            continue
        
        current_num_points = dilatation_fp.shape[1]
        if current_num_points == 0:
            candidate_high_deformation_points_all_parts[label] = np.zeros((num_frames, 0), dtype=bool)
            continue
            
        # Initialize boolean array for this part
        part_is_high_deformation = np.zeros((num_frames, current_num_points), dtype=bool)

        for t in range(num_frames):
            # Apply thresholds: point is candidate if EITHER dilatation OR shear is high
            # (Could also use AND or a combined metric)
            high_dilatation_mask = np.abs(dilatation_fp[t, :]) >= dilatation_threshold
            high_shear_mask = np.abs(max_shear_fp[t, :]) >= max_shear_threshold
            
            # Ensure masks are not all NaN before logical_or if NaNs are possible
            # (Current Stage C fills with NaN if KNN fails for a frame, for all points of that part)
            if np.all(np.isnan(dilatation_fp[t,:])) or np.all(np.isnan(max_shear_fp[t,:])):
                 part_is_high_deformation[t, :] = False # No valid data to make a decision
            else:
                part_is_high_deformation[t, :] = np.logical_or(
                    np.nan_to_num(high_dilatation_mask, nan=0.0).astype(bool), 
                    np.nan_to_num(high_shear_mask, nan=0.0).astype(bool)
                )
        candidate_high_deformation_points_all_parts[label] = part_is_high_deformation
        print(f"    Finished candidate identification for '{label}'. Found candidates on some frames: {np.any(part_is_high_deformation)}")

    # --- Temporal Consistency Analysis --- 
    print("\nPerforming Temporal Consistency Analysis...")
    consistent_high_deformation_points_all_parts = {}
    for label, candidate_deformation_tp in candidate_high_deformation_points_all_parts.items():
        num_frames_part, num_points_part = candidate_deformation_tp.shape
        part_is_consistent_deformation = np.zeros_like(candidate_deformation_tp, dtype=bool)

        if num_points_part == 0 or num_frames_part < temporal_consistency_window:
            consistent_high_deformation_points_all_parts[label] = part_is_consistent_deformation
            print(f"  Part '{label}': Not enough points or frames for temporal analysis. No consistent points found.")
            continue
        
        for p_idx in range(num_points_part):
            point_candidate_series_t = candidate_deformation_tp[:, p_idx]
            current_streak = 0
            for t_idx in range(num_frames_part):
                if point_candidate_series_t[t_idx]:
                    current_streak += 1
                else:
                    # If streak was long enough, mark it before resetting
                    if current_streak >= temporal_consistency_window:
                        for i in range(current_streak):
                            part_is_consistent_deformation[t_idx - 1 - i, p_idx] = True
                    current_streak = 0 # Reset streak
            
            # Check for a streak at the very end of the series
            if current_streak >= temporal_consistency_window:
                for i in range(current_streak):
                    part_is_consistent_deformation[num_frames_part - 1 - i, p_idx] = True
        
        consistent_high_deformation_points_all_parts[label] = part_is_consistent_deformation
        print(f"  Part '{label}': Temporal consistency applied. Consistent candidates found: {np.any(part_is_consistent_deformation)}")

    # --- Spatial Clustering --- 
    print("\nPerforming Spatial Clustering per frame...")
    
    # Load Stage B data to get actual point coordinates for clustering
    stage_b_tracks_path = stage_c_data.get('source_multipart_tracks_path', np.array(None)).item()
    if not stage_b_tracks_path or not os.path.exists(stage_b_tracks_path):
        print(f"Error: Path to Stage B tracks ('{stage_b_tracks_path}') not found in Stage C data or file missing. Cannot perform clustering.")
        # Save what we have so far if output path is given
        if output_movement_data_path:
            # Fill per_frame_regions with empty data if we can't cluster
            for t_idx in range(num_frames):
                all_movement_regions_data["per_frame_regions"].append(
                    {"frame_index": t_idx, "regions_for_part": {label: [] for label in part_labels}}
                )
            try:
                np.savez_compressed(output_movement_data_path, **all_movement_regions_data)
                print(f"Saved partial Stage D data (no clustering) to {output_movement_data_path}")
            except Exception as e_save:
                print(f"Error saving partial Stage D data: {e_save}")
        return

    try:
        stage_b_data_loaded = np.load(stage_b_tracks_path, allow_pickle=True)
        stage_b_tracks_data = {key: stage_b_data_loaded[key] for key in stage_b_data_loaded.files}
        print(f"  Successfully loaded Stage B tracks from: {stage_b_tracks_path}")
    except Exception as e_load_b:
        print(f"Error loading Stage B tracks from {stage_b_tracks_path}: {e_load_b}. Cannot perform clustering.")
        if output_movement_data_path:
            for t_idx in range(num_frames):
                all_movement_regions_data["per_frame_regions"].append(
                    {"frame_index": t_idx, "regions_for_part": {label: [] for label in part_labels}}
                )
            try:
                np.savez_compressed(output_movement_data_path, **all_movement_regions_data)
                print(f"Saved partial Stage D data (no clustering) to {output_movement_data_path}")
            except Exception as e_save:
                print(f"Error saving partial Stage D data: {e_save}")
        return

    actual_dbscan_min_samples = dbscan_min_samples if dbscan_min_samples is not None else min_region_points

    for t in range(num_frames):
        frame_regions_data = {"frame_index": t, "regions_for_part": {}}
        for label in part_labels:
            frame_regions_data["regions_for_part"][label] = [] # Initialize list for this part
            if label not in consistent_high_deformation_points_all_parts:
                continue
            
            # Get original indices of points that are consistently deforming for this part at this frame
            consistent_point_indices_for_part = np.where(consistent_high_deformation_points_all_parts[label][t, :])[0]
            
            if len(consistent_point_indices_for_part) < actual_dbscan_min_samples:
                # Not enough points to even attempt clustering for this part at this frame
                continue

            # Get the actual XY coordinates of these points from Stage B tracks
            part_tracks_key = f"{label}_tracks_xy"
            if part_tracks_key not in stage_b_tracks_data:
                print(f"Warning: Tracks for part '{label}' not found in Stage B data. Skipping clustering for this part at frame {t}.")
                continue
            
            # Coordinates of the points to be clustered for this part at this frame
            points_to_cluster_coords_xy = stage_b_tracks_data[part_tracks_key][t, consistent_point_indices_for_part, :]
            
            if points_to_cluster_coords_xy.shape[0] < actual_dbscan_min_samples:
                # After potential visibility issues if tracks were NaN for these points (though unlikely if consistent deformation)
                continue

            # Run DBSCAN
            try:
                db = DBSCAN(eps=dbscan_eps, min_samples=actual_dbscan_min_samples).fit(points_to_cluster_coords_xy)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                if hasattr(db, 'core_sample_indices_') and len(db.core_sample_indices_) > 0:
                    core_samples_mask[db.core_sample_indices_] = True # core_sample_indices_ are relative to points_to_cluster_coords_xy
                
                cluster_labels_for_points = db.labels_ # Labels for each point in points_to_cluster_coords_xy
                unique_clusters = set(cluster_labels_for_points)

                for k_cluster in unique_clusters:
                    if k_cluster == -1:
                        # -1 is noise in DBSCAN
                        continue
                    
                    # Get mask for points belonging to this cluster (relative to points_to_cluster_coords_xy)
                    class_member_mask = (cluster_labels_for_points == k_cluster)
                    num_points_in_cluster = np.sum(class_member_mask)
                    
                    if num_points_in_cluster >= min_region_points: # User-defined threshold for a significant region
                        # Get the original indices (relative to the full part point list) of points in this cluster
                        original_indices_in_cluster = consistent_point_indices_for_part[class_member_mask]
                        
                        # Calculate centroid of this cluster
                        cluster_coords = points_to_cluster_coords_xy[class_member_mask]
                        cluster_centroid = np.mean(cluster_coords, axis=0)
                        
                        frame_regions_data["regions_for_part"][label].append({
                            "cluster_id": int(k_cluster),
                            "point_indices_in_part": original_indices_in_cluster.tolist(),
                            "num_points": num_points_in_cluster,
                            "centroid_xy": cluster_centroid.tolist()
                        })
            except Exception as e_dbscan:
                print(f"Error during DBSCAN for part '{label}' at frame {t}: {e_dbscan}")

        all_movement_regions_data["per_frame_regions"].append(frame_regions_data)
    print("  Finished Spatial Clustering.")

    # --- Collect all persistent points for visualization ---
    persistent_points_for_visualization = {label: set() for label in part_labels}
    if "per_frame_regions" in all_movement_regions_data:
        for frame_data in all_movement_regions_data["per_frame_regions"]:
            for label, regions_in_frame in frame_data["regions_for_part"].items():
                if label in persistent_points_for_visualization: # Ensure label is valid
                    for region_details in regions_in_frame:
                        persistent_points_for_visualization[label].update(region_details["point_indices_in_part"])
    
    # For debugging what was collected
    # for label, indices in persistent_points_for_visualization.items():
    #     print(f"  Persistent points for {label}: {len(indices)} points")


    # --- Leverage Inter-Part Motion ---
    print("\nLeveraging Inter-Part Motion data from Stage C...")
    inter_part_motion_results = {"centroid_velocities": {}}
    if not part_labels:
        print("  No parts found, skipping inter-part motion analysis.")
    else:
        for label in part_labels:
            centroid_key = f"{label}_centroid_xy_t2"
            if centroid_key in stage_c_data:
                centroids_f2 = stage_c_data[centroid_key] # Should be (num_frames, 2)
                if centroids_f2.ndim == 2 and centroids_f2.shape[0] > 1 and centroids_f2.shape[1] == 2:
                    # Calculate velocity: displacement between consecutive frames
                    # Pad with zeros for the first frame's velocity
                    velocities_f2 = np.zeros_like(centroids_f2)
                    velocities_f2[1:] = np.diff(centroids_f2, axis=0) 
                    inter_part_motion_results["centroid_velocities"][label] = velocities_f2
                    print(f"  Calculated centroid velocities for part '{label}'.")
                else:
                    print(f"  Centroid data for part '{label}' has unexpected shape or not enough frames: {centroids_f2.shape}. Skipping velocity calculation.")
                    inter_part_motion_results["centroid_velocities"][label] = np.array([]) # Store empty
            else:
                print(f"  No centroid trajectory data found for part '{label}' (expected key: {centroid_key}).")
                inter_part_motion_results["centroid_velocities"][label] = np.array([]) # Store empty
    
    all_movement_regions_data["inter_part_motion"] = inter_part_motion_results
    print("  Finished inter-part motion analysis (currently calculating centroid velocities).")

    # --- Save Results --- 
    if output_movement_data_path:
        try:
            output_dir = os.path.dirname(output_movement_data_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            np.savez_compressed(output_movement_data_path, **all_movement_regions_data)
            print(f"Successfully saved Stage D movement data to: {output_movement_data_path}")
        except Exception as e:
            print(f"Error saving Stage D movement data: {e}")

    # --- Visualization --- 
    print("\nAttempting Visualization...")
    if output_visualization_path and (output_visualization_path.lower().endswith(('.mp4', '.avi', '.mov'))):
        print(f"  Preparing visualization to be saved to: {output_visualization_path}")
        
        # 1. Get original video path from Stage B data (via Stage C metadata)
        original_video_path = None
        if stage_b_tracks_path and os.path.exists(stage_b_tracks_path): # stage_b_tracks_path loaded for clustering
            # stage_b_tracks_data is already loaded if clustering happened
            if stage_b_tracks_data and 'original_video_path' in stage_b_tracks_data:
                original_video_path = stage_b_tracks_data['original_video_path'].item()
                if not isinstance(original_video_path, str) or not os.path.exists(original_video_path):
                    print(f"Warning: Original video path from Stage B data is invalid or file not found: {original_video_path}")
                    original_video_path = None
            else:
                print("Warning: 'original_video_path' key not found in Stage B data. Cannot create visualization.")
        else:
            print("Warning: Stage B tracks data not available. Cannot create visualization.")

        if original_video_path and num_frames > 0:
            try:
                cap = cv2.VideoCapture(original_video_path)
                if not cap.isOpened():
                    raise IOError(f"Cannot open video file: {original_video_path}")
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                output_vis_dir = os.path.dirname(output_visualization_path)
                if output_vis_dir and not os.path.exists(output_vis_dir):
                    os.makedirs(output_vis_dir, exist_ok=True)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Common codec for .mp4
                video_writer = cv2.VideoWriter(output_visualization_path, fourcc, fps, (width, height))
                if not video_writer.isOpened():
                    raise IOError(f"Cannot open video writer for: {output_visualization_path}")

                print(f"  Processing video frames for visualization ({width}x{height} @ {fps:.2f} FPS)...Output: {output_visualization_path}")

                for t in range(num_frames):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Warning: Could not read frame {t} from video. Stopping visualization early.")
                        break
                    
                    # Get currently active regions for this frame for quick lookup
                    current_frame_active_regions_by_part = {}
                    if t < len(all_movement_regions_data["per_frame_regions"]):
                         current_frame_active_regions_by_part = all_movement_regions_data["per_frame_regions"][t]["regions_for_part"]

                    # --- Draw all persistent points first (subtly if not active) ---
                    for label in part_labels:
                        part_color = get_color_for_label(label, part_labels)
                        dimmed_color = tuple(c // 2 for c in part_color) # Dimmed color for non-active persistent points

                        current_persistent_indices = list(persistent_points_for_visualization.get(label, set()))
                        if not current_persistent_indices:
                            continue

                        part_tracks_key = f"{label}_tracks_xy"
                        if stage_b_tracks_data and part_tracks_key in stage_b_tracks_data:
                            part_tracks_at_t = stage_b_tracks_data[part_tracks_key]
                            if part_tracks_at_t.shape[0] > t and (not current_persistent_indices or np.max(current_persistent_indices) < part_tracks_at_t.shape[1]):
                                points_xy_for_persistent = part_tracks_at_t[t, current_persistent_indices, :]
                                
                                for i, original_point_index in enumerate(current_persistent_indices):
                                    # original_point_index is the index within the specific part's full list of points.
                                    # We need to ensure this index is valid for the visibility array of the part.

                                    point_coord_xy = points_xy_for_persistent[i, :]
                                    if np.isnan(point_coord_xy[0]) or np.isnan(point_coord_xy[1]):
                                        continue # Skip NaN points

                                    # Check CoTracker's visibility for this point at this frame
                                    is_visible_in_stage_b = True # Default to true if visibility data is missing for some reason
                                    visibility_key = f"{label}_visibility"
                                    if stage_b_tracks_data and visibility_key in stage_b_tracks_data:
                                        part_visibility_at_t = stage_b_tracks_data[visibility_key]
                                        if part_visibility_at_t.shape[0] > t and original_point_index < part_visibility_at_t.shape[1]:
                                            is_visible_in_stage_b = part_visibility_at_t[t, original_point_index]
                                        # else:
                                            # Potentially log if visibility index is out of bounds, but be cautious of spam
                                            # print(f"Warning: Visibility index {original_point_index} for part '{label}' out of bounds at frame {t}.")
                                            # is_visible_in_stage_b = False # Treat as not visible if index is bad
                                    # else:
                                        # print(f"Warning: Visibility data for part '{label}' not found in Stage B data.")
                                        # is_visible_in_stage_b = False # Treat as not visible if key is missing
                                        
                                    if not is_visible_in_stage_b:
                                        continue # Skip drawing if CoTracker marked it not visible

                                    pt_x, pt_y = int(point_coord_xy[0]), int(point_coord_xy[1])
                                    
                                    # Check if this point is active in the current frame
                                    is_active_now = False
                                    for active_region in current_frame_active_regions_by_part.get(label, []):
                                        if original_point_index in active_region["point_indices_in_part"]:
                                            is_active_now = True
                                            break
                                    
                                    if not is_active_now: # Draw only if not active (active ones drawn below more prominently)
                                        cv2.circle(frame, (pt_x, pt_y), radius=1, color=dimmed_color, thickness=-1)
                        # else:
                            # print(f"Warning: Track data for persistent points for part '{label}' not found or inconsistent for frame {t}.")


                    # --- Draw currently active deforming regions (more prominently) ---
                    # frame_regions_summary = all_movement_regions_data["per_frame_regions"][t] # Already got this as current_frame_active_regions_by_part
                    for label in part_labels: # Iterate through labels again to ensure drawing order or use label_idx if preferred
                        part_color = get_color_for_label(label, part_labels)
                        
                        regions_for_part = current_frame_active_regions_by_part.get(label, [])
                        
                        for region in regions_for_part:
                            point_indices_in_part = np.array(region["point_indices_in_part"])
                            centroid_xy = tuple(map(int, region["centroid_xy"]))
                            
                            part_tracks_key = f"{label}_tracks_xy"
                            # Ensure point_indices_in_part is not empty before np.max
                            if stage_b_tracks_data and part_tracks_key in stage_b_tracks_data and \
                               stage_b_tracks_data[part_tracks_key].shape[0] > t and \
                               (point_indices_in_part.size == 0 or np.max(point_indices_in_part) < stage_b_tracks_data[part_tracks_key].shape[1]):
                                
                                if point_indices_in_part.size > 0: # Proceed only if there are points
                                    region_points_xy = stage_b_tracks_data[part_tracks_key][t, point_indices_in_part, :]
                                    for pt_idx in range(region_points_xy.shape[0]):
                                        if np.isnan(region_points_xy[pt_idx, 0]) or np.isnan(region_points_xy[pt_idx, 1]):
                                            continue
                                        pt_x, pt_y = int(region_points_xy[pt_idx, 0]), int(region_points_xy[pt_idx, 1])
                                        cv2.circle(frame, (pt_x, pt_y), radius=3, color=part_color, thickness=-1) # Active points brighter/larger
                                
                                cv2.drawMarker(frame, centroid_xy, color=(255,255,255), markerType=cv2.MARKER_STAR, markerSize=10, thickness=1) # Centroid for active
                                cv2.putText(frame, f"{label[:4]}_r{region['cluster_id']}", (centroid_xy[0]+6, centroid_xy[1]-6), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, part_color, 1, cv2.LINE_AA)
                            else:
                                if point_indices_in_part.size > 0 : # Only print warning if there were points to draw
                                     print(f"Warning: Could not retrieve points for ACTIVE region in part '{label}' at frame {t} for visualization. Track data missing or inconsistent. Max index: {np.max(point_indices_in_part) if point_indices_in_part.size > 0 else 'N/A'}, Track shape[1]: {stage_b_tracks_data[part_tracks_key].shape[1] if stage_b_tracks_data and part_tracks_key in stage_b_tracks_data else 'N/A'}")

                    video_writer.write(frame)
                
                cap.release()
                video_writer.release()
                print(f"  Successfully created visualization video: {output_visualization_path}")

            except Exception as e_vis:
                print(f"Error during visualization: {e_vis}")
                if 'cap' in locals() and cap.isOpened(): cap.release()
                if 'video_writer' in locals() and video_writer.isOpened(): video_writer.release()
        elif not (output_visualization_path and (output_visualization_path.lower().endswith(('.mp4', '.avi', '.mov')))):
            print("  Visualization skipped: Output path not provided or not a recognized video file type.")
        elif not original_video_path:
            print("  Visualization skipped: Original video path could not be determined.")
        elif num_frames == 0:
            print("  Visualization skipped: Number of frames is zero.")
    else:
        print("  Visualization skipped: No valid output_visualization_path provided for video.")
        if output_visualization_path:
             print(f"    (Note: Visualization path '{output_visualization_path}' was provided but not a recognized video format like .mp4, .avi, .mov)")

    print("Stage D finished.")


# Helper function for colors
def get_color_for_label(label, part_labels):
    """Generates a somewhat unique color for a label based on its index."""
    try:
        idx = part_labels.index(label)
    except ValueError:
        idx = -1 # Should not happen if label is from part_labels
    
    # Generate colors from a simple palette
    # BGR format for OpenCV
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),         # Blue, Green, Red
        (255, 255, 0), (255, 0, 255), (0, 255, 255),   # Cyan, Magenta, Yellow
        (128, 0, 0), (0, 128, 0), (0, 0, 128),         # Dark Blue, Dark Green, Dark Red
        (128, 128, 0), (128, 0, 128), (0, 128, 128),   # Teal, Purple, Olive
        (255, 128, 0), (255, 0, 128), (0, 255, 128),   # Orange-ish, Pink-ish, Spring Green-ish
        (128, 255, 0), (0, 128, 255), (128, 0, 255)    # Lime-ish, Sky Blue-ish, Violet-ish
    ]
    return colors[idx % len(colors)]


if __name__ == "__main__":
    print("Running Stage D: Identifying Movement Areas (Example)")

    DUMMY_STAGE_C_ANALYSIS_PATH = "examples/outputs/stage_c_deformation_analysis.npz"
    DUMMY_OUTPUT_MOVEMENT_DATA_PATH_D = "examples/outputs/stage_d_movement_data.npz"
    DUMMY_OUTPUT_VIS_PATH_D = "examples/outputs/stage_d_movement_visualization.mp4" # Changed to .mp4

    if not os.path.exists(DUMMY_STAGE_C_ANALYSIS_PATH):
        print(f"ERROR: Stage C analysis file not found at {DUMMY_STAGE_C_ANALYSIS_PATH}.")
        print("Please run Stage C example first or provide real analysis data.")
        exit()

    print(f"Attempting to identify movement areas from: {DUMMY_STAGE_C_ANALYSIS_PATH}")

    identify_movement_areas(
        stage_c_analysis_path=DUMMY_STAGE_C_ANALYSIS_PATH,
        output_visualization_path=DUMMY_OUTPUT_VIS_PATH_D,
        output_movement_data_path=DUMMY_OUTPUT_MOVEMENT_DATA_PATH_D,
        dilatation_threshold=0.10, # Increased from 0.05
        max_shear_threshold=0.10,  # Increased from 0.05
        temporal_consistency_window=3,
        min_region_points=5,
        dbscan_eps=20.0, 
        dbscan_min_samples=5 # Aligned with min_region_points, was 3
    )

    print("-" * 30)
    print("Stage D example run finished.")
    if DUMMY_OUTPUT_MOVEMENT_DATA_PATH_D and os.path.exists(DUMMY_OUTPUT_MOVEMENT_DATA_PATH_D):
        print(f"Movement data potentially saved to: {DUMMY_OUTPUT_MOVEMENT_DATA_PATH_D}")
        # try:
        #     data = np.load(DUMMY_OUTPUT_MOVEMENT_DATA_PATH_D, allow_pickle=True)
        #     print(f"  Loaded. Keys: {list(data.keys())}")
        #     if 'per_frame_regions' in data:
        #         print(f"  Number of frames processed in D: {len(data['per_frame_regions'])}")
        # except Exception as e:
        #     print(f"  Error loading output from Stage D: {e}")

    if DUMMY_OUTPUT_VIS_PATH_D and os.path.exists(DUMMY_OUTPUT_VIS_PATH_D):
        print(f"Dummy visualization info potentially saved to: {DUMMY_OUTPUT_VIS_PATH_D}")

    # For a real run:
    # identify_movement_areas(
    #     stage_c_analysis_path="examples/outputs/stage_c_deformation_analysis.npz",
    #     output_visualization_path="examples/outputs/stage_d_visualization.mp4", # e.g. a video
    #     output_movement_data_path="examples/outputs/stage_d_movement_regions.npz"
    # ) 