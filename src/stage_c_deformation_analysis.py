"""
Stage C: Deformation Analysis

Objective: To quantify the local and regional deformation (contraction, 
           expansion, shearing, twisting) of the animal's surface based on 
           the motion of the dense point cloud from Stage B, and to analyze
           relative motion between labeled parts.
Methods: Neighborhood definition, relative motion analysis, Jacobian, strain tensor.
Output: A per-frame (or temporally smoothed) deformation map for each part,
        and metrics for inter-part relative motion.
"""
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

# --- Configuration ---
DEFAULT_KNN_K = 10 # K for K-Nearest Neighbors
DEFAULT_VISIBILITY_THRESHOLD = 0.5 # Minimum visibility to consider a point valid

def calculate_deformation_gradient(ref_points_patch, current_points_patch):
    """
    Calculates the deformation gradient F for a patch of points.
    ref_points_patch: (N, 2) array of points in reference configuration (centroid subtracted).
    current_points_patch: (N, 2) array of points in current configuration (centroid subtracted).
    Returns: 2x2 deformation gradient matrix F.
    """
    if ref_points_patch.shape[0] < 2 or current_points_patch.shape[0] < 2:
        # Not enough points to define a plane or robustly calculate F
        return np.eye(2) # Return identity if patch is too small

    # Covariance matrix H = sum_{alpha=1 to N} (p_alpha_ref) (p_alpha_current)^T
    # We are solving F * P_ref_centered.T = P_curr_centered.T
    # F = P_curr_centered.T @ P_ref_centered @ (P_ref_centered.T @ P_ref_centered)^-1
    # F = (P_curr_centered.T @ P_ref_centered) @ inv(P_ref_centered.T @ P_ref_centered)
    
    # Let A = ref_points_patch.T, B = current_points_patch.T
    # We want to find F such that F A = B.  F = B A_pinv
    # Using least squares: F = (B @ A.T) @ inv(A @ A.T) for A: 2xN, B: 2xN
    
    A = ref_points_patch.T # Shape (2, N)
    B = current_points_patch.T # Shape (2, N)
    
    # Numerator: B @ A.T
    numerator = B @ A.T # Shape (2, 2)
    
    # Denominator: A @ A.T
    denominator = A @ A.T # Shape (2, 2)
    
    try:
        F = numerator @ np.linalg.inv(denominator)
    except np.linalg.LinAlgError:
        # If denominator is singular (e.g., points are collinear in ref frame)
        # Fallback to pseudo-inverse of the denominator or return identity
        # print(f"Warning: Singular matrix in deformation gradient calculation for patch. Using pseudo-inverse.")
        try:
            F = numerator @ np.linalg.pinv(denominator)
        except np.linalg.LinAlgError:
            # print(f"Warning: Pseudo-inverse also failed. Returning identity.")
            return np.eye(2) # Catastrophic failure, return identity
    return F

def analyze_intra_part_deformation(
    part_tracks_xy, # (T, N_points, 2)
    part_visibility, # (T, N_points)
    part_initial_query_points_yx, # (N_points, 2) - original yx for reference frame
    reference_frame_index: int,
    knn_k: int = DEFAULT_KNN_K,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD
):
    """
    Analyzes intra-part deformation for a single tracked part.

    Returns a dictionary of deformation metrics per point per frame.
    Example metric: Green-Lagrangian strain tensor E (T, N_points, 2, 2)
                    Dilatation (T, N_points)
    """
    num_frames, num_points, _ = part_tracks_xy.shape
    if num_points == 0:
        return {
            "green_lagrangian_strain_tensors_fpt22": np.empty((num_frames, 0, 2, 2)),
            "dilatation_fp": np.empty((num_frames, 0)),
            "principal_strains_fpt2": np.empty((num_frames, 0, 2)),
            "principal_directions_fpt22": np.empty((num_frames, 0, 2, 2)),
            "max_shear_strains_fp": np.empty((num_frames, 0))
        }

    # Initialize output arrays
    # Green-Lagrangian strain tensor E = 0.5 * (F.T @ F - I)
    green_lagrangian_strain_tensors = np.zeros((num_frames, num_points, 2, 2))
    dilatation = np.zeros((num_frames, num_points))
    principal_strains = np.zeros((num_frames, num_points, 2)) # For lambda_1, lambda_2
    # Store principal directions as 2x2 matrices where columns are eigenvectors
    principal_directions = np.zeros((num_frames, num_points, 2, 2)) 
    max_shear_strains = np.zeros((num_frames, num_points))

    # Reference point positions (at reference_frame_index)
    # Ensure initial query points are xy for consistency with tracks
    ref_points_xy_all = part_tracks_xy[reference_frame_index, :, :]
    ref_visibility_all = part_visibility[reference_frame_index, :]

    # Pre-calculate KNN for the reference frame if points don't change order (they do here)
    # So, KNN must be done per frame based on current visible points.

    for t in range(num_frames):
        current_points_xy_frame = part_tracks_xy[t, :, :]
        current_visibility_frame = part_visibility[t, :] 
        
        # Consider only points currently visible and also visible in reference frame for stability
        # This is a strict criterion. Alternatively, use only current visibility for KNN for current patch,
        # and only ref visibility for KNN for ref patch. For F calculation, need correspondence.
        valid_points_for_frame_indices = np.where(
            (current_visibility_frame >= visibility_threshold) &
            (ref_visibility_all >= visibility_threshold) # Ensure reference points were also good
        )[0]

        if len(valid_points_for_frame_indices) <= knn_k: # Need at least k+1 points for KNN
            # print(f"Frame {t}: Not enough valid points ({len(valid_points_for_frame_indices)}) for KNN. Skipping deformation calc.")
            # Leave metrics as zeros for this frame or fill with NaN
            green_lagrangian_strain_tensors[t, :, :, :] = np.nan
            dilatation[t, :] = np.nan
            principal_strains[t, :, :] = np.nan
            principal_directions[t, :, :, :] = np.nan
            max_shear_strains[t, :] = np.nan
            continue

        valid_current_points_xy = current_points_xy_frame[valid_points_for_frame_indices, :]
        valid_ref_points_xy = ref_points_xy_all[valid_points_for_frame_indices, :] # Corresponding points
        
        # KNN on the *currently valid points* for this frame
        # Note: n_neighbors includes the point itself, so ask for knn_k + 1
        if valid_current_points_xy.shape[0] <= knn_k:
            green_lagrangian_strain_tensors[t, :, :, :] = np.nan
            dilatation[t, :] = np.nan
            principal_strains[t, :, :] = np.nan
            principal_directions[t, :, :, :] = np.nan
            max_shear_strains[t, :] = np.nan
            continue # Not enough points to form neighborhoods
            
        nbrs = NearestNeighbors(n_neighbors=min(knn_k + 1, valid_current_points_xy.shape[0]), algorithm='ball_tree').fit(valid_current_points_xy)
        distances, indices_in_valid = nbrs.kneighbors(valid_current_points_xy)

        # Iterate through each valid point to calculate its deformation
        for i, point_original_idx in enumerate(valid_points_for_frame_indices):
            neighbor_indices_in_valid = indices_in_valid[i, :] # Indices relative to valid_current_points_xy
            
            # Get actual patch coordinates (these are already centered if we work with differences from centroid)
            ref_patch = valid_ref_points_xy[neighbor_indices_in_valid, :]
            current_patch = valid_current_points_xy[neighbor_indices_in_valid, :]
            
            # Centroids
            ref_centroid = np.mean(ref_patch, axis=0)
            current_centroid = np.mean(current_patch, axis=0)
            
            # Centered patches
            ref_patch_centered = ref_patch - ref_centroid
            current_patch_centered = current_patch - current_centroid
            
            if ref_patch_centered.shape[0] < 2 : # Need at least 2 points for F
                 F = np.eye(2)
            else:
                F = calculate_deformation_gradient(ref_patch_centered, current_patch_centered)
            
            # Green-Lagrangian Strain E = 0.5 * (F.T @ F - I)
            E_tensor = 0.5 * (F.T @ F - np.eye(2))
            green_lagrangian_strain_tensors[t, point_original_idx, :, :] = E_tensor
            
            # Dilatation (Area change J-1, where J = det(F))
            # J = det(F) gives ratio of current area to reference area for the patch.
            # Dilatation = J - 1
            dil = np.linalg.det(F) - 1.0
            dilatation[t, point_original_idx] = dil
            
            # TODO: Calculate principal strains (eigenvalues of E) and other metrics here.
            # Principal Strains and Directions
            try:
                # Eigenvalue decomposition of the Green-Lagrangian strain tensor E
                # For a 2x2 symmetric matrix, eigh returns eigenvalues in ascending order
                # and corresponding eigenvectors as columns in a matrix.
                eigenvalues, eigenvectors = np.linalg.eigh(E_tensor)
                principal_strains[t, point_original_idx, :] = eigenvalues # [lambda_min, lambda_max]
                principal_directions[t, point_original_idx, :, :] = eigenvectors
                
                # Max Shear Strain: abs(lambda_max - lambda_min) for 2D
                # (Some definitions use (lambda_max - lambda_min)/2)
                max_shear_strains[t, point_original_idx] = np.abs(eigenvalues[1] - eigenvalues[0])

            except np.linalg.LinAlgError:
                # print(f"Warning: Eigenvalue decomposition failed for point {point_original_idx} at frame {t}. Setting NaNs.")
                principal_strains[t, point_original_idx, :] = np.nan
                principal_directions[t, point_original_idx, :, :] = np.nan
                max_shear_strains[t, point_original_idx] = np.nan

    return {
        "green_lagrangian_strain_tensors_fpt22": green_lagrangian_strain_tensors, # F=frame, P=point, T=tensor(2x2)
        "dilatation_fp": dilatation, # F=frame, P=point
        "principal_strains_fpt2": principal_strains, # F=frame, P=point, 2=strains
        "principal_directions_fpt22": principal_directions, # F=frame, P=point, Eigenvectors as columns
        "max_shear_strains_fp": max_shear_strains # F=frame, P=point
    }

def analyze_inter_part_relative_motion(all_tracking_data, reference_frame_index):
    """
    Analyzes relative motion between different tracked parts.
    Calculates centroid trajectories and relative displacements.
    """
    part_labels = [key.replace("_tracks_xy", "") for key in all_tracking_data.keys() if key.endswith("_tracks_xy")]
    if not part_labels:
        return {}

    num_frames = 0
    # Determine number of frames from the first part found
    for label in part_labels:
        if f"{label}_tracks_xy" in all_tracking_data:
            num_frames = all_tracking_data[f"{label}_tracks_xy"].shape[0]
            break
    if num_frames == 0:
        return {}

    centroid_trajectories = {}
    for label in part_labels:
        tracks_xy = all_tracking_data.get(f"{label}_tracks_xy")
        visibility = all_tracking_data.get(f"{label}_visibility")
        
        if tracks_xy is None or visibility is None:
            print(f"Warning: Missing tracks or visibility for part '{label}' in inter-part analysis.")
            continue
            
        current_part_centroids = np.full((num_frames, 2), np.nan) # T x 2 (x,y)
        for t in range(num_frames):
            visible_points_indices = np.where(visibility[t, :] >= DEFAULT_VISIBILITY_THRESHOLD)[0]
            if len(visible_points_indices) > 0:
                visible_points_coords = tracks_xy[t, visible_points_indices, :]
                current_part_centroids[t, :] = np.mean(visible_points_coords, axis=0)
        centroid_trajectories[f"{label}_centroid_xy_t2"] = current_part_centroids

    # TODO: Calculate relative displacements between centroids of key part pairs
    # TODO: Estimate relative rotations/orientations if possible

    analysis_results = {}
    analysis_results.update(centroid_trajectories)
    return analysis_results


def analyze_deformation_and_relative_motion(
    multipart_tracks_path: str,  # .npz file from Stage B
    output_analysis_path: str,   # New .npz file to store Stage C results
    knn_k: int = DEFAULT_KNN_K,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD
):
    """
    Main function for Stage C. Loads tracked points for multiple parts, 
    analyzes intra-part deformation and inter-part relative motion.
    """
    print(f"Starting Stage C: Deformation and Relative Motion Analysis")
    print(f"  Loading multi-part tracks from: {multipart_tracks_path}")
    try:
        all_tracking_data_loaded = np.load(multipart_tracks_path, allow_pickle=True)
        # Convert to a standard dict for easier manipulation if it's a NpzFile object
        all_tracking_data = {key: all_tracking_data_loaded[key] for key in all_tracking_data_loaded.files}
    except Exception as e:
        print(f"Error loading tracks from {multipart_tracks_path}: {e}. Aborting Stage C.")
        return

    print(f"  Found data for keys: {list(all_tracking_data.keys())}")

    reference_frame_index = all_tracking_data.get("reference_frame_index", np.array(0)).item() # .item() if it's a 0-d array
    if not isinstance(reference_frame_index, int):
        try: reference_frame_index = int(reference_frame_index)
        except: reference_frame_index = 0; print("Warning: could not parse reference_frame_index, defaulting to 0.")

    print(f"  Using reference frame index: {reference_frame_index}")

    stage_c_results = {
        "source_multipart_tracks_path": multipart_tracks_path,
        "reference_frame_index": reference_frame_index,
        "knn_k_for_intra_part": knn_k,
        "visibility_threshold_for_intra_part": visibility_threshold
    }

    # 1. Intra-Part Deformation Analysis
    part_labels_found = sorted(list(set([k.split('_tracks_xy')[0] for k in all_tracking_data if k.endswith('_tracks_xy')])))
    print(f"\nPerforming Intra-Part Deformation Analysis for parts: {part_labels_found}")
    for label in part_labels_found:
        print(f"  Analyzing part: '{label}'")
        part_tracks_key = f"{label}_tracks_xy"
        part_visibility_key = f"{label}_visibility"
        part_initial_points_key = f"{label}_initial_query_points_yx" # From Stage B

        if not (part_tracks_key in all_tracking_data and 
                  part_visibility_key in all_tracking_data and
                  part_initial_points_key in all_tracking_data):
            print(f"    Skipping '{label}': Missing required data (tracks, visibility, or initial_query_points).")
            continue

        part_tracks_xy = all_tracking_data[part_tracks_key]
        part_visibility = all_tracking_data[part_visibility_key]
        part_initial_query_points_yx = all_tracking_data[part_initial_points_key]

        if part_tracks_xy.ndim != 3 or part_tracks_xy.shape[2] != 2:
            print(f"    Skipping '{label}': Tracks data has unexpected shape {part_tracks_xy.shape}.")
            continue
        if part_tracks_xy.shape[1] == 0: # No points tracked for this part
            print(f"    Skipping '{label}': No points were tracked for this part.")
            stage_c_results[f"{label}_intra_deformation"] = { # Store empty/default metrics
                 "green_lagrangian_strain_tensors_fpt22": np.empty((part_tracks_xy.shape[0], 0, 2, 2)),
                 "dilatation_fp": np.empty((part_tracks_xy.shape[0], 0)),
                 "principal_strains_fpt2": np.empty((part_tracks_xy.shape[0], 0, 2)),
                 "principal_directions_fpt22": np.empty((part_tracks_xy.shape[0], 0, 2, 2)),
                 "max_shear_strains_fp": np.empty((part_tracks_xy.shape[0], 0))
            }
            continue

        deformation_metrics = analyze_intra_part_deformation(
            part_tracks_xy, part_visibility, 
            part_initial_query_points_yx,
            reference_frame_index,
            knn_k=knn_k, 
            visibility_threshold=visibility_threshold
        )
        stage_c_results[f"{label}_intra_deformation"] = deformation_metrics
        print(f"    Finished deformation analysis for '{label}'.")

    # 2. Inter-Part Relative Motion Analysis
    print(f"\nPerforming Inter-Part Relative Motion Analysis...")
    relative_motion_results = analyze_inter_part_relative_motion(all_tracking_data, reference_frame_index)
    stage_c_results.update(relative_motion_results) # Add centroid trajectories etc.
    print(f"  Finished relative motion analysis.")

    # Save Stage C results
    try:
        output_dir = os.path.dirname(output_analysis_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory for Stage C results: {output_dir}")
        
        np.savez_compressed(output_analysis_path, **stage_c_results)
        print(f"\nSuccessfully saved Stage C analysis results to: {output_analysis_path}")
        print(f"  Contains keys: {list(stage_c_results.keys())}")

    except Exception as e:
        print(f"Error saving Stage C analysis results to {output_analysis_path}: {e}")

if __name__ == "__main__":
    print("Running Stage C: Deformation and Relative Motion Analysis (Example)")

    # This example assumes Stage B has run and produced a multipart_tracks.npz file.
    DUMMY_MULTIPART_TRACKS_PATH_C = "examples/outputs/stage_b_multipart_tracks.npz"
    DUMMY_OUTPUT_ANALYSIS_PATH_C = "examples/outputs/stage_c_deformation_analysis.npz"

    # Ensure Stage B output exists
    if not os.path.exists(DUMMY_MULTIPART_TRACKS_PATH_C):
        print(f"ERROR: Multi-part tracks file for Stage C not found at {DUMMY_MULTIPART_TRACKS_PATH_C}.")
        print("Please run Stage B example first to create it, or provide real tracks data.")
        exit()

    print(f"Attempting to analyze tracks from: {DUMMY_MULTIPART_TRACKS_PATH_C}")
    print(f"Saving analysis to: {DUMMY_OUTPUT_ANALYSIS_PATH_C}")

    try:
        analyze_deformation_and_relative_motion(
            multipart_tracks_path=DUMMY_MULTIPART_TRACKS_PATH_C,
            output_analysis_path=DUMMY_OUTPUT_ANALYSIS_PATH_C,
            knn_k=5, # Smaller K for potentially sparse dummy data
            visibility_threshold=0.1 # Lower threshold for dummy data
        )
        print("-" * 30)
        print("Stage C example run finished.")
        if os.path.exists(DUMMY_OUTPUT_ANALYSIS_PATH_C):
            print(f"Output potentially saved to {DUMMY_OUTPUT_ANALYSIS_PATH_C}")
            # Verify contents (optional)
            try:
                loaded_analysis_data = np.load(DUMMY_OUTPUT_ANALYSIS_PATH_C, allow_pickle=True)
                print(f"  Successfully loaded Stage C results. Contains keys: {list(loaded_analysis_data.keys())}")
                # Example: Check for a specific part's deformation if available
                key_to_check = None
                for k in loaded_analysis_data.keys():
                    if k.endswith("_intra_deformation"):
                        key_to_check = k
                        break
                if key_to_check:
                    part_analysis = loaded_analysis_data[key_to_check].item() # it's a dict
                    if 'dilatation_fp' in part_analysis:
                         print(f"    Sample analysis for '{key_to_check}': dilatation_fp shape {part_analysis['dilatation_fp'].shape}")
                    if 'principal_strains_fpt2' in part_analysis:
                         print(f"    Sample analysis for '{key_to_check}': principal_strains_fpt2 shape {part_analysis['principal_strains_fpt2'].shape}")
            except Exception as e:
                print(f"  Error loading or inspecting Stage C output file: {e}")
        else:
            print(f"  Output file {DUMMY_OUTPUT_ANALYSIS_PATH_C} was NOT created. Check logs for errors.")

    except Exception as e:
        print(f"An error occurred during the Stage C example run: {e}")
        import traceback
        traceback.print_exc()
        print("This might be due to issues with data loading, numerical stability, or calculations.")

    # For a real run:
    # analyze_deformation_and_relative_motion(
    #     multipart_tracks_path="examples/outputs/stage_b_multipart_tracks.npz",
    #     output_analysis_path="examples/outputs/stage_c_deformation_analysis.npz"
    # ) 