"""
Stage B: Dense Point Cloud Generation & Tracking

Objective: To obtain a dense representation of the animal's visible surface 
           and track how this surface deforms over time.
Potential Tools: CoTracker3, RAFT, PWC-Net, etc.
Output: A time series of 2D or (preferably) 3D coordinates for a dense set 
        of points representing the animal's deforming surface.
"""

import torch
import imageio.v3 as iio
import numpy as np
import os
import cv2
from pathlib import Path

# Attempt to import CoTrackerPredictor
COTRACKER_AVAILABLE = False
try:
    # Assuming CoTracker is installed and predictor can be imported
    # The exact import path might depend on how CoTracker is structured/installed.
    # This matches the structure suggested by some CoTracker examples if it's a local import.
    # If installed as a package, it might be different e.g., `from cotracker.predictor import CoTrackerPredictor`
    from cotracker.predictor import CoTrackerPredictor 
    from cotracker.utils.visualizer import Visualizer, read_video_from_path # Visualizer might be useful later
    COTRACKER_AVAILABLE = True
except ImportError as e:
    print(f"CoTracker library not found or import error: {e}. Please ensure it is installed correctly.")
    # Define a dummy class if not available to prevent runtime errors on import
    class CoTrackerPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("CoTrackerPredictor is not available.")
        def __call__(self, *args, **kwargs):
            raise ImportError("CoTrackerPredictor is not available.")

# --- Configuration ---
# Path to CoTracker checkpoint. This should be set by the user or via an environment variable.
COTRACKER_CHECKPOINT_PATH = os.getenv("COTRACKER_CHECKPOINT_PATH", "checkpoints/scaled_offline.pth")
DEFAULT_POINTS_PER_PART = 256 # Target number of points to sample per mask part

def load_video_frames_for_cotracker_bthwc_float(
    video_path: str
) -> tuple[torch.Tensor | None, list[np.ndarray] | None]:
    """
    Loads all video frames as a B T H W C float tensor (B=1) and a list of raw HWC RGB frames.

    Args:
        video_path (str): Path to the video file.

    Returns:
        tuple[torch.Tensor | None, list[np.ndarray] | None]:
            - video_tensor_bthwc (torch.Tensor | None): The video frames as a B T H W C float tensor.
            - raw_frames_list_hwc_rgb (list[np.ndarray] | None): The list of raw frames for visualization.
    """
    try:
        raw_frames_list_hwc_rgb = []
        # Use imageio.v3 directly for consistent loading
        for frame_data in iio.imiter(video_path, plugin="FFMPEG"):
            if frame_data.ndim == 2: frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
            if frame_data.shape[2] == 4: frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGBA2RGB)
            # CoTracker examples often use uint8 HWC numpy arrays as input to its own video reading utils,
            # which then convert to float tensors. Here we prepare the list of raw frames too.
            raw_frames_list_hwc_rgb.append(frame_data.astype(np.uint8))
        
        if not raw_frames_list_hwc_rgb:
            print(f"No frames loaded from video: {video_path}")
            return None, None

        # Convert list of HWC uint8 to B T H W C float tensor (0-255 range as per some CoTracker utils before normalization)
        video_tensor_bthwc = torch.from_numpy(np.stack(raw_frames_list_hwc_rgb)).unsqueeze(0).float()
        # CoTracker internally handles normalization if needed based on its training.
        # For `predictor(video_tensor_bthwc, ...)` call, it expects B T H W C, channel last.
        # If it expects CHW, then permute(0, 1, 4, 2, 3) would be needed for C H W
        # The CoTracker demo `predict_video` seems to work with HWC directly when building video tensor.
        # Let's ensure it's B T H W C as this is common for video models processing frames directly.
        # If CoTracker's `CoTrackerPredictor` call expects B T C H W, we will need to permute.
        # The provided `read_video_from_path` in CoTracker utils returns B T H W C.
        # Let's assume B T H W C from here, which means channels are last.

        print(f"Video {video_path} loaded into tensor of shape: {video_tensor_bthwc.shape}")
        return video_tensor_bthwc, raw_frames_list_hwc_rgb

    except Exception as e:
        print(f"Error loading video {video_path} for CoTracker: {e}")
        return None, None

def sample_points_from_mask(
    mask_array: np.ndarray, 
    num_points: int = DEFAULT_POINTS_PER_PART, 
    strategy: str = "uniform_random"
) -> np.ndarray:
    """
    Samples (y, x) points from a boolean mask.

    Args:
        mask_array (np.ndarray): The boolean mask array.
        num_points (int): The number of points to sample.
        strategy (str): The sampling strategy.

    Returns:
        np.ndarray: The sampled points as (y, x) coordinates.
    """
    if not mask_array.any():
        return np.empty((0, 2), dtype=int) # No points if mask is empty

    fg_pixels_yx = np.argwhere(mask_array) # Get (row, col) or (y, x) of True pixels
    if fg_pixels_yx.shape[0] == 0:
        return np.empty((0, 2), dtype=int)

    if strategy == "uniform_random":
        if fg_pixels_yx.shape[0] <= num_points:
            return fg_pixels_yx # Return all if less than or equal to num_points
        else:
            indices = np.random.choice(fg_pixels_yx.shape[0], size=num_points, replace=False)
            return fg_pixels_yx[indices]
    # Add other strategies like 'grid_in_bbox' if needed later
    else:
        raise ValueError(f"Unknown point sampling strategy: {strategy}")

def track_multipart_points(
    original_video_path: str,
    initial_masks_path: str, # .npz file from Stage A
    output_tracks_path: str, # .npz file to save all tracks
    cotracker_checkpoint: str = COTRACKER_CHECKPOINT_PATH,
    reference_frame_index: int = 0, # Frame index used for initial_masks
    points_per_part: int = DEFAULT_POINTS_PER_PART,
    point_sampling_strategy: str = "uniform_random"
):
    """
    Tracks points for multiple parts defined by initial masks from Stage A using CoTracker.

    Args:
        original_video_path (str): Path to the original video file.
        initial_masks_path (str): Path to the .npz file from Stage A containing labeled masks.
        output_tracks_path (str): Path to save the output .npz file.
                                  This file will contain tracks, visibility, initial masks,
                                  and initial query points for each labeled part.
        cotracker_checkpoint (str): Path to CoTracker model checkpoint.
        reference_frame_index (int): The frame index from which initial masks were generated.
        points_per_part (int): Target number of points to sample and track for each part.
        point_sampling_strategy (str): Strategy for sampling points from masks ('uniform_random').
    """
    if not COTRACKER_AVAILABLE:
        print("CoTracker is not available. Skipping multi-part point tracking.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CoTracker model
    print(f"Loading CoTracker model from checkpoint: {cotracker_checkpoint}")
    if not os.path.exists(cotracker_checkpoint):
        print(f"ERROR: CoTracker checkpoint not found at {cotracker_checkpoint}. Aborting.")
        return
    try:
        # Ensure checkpoint path is Path object if CoTracker expects it
        cotracker_model = CoTrackerPredictor(checkpoint=Path(cotracker_checkpoint)).to(device)
    except Exception as e:
        print(f"Error loading CoTracker model: {e}. Aborting.")
        return
    print("CoTracker model loaded.")

    # Load video frames
    # CoTracker's default `read_video_from_path` returns B T H W C, float, range 0-1
    # Let's use our loader which gives B T H W C, float, range 0-255, and also raw frames
    # video_tensor_bthwc, _raw_frames = load_video_frames_for_cotracker_bthwc_float(original_video_path)
    # The CoTrackerPredictor call seems to handle video path directly too, let's try that for simplicity
    # as it might do specific preprocessing. If not, we use the tensor loader.
    # For now, we need the video tensor for the main call if providing queries.
    
    # CoTracker's `read_video_from_path` (often used in its demos) returns a BGR tensor, range 0-1.
    # Let's use that utility if available for consistency with CoTracker examples.
    try:
        print(f"Loading video using CoTracker's read_video_from_path for consistency: {original_video_path}")
        video_thwc_np = read_video_from_path(original_video_path) # Assuming T H W C numpy array, range 0-1
        if video_thwc_np is None:
            print(f"Failed to load video: {original_video_path}")
            return
        
        # Convert NumPy array to PyTorch tensor
        video_thwc_tensor = torch.from_numpy(video_thwc_np).float() # T H W C

        # Add batch dimension: B T H W C (B=1)
        video_bthwc_tensor = video_thwc_tensor.unsqueeze(0) # 1 T H W C
        
        # Permute and move to device: B T C H W
        video_for_predictor = video_bthwc_tensor.permute(0, 1, 4, 2, 3).to(device) # 1 T C H W
        print(f"Video tensor prepared for CoTracker predictor: {video_for_predictor.shape}, device: {video_for_predictor.device}")

    except Exception as e:
        print(f"Error loading video with CoTracker utils or preparing tensor: {e}. Aborting.")
        return

    # Load initial masks from Stage A
    print(f"Loading initial masks from: {initial_masks_path}")
    try:
        masks_data = np.load(initial_masks_path)
        labeled_masks = {label: masks_data[label] for label in masks_data.files}
        if not labeled_masks:
            print("No masks found in the .npz file. Aborting.")
            return
        print(f"Loaded {len(labeled_masks)} labeled masks: {list(labeled_masks.keys())}")
    except Exception as e:
        print(f"Error loading masks from {initial_masks_path}: {e}. Aborting.")
        return

    all_tracking_data = {
        "original_video_path": original_video_path,
        "initial_masks_path": initial_masks_path,
        "reference_frame_index": reference_frame_index,
        "cotracker_checkpoint": cotracker_checkpoint,
        "points_per_part_requested": points_per_part
    }

    # Iterate through each part, sample points, and track
    for label, mask_array_hw in labeled_masks.items():
        print(f"\nProcessing part: '{label}'")
        if not isinstance(mask_array_hw, np.ndarray) or mask_array_hw.ndim != 2:
            print(f"  Skipping '{label}': mask is not a 2D numpy array.")
            continue
        if mask_array_hw.dtype != bool:
            mask_array_hw = mask_array_hw.astype(bool) # Ensure boolean

        # 1. Sample initial (y,x) points from the mask for the reference frame
        initial_points_yx = sample_points_from_mask(mask_array_hw, points_per_part, point_sampling_strategy)
        if initial_points_yx.shape[0] == 0:
            print(f"  No points sampled for '{label}' from its mask. Skipping tracking for this part.")
            all_tracking_data[f"{label}_tracks"] = np.empty((video_for_predictor.shape[1], 0, 2), dtype=float)
            all_tracking_data[f"{label}_visibility"] = np.empty((video_for_predictor.shape[1], 0), dtype=bool)
            all_tracking_data[f"{label}_initial_mask"] = mask_array_hw
            all_tracking_data[f"{label}_initial_query_points_yx"] = np.empty((0,2), dtype=int)
            continue
        
        num_sampled_points = initial_points_yx.shape[0]
        print(f"  Sampled {num_sampled_points} points for '{label}'. Coords (y,x) sample: {initial_points_yx[:min(3, num_sampled_points)]}")

        # 2. Prepare queries for CoTracker: (1, N_points, 3) tensor [t, x, y]
        # CoTracker expects (x,y) coordinates for queries.
        initial_points_xy = initial_points_yx[:, ::-1] # Convert (y,x) to (x,y)
        queries_txy_list = []
        for point_xy in initial_points_xy:
            queries_txy_list.append([reference_frame_index, point_xy[0], point_xy[1]])
        
        queries_txy_tensor = torch.tensor(queries_txy_list, dtype=torch.float, device=device).unsqueeze(0) # Batch size 1: (1, N_points, 3)
        print(f"  Prepared queries tensor for CoTracker: {queries_txy_tensor.shape}")

        # 3. Run CoTracker for this part's points
        try:
            with torch.no_grad(): # Inference mode
                # The predictor call expects `video` (B T C H W) and `queries` (B N 3) [t, x, y]
                # Output `tracks` are B T N 2 (x,y) and `visibility` is B T N
                pred_tracks_b_t_n_xy, pred_visibility_b_t_n = cotracker_model(
                    video_for_predictor, 
                    queries=queries_txy_tensor
                )
            
            # Squeeze batch dimension as we process one video
            part_tracks_t_n_xy = pred_tracks_b_t_n_xy.squeeze(0).cpu().numpy()
            part_visibility_t_n = pred_visibility_b_t_n.squeeze(0).cpu().numpy()
            print(f"  Tracking complete for '{label}'. Tracks shape: {part_tracks_t_n_xy.shape}, Visibility shape: {part_visibility_t_n.shape}")

            # Store results for this part
            all_tracking_data[f"{label}_tracks_xy"] = part_tracks_t_n_xy
            all_tracking_data[f"{label}_visibility"] = part_visibility_t_n
            all_tracking_data[f"{label}_initial_mask_hw"] = mask_array_hw # H, W boolean
            all_tracking_data[f"{label}_initial_query_points_yx"] = initial_points_yx # N, 2 (y,x) int

        except Exception as e:
            print(f"  Error during CoTracker prediction for part '{label}': {e}")
            import traceback
            traceback.print_exc()
            # Store empty arrays or indicate failure for this part
            all_tracking_data[f"{label}_tracks_xy"] = np.empty((video_for_predictor.shape[1], num_sampled_points, 2), dtype=float)
            all_tracking_data[f"{label}_visibility"] = np.zeros((video_for_predictor.shape[1], num_sampled_points), dtype=bool)
            all_tracking_data[f"{label}_initial_mask_hw"] = mask_array_hw
            all_tracking_data[f"{label}_initial_query_points_yx"] = initial_points_yx
            all_tracking_data[f"{label}_error"] = str(e)

    # 4. Save all aggregated tracking data
    try:
        output_dir = os.path.dirname(output_tracks_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        np.savez_compressed(output_tracks_path, **all_tracking_data)
        print(f"\nSuccessfully saved all multi-part tracking data to: {output_tracks_path}")
        print(f"  Contains data for parts: {list(labeled_masks.keys())}")
        print(f"  Main data keys per part: '_tracks_xy', '_visibility', '_initial_mask_hw', '_initial_query_points_yx'")

    except Exception as e:
        print(f"Error saving aggregated tracking data to {output_tracks_path}: {e}")


if __name__ == "__main__":
    print("Running Stage B: Multi-Part Point Tracking (Example)")

    # --- Configuration for Example Usage ---
    # This example assumes Stage A has run and produced an initial_masks.npz file.
    # It also needs a video and a CoTracker checkpoint.

    DUMMY_VIDEO_PATH_B = "media/crow.mp4" # Same dummy video as Stage A
    # This is the output from Stage A's example run
    DUMMY_INITIAL_MASKS_PATH_B = "examples/outputs/stage_a_initial_masks.npz" 
    DUMMY_OUTPUT_TRACKS_PATH_B = "examples/outputs/stage_b_multipart_tracks.npz"

    # Ensure dummy video exists (Stage A example should create it)
    if not os.path.exists(DUMMY_VIDEO_PATH_B):
        print(f"ERROR: Dummy video for Stage B not found at {DUMMY_VIDEO_PATH_B}.")
        print("Please run Stage A example first to create it, or provide a real video.")
        if not COTRACKER_AVAILABLE or not os.path.exists(COTRACKER_CHECKPOINT_PATH):
             exit() # Exit if key components are missing and can't even test structure
    
    # Ensure initial masks file exists (Stage A example should create it)
    if not os.path.exists(DUMMY_INITIAL_MASKS_PATH_B):
        print(f"ERROR: Initial masks file for Stage B not found at {DUMMY_INITIAL_MASKS_PATH_B}.")
        print("Please run Stage A example first to create it, or provide real masks.")
        if not COTRACKER_AVAILABLE or not os.path.exists(COTRACKER_CHECKPOINT_PATH):
             exit()

    # Check for CoTracker checkpoint
    cotracker_ckpt_b = COTRACKER_CHECKPOINT_PATH
    if not os.path.exists(cotracker_ckpt_b):
        print("-" * 50)
        print(f"WARNING: CoTracker checkpoint not found at: {cotracker_ckpt_b}")
        print("The script will fail during CoTracker model initialization if this is not correct.")
        print("Please download CoTracker models and update COTRACKER_CHECKPOINT_PATH environment variable or argument.")
        print("Skipping example run if CoTracker components are missing.")
        print("-" * 50)
        if not (os.path.exists(DUMMY_VIDEO_PATH_B) and os.path.exists(DUMMY_INITIAL_MASKS_PATH_B)):
            exit() # Can't run without inputs and model

    print(f"Attempting to track points for parts defined in: {DUMMY_INITIAL_MASKS_PATH_B}")
    print(f"Using video: {DUMMY_VIDEO_PATH_B}")
    print(f"Saving tracks to: {DUMMY_OUTPUT_TRACKS_PATH_B}")

    # Only run if primary components seem available, even if models might fail later
    if COTRACKER_AVAILABLE and os.path.exists(DUMMY_VIDEO_PATH_B) and os.path.exists(DUMMY_INITIAL_MASKS_PATH_B) and os.path.exists(cotracker_ckpt_b):
        try:
            track_multipart_points(
                original_video_path=DUMMY_VIDEO_PATH_B,
                initial_masks_path=DUMMY_INITIAL_MASKS_PATH_B,
                output_tracks_path=DUMMY_OUTPUT_TRACKS_PATH_B,
                cotracker_checkpoint=cotracker_ckpt_b,
                reference_frame_index=0, # Assuming masks from Stage A example were for frame 0
                points_per_part=64 # Keep it low for dummy example
            )
            print("-" * 30)
            print("Stage B example run finished.")
            if os.path.exists(DUMMY_OUTPUT_TRACKS_PATH_B):
                print(f"Output potentially saved to {DUMMY_OUTPUT_TRACKS_PATH_B}")
                # Verify contents (optional)
                try:
                    loaded_tracks_data = np.load(DUMMY_OUTPUT_TRACKS_PATH_B, allow_pickle=True)
                    print(f"  Successfully loaded. Contains keys: {list(loaded_tracks_data.keys())}")
                    # Example: Check for a specific part's tracks if masks were generated by Stage A example
                    if 'bird head_tracks_xy' in loaded_tracks_data:
                        print(f"    'bird head_tracks_xy' shape: {loaded_tracks_data['bird head_tracks_xy'].shape}")
                    elif loaded_tracks_data.files:
                        # Check the first actual data key if 'bird head' isn't there (e.g. if prompts changed)
                        first_data_key = [k for k in loaded_tracks_data.files if k.endswith("_tracks_xy")]
                        if first_data_key:
                             print(f"    Sample data '{first_data_key[0]}' shape: {loaded_tracks_data[first_data_key[0]].shape}")
                except Exception as e:
                    print(f"  Error loading or inspecting Stage B output file: {e}")
            else:
                print(f"  Output file {DUMMY_OUTPUT_TRACKS_PATH_B} was NOT created. Check logs for errors.")

        except ImportError as ie:
            print(f"An ImportError occurred: {ie}. This often means a key library (like CoTracker) is not installed correctly.")
        except Exception as e:
            print(f"An error occurred during the Stage B example run: {e}")
            import traceback
            traceback.print_exc()
            print("This might be due to issues with model loading, video processing, or CoTracker execution.")
    else:
        print("Skipping Stage B example execution due to missing components (CoTracker lib, video, masks, or checkpoint).")
        print(f"  CoTracker Available: {COTRACKER_AVAILABLE}")
        print(f"  Video Exists ({DUMMY_VIDEO_PATH_B}): {os.path.exists(DUMMY_VIDEO_PATH_B)}")
        print(f"  Masks Exist ({DUMMY_INITIAL_MASKS_PATH_B}): {os.path.exists(DUMMY_INITIAL_MASKS_PATH_B)}")
        print(f"  CoTracker Ckpt Exists ({cotracker_ckpt_b}): {os.path.exists(cotracker_ckpt_b)}")

    # For a real run:
    # track_multipart_points(
    #     original_video_path="path/to/your/video.mp4",
    #     initial_masks_path="output/stage_a_masks.npz",
    #     output_tracks_path="output/stage_b_multitracks.npz",
    #     # reference_frame_index defaults to 0
    #     # points_per_part defaults to 256
    # )

    # To clean up dummy tracks (optional):
    # if os.path.exists(DUMMY_TRACKS_OUTPUT_PATH):
    #     os.remove(DUMMY_TRACKS_OUTPUT_PATH)
    # output_dir = os.path.dirname(DUMMY_TRACKS_OUTPUT_PATH)
    # if output_dir and os.path.exists(output_dir) and not os.listdir(output_dir):
    #     os.rmdir(output_dir) 