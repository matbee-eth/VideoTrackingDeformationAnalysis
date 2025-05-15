import torch
import imageio.v3 as iio
import numpy as np
import os
import cv2 # For image processing if needed
from PIL import Image
import hydra
from sam2.build_sam import build_sam2
from omegaconf import OmegaConf

# Attempt to import SAM2 and Florence-2 components
SAM2_AVAILABLE = False
FLORENCE2_AVAILABLE = False

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("SAM2 library not found. Please ensure it is installed correctly (e.g., from git).")

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    FLORENCE2_AVAILABLE = True
except ImportError:
    print("Transformers library not found. Please install it for Florence-2.")

# hydra is initialized on import of sam2, which sets the search path which can't be modified
# so we need to clear the hydra instance
# hydra.core.global_hydra.GlobalHydra.instance().clear()
# reinit hydra with a new search path for configs
# hydra.initialize_config_module('/home/acidhax/dev/VideoTrackingDeformationAnalysis/configs/sam2.1', version_base='1.2')
from omegaconf import DictConfig
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

def build_sam2_(
    config_file,  # Can be a config name (str) or a DictConfig
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs
):
    # Allow passing an actual config object
    if isinstance(config_file, DictConfig):
        cfg = config_file
    else:
        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            ]

        cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)

    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    if mode == "train":
        model.train()
    else:
        model.eval()

    return model


# --- Model Configuration ---
# These should ideally be configurable or passed as arguments
FLORENCE2_MODEL_ID = "microsoft/Florence-2-base"
# Make sure SAM2_CHECKPOINT and SAM2_CONFIG paths are correct for your setup
# For now, assuming they might be in a 'checkpoints' or 'configs' directory relative to the project root
# or that the SAM2 library handles finding them if installed system-wide.
# These paths will likely need adjustment based on actual installation.
SAM2_CHECKPOINT_PATH = os.getenv("SAM2_CHECKPOINT_PATH", "checkpoints/sam2.1_hiera_large.pt") # Or your actual path
SAM2_CONFIG_PATH = "sam2.1_hiera_l.yaml" # Or your actual path

# Task prompt for Florence-2. Open Vocabulary Detection seems most appropriate for "find 'object'"
FLORENCE2_TASK_PROMPT = "<OPEN_VOCABULARY_DETECTION>"

# Helper for Stage A visualization
def get_color_for_mask_label(label_index, num_labels, s=0.8, v=0.9):
    # Use HSV color space for more distinct colors
    # Calculate hue in 0-360 range, then map to 0-179 for OpenCV uint8 HSV
    hue_360 = label_index * (360.0 / num_labels)
    hue_180 = int(hue_360 % 180) # Scale to 0-179 range for OpenCV uint8
    
    saturation = int(s * 255)
    value = int(v * 255)
    
    color_hsv = np.array([[[hue_180, saturation, value]]], dtype=np.uint8)
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])) # BGR tuple

def visualize_and_save_masks_on_frame(
    frame_np_hwc_rgb: np.ndarray,
    labeled_masks: dict, # {label_str: mask_bool_array_hw}
    output_image_path: str,
    alpha: float = 0.4 # Transparency of masks
):
    """Overlays masks on the frame and saves the image."""
    if not labeled_masks:
        print("No masks to visualize.")
        return

    vis_image_bgr = cv2.cvtColor(frame_np_hwc_rgb, cv2.COLOR_RGB2BGR)
    overlay = vis_image_bgr.copy()

    num_actual_masks = len(labeled_masks)
    mask_labels = list(labeled_masks.keys())

    for i, label in enumerate(mask_labels):
        mask_hw = labeled_masks[label]
        if not mask_hw.any(): # Skip empty masks
            continue
        
        color_bgr = get_color_for_mask_label(i, num_actual_masks)
        
        # Apply color to mask region on the overlay
        overlay[mask_hw] = color_bgr

        # Add text label near mask centroid
        try:
            moments = cv2.moments(mask_hw.astype(np.uint8))
            if moments["m00"] > 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
                cv2.putText(vis_image_bgr, label, (center_x, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA) # White text with black outline (effectively)
                cv2.putText(vis_image_bgr, label, (center_x, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)
        except Exception as e_centroid:
            print(f"    Could not calculate centroid or place text for label '{label}': {e_centroid}")

    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, vis_image_bgr, 1 - alpha, 0, vis_image_bgr)
    
    try:
        output_dir = os.path.dirname(output_image_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_image_path, vis_image_bgr)
        print(f"Saved Stage A mask visualization to: {output_image_path}")
    except Exception as e_save:
        print(f"Error saving Stage A mask visualization: {e_save}")

def load_reference_frame(video_path: str, frame_index: int = 0) -> np.ndarray | None:
    """Loads a specific frame from a video path."""
    try:
        frames = []
        for i, frame_data in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
            if i == frame_index:
                # Ensure frame is HWC, RGB
                if frame_data.ndim == 2: frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
                if frame_data.shape[2] == 4: frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGBA2RGB)
                if frame_data.dtype != np.uint8: frame_data = frame_data.astype(np.uint8)
                frames.append(frame_data)
                break
        if frames:
            print(f"Loaded frame {frame_index} from {video_path}")
            return frames[0]
        else:
            print(f"Frame {frame_index} not found or video is shorter.")
            return None
    except Exception as e:
        print(f"Error loading frame {frame_index} from video {video_path}: {e}")
        return None

def initialize_models(device_str: str):
    """Initializes and returns Florence-2 and SAM2 models and predictors."""
    if not SAM2_AVAILABLE or not FLORENCE2_AVAILABLE:
        raise RuntimeError("SAM2 or Florence-2 libraries are not available. Please check installation.")

    # Initialize Florence-2
    print(f"Initializing Florence-2 model: {FLORENCE2_MODEL_ID} on device: {device_str}")
    florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device_str)
    florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)
    print("Florence-2 initialized.")

    # Initialize SAM2
    print(f"Initializing SAM2 model from checkpoint: {SAM2_CHECKPOINT_PATH} and config: {SAM2_CONFIG_PATH} on device: {device_str}")
    if not os.path.exists(SAM2_CHECKPOINT_PATH):
        raise FileNotFoundError(f"SAM2 Checkpoint not found at {SAM2_CHECKPOINT_PATH}. Please set SAM2_CHECKPOINT_PATH or place it correctly.")
    if not os.path.exists(SAM2_CONFIG_PATH):
         raise FileNotFoundError(f"SAM2 Config not found at {SAM2_CONFIG_PATH}. Please set SAM2_CONFIG_PATH or place it correctly.")
    # Load the YAML config manually from file
    yaml_path = os.path.abspath("/home/acidhax/dev/VideoTrackingDeformationAnalysis/configs/sam2.1/sam2.1_hiera_l.yaml")
    cfg = OmegaConf.load(yaml_path)
    sam2_checkpoint = "/home/acidhax/dev/VideoTrackingDeformationAnalysis/checkpoints/sam2.1_hiera_large.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2_model_torch = build_sam2_(cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model_torch)
    print("SAM2ImagePredictor initialized.")

    return florence2_model, florence2_processor, sam2_predictor

def run_florence2_od(
    model, processor, image_pil, text_input, task_prompt=FLORENCE2_TASK_PROMPT
):
    """Runs Florence-2 for object detection based on text input."""
    device = model.device
    prompt = task_prompt + text_input # Florence-2 expects the prompt this way for OD

    inputs = processor(text=prompt, images=image_pil, return_tensors="pt").to(device)
    if hasattr(inputs, 'pixel_values') and inputs.pixel_values.dtype == torch.float32 and device.type == 'cuda':
        inputs.pixel_values = inputs.pixel_values.to(torch.bfloat16) # Or float16 if bfloat16 not supported

    # Debug: Print input IDs and pixel values shapes
    print(f"    [F2 Debug] input_ids shape: {inputs['input_ids'].shape}")
    print(f"    [F2 Debug] pixel_values shape: {inputs['pixel_values'].shape}, dtype: {inputs['pixel_values'].dtype}")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(f"    [F2 Debug] Generated text (raw): {generated_text}") # DEBUG

    # The post_process_generation for OD returns a dict like: {'<OD>': {'bboxes': [[x1,y1,x2,y2], ...], 'labels': ["label1", ...]}}
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image_pil.width, image_pil.height)
    )
    print(f"    [F2 Debug] Parsed answer from post_process_generation: {parsed_answer}") # DEBUG
    return parsed_answer


def generate_initial_multipart_masks(
    video_path: str,
    text_prompts: list[str],
    output_masks_path: str,
    reference_frame_index: int = 0,
    sam_checkpoint: str = SAM2_CHECKPOINT_PATH, # Allow override
    sam_config: str = SAM2_CONFIG_PATH, # Allow override
    florence_model_id: str = FLORENCE2_MODEL_ID, # Allow override
    visual_preview: bool = True # User changed default to True
):
    """
    Generates initial segmentation masks for multiple parts based on text prompts
    from a reference frame in a video.

    Args:
        video_path (str): Path to the input video file.
        text_prompts (list[str]): A list of text descriptions for parts to segment
                                  (e.g., ["head", "left leg", "tail"]).
        output_masks_path (str): Path to save the output .npz file containing labeled masks.
                                 The .npz file will store a dictionary where keys are prompts
                                 and values are boolean mask arrays.
        reference_frame_index (int): Index of the frame in the video to use for segmentation.
        sam_checkpoint (str): Path to SAM2 model checkpoint.
        sam_config (str): Path to SAM2 model config.
        florence_model_id (str): HuggingFace model ID for Florence-2.
        visual_preview (bool): Whether to display a visual preview of the reference frame.
    """
    if not SAM2_AVAILABLE or not FLORENCE2_AVAILABLE:
        print("Error: SAM2 or Florence-2 dependencies are not met. Cannot proceed.")
        return

    # Update global paths if overridden
    global SAM2_CHECKPOINT_PATH, SAM2_CONFIG_PATH, FLORENCE2_MODEL_ID
    SAM2_CHECKPOINT_PATH = sam_checkpoint
    SAM2_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml" # Keep specific path for now
    FLORENCE2_MODEL_ID = florence_model_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set up torch environment settings for CUDA if available - outside autocast context
    if device == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8: # Ampere+
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _generate_masks_logic():
        frame_np_hwc_rgb = load_reference_frame(video_path, reference_frame_index)
        if frame_np_hwc_rgb is None:
            print(f"Could not load reference frame {reference_frame_index} from {video_path}. Aborting.")
            return

        # --- VISUAL PREVIEW ---
        if visual_preview:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,6))
            plt.imshow(frame_np_hwc_rgb)
            plt.title(f"Reference Frame {reference_frame_index}")
            plt.axis('off')
            plt.show()
        # --- END VISUAL PREVIEW ---

        image_pil = Image.fromarray(frame_np_hwc_rgb)

        try:
            florence2_model, florence2_processor, sam2_predictor = initialize_models(device)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Error initializing models: {e}. Aborting.")
            return

        labeled_masks = {}
        sam2_predictor.set_image(frame_np_hwc_rgb) # Set image once for SAM2

        print(f"Processing {len(text_prompts)} text prompts: {text_prompts}")
        for text_prompt in text_prompts:
            print(f"  Processing prompt: '{text_prompt}'")
            try:
                # Run Florence-2 to get bounding boxes for the current text prompt
                florence_results = run_florence2_od(
                    florence2_model, florence2_processor, image_pil, text_input=text_prompt
                )
                print(f"    [Main Debug] Full Florence-2 results for '{text_prompt}': {florence_results}") # DEBUG
                
                detections = florence_results.get(FLORENCE2_TASK_PROMPT, {})
                bboxes_all = detections.get("bboxes", [])
                labels_all = detections.get("bboxes_labels", [])
                print(f"    [Main Debug] Extracted bboxes_all: {bboxes_all}") # DEBUG
                print(f"    [Main Debug] Extracted labels_all: {labels_all}") # DEBUG

                input_boxes_for_prompt = []
                for i, label in enumerate(labels_all):
                    # Flexible label matching: check if prompt is IN the label
                    if text_prompt.lower() in label.lower() and i < len(bboxes_all):
                        input_boxes_for_prompt.append(bboxes_all[i])
                
                if not input_boxes_for_prompt:
                    print(f"    Florence-2 did not return a bounding box for prompt: '{text_prompt}'. Skipping.")
                    continue

                sam_input_box_np = np.array(input_boxes_for_prompt[0]).reshape(1, 4)
                print(f"    Got box from Florence-2: {sam_input_box_np}")

                masks_sam, scores_sam, _ = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=sam_input_box_np,
                    multimask_output=False,
                )
                
                if masks_sam.ndim == 3 and masks_sam.shape[0] == 1:
                    final_mask = masks_sam.squeeze(0)
                    labeled_masks[text_prompt] = final_mask.astype(bool)
                    print(f"    Generated and stored mask for '{text_prompt}'. Mask shape: {final_mask.shape}, Non-zero elements: {np.sum(final_mask)}")
                else:
                    print(f"    SAM2 did not return a single valid mask for prompt '{text_prompt}' with box {sam_input_box_np}. Mask shape: {masks_sam.shape}. Skipping.")

            except Exception as e:
                print(f"    Error processing prompt '{text_prompt}': {e}")
                import traceback
                traceback.print_exc()

        if labeled_masks:
            output_dir = os.path.dirname(output_masks_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")

            np.savez_compressed(output_masks_path, **labeled_masks)
            print(f"Saved {len(labeled_masks)} labeled masks to: {output_masks_path}")
            print(f"Labels saved: {list(labeled_masks.keys())}")

            # --- Add visualization call here ---
            vis_output_path = output_masks_path.replace(".npz", "_visualization.png")
            visualize_and_save_masks_on_frame(frame_np_hwc_rgb, labeled_masks, vis_output_path)
            # --- End visualization call ---

        else:
            print("No masks were generated. Output file not saved.")

    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _generate_masks_logic()
    else:
        _generate_masks_logic()


if __name__ == "__main__":
    print("Running Stage A: Initial Multi-Part Mask Generation (Example)")

    # --- Configuration for Example Usage ---
    # Create dummy video and checkpoint/config files if they don't exist for a basic run
    DUMMY_VIDEO_PATH = "media/crow.mp4"
    DUMMY_OUTPUT_MASKS_PATH = "examples/outputs/stage_a_initial_masks.npz"
    DUMMY_OUTPUT_VIS_PATH_A = "examples/outputs/stage_a_initial_masks_visualization.png" # For clarity
    
    # Ensure example directories exist
    os.makedirs(os.path.dirname(DUMMY_VIDEO_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(DUMMY_OUTPUT_MASKS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(DUMMY_OUTPUT_VIS_PATH_A), exist_ok=True)

    # Create a small dummy MP4 video if it doesn't exist
    if not os.path.exists(DUMMY_VIDEO_PATH):
        print(f"Creating dummy video at {DUMMY_VIDEO_PATH}")
        try:
            # Create 10 frames of a 100x100 black image
            frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)]
            # Add a moving white square for basic testing if models were real
            for i in range(10):
                frames[i][i*5:i*5+20, i*5:i*5+20, :] = 255
            iio.mimwrite(DUMMY_VIDEO_PATH, frames, fps=5, plugin="FFMPEG", codec="libx264")
            print("Dummy video created.")
        except Exception as e:
            print(f"Could not create dummy video (ffmpeg/libx264 may not be installed/configured): {e}")
            print("Please ensure ffmpeg is installed and accessible by imageio, or provide a real video.")
            # If dummy video creation fails, the script might not run without a real video.

    # --- Check for SAM2 checkpoint and config ---
    # This example won't run without actual models.
    # The user needs to download SAM2 checkpoints and provide correct paths.
    # e.g., from https://github.com/facebookresearch/sam2
    # And ensure Florence-2 can be downloaded by HuggingFace transformers.

    sam_ckpt = SAM2_CHECKPOINT_PATH
    sam_cfg = SAM2_CONFIG_PATH

    if not (os.path.exists(sam_ckpt) and os.path.exists(sam_cfg)):
        print("-" * 50)
        print("WARNING: SAM2 checkpoint or config file not found at specified paths:")
        print(f"  Checkpoint: {sam_ckpt} (exists: {os.path.exists(sam_ckpt)})")
        print(f"  Config: {sam_cfg} (exists: {os.path.exists(sam_cfg)})")
        print("The script will likely fail during model initialization if these are not correct.")
        print("Please download SAM2 models and update SAM2_CHECKPOINT_PATH and SAM2_CONFIG_PATH environment variables or arguments.")
        print("Skipping example run if dummy video also doesn't exist and models are missing.")
        print("-" * 50)
        if not os.path.exists(DUMMY_VIDEO_PATH):
             exit() # Exit if no video and no models to avoid crashing later

    # Example text prompts
    # These are illustrative. The success depends entirely on Florence-2's ability
    # to detect these specific terms in the given image.
    prompts = ["body", "head", "tail", "leg", "eye", "wing", "beak", "foot", "neck", "back", "breast", "belly", "underbelly", "back", "breast", "belly", "underbelly", "back", "breast", "belly", "underbelly"]
    # prompts = ["a small red apple", "a green banana"] # More generic prompts

    print(f"Attempting to generate masks for prompts: {prompts}")
    print(f"Using video: {DUMMY_VIDEO_PATH}")
    print(f"Saving masks to: {DUMMY_OUTPUT_MASKS_PATH}")

    # Check if the dummy video exists before trying to process it
    if os.path.exists(DUMMY_VIDEO_PATH):
        try:
            generate_initial_multipart_masks(
                video_path=DUMMY_VIDEO_PATH,
                text_prompts=prompts,
                output_masks_path=DUMMY_OUTPUT_MASKS_PATH,
                reference_frame_index=0, # Use the first frame
                sam_checkpoint=sam_ckpt,
                sam_config=sam_cfg
            )
            print("-" * 30)
            print("Example run finished.")
            if os.path.exists(DUMMY_OUTPUT_MASKS_PATH):
                print(f"Output potentially saved to {DUMMY_OUTPUT_MASKS_PATH}")
                # Verify contents (optional)
                try:
                    loaded_data = np.load(DUMMY_OUTPUT_MASKS_PATH)
                    print(f"  Successfully loaded. Contains keys: {list(loaded_data.keys())}")
                    for key in loaded_data.keys():
                        print(f"    Mask '{key}' shape: {loaded_data[key].shape}, type: {loaded_data[key].dtype}")
                    # Check for visualization file as well
                    if os.path.exists(DUMMY_OUTPUT_MASKS_PATH.replace(".npz", "_visualization.png")):
                        print(f"  Visualization potentially saved to: {DUMMY_OUTPUT_MASKS_PATH.replace('.npz', '_visualization.png')}")
                except Exception as e:
                    print(f"  Error loading or inspecting output file: {e}")
            else:
                print(f"  Output file {DUMMY_OUTPUT_MASKS_PATH} was NOT created. Check logs for errors.")

        except Exception as e:
            print(f"An error occurred during the example run of generate_initial_multipart_masks: {e}")
            import traceback
            traceback.print_exc()
            print("This likely means models (SAM2/Florence-2) could not be loaded or run.")
            print("Please ensure checkpoints are correctly paths and all dependencies are installed.")
    else:
        print(f"Dummy video {DUMMY_VIDEO_PATH} does not exist, and SAM models might be missing. Skipping example execution.")
        print("Please provide a video and ensure models are set up to run this stage.")

    # For a real run, you would call:
    # generate_initial_multipart_masks(
    #     video_path="path/to/your/video.mp4",
    #     text_prompts=["main subject", "left limb", "object of interest"],
    #     output_masks_path="output/your_experiment_masks.npz",
    #     reference_frame_index=0,
    #     # Optionally override sam_checkpoint, sam_config, florence_model_id
    # )
