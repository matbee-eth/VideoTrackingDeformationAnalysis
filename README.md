# Animal Movement Deformation Analysis & Robot Mapping Concept

**⚠️ Project Status: Under Heavy Development & Unstable ⚠️**

This project is a work-in-progress exploring methods to analyze animal movement from videos, focusing on surface deformation to infer kinematic insights. The long-term vision is to utilize these insights for mapping movements to robotic systems.

**Please note:** The codebase is currently experimental and subject to frequent changes. APIs and data formats may break between updates. It is not yet recommended for production use.

## 1. Introduction and Goal

The primary goal of this project is to analyze video footage of animal movement to identify and map regions of significant surface deformation (contraction, expansion, shear). These deformation patterns can serve as strong indicators of underlying biological joints or areas of notable flexibility. The ultimate, long-term ambition is to leverage this understanding to inform how a robot's joints might be actuated to achieve a more articulated, holistic, and naturalistic imitation of the observed animal's motion, moving beyond simple end-effector path mimicry.

For a detailed breakdown of the project's conceptual stages, methodologies, and evolution, please see [PROJECT.MD](PROJECT.MD).

## 2. Project Stages Overview

The project is conceptually divided into the following stages:

*   **Stage A: Animal Segmentation (Initial Multi-Part Mask Generation)**
    *   Objective: Isolate multiple, semantically defined parts of the target animal from an initial reference frame using text prompts (e.g., Florence-2 for text-to-bounding-box and SAM2 for segmentation).
*   **Stage B: Dense Point Cloud Generation & Multi-Part Tracking**
    *   Objective: Initialize and track distinct sets of dense points on each segmented animal part through the video (e.g., using CoTracker3).
*   **Stage C: Deformation Analysis**
    *   Objective: Quantify local intra-part deformation (strain, dilatation, shear based on K-Nearest Neighbors) and inter-part relative motion (e.g., centroid trajectories).
*   **Stage D: Identifying "Areas of Movement"**
    *   Objective: Pinpoint regions exhibiting consistent, significant deformation, potentially indicating joints or flexible areas. Visualize these findings, highlighting active and persistent tracks, and utilizing tracker visibility flags.
*   **Stage E: Conceptual Mapping to Robot Joints**
    *   Objective (Long-Term): Utilize insights from Stage D to inform control strategies for robotic imitation.

## 3. Current Functionality

The project currently includes Python scripts for:

*   `src/stage_a_segmentation.py`: Generates labeled masks for animal parts from a reference frame using text prompts, employing Florence-2 for text-to-bounding-box mapping and SAM2 for detailed segmentation. Includes a visual preview option for the reference frame.
*   `src/stage_b_tracking.py`: Initializes points within the masks from Stage A and tracks these distinct sets of points across the video using CoTracker3.
*   `src/stage_c_deformation_analysis.py`: Calculates intra-part deformation metrics (such as dilatation and shear strains using K-Nearest Neighbors to define local patches) and computes inter-part centroid trajectories over time.
*   `src/stage_d_movement_identification.py`: Identifies and visualizes areas of high deformation by thresholding Stage C metrics, applying temporal consistency filters to individual point deformations, and performing spatial clustering (DBSCAN) on these filtered points. The visualization distinguishes between actively deforming regions and persistently tracked significant points, incorporating CoTracker's visibility flags to reduce visual clutter from unreliable tracks.
*   `src/stage_e_robot_mapping.py`: Placeholder for future development.

Data is primarily passed between stages using `.npz` files. Example scripts and dummy data might be found in the `examples/` directory.

## 4. Prerequisites and Setup

1.  **Python Environment:** Ensure you have Python 3.8+ installed.
2.  **Dependencies:** Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Some dependencies like PyTorch (for SAM2, CoTracker3, Florence-2) might have specific installation instructions based on your CUDA version if GPU support is desired. Please refer to their official documentation.*
3.  **Pre-trained Models:** The scripts rely on pre-trained models. Libraries like Hugging Face Transformers (for Florence-2, SAM2) typically download these automatically to a user cache directory upon first use. However, for some models like CoTracker, you may need to manually download a specific checkpoint file (e.g., `scaled_offline.pth` for CoTracker3) and place it in a designated `checkpoints/` directory within the project root. Ensure you have an internet connection for initial automatic downloads.

## 5. How to Run (Example Workflow)

Each stage is typically run sequentially. Output paths for data and visualizations are configured within the `if __name__ == "__main__":` blocks of the scripts or can be modified as needed.

1.  **Stage A:**
    *   Modify parameters in `src/stage_a_segmentation.py` (e.g., video path, text prompts, `visual_preview=True`).
    *   Run: `python src/stage_a_segmentation.py`
2.  **Stage B:**
    *   Ensure Stage A's output `.npz` mask file path is correctly set as input in `src/stage_b_tracking.py`.
    *   Verify the `COTRACKER_CHECKPOINT_PATH` in the script points to your downloaded CoTracker checkpoint (e.g., `checkpoints/scaled_offline.pth`).
    *   Run: `python src/stage_b_tracking.py`
3.  **Stage C:**
    *   Ensure Stage B's output `.npz` tracks file path is correctly set as input in `src/stage_c_deformation_analysis.py`.
    *   Run: `python src/stage_c_deformation_analysis.py`
4.  **Stage D:**
    *   Ensure Stage C's output `.npz` analysis file path is correctly set as input in `src/stage_d_movement_identification.py`.
    *   Ensure the visualization output path (e.g., `DUMMY_OUTPUT_VIS_PATH_D`) is set to a video file format (e.g., `.mp4`) if video output is desired.
    *   Run: `python src/stage_d_movement_identification.py`

Refer to the `if __name__ == "__main__":` blocks within each script for example usage and configurable parameters. The `examples/outputs/` directory is often used for storing results in these examples.

## 6. Contributing

This project is currently in a highly experimental phase and primarily serves as a research exploration. While feedback and suggestions are welcome via GitHub Issues, we are not actively seeking direct code contributions at this moment due to the rapid pace of change.

## 7. License

To be determined. For now, please consider all code proprietary unless explicitly stated otherwise.

---

Thank you for your interest in this project! 