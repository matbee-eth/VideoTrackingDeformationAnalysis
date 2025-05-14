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
    *   Objective: Isolate multiple, semantically defined parts of the target animal from an initial reference frame using text prompts (e.g., Florence-2 + SAM2).
*   **Stage B: Dense Point Cloud Generation & Multi-Part Tracking**
    *   Objective: Initialize and track distinct sets of dense points on each segmented animal part through the video (e.g., using CoTracker3).
*   **Stage C: Deformation Analysis**
    *   Objective: Quantify local intra-part deformation (strain, dilatation, shear) and inter-part relative motion.
*   **Stage D: Identifying "Areas of Movement"**
    *   Objective: Pinpoint regions exhibiting consistent, significant deformation, potentially indicating joints or flexible areas, and visualize these findings.
*   **Stage E: Conceptual Mapping to Robot Joints**
    *   Objective (Long-Term): Utilize insights from Stage D to inform control strategies for robotic imitation.

## 3. Current Functionality

The project currently includes Python scripts for:

*   `src/stage_a_segmentation.py`: Generates labeled masks for animal parts from a reference frame using text prompts.
*   `src/stage_b_tracking.py`: Tracks points initialized within these labeled masks across a video.
*   `src/stage_c_deformation_analysis.py`: Calculates intra-part deformation metrics and inter-part centroid trajectories.
*   `src/stage_d_movement_identification.py`: Identifies and visualizes areas of high deformation based on Stage C's output.
*   `src/stage_e_robot_mapping.py`: Placeholder for future development.

Data is primarily passed between stages using `.npz` files. Example scripts and dummy data might be found in the `examples/` directory.

## 4. Prerequisites and Setup

1.  **Python Environment:** Ensure you have Python 3.8+ installed.
2.  **Dependencies:** Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Some dependencies like PyTorch (for SAM2, CoTracker3, Florence-2) might have specific installation instructions based on your CUDA version if GPU support is desired. Please refer to their official documentation.*
3.  **Pre-trained Models:** The scripts rely on pre-trained models (SAM2, CoTracker3, Florence-2). These are typically downloaded automatically by the respective libraries upon first use to a cache directory. Ensure you have an internet connection when running for the first time.

## 5. How to Run (Example Workflow)

Each stage is typically run sequentially. Output paths for data and visualizations are configured within the scripts or can be passed as arguments.

1.  **Stage A:**
    *   Modify parameters in `src/stage_a_segmentation.py` (e.g., video path, text prompts).
    *   Run: `python src/stage_a_segmentation.py`
2.  **Stage B:**
    *   Update input path in `src/stage_b_tracking.py` to point to Stage A's output.
    *   Run: `python src/stage_b_tracking.py`
3.  **Stage C:**
    *   Update input path in `src/stage_c_deformation_analysis.py` for Stage B's output.
    *   Run: `python src/stage_c_deformation_analysis.py`
4.  **Stage D:**
    *   Update input path in `src/stage_d_movement_identification.py` for Stage C's output.
    *   Run: `python src/stage_d_movement_identification.py`

Refer to the `if __name__ == "__main__":` blocks within each script for example usage and configurable parameters. The `examples/outputs/` directory is often used for storing results in these examples.

## 6. Contributing

This project is currently in a highly experimental phase and primarily serves as a research exploration. While feedback and suggestions are welcome via GitHub Issues, we are not actively seeking direct code contributions at this moment due to the rapid pace of change.

## 7. License

To be determined. For now, please consider all code proprietary unless explicitly stated otherwise.

---

Thank you for your interest in this project! 