# BADAS-Open-CLI: Ego-Centric Collision Prediction with attention/grad rollout

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2025.xxxxx)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/nexar-ai/badas-open)
[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen)](https://nexar-ai.github.io/badas-open)

</div>

<div align="center">
  <p align="center">
    <img src="./data/attn.gif" alt="Attention Rollout" width="400"/>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <br><br>
    <strong>Attention Rollout</strong> <em>(Left)</em> : Model's attention visualization<br>
  </p>
</div>

## Quick Start

### Installation

Python version : 3.8, 3.9, 3.10, 3.11

```bash
# Clone this repository
git clone https://gitlabee.dt.renault.com/dire/dea-ir/ai_emerging_tech/assisted_driving/badas-open.git

# Create a venv
python -m venv venv

# On Windows
.\venv\Scripts\activate
# On Mac/Linux
source .venv/bin/activate

# Install requirements.txt
pip install -r requirements.txt
```

### Usage

Use `inference.py` to run the model on a specific video

```bash
python inference.py --video-path dash_cam.mp4 --export-video output.mp4 --attn-rollout
```

**Available Flags:**
* `--video-path` *(Required)* : Path to the video file you want to analyze.
* `--real-time` : Enables an OpenCV GUI window showing the video playback. (Not working for now)
* `--time` : Display the average runtime for a prediction (16 frames processing).
* `--export-video` : Saves a new video with the risk gauge overlay (e.g., output.mp4).
* `--grad-rollout` : Active Gradient Rollout heat map (`--export-video` must be on).
* `--attn-rollout` : Active Attention Rollout heat map (`--export-video` must be on).

### Basic Usage

```python
from badas import BADASModel

# Initialize model
model = BADASModel(device="cuda")  # or "cpu" for CPU inference

# Predict on video
predictions = model.predict("dashcam_video.mp4")

# Get collision risk for each frame window
for i, prob in enumerate(predictions):
    if prob > 0.8:
        print(f"⚠️ High collision risk at {i*0.125:.1f}s: {prob:.2%}")
```

### Advanced Usage

```python
import torch
from badas import load_badas_model, preprocess_video

# Load model with custom configuration
model = load_badas_model(
    device="cuda",
    checkpoint_path="path/to/custom_checkpoint.pth"  # Optional
)

# Preprocess video manually for batch processing
frames = preprocess_video(
    "dashcam_video.mp4",
    target_fps=8,
    num_frames=16,
    img_size=224
)

# Run inference
with torch.no_grad():
    collision_probs = model(frames)
    
# Estimate time to collision
tta = model.estimate_time_to_accident(collision_probs, fps=8.0)
if tta is not None:
    print(f"🚨 Collision in {tta:.1f} seconds!")
```

### Training

Run the following command for finetuning end-to-end : 
```bash
python ./badas/train/video_training.py --model_name "facebook/vjepa2-vitl-fpc16-256-ssv2" --data_root "./data" --output_dir "./results" --head_type "linear" --use_custom_head --temporal_method "mean" --epochs 10 
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **V-JEPA2 Foundation Model** by Meta AI Research
- **Nexar Driver Community** for dataset contribution
- **Academic Partners** for benchmark annotations
```bibtex
@article{goldshmidt2025badas,
  title={BADAS: Context-Aware Collision Prediction Using Real-World Dashcam Data},
  author={Goldshmidt, Roni and Scott, Hamish and Niccolini, Lorenzo and 
          Zhu, Shizhan and Moura, Daniel and Zvitia, Orly},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## 🔗 Links

- 🌐 **Website**: [nexar.ai/badas](https://nexar-data.webflow.io/badas)

## ⚠️ Disclaimer

This model is intended for research and development purposes. It should not be used as the sole decision-making system for vehicle safety. Always maintain full attention while driving and follow all traffic laws and safety regulations.

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://nexar.ai">Nexar AI Research</a></sub>
</div>
