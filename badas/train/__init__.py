"""
BADAS: Training Module

Training utilities and video-based model training logic
for the BADAS (V-JEPA2) system.
"""

__version__ = "1.0.0"
__author__ = "Nexar AI Research"
__license__ = "Apache 2.0"

from .video_training import (
    detect_model_type,
    EnhancedVideoClassifier
)

__all__ = [
    "detect_model_type",
    "EnhancedVideoClassifier"
]
