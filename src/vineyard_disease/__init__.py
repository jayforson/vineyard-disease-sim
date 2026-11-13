"""CPU-first PlantVillage grape-leaf disease screening pipeline.

The package prepares bounded, reproducible manifests from public grape-leaf
images, runs a pretrained image classifier on single leaf crops, and evaluates
its predictions. Scores are uncalibrated classifier outputs, not a diagnosis.

Heavy dependencies (Pillow, torch, transformers, rclpy) are imported lazily by
the functions that need them, so importing this package is cheap.
"""

__version__ = "0.1.0"

DISCLAIMER = (
    "Screening output only: scores are uncalibrated classifier probabilities "
    "on a single leaf crop, not a plant-health diagnosis. No action is taken."
)

__all__ = ["DISCLAIMER", "__version__"]
