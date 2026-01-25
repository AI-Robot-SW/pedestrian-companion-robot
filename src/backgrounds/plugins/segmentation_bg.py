import logging
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.segmentation_provider import SegmentationProvider


class SegmentationConfig(BackgroundConfig):
    """
    Configuration for Segmentation Background.

    Parameters
    ----------
    engine_path : Optional[str]
        Path to TensorRT engine file (default: "").
    """

    engine_path: Optional[str] = Field(
        default="", description="Path to TensorRT engine file"
    )


class SegmentationBg(Background[SegmentationConfig]):
    """
    Segmentation Background.

    Initializes and starts the SegmentationProvider in the background.
    """

    def __init__(self, config: SegmentationConfig):
        """
        Initialize the Segmentation Background.

        Parameters
        ----------
        config : SegmentationConfig
            Configuration for the background task.
        """
        super().__init__(config)

        engine_path = self.config.engine_path or ""

        # Initialize Provider (singleton, so same instance shared)
        self.segmentation_provider = SegmentationProvider(engine_path=engine_path)

        # Start Provider
        self.segmentation_provider.start()

        logging.info(
            f"Segmentation Provider initialized in background (engine_path: {engine_path})"
        )
