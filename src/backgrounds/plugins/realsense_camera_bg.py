import logging
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.realsense_camera_provider import RealSenseCameraProvider


class RealSenseCameraConfig(BackgroundConfig):
    """
    Configuration for RealSense Camera Background.

    Parameters
    ----------
    camera_index : Optional[int]
        Camera index for RealSense device (default: 0).
    """

    camera_index: Optional[int] = Field(
        default=0, description="Camera index for RealSense device"
    )


class RealSenseCameraBg(Background[RealSenseCameraConfig]):
    """
    RealSense Camera Background.

    Initializes and starts the RealSenseCameraProvider in the background.
    """

    def __init__(self, config: RealSenseCameraConfig):
        """
        Initialize the RealSense Camera Background.

        Parameters
        ----------
        config : RealSenseCameraConfig
            Configuration for the background task.
        """
        super().__init__(config)

        camera_index = self.config.camera_index or 0

        # Initialize Provider (singleton, so same instance shared)
        self.realsense_camera_provider = RealSenseCameraProvider(
            camera_index=camera_index
        )

        # Start Provider
        self.realsense_camera_provider.start()

        logging.info(
            f"RealSense Camera Provider initialized in background (camera_index: {camera_index})"
        )
