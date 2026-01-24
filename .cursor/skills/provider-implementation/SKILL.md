---
name: provider-implementation
description: Rules and guidelines for implementing Providers in OM Cortex Runtime
---

# Provider Implementation Guide

This document provides rules, naming conventions, and structural guidelines for implementing Providers in OM Cortex Runtime. Use this guide to review and refactor your Provider implementations.

## Quick Checklist

Before submitting your Provider code, verify:

- [ ] Uses `@singleton` decorator
- [ ] File name: `{name}_provider.py` (snake_case)
- [ ] Class name: `{Name}Provider` (PascalCase)
- [ ] Implements `data` property for data access
- [ ] Implements `start()` and `stop()` methods for lifecycle management
- [ ] Follows Single Responsibility Principle classification
- [ ] Proper error handling and logging
- [ ] Complete docstrings

## 1. Core Structure

### 1.1 Singleton Pattern

**REQUIRED**: All Providers MUST use the `@singleton` decorator.

```python
from providers.singleton import singleton

@singleton
class GpsProvider:
    """
    GPS Provider.
    
    This class implements a singleton pattern to manage GPS data from serial.
    """
    def __init__(self, serial_port: str = ""):
        # Initialization logic
        pass
```

**Why**: 
- Single instance across the system for resource sharing
- Shared communication sessions (if using ROS2/Zenoh), hardware connections
- Consistent state across Sensor/Action/Background usage

### 1.2 File Location and Naming

- **File path**: `src/providers/{name}_provider.py`
- **Class name**: `{Name}Provider` (PascalCase)
- **Module structure**: Each Provider in a separate file

**Examples**:
- `src/providers/gps_provider.py` → `GpsProvider`
- `src/providers/realsense_camera_provider.py` → `RealSenseCameraProvider`
- `src/providers/unitree_go2_navigation_provider.py` → `UnitreeGo2NavigationProvider`

### 1.3 Data Access Interface

**REQUIRED**: Provider MUST expose data via `data` property.

```python
@property
def data(self) -> Optional[dict]:
    """
    Get the current provider data.
    
    Returns
    -------
    Optional[dict]
        Current data from the provider, or None if not available.
    """
    return self._data
```

### 1.4 Lifecycle Management

**REQUIRED**: Provider MUST implement `start()` and `stop()` methods.

```python
def start(self):
    """
    Start the provider.
    
    Initializes and starts the provider thread/process if not already running.
    """
    if self.running:
        return
    
    self.running = True
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

def stop(self):
    """
    Stop the provider.
    
    Stops the provider thread/process and cleans up resources.
    """
    self.running = False
    if self._thread:
        self._thread.join(timeout=5)
```

## 2. Provider Classification

Providers are classified based on **Single Responsibility Principle**:

### 2.1 Hardware/Sensor Level
**One physical sensor = One Provider**

- `GpsProvider`: GPS sensor only
- `OdomProvider`: Odometry sensor only
- `RPLidarProvider`: RPLidar sensor only
- `RealsenseProvider`: RealSense camera only

**Reason**: Each sensor is independent hardware with different protocols and data formats.

### 2.2 Robot Platform Level
**Robot-specific functionality = Separate Provider**

- `UnitreeGo2NavigationProvider`: Unitree Go2 specific navigation
- `UnitreeG1NavigationProvider`: Unitree G1 specific navigation
- `TurtleBot4CameraVLMProvider`: TurtleBot4 specific camera VLM

**Reason**: Different robots use different communication protocols (CycloneDDS vs Zenoh) and APIs.

### 2.3 Functionality/Service Level
**One functionality/service = One Provider**

- `VLMGeminiProvider`: Gemini VLM service only
- `VLMOpenAIProvider`: OpenAI VLM service only
- `ElevenLabsTTSProvider`: ElevenLabs TTS service only
- `GPSNavProvider`: GPS Navigation functionality only
- `DWANavProvider`: DWA Navigation functionality only

**Reason**: Different services have different APIs, configurations, and data formats.

### 2.4 Communication Protocol Level
**Communication method = Separate Provider (when needed)**

- `ZenohPublisherProvider`: Zenoh publish only
- `ZenohListenerProvider`: Zenoh subscribe only
- `ROS2PublisherProvider`: ROS2 publish only

**Reason**: Different communication protocols and message formats.

### 2.5 System/Utility Level
**System-wide functionality = Separate Provider**

- `IOProvider`: I/O data management (singleton)
- `ConfigProvider`: Config broadcasting
- `ContextProvider`: Context management

**Reason**: System-wide shared resources, not tied to specific hardware/service.

## 3. Complete Template

```python
import logging
import threading
import time
from typing import Optional

from providers.singleton import singleton


@singleton
class ExampleProvider:
    """
    Example Provider.
    
    This class implements a singleton pattern to manage [functionality description].
    
    Parameters
    ----------
    config_param : str
        Configuration parameter.
    """
    
    def __init__(self, config_param: str = ""):
        """
        Initialize the Example Provider.
        
        Parameters
        ----------
        config_param : str
            Configuration parameter.
        """
        logging.info(f"ExampleProvider booting at: {config_param}")
        
        # State variables
        self._data: Optional[dict] = None
        self.running = False
        self._thread: Optional[threading.Thread] = None
        
        # Auto-start after initialization (optional)
        self.start()
    
    def start(self):
        """Start the provider."""
        if self._thread and self._thread.is_alive():
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("ExampleProvider started")
    
    def _run(self):
        """Main loop for the provider."""
        while self.running:
            try:
                # Data collection/processing logic
                self._update_data()
            except Exception as e:
                logging.error(f"Error in provider loop: {e}")
            time.sleep(0.1)  # Adjust interval as needed
    
    def _update_data(self):
        """Update internal data."""
        # Data update logic
        self._data = {
            "field1": value1,
            "field2": value2,
        }
    
    def stop(self):
        """Stop the provider."""
        self.running = False
        if self._thread:
            logging.info("Stopping ExampleProvider")
            self._thread.join(timeout=5)
    
    @property
    def data(self) -> Optional[dict]:
        """
        Get the current provider data.
        
        Returns
        -------
        Optional[dict]
            Current data from the provider, or None if not available.
        """
        return self._data
```

## 4. Vendor Code Integration

When integrating vendor code into Providers:

### 4.1 Hardware Abstraction
**Principle**: Hardware communication goes in Provider

Examples:
- GPS serial port reading → `GpsProvider`
- RealSense camera data collection → `RealSenseCameraProvider`
- ROS2 topic publish/subscribe → `ROS2PublisherProvider`, `ROS2ListenerProvider`

### 4.2 Computation Logic in Provider
**Principle**: Provider can include computation logic for data generation/transformation

Examples:
- `BEVOccupancyGridProvider`: BEV Occupancy Grid generation logic
- `PointCloudProvider`: PointCloud generation logic
- `SegmentationProvider`: TensorRT model execution and segmentation

```python
@singleton
class BEVOccupancyGridProvider:
    def __init__(self):
        # CUDA initialization, model loading, etc.
        self._init_cuda()
        self._load_model()
    
    def _generate_bev(self, camera_data):
        # BEV generation computation logic (ported from vendor code)
        # ...
        return bev_grid
    
    @property
    def data(self):
        return self._bev_grid
```

### 4.3 Communication Protocol Considerations (If Using ROS2)
**Note**: This section applies only if you are using ROS2 in your implementation.

**Principle**: If using ROS2, only hardware direct control should use ROS2 nodes

- ✅ **Use ROS2 nodes** (if using ROS2): RealSense driver, Unitree Go2 control bridge
- ❌ **Don't use ROS2 nodes**: Computation logic, business logic, data processing

**Note**: ROS2 is optional. Providers can use any communication protocol (serial, websocket, HTTP, SDK, etc.) or no external communication at all.

## 5. Usage Patterns

### 5.1 Sensor Access Pattern
Sensors read from Provider's `data` property (read-only):

```python
# In Sensor implementation
class Gps(FuserInput[SensorConfig, Optional[dict]]):
    def __init__(self, config: SensorConfig):
        self.gps = GpsProvider()  # Singleton instance
    
    async def _poll(self) -> Optional[dict]:
        return self.gps.data  # Read-only access
```

### 5.2 Action Access Pattern
Actions call Provider methods for write/control:

```python
# In Action Connector implementation
class NavigateConnector(ActionConnector[...]):
    def __init__(self, config: ...):
        self.navigation_provider = NavigationProvider()
    
    async def connect(self, input: NavigateInput):
        # Business logic: parse input, transform data
        goal_pose = PoseStamped(...)
        
        # Provider call: actual communication
        self.navigation_provider.publish_goal_pose(goal_pose)
```

## 6. Common Patterns

### 6.1 Thread-based Provider
```python
@singleton
class ThreadBasedProvider:
    def __init__(self):
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._data: Optional[dict] = None
        self.start()
    
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def _run(self):
        while self.running:
            # Main loop
            time.sleep(0.1)
    
    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
```

### 6.2 Callback-based Provider
```python
@singleton
class CallbackBasedProvider:
    def __init__(self):
        self._callbacks: List[Callable] = []
        self._data: Optional[dict] = None
    
    def register_callback(self, callback: Callable):
        self._callbacks.append(callback)
    
    def _notify_callbacks(self):
        for callback in self._callbacks:
            callback(self._data)
```

## 7. Review Checklist

When reviewing your Provider implementation:

- [ ] **Singleton**: Uses `@singleton` decorator
- [ ] **File naming**: `{name}_provider.py` (snake_case)
- [ ] **Class naming**: `{Name}Provider` (PascalCase)
- [ ] **Data property**: Implements `data` property
- [ ] **Lifecycle**: Implements `start()` and `stop()` methods
- [ ] **Classification**: Follows appropriate classification (hardware/robot/functionality/communication/system)
- [ ] **Error handling**: Proper exception handling
- [ ] **Logging**: Appropriate logging at info/warning/error levels
- [ ] **Documentation**: Complete docstrings for class and methods
- [ ] **Type hints**: Proper type annotations
- [ ] **Thread safety**: Thread-safe if using threads
- [ ] **Resource cleanup**: Proper cleanup in `stop()` method

## 8. Reference Examples

- `src/providers/gps_provider.py`: GPS Provider with serial communication
- `src/providers/odom_provider.py`: Odometry Provider with ROS2/Zenoh
- `src/providers/asr_provider.py`: ASR Provider with websocket communication
- `src/providers/unitree_go2_navigation_provider.py`: Navigation Provider for Unitree Go2
