---
name: background-implementation
description: Rules and guidelines for implementing Backgrounds in OM Cortex Runtime
---

# Background Implementation Guide

This document provides rules, naming conventions, and structural guidelines for implementing Backgrounds in OM Cortex Runtime. Use this guide to review and refactor your Background implementations.

## Quick Checklist

Before submitting your Background code, verify:

- [ ] Inherits from `Background[ConfigType]`
- [ ] File name: `{name}.py` (snake_case) in `src/backgrounds/plugins/`
- [ ] Class name: `{Name}` (PascalCase, no "Provider" suffix)
- [ ] Config class: `{Name}Config` inheriting from `BackgroundConfig`
- [ ] Initializes Provider(s) in `__init__`
- [ ] Implements `run()` method
- [ ] Proper error handling and logging
- [ ] Complete docstrings

## 1. Core Structure

### 1.1 Base Class Inheritance

**REQUIRED**: All Backgrounds MUST inherit from `Background[ConfigType]`.

```python
from backgrounds.base import Background, BackgroundConfig
from providers.example_provider import ExampleProvider

class ExampleConfig(BackgroundConfig):
    """Configuration for Example Background."""
    config_param: Optional[str] = Field(default=None)

class Example(Background[ExampleConfig]):
    """Example Background."""
    def __init__(self, config: ExampleConfig):
        super().__init__(config)
        # Provider initialization
```

### 1.2 File Location and Naming

- **File path**: `src/backgrounds/plugins/{name}.py`
- **Class name**: `{Name}` (PascalCase, no "Provider" suffix)
- **Config class**: `{Name}Config` (PascalCase)

**Examples**:
- `src/backgrounds/plugins/gps.py` → `Gps` (for `GpsProvider`)
- `src/backgrounds/plugins/realsense_camera.py` → `RealSenseCamera` (for `RealSenseCameraProvider`)
- `src/backgrounds/plugins/odom.py` → `Odom` (for `OdomProvider`)

### 1.3 Configuration Class

**REQUIRED**: Background MUST have a Config class inheriting from `BackgroundConfig`.

```python
from pydantic import Field
from backgrounds.base import BackgroundConfig

class ExampleConfig(BackgroundConfig):
    """
    Configuration for Example Background.
    
    Parameters
    ----------
    config_param : Optional[str]
        Configuration parameter.
    """
    config_param: Optional[str] = Field(
        default=None,
        description="Configuration parameter"
    )
```

## 2. Provider Initialization

### 2.1 Basic Pattern: 1:1 Relationship

**Most common**: One Background = One Provider

```python
class Gps(Background[GpsConfig]):
    """
    Reads GPS and Magnetometer data from GPS provider.
    """
    
    def __init__(self, config: GpsConfig):
        super().__init__(config)
        
        # Validate config
        port = self.config.serial_port
        if port is None:
            logging.error("GPS serial port not specified in config")
            return
        
        # Initialize Provider
        self.gps_provider = GpsProvider(serial_port=port)
        logging.info(f"Initiated GPS Provider with serial port: {port} in background")
    
    def run(self) -> None:
        """Background process loop."""
        time.sleep(60)  # Default: sleep, override if needed
```

### 2.2 Complex Pattern: 1:N Relationship

**For composite functionality**: One Background = Multiple Providers

```python
class RFmapper(Background[RFmapperConfig]):
    """
    Collects and sends location data to Fabric network.
    """
    
    def __init__(self, config: RFmapperConfig):
        super().__init__(config)
        
        # Initialize multiple Providers
        self.gps_provider = GpsProvider(...)
        self.rtk_provider = RtkProvider(...)
        self.odom_provider = OdomProvider(...)
        logging.info("RFmapper initialized with multiple providers")
    
    def run(self) -> None:
        """Collect data from multiple providers and send to Fabric."""
        gps_data = self.gps_provider.data
        rtk_data = self.rtk_provider.data
        odom_data = self.odom_provider.data
        # Process and send to Fabric network
        time.sleep(1)  # Adjust as needed
```

### 2.3 Initialization Responsibility

**Background is responsible for Provider initialization**

**Why**:
1. **Lifecycle management**: Background ensures Provider is ready before Sensor/Action use
2. **Resource management**: Background manages Provider lifecycle (start/stop)
3. **Configuration**: Background passes configuration to Provider
4. **Separation of concerns**: Background handles initialization, Sensor/Action handle usage

## 3. Complete Template

### 3.1 Simple Background Template

```python
import logging
import time
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.example_provider import ExampleProvider


class ExampleConfig(BackgroundConfig):
    """
    Configuration for Example Background.
    
    Parameters
    ----------
    config_param : Optional[str]
        Configuration parameter.
    """
    
    config_param: Optional[str] = Field(
        default=None,
        description="Configuration parameter"
    )


class Example(Background[ExampleConfig]):
    """
    Example Background.
    
    Initializes and starts Example Provider.
    """
    
    def __init__(self, config: ExampleConfig):
        super().__init__(config)
        
        # Validate configuration
        param = self.config.config_param
        if param is None:
            logging.error("Config parameter not specified in config")
            return
        
        # Initialize Provider
        self.example_provider = ExampleProvider(config_param=param)
        logging.info(f"Example Provider initialized with param: {param} in background")
    
    def run(self) -> None:
        """
        Background process loop.
        
        Can be used for continuous monitoring or periodic tasks.
        Override this method if you need custom background behavior.
        """
        time.sleep(60)  # Default: sleep, override if needed
```

### 3.2 Background with Custom Run Logic

```python
class MonitoringBackground(Background[MonitoringConfig]):
    """
    Background with custom monitoring logic.
    """
    
    def __init__(self, config: MonitoringConfig):
        super().__init__(config)
        self.provider = MonitoringProvider(...)
        self._last_check = time.time()
    
    def run(self) -> None:
        """Custom background loop with periodic checks."""
        current_time = time.time()
        
        # Perform periodic check every 30 seconds
        if current_time - self._last_check >= 30:
            self._perform_check()
            self._last_check = current_time
        
        time.sleep(1)  # Short sleep to avoid busy waiting
    
    def _perform_check(self):
        """Perform periodic monitoring check."""
        data = self.provider.data
        if data is None:
            logging.warning("No data available from provider")
            return
        # Process data
        logging.info(f"Monitoring check completed: {data}")
```

## 4. Common Patterns

### 4.1 Provider Initialization with Error Handling

```python
class RobustBackground(Background[RobustConfig]):
    def __init__(self, config: RobustConfig):
        super().__init__(config)
        
        try:
            self.provider = RobustProvider(
                param1=config.param1,
                param2=config.param2
            )
            logging.info("Robust Provider initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Robust Provider: {e}")
            self.provider = None
    
    def run(self) -> None:
        if self.provider is None:
            logging.warning("Provider not initialized, skipping run")
            time.sleep(60)
            return
        # Normal operation
        time.sleep(60)
```

### 4.2 Background with Multiple Providers

```python
class CompositeBackground(Background[CompositeConfig]):
    def __init__(self, config: CompositeConfig):
        super().__init__(config)
        
        # Initialize multiple providers
        self.provider1 = Provider1(...)
        self.provider2 = Provider2(...)
        self.provider3 = Provider3(...)
    
    def run(self) -> None:
        """Collect and process data from multiple providers."""
        data1 = self.provider1.data
        data2 = self.provider2.data
        data3 = self.provider3.data
        
        # Process combined data
        if all([data1, data2, data3]):
            self._process_combined_data(data1, data2, data3)
        
        time.sleep(1)
    
    def _process_combined_data(self, d1, d2, d3):
        """Process data from all providers."""
        # Processing logic
        pass
```

## 5. Background-Provider Relationship Rules

### 5.1 Initialization Order

1. Background `__init__` is called by BackgroundOrchestrator
2. Background initializes Provider(s) in its `__init__`
3. Provider starts automatically (if `start()` is called in Provider `__init__`)
4. Background `run()` is called in separate thread by BackgroundOrchestrator

### 5.2 Lifecycle Management

- **Background initialization**: Happens once at system startup
- **Provider initialization**: Happens in Background `__init__`
- **Background run loop**: Runs continuously in separate thread
- **Provider lifecycle**: Managed by Provider itself (start/stop)

### 5.3 Configuration Flow

```
Config File → BackgroundConfig → Background.__init__() → Provider.__init__()
```

## 6. Review Checklist

When reviewing your Background implementation:

- [ ] **Inheritance**: Inherits from `Background[ConfigType]`
- [ ] **File naming**: `{name}.py` in `src/backgrounds/plugins/`
- [ ] **Class naming**: `{Name}` (PascalCase, no "Provider" suffix)
- [ ] **Config class**: `{Name}Config` inheriting from `BackgroundConfig`
- [ ] **Super init**: Calls `super().__init__(config)`
- [ ] **Provider init**: Initializes Provider(s) in `__init__`
- [ ] **Config validation**: Validates config parameters before Provider init
- [ ] **Error handling**: Handles Provider initialization errors
- [ ] **Logging**: Logs Provider initialization
- [ ] **Run method**: Implements `run()` method
- [ ] **Documentation**: Complete docstrings for class and methods
- [ ] **Type hints**: Proper type annotations

## 7. Reference Examples

- `src/backgrounds/plugins/gps.py`: Simple Background with single Provider
- `src/backgrounds/plugins/odom.py`: Background with conditional Provider initialization
- `src/backgrounds/plugins/rf_mapper.py`: Complex Background with multiple Providers

## 8. Anti-patterns to Avoid

### ❌ Don't: Initialize Provider outside Background

```python
# WRONG: Provider initialized in Sensor/Action
class GpsSensor(FuserInput[...]):
    def __init__(self, config):
        self.provider = GpsProvider(...)  # Should be in Background
```

### ❌ Don't: Skip Config validation

```python
# WRONG: No validation
class BadBackground(Background[BadConfig]):
    def __init__(self, config):
        super().__init__(config)
        self.provider = Provider(config.param)  # May fail if param is None
```

### ✅ Do: Validate config and handle errors

```python
# CORRECT: Validate and handle errors
class GoodBackground(Background[GoodConfig]):
    def __init__(self, config):
        super().__init__(config)
        if config.param is None:
            logging.error("Param not specified")
            return
        try:
            self.provider = Provider(config.param)
        except Exception as e:
            logging.error(f"Failed to initialize: {e}")
```
