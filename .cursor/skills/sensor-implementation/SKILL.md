---
name: sensor-implementation
description: Rules and guidelines for implementing Sensors (Inputs) in OM Cortex Runtime
---

# Sensor (Input) Implementation Guide

This document provides rules, naming conventions, and structural guidelines for implementing Sensors (Inputs) in OM Cortex Runtime. Use this guide to review and refactor your Sensor implementations.

## Quick Checklist

Before submitting your Sensor code, verify:

- [ ] Inherits from `FuserInput[ConfigType, R]`
- [ ] File name: `{name}.py` (snake_case) in `src/inputs/plugins/`
- [ ] Class name: `{Name}` (PascalCase)
- [ ] Config class: `{Name}Config` inheriting from `SensorConfig` (optional)
- [ ] Implements `async def _poll()` method
- [ ] Implements `async def _raw_to_text()` method
- [ ] Implements `formatted_latest_buffer()` method
- [ ] Proper error handling and logging
- [ ] Complete docstrings

## 1. Core Structure

### 1.1 Base Class Inheritance

**REQUIRED**: All Sensors MUST inherit from `FuserInput[ConfigType, R]`.

```python
from inputs.base import SensorConfig, Message
from inputs.base.loop import FuserInput
from providers.example_provider import ExampleProvider

class Example(FuserInput[SensorConfig, Optional[dict]]):
    """
    Example Sensor.
    
    Reads data from Example Provider and converts to text format.
    """
    pass
```

### 1.2 File Location and Naming

- **File path**: `src/inputs/plugins/{name}.py`
- **Class name**: `{Name}` (PascalCase)
- **Config class**: `{Name}Config` (optional, inherits from `SensorConfig`)

**Examples**:
- `src/inputs/plugins/gps.py` → `Gps`
- `src/inputs/plugins/odom.py` → `Odom`
- `src/inputs/plugins/vlm_gemini.py` → `VLMGemini`

### 1.3 Type Parameters

- **ConfigType**: Configuration class (usually `SensorConfig` or custom config)
- **R**: Raw input type (e.g., `Optional[dict]`, `Optional[str]`)

## 2. Required Methods

### 2.1 `_poll()` Method

**REQUIRED**: MUST implement `async def _poll()` to poll for new data.

```python
async def _poll(self) -> Optional[dict]:
    """
    Poll for new messages from the Provider.
    
    Returns
    -------
    Optional[dict]
        The next message from the provider if available, None otherwise
    """
    await asyncio.sleep(0.5)  # Prevent excessive CPU usage
    
    try:
        return self.provider.data  # Read from Provider
    except Exception as e:
        logging.error(f"Error polling provider: {e}")
        return None
```

### 2.2 `_raw_to_text()` Method

**REQUIRED**: MUST implement `async def _raw_to_text()` to convert raw data to Message.

```python
async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
    """
    Process raw input to generate a timestamped message.
    
    Converts raw provider data into a formatted Message object
    that can be consumed by the Fuser.
    
    Parameters
    ----------
    raw_input : Optional[dict]
        Raw input from provider
    
    Returns
    -------
    Optional[Message]
        A timestamped message containing the processed input, or None
    """
    if not raw_input:
        return None
    
    # Process and format raw data
    formatted_text = self._format_data(raw_input)
    
    if formatted_text:
        return Message(timestamp=time.time(), message=formatted_text)
    return None
```

### 2.3 `formatted_latest_buffer()` Method

**REQUIRED**: MUST implement `formatted_latest_buffer()` to format buffer for Fuser.

```python
def formatted_latest_buffer(self) -> Optional[str]:
    """
    Format and clear the latest buffer contents.
    
    Formats the most recent message with descriptor and class name,
    adds it to the IO provider, then clears the buffer.
    
    Returns
    -------
    Optional[str]
        Formatted string of buffer contents or None if buffer is empty
    """
    if len(self.messages) == 0:
        return None
    
    latest_message = self.messages[-1]
    
    result = (
        f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
        f"{latest_message.message}\n// END\n"
    )
    
    self.io_provider.add_input(
        self.__class__.__name__,
        latest_message.message,
        latest_message.timestamp
    )
    self.messages = []  # Clear buffer
    
    return result
```

## 3. Complete Template

```python
import asyncio
import logging
import time
from typing import Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.example_provider import ExampleProvider
from providers.io_provider import IOProvider


class Example(FuserInput[SensorConfig, Optional[dict]]):
    """
    Example Sensor.
    
    Reads data from Example Provider and converts to text format for LLM.
    """
    
    def __init__(self, config: SensorConfig):
        """
        Initialize the Example Sensor.
        
        Parameters
        ----------
        config : SensorConfig
            Sensor configuration.
        """
        super().__init__(config)
        
        # Initialize Provider (singleton)
        self.provider = ExampleProvider()
        
        # Initialize IO Provider
        self.io_provider = IOProvider()
        
        # Message buffer
        self.messages: list[Message] = []
        
        # Descriptor for LLM (used in formatted_latest_buffer)
        self.descriptor_for_LLM = "Example Sensor"
    
    async def _poll(self) -> Optional[dict]:
        """
        Poll for new messages from the Provider.
        
        Returns
        -------
        Optional[dict]
            The next message from the provider if available, None otherwise
        """
        await asyncio.sleep(0.5)  # Prevent excessive CPU usage
        
        try:
            return self.provider.data  # Read from Provider
        except Exception as e:
            logging.error(f"Error polling provider: {e}")
            return None
    
    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        """
        Process raw input to generate a timestamped message.
        
        Parameters
        ----------
        raw_input : Optional[dict]
            Raw input from provider
        
        Returns
        -------
        Optional[Message]
            A timestamped message containing the processed input, or None
        """
        if not raw_input:
            return None
        
        logging.debug(f"Example sensor: {raw_input}")
        
        # Process and format raw data
        formatted_text = self._format_data(raw_input)
        
        if formatted_text:
            return Message(timestamp=time.time(), message=formatted_text)
        return None
    
    def _format_data(self, raw_data: dict) -> Optional[str]:
        """
        Format raw data into text string.
        
        Parameters
        ----------
        raw_data : dict
            Raw data from provider
        
        Returns
        -------
        Optional[str]
            Formatted text string, or None if data is invalid
        """
        # Extract and format data
        try:
            field1 = raw_data.get("field1", "unknown")
            field2 = raw_data.get("field2", "unknown")
            return f"Field1 is {field1}, Field2 is {field2}"
        except Exception as e:
            logging.error(f"Error formatting data: {e}")
            return None
    
    async def raw_to_text(self, raw_input: Optional[dict]):
        """
        Update message buffer.
        
        Parameters
        ----------
        raw_input : Optional[dict]
            Raw input to be processed
        """
        pending_message = await self._raw_to_text(raw_input)
        
        if pending_message is not None:
            self.messages.append(pending_message)
    
    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.
        
        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None
        
        latest_message = self.messages[-1]
        
        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )
        
        self.io_provider.add_input(
            self.__class__.__name__,
            latest_message.message,
            latest_message.timestamp
        )
        self.messages = []  # Clear buffer
        
        return result
```

## 4. Provider Usage Patterns

### 4.1 Pattern 1: Direct Property Access (Most Common)

```python
class Gps(FuserInput[SensorConfig, Optional[dict]]):
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.gps = GpsProvider()  # Singleton instance
    
    async def _poll(self) -> Optional[dict]:
        return self.gps.data  # Direct property access
```

### 4.2 Pattern 2: Method Call

```python
class LocationsInput(FuserInput[SensorConfig, Optional[str]]):
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.locations_provider = LocationsProvider(...)
    
    async def _poll(self) -> Optional[str]:
        locations = self.locations_provider.get_all_locations()  # Method call
        return formatted_locations
```

### 4.3 Pattern 3: Callback Registration (Async Events)

```python
class VLMGemini(FuserInput[VLMGeminiConfig, Optional[str]]):
    def __init__(self, config: VLMGeminiConfig):
        super().__init__(config)
        self.vlm_provider = VLMGeminiProvider(...)
        
        # Register callback for async events
        self.vlm_provider.register_message_callback(self._on_vlm_result)
        self._latest_result: Optional[str] = None
    
    def _on_vlm_result(self, result: str):
        """Callback for VLM results."""
        self._latest_result = result
    
    async def _poll(self) -> Optional[str]:
        result = self._latest_result
        self._latest_result = None  # Clear after reading
        return result
```

## 5. Data Formatting Best Practices

### 5.1 Format for LLM Consumption

Format data in a way that's natural for LLM to understand:

```python
def _format_data(self, raw_data: dict) -> Optional[str]:
    """Format GPS data for LLM."""
    lat = raw_data.get("gps_lat", 0)
    lon = raw_data.get("gps_lon", 0)
    alt = raw_data.get("gps_alt", 0)
    
    # Natural language format
    return f"Your GPS location is {lat}° North, {lon}° East at {alt}m altitude."
```

### 5.2 Handle Missing or Invalid Data

```python
def _format_data(self, raw_data: dict) -> Optional[str]:
    """Format data with validation."""
    if not raw_data:
        return None
    
    # Validate required fields
    required_fields = ["field1", "field2"]
    if not all(field in raw_data for field in required_fields):
        logging.warning("Missing required fields in raw data")
        return None
    
    # Format valid data
    return f"Field1: {raw_data['field1']}, Field2: {raw_data['field2']}"
```

## 6. Common Patterns

### 6.1 Simple Polling Sensor

```python
class SimpleSensor(FuserInput[SensorConfig, Optional[dict]]):
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.provider = SimpleProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "Simple Sensor"
        self.io_provider = IOProvider()
    
    async def _poll(self) -> Optional[dict]:
        await asyncio.sleep(0.5)
        return self.provider.data
    
    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        if not raw_input:
            return None
        return Message(timestamp=time.time(), message=str(raw_input))
    
    def formatted_latest_buffer(self) -> Optional[str]:
        if not self.messages:
            return None
        msg = self.messages[-1]
        self.messages = []
        return f"\nINPUT: {self.descriptor_for_LLM}\n// START\n{msg.message}\n// END\n"
```

### 6.2 Sensor with Custom Config

```python
from pydantic import Field
from inputs.base import SensorConfig

class CustomSensorConfig(SensorConfig):
    """Configuration for Custom Sensor."""
    custom_param: str = Field(default="default", description="Custom parameter")

class CustomSensor(FuserInput[CustomSensorConfig, Optional[dict]]):
    def __init__(self, config: CustomSensorConfig):
        super().__init__(config)
        self.custom_param = config.custom_param
        # ... rest of implementation
```

## 7. Review Checklist

When reviewing your Sensor implementation:

- [ ] **Inheritance**: Inherits from `FuserInput[ConfigType, R]`
- [ ] **File naming**: `{name}.py` in `src/inputs/plugins/`
- [ ] **Class naming**: `{Name}` (PascalCase)
- [ ] **Super init**: Calls `super().__init__(config)`
- [ ] **Provider init**: Initializes Provider(s) in `__init__`
- [ ] **Message buffer**: Initializes `self.messages: list[Message] = []`
- [ ] **Descriptor**: Sets `self.descriptor_for_LLM`
- [ ] **IO Provider**: Initializes `self.io_provider = IOProvider()`
- [ ] **Poll method**: Implements `async def _poll()`
- [ ] **Raw to text**: Implements `async def _raw_to_text()`
- [ ] **Formatted buffer**: Implements `formatted_latest_buffer()`
- [ ] **Error handling**: Proper exception handling
- [ ] **Logging**: Appropriate logging at debug/info/warning/error levels
- [ ] **Documentation**: Complete docstrings for class and methods
- [ ] **Type hints**: Proper type annotations

## 8. Reference Examples

- `src/inputs/plugins/gps.py`: GPS sensor with Provider data access
- `src/inputs/plugins/odom.py`: Odometry sensor
- `src/inputs/plugins/vlm_gemini.py`: VLM sensor with callback pattern
- `src/inputs/plugins/amcl_localization_input.py`: Localization sensor

## 9. Anti-patterns to Avoid

### ❌ Don't: Access Provider without initialization

```python
# WRONG: Provider not initialized
class BadSensor(FuserInput[...]):
    async def _poll(self):
        return self.provider.data  # AttributeError if not initialized
```

### ❌ Don't: Skip message buffer management

```python
# WRONG: No buffer management
class BadSensor(FuserInput[...]):
    async def raw_to_text(self, raw_input):
        # Missing: self.messages.append(...)
        pass
```

### ✅ Do: Proper initialization and buffer management

```python
# CORRECT: Proper initialization and buffer
class GoodSensor(FuserInput[...]):
    def __init__(self, config):
        super().__init__(config)
        self.provider = Provider()
        self.messages: list[Message] = []
        self.io_provider = IOProvider()
        self.descriptor_for_LLM = "Good Sensor"
    
    async def raw_to_text(self, raw_input):
        message = await self._raw_to_text(raw_input)
        if message:
            self.messages.append(message)
```
