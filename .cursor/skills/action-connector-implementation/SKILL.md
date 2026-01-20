---
name: action-connector-implementation
description: Rules and guidelines for implementing Action Connectors in OM Cortex Runtime
---

# Action Connector Implementation Guide

This document provides rules, naming conventions, and structural guidelines for implementing Action Connectors in OM Cortex Runtime. Use this guide to review and refactor your Action Connector implementations.

## Quick Checklist

Before submitting your Action Connector code, verify:

- [ ] Inherits from `ActionConnector[ConfigType, InputType]`
- [ ] File location: `src/actions/{action_name}/connector/{protocol}.py`
- [ ] Class name: `{ActionName}Connector` (PascalCase)
- [ ] Config class: `{ActionName}Config` inheriting from `ActionConfig`
- [ ] Interface file: `src/actions/{action_name}/interface.py` exists
- [ ] Implements `async def connect()` method
- [ ] Separates business logic from provider communication
- [ ] Proper error handling and logging
- [ ] Complete docstrings

## 1. Core Structure

### 1.1 Base Class Inheritance

**REQUIRED**: All Action Connectors MUST inherit from `ActionConnector[ConfigType, InputType]`.

```python
from actions.base import ActionConfig, ActionConnector
from actions.example_action.interface import ExampleActionInput

class ExampleActionConfig(ActionConfig):
    """Configuration for Example Action connector."""
    pass

class ExampleActionConnector(
    ActionConnector[ExampleActionConfig, ExampleActionInput]
):
    """Example Action connector."""
    pass
```

### 1.2 File Location and Structure

**REQUIRED**: Action Connector MUST follow this structure:

```
src/actions/
└── {action_name}/
    ├── connector/
    │   └── {connector_name}.py    # Connector implementation
    └── interface.py                # Input interface definition
```

**Connector naming**: Use descriptive names based on implementation approach:
- Protocol-based: `ros2.py`, `zenoh.py` (if using specific protocols)
- Platform-based: `unitree_go2_nav.py`, `unitree_g1_nav.py` (if robot-specific)
- Service-based: `elevenlabs_tts.py`, `riva_tts.py` (if service-specific)
- SDK-based: `unitree_sdk.py`, `serial_arduino.py` (if using specific SDKs)
- Generic: `{action_name}_connector.py` (if no specific naming needed)

**Examples**:
- `src/actions/navigate_location/connector/unitree_go2_nav.py` → `UnitreeGo2NavConnector`
- `src/actions/speak/connector/elevenlabs_tts.py` → `ElevenLabsTTSConnector`
- `src/actions/move/connector/unitree_sdk.py` → `UnitreeSDKConnector`

### 1.3 Naming Conventions

- **Class name**: `{ActionName}Connector` or `{SpecificName}Connector` (PascalCase)
- **Config class**: `{ActionName}Config` or `{SpecificName}Config` (PascalCase)
- **File name**: `{connector_name}.py` (snake_case, descriptive of implementation)
- **Directory**: `{action_name}` (snake_case)

**Examples**:
- `NavigateLocationInput` → `UnitreeGo2NavConnector` (in `unitree_go2_nav.py`)
- `SpeakInput` → `ElevenLabsTTSConnector` (in `elevenlabs_tts.py`)
- `MoveInput` → `UnitreeSDKConnector` (in `unitree_sdk.py`)

## 2. Interface Definition

### 2.1 Interface File Structure

**REQUIRED**: Each action MUST have an `interface.py` file.

```python
# src/actions/example_action/interface.py
from dataclasses import dataclass
from actions.base import Interface

@dataclass
class ExampleActionInput:
    """
    Input payload for example action.
    
    Parameters
    ----------
    action : str
        Action command or data.
    """
    action: str

@dataclass
class ExampleAction(Interface[ExampleActionInput, ExampleActionInput]):
    """
    Example action interface.
    
    Description of what this action does.
    """
    input: ExampleActionInput
    output: ExampleActionInput
```

## 3. Configuration Class

### 3.1 Config Structure

**REQUIRED**: Action Connector MUST have a Config class.

```python
from pydantic import Field
from actions.base import ActionConfig

class ExampleActionConfig(ActionConfig):
    """
    Configuration for Example Action connector.
    
    Parameters
    ----------
    config_param : str
        Configuration parameter.
    timeout : int
        Timeout in seconds.
    """
    
    config_param: str = Field(
        default="default_value",
        description="Configuration parameter"
    )
    timeout: int = Field(
        default=5,
        description="Timeout in seconds"
    )
```

## 4. Connector Implementation

### 4.1 Basic Structure

```python
import logging
from actions.base import ActionConfig, ActionConnector
from actions.example_action.interface import ExampleActionInput
from providers.example_provider import ExampleProvider

class ExampleActionConfig(ActionConfig):
    """Configuration for Example Action connector."""
    pass

class ExampleActionConnector(
    ActionConnector[ExampleActionConfig, ExampleActionInput]
):
    """
    Example Action connector.
    
    Description of what this connector does.
    """
    
    def __init__(self, config: ExampleActionConfig):
        """
        Initialize the ExampleActionConnector.
        
        Parameters
        ----------
        config : ExampleActionConfig
            Configuration for the action connector.
        """
        super().__init__(config)
        
        # Initialize Providers
        self.example_provider = ExampleProvider(...)
        logging.info("ExampleActionConnector initialized")
    
    async def connect(self, output_interface: ExampleActionInput) -> None:
        """
        Connect the input protocol to the example action.
        
        Parameters
        ----------
        output_interface : ExampleActionInput
            The input protocol containing the action details.
        """
        # Implementation
        pass
```

### 4.2 Connect Method Implementation

**REQUIRED**: MUST implement `async def connect()` method.

```python
async def connect(self, output_interface: ExampleActionInput) -> None:
    """
    Execute the action.
    
    This method should:
    1. Parse and validate input
    2. Transform data if needed
    3. Call Provider methods for actual communication/control
    4. Handle errors appropriately
    """
    # Step 1: Parse input
    action_data = output_interface.action
    
    # Step 2: Business logic (validation, transformation)
    if not self._validate_input(action_data):
        logging.warning(f"Invalid input: {action_data}")
        return
    
    transformed_data = self._transform_data(action_data)
    
    # Step 3: Provider call (actual communication)
    try:
        self.example_provider.execute_action(transformed_data)
        logging.info(f"Action executed: {action_data}")
    except Exception as e:
        logging.error(f"Error executing action: {e}")
```

## 5. Responsibility Separation

### 5.1 Business Logic (Action Connector)

**Action Connector handles**:
- Input parsing: "go to kitchen" → "kitchen"
- Data transformation: location name → message format (e.g., PoseStamped, custom message)
- Validation: Check if location exists, validate coordinates
- Error handling: Handle missing locations, invalid inputs

### 5.2 Provider Communication (Provider)

**Provider handles**:
- Communication protocol (ROS2, Zenoh, HTTP, WebSocket, etc.)
- Hardware communication
- State management
- Message serialization/deserialization

### 5.3 Example: Proper Separation

```python
class NavigateLocationConnector(
    ActionConnector[NavigateLocationConfig, NavigateLocationInput]
):
    def __init__(self, config: NavigateLocationConfig):
        super().__init__(config)
        self.location_provider = LocationProvider(...)
        self.navigation_provider = NavigationProvider()
    
    async def connect(self, input: NavigateLocationInput) -> None:
        # BUSINESS LOGIC: Parse input
        label = input.action.lower().strip()
        for prefix in ["go to the ", "go to ", "navigate to "]:
            if label.startswith(prefix):
                label = label[len(prefix):].strip()
                break
        
        # BUSINESS LOGIC: Lookup location
        loc = self.location_provider.get_location(label)
        if loc is None:
            logging.warning(f"Location '{label}' not found")
            return
        
        # BUSINESS LOGIC: Transform data
        pose = loc.get("pose") or {}
        goal_pose = PoseStamped(...)  # Create ROS2 message
        
        # PROVIDER CALL: Actual ROS2 communication
        self.navigation_provider.publish_goal_pose(goal_pose, label)
```

## 6. Complete Template

```python
import logging
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.example_action.interface import ExampleActionInput
from providers.example_provider import ExampleProvider


class ExampleActionConfig(ActionConfig):
    """
    Configuration for Example Action connector.
    
    Parameters
    ----------
    config_param : str
        Configuration parameter.
    timeout : int
        Timeout in seconds.
    """
    
    config_param: str = Field(
        default="default_value",
        description="Configuration parameter"
    )
    timeout: int = Field(
        default=5,
        description="Timeout in seconds"
    )


class ExampleActionConnector(
    ActionConnector[ExampleActionConfig, ExampleActionInput]
):
    """
    Example Action connector.
    
    Description of what this connector does.
    """
    
    def __init__(self, config: ExampleActionConfig):
        """
        Initialize the ExampleActionConnector.
        
        Parameters
        ----------
        config : ExampleActionConfig
            Configuration for the action connector.
        """
        super().__init__(config)
        
        # Initialize Providers
        self.example_provider = ExampleProvider(
            param=self.config.config_param
        )
        logging.info("ExampleActionConnector initialized")
    
    async def connect(self, output_interface: ExampleActionInput) -> None:
        """
        Connect the input protocol to the example action.
        
        Parameters
        ----------
        output_interface : ExampleActionInput
            The input protocol containing the action details.
        """
        # Step 1: Parse and validate input
        action_data = output_interface.action
        
        if not action_data:
            logging.warning("Empty action data received")
            return
        
        # Step 2: Business logic - transform data
        try:
            transformed_data = self._transform_data(action_data)
        except Exception as e:
            logging.error(f"Failed to transform data: {e}")
            return
        
        # Step 3: Provider call - actual communication/control
        try:
            self.example_provider.execute_action(transformed_data)
            logging.info(f"Action executed successfully: {action_data}")
        except Exception as e:
            logging.error(f"Error executing action: {e}")
            raise
    
    def _transform_data(self, input_data: str):
        """
        Transform input data for provider.
        
        Parameters
        ----------
        input_data : str
            Raw input data.
        
        Returns
        -------
        Transformed data for provider.
        """
        # Transformation logic
        return transformed_data
    
    def _validate_input(self, input_data: str) -> bool:
        """
        Validate input data.
        
        Parameters
        ----------
        input_data : str
            Input data to validate.
        
        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        # Validation logic
        return True
```

## 7. Common Patterns

### 7.1 Input Parsing Pattern

```python
async def connect(self, input: NavigateLocationInput) -> None:
    # Parse location name from various command formats
    label = input.action.lower().strip()
    
    # Remove common prefixes
    prefixes = ["go to the ", "go to ", "navigate to ", "move to "]
    for prefix in prefixes:
        if label.startswith(prefix):
            label = label[len(prefix):].strip()
            break
    
    # Use cleaned label
    location = self.location_provider.get_location(label)
```

### 7.2 Error Handling Pattern

```python
async def connect(self, input: ExampleActionInput) -> None:
    try:
        # Validate input
        if not self._validate(input):
            logging.warning(f"Invalid input: {input.action}")
            return
        
        # Execute action
        result = self.provider.execute(input.action)
        logging.info(f"Action completed: {result}")
        
    except ValueError as e:
        logging.error(f"Validation error: {e}")
        return
    except ConnectionError as e:
        logging.error(f"Connection error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise
```

### 7.3 Data Transformation Pattern

```python
def _create_message(self, data: dict) -> PoseStamped:
    """Transform dict data to message format (e.g., PoseStamped)."""
    position = data.get("position", {})
    orientation = data.get("orientation", {})
    
    return PoseStamped(
        header=Header(
            stamp=Time(sec=int(time.time()), nanosec=0),
            frame_id="map"
        ),
        pose=Pose(
            position=Point(
                x=float(position.get("x", 0.0)),
                y=float(position.get("y", 0.0)),
                z=float(position.get("z", 0.0))
            ),
            orientation=Quaternion(
                x=float(orientation.get("x", 0.0)),
                y=float(orientation.get("y", 0.0)),
                z=float(orientation.get("z", 0.0)),
                w=float(orientation.get("w", 1.0))
            )
        )
    )
```

## 8. Vendor Code Integration

When porting vendor code algorithms to Action Connectors:

### 8.1 Algorithm Porting

**Principle**: Computation logic goes in Action Connector

```python
class NavigateGPSConnector(ActionConnector[...]):
    def __init__(self, config: ...):
        super().__init__(config)
        self.gps_provider = GpsProvider(...)
        self.odom_provider = OdomProvider(...)
    
    async def connect(self, input: NavigateGPSInput):
        # Ported vendor algorithm: goal_to_xy
        goal_xy = self._goal_to_xy(input.goal_lat, input.goal_lon)
        
        # Ported vendor algorithm: PriorityPD
        control = self._priority_pd_step(goal_xy)
        
        # Provider call: hardware control
        self.unitree_go2_provider.publish_cmd(control)
    
    def _goal_to_xy(self, lat: float, lon: float):
        """Ported from vendor GPS navigation code."""
        # Algorithm implementation
        pass
    
    def _priority_pd_step(self, goal_xy):
        """Ported from vendor PriorityPD controller."""
        # Algorithm implementation
        pass
```

## 9. Review Checklist

When reviewing your Action Connector implementation:

- [ ] **Inheritance**: Inherits from `ActionConnector[ConfigType, InputType]`
- [ ] **File location**: Correct path `src/actions/{action_name}/connector/{protocol}.py`
- [ ] **Class naming**: `{ActionName}Connector` (PascalCase)
- [ ] **Config class**: `{ActionName}Config` inheriting from `ActionConfig`
- [ ] **Interface file**: `interface.py` exists with proper Interface definition
- [ ] **Super init**: Calls `super().__init__(config)`
- [ ] **Provider init**: Initializes Provider(s) in `__init__`
- [ ] **Connect method**: Implements `async def connect()`
- [ ] **Business logic**: Separates business logic from provider communication
- [ ] **Error handling**: Proper exception handling
- [ ] **Logging**: Appropriate logging at info/warning/error levels
- [ ] **Documentation**: Complete docstrings for class and methods
- [ ] **Type hints**: Proper type annotations
- [ ] **Input validation**: Validates input before processing
- [ ] **Data transformation**: Transforms data appropriately

## 10. Reference Examples

- `src/actions/navigate_location/connector/unitree_go2_nav.py`: Navigation connector with location lookup
- `src/actions/speak/connector/elevenlabs_tts.py`: Speech action with TTS service
- `src/actions/move/connector/unitree_sdk.py`: Movement action with SDK

## 11. Anti-patterns to Avoid

### ❌ Don't: Put business logic in Provider

```python
# WRONG: Business logic in Provider
class BadProvider:
    def navigate_to_location(self, user_command: str):
        # Business logic should be in Action Connector
        label = user_command.lower().strip()
        for prefix in ["go to ", "navigate to "]:
            if label.startswith(prefix):
                label = label[len(prefix):]
                break
        # ...
```

### ❌ Don't: Skip input validation

```python
# WRONG: No validation
async def connect(self, input: ExampleActionInput):
    self.provider.execute(input.action)  # May fail if invalid
```

### ✅ Do: Validate and handle errors

```python
# CORRECT: Validate and handle errors
async def connect(self, input: ExampleActionInput):
    if not self._validate(input.action):
        logging.warning("Invalid input")
        return
    try:
        self.provider.execute(input.action)
    except Exception as e:
        logging.error(f"Execution failed: {e}")
        raise
```
