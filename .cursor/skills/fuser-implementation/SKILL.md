---
name: fuser-implementation
description: Rules and guidelines for implementing Fuser in OM Cortex Runtime
---

# Fuser Implementation Guide

This document provides rules, naming conventions, and structural guidelines for implementing Fuser in OM Cortex Runtime. Use this guide to review and refactor your Fuser implementations.

## Quick Checklist

Before submitting your Fuser code, verify:

- [ ] Inherits from `Fuser` class
- [ ] File location: `src/fuser/__init__.py` (usually)
- [ ] Implements `fuse()` method
- [ ] Combines system prompts, inputs, and actions
- [ ] Records timing in IOProvider
- [ ] Proper error handling and logging
- [ ] Complete docstrings

## 1. Core Structure

### 1.1 Base Class

**REQUIRED**: Fuser MUST inherit from `Fuser` class.

```python
from fuser import Fuser
from runtime.single_mode.config import RuntimeConfig

class CustomFuser(Fuser):
    """
    Custom Fuser implementation.
    
    Combines multiple inputs into a single formatted prompt.
    """
    pass
```

**Note**: In most cases, the default `Fuser` class is sufficient and doesn't need customization. Only implement a custom Fuser if you need special prompt formatting logic.

### 1.2 File Location

- **File path**: Usually `src/fuser/__init__.py` (default implementation)
- **Custom Fuser**: `src/fuser/custom_fuser.py` (if needed)

## 2. Required Methods

### 2.1 `fuse()` Method

**REQUIRED**: MUST implement `fuse()` to combine inputs into a single prompt.

```python
def fuse(self, inputs: list[Sensor], finished_promises: list[T.Any]) -> str:
    """
    Combine all inputs into a single formatted prompt string.
    
    Integrates system prompts, input buffers, action descriptions, and
    command prompts into a structured format for LLM processing.
    
    Parameters
    ----------
    inputs : list[Sensor]
        List of agent input objects containing latest input buffers.
    finished_promises : list[Any]
        List of completed promises from previous actions.
    
    Returns
    -------
    str
        Fused prompt string combining all inputs and context.
    """
    # Implementation
    pass
```

## 3. Complete Template

```python
import logging
import time
import typing as T

from actions import describe_action
from inputs.base import Sensor
from providers.io_provider import IOProvider
from runtime.single_mode.config import RuntimeConfig


class CustomFuser(Fuser):
    """
    Custom Fuser implementation.
    
    Combines multiple agent inputs into a single formatted prompt.
    
    Responsible for integrating system prompts, input streams, action descriptions,
    and command prompts into a coherent format for LLM processing.
    
    Parameters
    ----------
    config : RuntimeConfig
        Runtime configuration containing system prompts and agent actions.
    """
    
    def __init__(self, config: RuntimeConfig):
        """
        Initialize the Fuser with runtime configuration.
        
        Parameters
        ----------
        config : RuntimeConfig
            Runtime configuration object.
        """
        self.config = config
        self.io_provider = IOProvider()
    
    def fuse(self, inputs: list[Sensor], finished_promises: list[T.Any]) -> str:
        """
        Combine all inputs into a single formatted prompt string.
        
        Integrates system prompts, input buffers, action descriptions, and
        command prompts into a structured format for LLM processing.
        
        Parameters
        ----------
        inputs : list[Sensor]
            List of agent input objects containing latest input buffers.
        finished_promises : list[Any]
            List of completed promises from previous actions.
        
        Returns
        -------
        str
            Fused prompt string combining all inputs and context.
        """
        # Record the timestamp of the input
        self.io_provider.fuser_start_time = time.time()
        
        # Collect formatted input buffers
        input_strings = [input.formatted_latest_buffer() for input in inputs]
        logging.debug(f"InputMessageArray: {input_strings}")
        
        # Build system prompt
        system_prompt = self._build_system_prompt()
        
        # Combine inputs
        inputs_fused = " ".join([s for s in input_strings if s is not None])
        
        # Build actions description
        actions_fused = self._build_actions_description()
        
        # Build question prompt
        question_prompt = "What will you do? Actions:"
        
        # Combine everything
        fused_prompt = (
            f"{system_prompt}\n\n"
            f"AVAILABLE INPUTS:\n{inputs_fused}\n"
            f"AVAILABLE ACTIONS:\n\n{actions_fused}\n\n"
            f"{question_prompt}"
        )
        
        logging.debug(f"FINAL PROMPT: {fused_prompt}")
        
        # Record in IO provider
        self._record_to_io_provider(system_prompt, inputs_fused, actions_fused, question_prompt)
        
        # Record the timestamp of the output
        self.io_provider.fuser_end_time = time.time()
        
        return fused_prompt
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt from config.
        
        Returns
        -------
        str
            System prompt string
        """
        system_prompt = "\nBASIC CONTEXT:\n" + self.config.system_prompt_base + "\n"
        
        # Add governance/laws if not in inputs
        # (This is a simplified check - actual implementation may be more complex)
        if "Universal Laws" not in system_prompt:
            system_prompt += "\nLAWS:\n" + self.config.system_governance
        
        # Add examples if available
        if self.config.system_prompt_examples:
            system_prompt += "\n\nEXAMPLES:\n" + self.config.system_prompt_examples
        
        return system_prompt
    
    def _build_actions_description(self) -> str:
        """
        Build the actions description from config.
        
        Returns
        -------
        str
            Actions description string
        """
        actions_fused = ""
        
        for action in self.config.agent_actions:
            desc = describe_action(
                action.name, action.llm_label, action.exclude_from_prompt
            )
            if desc:
                actions_fused += desc + "\n\n"
        
        return actions_fused
    
    def _record_to_io_provider(
        self,
        system_prompt: str,
        inputs_fused: str,
        actions_fused: str,
        question_prompt: str
    ):
        """
        Record fused components to IO provider.
        
        Parameters
        ----------
        system_prompt : str
            System prompt string
        inputs_fused : str
            Combined inputs string
        actions_fused : str
            Actions description string
        question_prompt : str
            Question prompt string
        """
        self.io_provider.set_fuser_system_prompt(system_prompt)
        self.io_provider.set_fuser_inputs(inputs_fused)
        self.io_provider.set_fuser_available_actions(
            f"AVAILABLE ACTIONS:\n{actions_fused}\n\n{question_prompt}"
        )
```

## 4. Default Fuser Structure

The default Fuser implementation follows this structure:

```
1. System Prompt
   - BASIC CONTEXT: system_prompt_base
   - LAWS: system_governance (if not in inputs)
   - EXAMPLES: system_prompt_examples (if available)

2. Available Inputs
   - Combined formatted buffers from all sensors

3. Available Actions
   - Descriptions of all available actions

4. Question Prompt
   - "What will you do? Actions:"
```

## 5. Customization Patterns

### 5.1 Custom Prompt Formatting

```python
class CustomFormattedFuser(Fuser):
    def fuse(self, inputs: list[Sensor], finished_promises: list[T.Any]) -> str:
        # Custom formatting logic
        system_prompt = self._custom_system_prompt()
        inputs_section = self._custom_inputs_format(inputs)
        actions_section = self._custom_actions_format()
        
        return f"{system_prompt}\n{inputs_section}\n{actions_section}"
    
    def _custom_system_prompt(self) -> str:
        # Custom system prompt building
        pass
    
    def _custom_inputs_format(self, inputs: list[Sensor]) -> str:
        # Custom inputs formatting
        pass
    
    def _custom_actions_format(self) -> str:
        # Custom actions formatting
        pass
```

### 5.2 Conditional Prompt Building

```python
class ConditionalFuser(Fuser):
    def fuse(self, inputs: list[Sensor], finished_promises: list[T.Any]) -> str:
        # Check conditions and build prompt accordingly
        if self._has_vision_input(inputs):
            system_prompt = self._build_vision_aware_prompt()
        else:
            system_prompt = self._build_standard_prompt()
        
        # Continue with fusion
        pass
    
    def _has_vision_input(self, inputs: list[Sensor]) -> bool:
        # Check if any input is vision-related
        return any("vision" in input.__class__.__name__.lower() for input in inputs)
```

### 5.3 Multi-Mode Fuser

```python
class MultiModeFuser(Fuser):
    def __init__(self, config: RuntimeConfig):
        super().__init__(config)
        self.mode = config.mode  # e.g., "conversation", "autonomous_navigation"
    
    def fuse(self, inputs: list[Sensor], finished_promises: list[T.Any]) -> str:
        # Build mode-specific prompt
        if self.mode == "conversation":
            return self._fuse_conversation_mode(inputs, finished_promises)
        elif self.mode == "autonomous_navigation":
            return self._fuse_navigation_mode(inputs, finished_promises)
        else:
            return super().fuse(inputs, finished_promises)
    
    def _fuse_conversation_mode(self, inputs, finished_promises):
        # Conversation-specific fusion
        pass
    
    def _fuse_navigation_mode(self, inputs, finished_promises):
        # Navigation-specific fusion
        pass
```

## 6. Timing and IO Provider Integration

### 6.1 Timing Recording

**REQUIRED**: MUST record timing in IOProvider.

```python
def fuse(self, inputs: list[Sensor], finished_promises: list[T.Any]) -> str:
    # Record start time
    self.io_provider.fuser_start_time = time.time()
    
    # ... fusion logic ...
    
    # Record end time
    self.io_provider.fuser_end_time = time.time()
    
    return fused_prompt
```

### 6.2 IO Provider Recording

**REQUIRED**: MUST record fused components to IOProvider.

```python
def fuse(self, inputs: list[Sensor], finished_promises: list[T.Any]) -> str:
    # ... build components ...
    
    # Record to IO provider
    self.io_provider.set_fuser_system_prompt(system_prompt)
    self.io_provider.set_fuser_inputs(inputs_fused)
    self.io_provider.set_fuser_available_actions(actions_fused)
    
    return fused_prompt
```

## 7. Review Checklist

When reviewing your Fuser implementation:

- [ ] **Inheritance**: Inherits from `Fuser` class
- [ ] **Init**: Initializes with `RuntimeConfig` and `IOProvider`
- [ ] **Fuse method**: Implements `fuse()` method
- [ ] **System prompt**: Builds system prompt from config
- [ ] **Inputs**: Collects and formats input buffers
- [ ] **Actions**: Builds actions description
- [ ] **Timing**: Records `fuser_start_time` and `fuser_end_time`
- [ ] **IO Provider**: Records system prompt, inputs, and actions
- [ ] **Error handling**: Handles None inputs gracefully
- [ ] **Logging**: Appropriate logging at debug/info levels
- [ ] **Documentation**: Complete docstrings for class and methods
- [ ] **Type hints**: Proper type annotations

## 8. Reference Examples

- `src/fuser/__init__.py`: Default Fuser implementation

## 9. When to Customize Fuser

**Most cases**: Use the default Fuser implementation.

**Customize only if**:
- You need special prompt formatting
- You need mode-specific prompt structures
- You need conditional prompt building based on inputs
- You need to integrate additional context (e.g., finished_promises)

## 10. Anti-patterns to Avoid

### ❌ Don't: Skip timing recording

```python
# WRONG: No timing
def fuse(self, inputs, finished_promises):
    # Missing: self.io_provider.fuser_start_time = time.time()
    fused_prompt = self._build_prompt(inputs)
    # Missing: self.io_provider.fuser_end_time = time.time()
    return fused_prompt
```

### ❌ Don't: Ignore None inputs

```python
# WRONG: May crash on None
def fuse(self, inputs, finished_promises):
    input_strings = [input.formatted_latest_buffer() for input in inputs]
    inputs_fused = " ".join(input_strings)  # May fail if None in list
```

### ✅ Do: Record timing and handle None

```python
# CORRECT: Proper timing and None handling
def fuse(self, inputs, finished_promises):
    self.io_provider.fuser_start_time = time.time()
    
    input_strings = [input.formatted_latest_buffer() for input in inputs]
    inputs_fused = " ".join([s for s in input_strings if s is not None])
    
    fused_prompt = self._build_prompt(inputs_fused)
    
    self.io_provider.fuser_end_time = time.time()
    return fused_prompt
```
