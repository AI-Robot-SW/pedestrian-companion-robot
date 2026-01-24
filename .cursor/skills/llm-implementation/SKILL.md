---
name: llm-implementation
description: Rules and guidelines for implementing LLM plugins in OM Cortex Runtime
---

# LLM Implementation Guide

This document provides rules, naming conventions, and structural guidelines for implementing LLM plugins in OM Cortex Runtime. Use this guide to review and refactor your LLM implementations.

## Quick Checklist

Before submitting your LLM code, verify:

- [ ] Inherits from `LLM[R]` where R is response model type
- [ ] File name: `{name}_llm.py` (snake_case) in `src/llm/plugins/`
- [ ] Class name: `{Name}LLM` (PascalCase)
- [ ] Config class: `{Name}LLMConfig` inheriting from `LLMConfig` (optional)
- [ ] Implements `async def ask()` method
- [ ] Handles function calling if available_actions provided
- [ ] Proper error handling and logging
- [ ] Complete docstrings

## 1. Core Structure

### 1.1 Base Class Inheritance

**REQUIRED**: All LLM implementations MUST inherit from `LLM[R]`.

```python
import typing as T
from pydantic import BaseModel
from llm import LLM, LLMConfig

R = T.TypeVar("R", bound=BaseModel)

class ExampleLLM(LLM[R]):
    """
    Example LLM implementation.
    
    Description of the LLM service.
    """
    pass
```

### 1.2 File Location and Naming

- **File path**: `src/llm/plugins/{name}_llm.py`
- **Class name**: `{Name}LLM` (PascalCase)
- **Config class**: `{Name}LLMConfig` (optional, inherits from `LLMConfig`)

**Examples**:
- `src/llm/plugins/openai_llm.py` → `OpenAILLM`
- `src/llm/plugins/gemini_llm.py` → `GeminiLLM`
- `src/llm/plugins/deepseek_llm.py` → `DeepSeekLLM`

### 1.3 Type Parameter

- **R**: Response model type (must be bound to `BaseModel`)
  - Usually `CortexOutputModel` for function calling
  - Or custom Pydantic model for structured output

## 2. Required Methods

### 2.1 `ask()` Method

**REQUIRED**: MUST implement `async def ask()` to send prompts and receive responses.

```python
async def ask(
    self, prompt: str, messages: T.List[T.Dict[str, str]] = []
) -> T.Optional[R]:
    """
    Execute LLM query and parse response.
    
    Parameters
    ----------
    prompt : str
        The input prompt to send to the model.
    messages : List[Dict[str, str]]
        List of message dictionaries (conversation history).
    
    Returns
    -------
    R or None
        Parsed response matching the output_model structure, or None if
        parsing fails.
    """
    # Implementation
    pass
```

## 3. Complete Template

```python
import logging
import time
import typing as T

import openai
from pydantic import BaseModel

from llm import LLM, LLMConfig
from llm.function_schemas import convert_function_calls_to_actions
from llm.output_model import CortexOutputModel
from providers.avatar_llm_state_provider import AvatarLLMState
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


class ExampleLLM(LLM[R]):
    """
    Example LLM implementation using OpenAI-compatible API.
    
    Handles authentication and response parsing for Example endpoints.
    
    Parameters
    ----------
    config : LLMConfig
        Configuration object containing API settings.
    available_actions : list[AgentAction], optional
        List of available actions for function call generation.
    """
    
    def __init__(
        self,
        config: LLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the Example LLM instance.
        
        Parameters
        ----------
        config : LLMConfig
            Configuration settings for the LLM.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)
        
        # Validate required config
        if not config.api_key:
            raise ValueError("config file missing api_key")
        if not config.model:
            self._config.model = "default-model"
        
        # Initialize API client
        self._client = openai.AsyncOpenAI(
            base_url=config.base_url or "https://api.example.com",
            api_key=config.api_key,
        )
        
        # Initialize history manager (if needed)
        self.history_manager = LLMHistoryManager(self._config, self._client)
    
    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, str]] = []
    ) -> T.Optional[R]:
        """
        Execute LLM query and parse response.
        
        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        messages : List[Dict[str, str]]
            List of message dictionaries to send to the model.
        
        Returns
        -------
        R or None
            Parsed response matching the output_model structure, or None if
            parsing fails.
        """
        try:
            logging.debug(f"Example LLM input: {prompt}")
            logging.debug(f"Example LLM messages: {messages}")
            
            # Record timing
            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)
            
            # Format messages
            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]
            formatted_messages.append({"role": "user", "content": prompt})
            
            # Call API
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=T.cast(T.Any, formatted_messages),
                tools=T.cast(T.Any, self.function_schemas) if self.function_schemas else None,
                tool_choice="auto" if self.function_schemas else None,
                timeout=self._config.timeout,
            )
            
            message = response.choices[0].message
            self.io_provider.llm_end_time = time.time()
            
            # Handle function calls
            if message.tool_calls:
                logging.info(f"Received {len(message.tool_calls)} function calls")
                actions = convert_function_calls_to_actions(
                    message.tool_calls, self._available_actions
                )
                return T.cast(R, CortexOutputModel(actions=actions))
            
            # Handle text response
            if message.content:
                logging.info(f"Received text response: {message.content}")
                # Parse structured output if needed
                return self._parse_response(message.content)
            
            logging.warning("No content or tool calls in response")
            return None
            
        except Exception as e:
            logging.error(f"Error in LLM ask: {e}")
            self.io_provider.llm_end_time = time.time()
            return None
    
    def _parse_response(self, content: str) -> T.Optional[R]:
        """
        Parse text response into structured output.
        
        Parameters
        ----------
        content : str
            Raw text response from LLM
        
        Returns
        -------
        R or None
            Parsed response model, or None if parsing fails
        """
        # Implementation depends on output model type
        # For function calling, usually returns CortexOutputModel
        # For structured output, parse JSON and validate against model
        pass
```

## 4. Function Calling Support

### 4.1 Automatic Function Schema Generation

Function schemas are automatically generated from `available_actions` in `__init__`:

```python
def __init__(self, config: LLMConfig, available_actions: T.Optional[T.List] = None):
    super().__init__(config, available_actions)
    # self.function_schemas is automatically populated
    # if available_actions is provided
```

### 4.2 Handling Function Calls

```python
async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]] = []):
    response = await self._client.chat.completions.create(...)
    message = response.choices[0].message
    
    if message.tool_calls:
        # Convert function calls to actions
        actions = convert_function_calls_to_actions(
            message.tool_calls, self._available_actions
        )
        return T.cast(R, CortexOutputModel(actions=actions))
```

## 5. Common Patterns

### 5.1 OpenAI-Compatible API

```python
class OpenAICompatibleLLM(LLM[R]):
    def __init__(self, config: LLMConfig, available_actions: T.Optional[T.List] = None):
        super().__init__(config, available_actions)
        
        if not config.api_key:
            raise ValueError("config file missing api_key")
        
        self._client = openai.AsyncOpenAI(
            base_url=config.base_url or "https://api.example.com",
            api_key=config.api_key,
        )
        self.history_manager = LLMHistoryManager(self._config, self._client)
    
    @AvatarLLMState.trigger_thinking()
    @LLMHistoryManager.update_history()
    async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]] = []):
        # Implementation
        pass
```

### 5.2 Custom API Client

```python
class CustomAPILLM(LLM[R]):
    def __init__(self, config: LLMConfig, available_actions: T.Optional[T.List] = None):
        super().__init__(config, available_actions)
        
        # Custom API client initialization
        self._client = CustomAPIClient(
            api_key=config.api_key,
            base_url=config.base_url
        )
    
    async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]] = []):
        # Custom API call
        response = await self._client.generate(
            prompt=prompt,
            messages=messages,
            tools=self.function_schemas
        )
        # Parse response
        return self._parse_response(response)
```

### 5.3 Multi-LLM Pattern

```python
class MultiLLM(LLM[R]):
    """
    Orchestrates multiple LLM instances.
    """
    def __init__(self, config: LLMConfig, available_actions: T.Optional[T.List] = None):
        super().__init__(config, available_actions)
        
        # Initialize multiple LLM instances
        self.primary_llm = PrimaryLLM(config, available_actions)
        self.fallback_llm = FallbackLLM(config, available_actions)
    
    async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]] = []):
        try:
            return await self.primary_llm.ask(prompt, messages)
        except Exception as e:
            logging.warning(f"Primary LLM failed: {e}, using fallback")
            return await self.fallback_llm.ask(prompt, messages)
```

## 6. Decorators and State Management

### 6.1 Avatar LLM State

Use `@AvatarLLMState.trigger_thinking()` to trigger thinking state:

```python
from providers.avatar_llm_state_provider import AvatarLLMState

@AvatarLLMState.trigger_thinking()
async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]] = []):
    # Implementation
    pass
```

### 6.2 History Management

Use `@LLMHistoryManager.update_history()` to manage conversation history:

```python
from providers.llm_history_manager import LLMHistoryManager

@LLMHistoryManager.update_history()
async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]] = []):
    # Implementation
    pass
```

### 6.3 Combined Decorators

```python
@AvatarLLMState.trigger_thinking()
@LLMHistoryManager.update_history()
async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]] = []):
    # Implementation with both decorators
    pass
```

## 7. Error Handling

### 7.1 API Error Handling

```python
async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]] = []):
    try:
        response = await self._client.chat.completions.create(...)
        # Process response
    except openai.APIError as e:
        logging.error(f"API error: {e}")
        return None
    except openai.Timeout as e:
        logging.error(f"Request timeout: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None
```

### 7.2 Response Validation

```python
async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]] = []):
    response = await self._client.chat.completions.create(...)
    message = response.choices[0].message
    
    if not message:
        logging.warning("Empty response from LLM")
        return None
    
    # Validate response structure
    if message.tool_calls:
        return self._handle_function_calls(message.tool_calls)
    elif message.content:
        return self._handle_text_response(message.content)
    else:
        logging.warning("No content or tool calls in response")
        return None
```

## 8. Review Checklist

When reviewing your LLM implementation:

- [ ] **Inheritance**: Inherits from `LLM[R]` where R is response model type
- [ ] **File naming**: `{name}_llm.py` in `src/llm/plugins/`
- [ ] **Class naming**: `{Name}LLM` (PascalCase)
- [ ] **Super init**: Calls `super().__init__(config, available_actions)`
- [ ] **Config validation**: Validates required config (api_key, model, etc.)
- [ ] **Client initialization**: Initializes API client properly
- [ ] **History manager**: Initializes LLMHistoryManager if needed
- [ ] **Ask method**: Implements `async def ask()`
- [ ] **Function calling**: Handles function calls if available_actions provided
- [ ] **Decorators**: Uses `@AvatarLLMState.trigger_thinking()` and `@LLMHistoryManager.update_history()` if needed
- [ ] **Error handling**: Proper exception handling
- [ ] **Logging**: Appropriate logging at debug/info/warning/error levels
- [ ] **Timing**: Records `llm_start_time` and `llm_end_time` in io_provider
- [ ] **Documentation**: Complete docstrings for class and methods
- [ ] **Type hints**: Proper type annotations

## 9. Reference Examples

- `src/llm/plugins/openai_llm.py`: OpenAI LLM with function calling
- `src/llm/plugins/gemini_llm.py`: Gemini LLM implementation
- `src/llm/plugins/deepseek_llm.py`: DeepSeek LLM implementation
- `src/llm/plugins/multi_llm.py`: Multi-LLM orchestration

## 10. Anti-patterns to Avoid

### ❌ Don't: Skip config validation

```python
# WRONG: No validation
class BadLLM(LLM[R]):
    def __init__(self, config: LLMConfig, available_actions=None):
        super().__init__(config, available_actions)
        self._client = openai.AsyncOpenAI(api_key=config.api_key)  # May fail
```

### ❌ Don't: Ignore function calls

```python
# WRONG: Ignoring function calls
async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]] = []):
    response = await self._client.chat.completions.create(...)
    message = response.choices[0].message
    return message.content  # Ignores tool_calls
```

### ✅ Do: Validate config and handle function calls

```python
# CORRECT: Validate and handle properly
class GoodLLM(LLM[R]):
    def __init__(self, config: LLMConfig, available_actions=None):
        super().__init__(config, available_actions)
        if not config.api_key:
            raise ValueError("api_key required")
        self._client = openai.AsyncOpenAI(api_key=config.api_key)
    
    async def ask(self, prompt: str, messages: T.List[T.Dict[str, str]] = []):
        response = await self._client.chat.completions.create(...)
        message = response.choices[0].message
        
        if message.tool_calls:
            return self._handle_function_calls(message.tool_calls)
        return self._handle_text_response(message.content)
```
