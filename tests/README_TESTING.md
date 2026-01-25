# Testing Guide

이 문서는 OM Cortex Runtime의 테스트 실행 방법을 설명합니다.

## 테스트 파일 구조

```
tests/
├── providers/
│   ├── test_realsense_camera_provider.py
│   ├── test_segmentation_provider.py
│   ├── test_pointcloud_provider.py
│   └── test_bev_occupancy_grid_provider.py
└── backgrounds/
    ├── test_realsense_camera_bg.py
    ├── test_segmentation_bg.py
    ├── test_pointcloud_bg.py
    └── test_bev_occupancy_grid_bg.py
```

## 테스트 실행 방법

### 1. 전체 테스트 실행

```bash
# 모든 테스트 실행
pytest

# 또는
uv run pytest
```

### 2. 특정 Provider 테스트 실행

```bash
# RealSenseCameraProvider 테스트
pytest tests/providers/test_realsense_camera_provider.py

# SegmentationProvider 테스트
pytest tests/providers/test_segmentation_provider.py

# PointCloudProvider 테스트
pytest tests/providers/test_pointcloud_provider.py

# BEVOccupancyGridProvider 테스트
pytest tests/providers/test_bev_occupancy_grid_provider.py
```

### 3. 특정 Background 테스트 실행

```bash
# RealSenseCameraBg 테스트
pytest tests/backgrounds/test_realsense_camera_bg.py

# SegmentationBg 테스트
pytest tests/backgrounds/test_segmentation_bg.py

# PointCloudBg 테스트
pytest tests/backgrounds/test_pointcloud_bg.py

# BEVOccupancyGridBg 테스트
pytest tests/backgrounds/test_bev_occupancy_grid_bg.py
```

### 4. 특정 테스트 함수 실행

```bash
# 특정 테스트 함수만 실행
pytest tests/providers/test_realsense_camera_provider.py::test_initialization

# 여러 테스트 함수 실행
pytest tests/providers/test_realsense_camera_provider.py::test_initialization tests/providers/test_realsense_camera_provider.py::test_singleton_pattern
```

### 5. Verbose 모드로 실행

```bash
# 상세한 출력과 함께 실행
pytest -v

# 매우 상세한 출력
pytest -vv

# 테스트 이름 표시
pytest -v -s
```

### 6. 특정 패턴의 테스트만 실행

```bash
# 모든 Provider 테스트 실행
pytest tests/providers/

# 모든 Background 테스트 실행
pytest tests/backgrounds/

# 특정 이름 패턴의 테스트만 실행
pytest -k "realsense"
pytest -k "segmentation"
pytest -k "pointcloud"
pytest -k "bev"
```

### 7. 커버리지 리포트 생성

```bash
# 커버리지 리포트 생성
pytest --cov=src/providers --cov=src/backgrounds

# HTML 리포트 생성
pytest --cov=src/providers --cov=src/backgrounds --cov-report=html
```

## 테스트 작성 가이드

### Provider 테스트 기본 구조

```python
import pytest
from providers.example_provider import ExampleProvider

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    ExampleProvider.reset()
    yield
    ExampleProvider.reset()

def test_initialization():
    """Test provider initialization."""
    provider = ExampleProvider()
    assert provider._data is None
    assert not provider.running

def test_singleton_pattern():
    """Test singleton pattern."""
    provider1 = ExampleProvider()
    provider2 = ExampleProvider()
    assert provider1 is provider2

def test_start():
    """Test starting the provider."""
    provider = ExampleProvider()
    provider.start()
    assert provider.running
    provider.stop()

def test_stop():
    """Test stopping the provider."""
    provider = ExampleProvider()
    provider.start()
    provider.stop()
    assert not provider.running

def test_data_property():
    """Test data property access."""
    provider = ExampleProvider()
    assert provider.data is None
```

### Background 테스트 기본 구조

```python
import pytest
from unittest.mock import patch, MagicMock
from backgrounds.plugins.example_bg import ExampleBg, ExampleConfig

@pytest.fixture
def config():
    return ExampleConfig()

@patch("backgrounds.plugins.example_bg.ExampleProvider")
def test_background_initialization(mock_provider_class, config):
    """Test background initialization."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance
    
    background = ExampleBg(config=config)
    
    mock_provider_class.assert_called_once()
    mock_provider_instance.start.assert_called_once()
```

## 주의사항

1. **Singleton Reset**: Provider 테스트에서는 `autouse=True` fixture로 singleton을 reset해야 합니다.
2. **Thread Cleanup**: `start()`를 호출한 테스트에서는 반드시 `stop()`을 호출하여 thread를 정리해야 합니다.
3. **Mock 사용**: Background 테스트에서는 Provider를 mock하여 실제 초기화를 방지합니다.

## 예제 테스트 실행

```bash
# RealSenseCameraProvider 전체 테스트
pytest tests/providers/test_realsense_camera_provider.py -v

# SegmentationBg 초기화 테스트만
pytest tests/backgrounds/test_segmentation_bg.py::test_background_initialization -v

# 모든 vision pipeline 관련 테스트
pytest -k "realsense or segmentation or pointcloud or bev" -v
```
