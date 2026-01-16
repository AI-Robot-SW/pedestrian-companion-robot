# Vendor 코드 통합 설계 (재설계)

이 문서는 KIST의 Interaction_AutonomousNavigation 레포지토리 코드를 OM1 플랫폼에 **완전히 통합**하는 상세 설계를 담고 있습니다.

**통합 대상**:
- `Autonomous_Navigation`: 자율 주행 시스템 (GPS, DWA, BEV 등)
- `LLMagent`: 대화 중심 시스템 (RAG, 일반 대화, 간단한 로봇 제어)

## 핵심 설계 원칙

1. **연산 로직은 OM1으로 포팅**: GPS Planner, DWA Planner의 알고리즘을 OM1 Action Connector로 이동
2. **ROS2 노드는 최소화**: 하드웨어 직접 제어만 ROS2 노드로 유지 (RealSense 드라이버, Unitree Go2 제어)
3. **Provider는 하드웨어 추상화**: GPS 시리얼 포트 직접 읽기, BEV/PointCloud 연산 등
4. **완전한 OM1 통합**: 모든 연산 로직이 OM1 내부에서 실행되어 디버깅과 확장이 용이
5. **Multi Mode 통합**: LLMagent와 Autonomous_Navigation을 OM1 Multi Mode로 통합

## 목차

1. [전체 아키텍처 (Multi Mode 중심)](#1-전체-아키텍처-multi-mode-중심)
2. [Input Sensors (PER-INPUT)](#2-input-sensors-per-input)
3. [Fuser System](#3-fuser-system)
4. [LLM System (AGT-LLM)](#4-llm-system-agt-llm)
5. [Action Connectors (NAV-ACTION)](#5-action-connectors-nav-action)
6. [Providers (SYS-PROVIDER)](#6-providers-sys-provider)
7. [Backgrounds (SYS-BG)](#7-backgrounds-sys-bg)
8. [LLMagent 통합 설계](#8-llmagent-통합-설계)
9. [Multi Mode 구성](#9-multi-mode-구성)
10. [연산 로직 포팅 계획](#10-연산-로직-포팅-계획)
11. [Config 파일 구조](#11-config-파일-구조)
12. [ROS2 Workspace 구조 (최소화)](#12-ros2-workspace-구조-최소화)
13. [구현 단계별 계획](#13-구현-단계별-계획)
14. [기술적 고려사항](#14-기술적-고려사항)
15. [파일 구조 요약](#15-파일-구조-요약)
16. [모듈 매핑 요약](#16-모듈-매핑-요약)
17. [데이터 플로우 상세](#17-데이터-플로우-상세)
18. [모듈 매핑 요약 (LLMagent 포함)](#18-모듈-매핑-요약-llmagent-포함)
19. [다음 단계](#19-다음-단계)

---

## 1. 전체 아키텍처 (Multi Mode 중심)

### 제안 폴더 구조

```
src/
├── ros2_ws/                          # ROS2 workspace (최소한만)
│   ├── src/
│   │   ├── realsense2_camera/       # RealSense 드라이버 (하드웨어 직접 제어)
│   │   └── unitree_go2_bridge/      # Unitree Go2 제어 브리지 (하드웨어 직접 제어)
│   ├── install/
│   ├── build/
│   └── log/
│
└── vendor_algorithms/                # Vendor 연산 로직 (OM1으로 포팅할 소스)
    ├── gps_nav/
    │   ├── controller.py            # PriorityPD, goal_to_xy 등
    │   └── nav_utils.py             # LinearPath, haversine 등
    ├── dwa_nav/
    │   ├── dwa_algorithm.py         # DWA 코스트 계산, 셀 선택
    │   └── distmap_def.py           # CUDA 거리 맵 생성
    ├── pointcloud_xyzrgb/
    │   └── pointcloud_generator.py  # PointCloud 생성 로직
    └── bev_cuda/
        └── bev_generator.py        # BEV Occupancy Grid 생성 로직
```

**이유**:
- `ros2_ws`: 하드웨어 직접 제어만 포함 (RealSense 드라이버, Go2 제어)
- `vendor_algorithms`: 연산 로직 소스 코드 (OM1으로 포팅할 참고용)
- ROS2 의존성 최소화

---

### 시스템 아키텍처 다이어그램 (Multi Mode 중심)

```
┌─────────────────────────────────────────────────────────────┐
│              OpenMind Platform (Multi Mode)                │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Mode Manager (SYS-CORE)                              │ │
│  │  - conversation mode / autonomous_navigation mode     │ │
│  │  - Mode 전환 규칙 관리                                │ │
│  └───────────────────────────────────────────────────────┘ │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  1. Input Sensors (PER-INPUT)                        │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │ │
│  │  │RealSense │  │PointCloud │  │   BEV    │          │ │
│  │  │  Sensor  │  │  Sensor   │  │  Sensor  │          │ │
│  │  │  ASR     │  │  GPS      │  │  Odom   │          │ │
│  │  └────┬─────┘  └─────┬────┘  └─────┬────┘          │ │
│  │       │              │             │                │ │
│  │       └──────┬───────┴─────────────┘                │ │
│  │              │                                      │ │
│  │              ▼                                      │ │
│  │  ┌──────────────────────────────────────┐          │ │
│  │  │  Providers (SYS-PROVIDER)             │          │ │
│  │  │  ┌──────────┐  ┌──────────┐          │          │ │
│  │  │  │RealSense │  │PointCloud│  ...     │          │ │
│  │  │  │ Provider │  │ Provider │          │          │ │
│  │  │  └────┬─────┘  └─────┬────┘          │          │ │
│  │  │       │              │                │          │ │
│  │  │       │              │ (연산 로직 포함) │          │ │
│  │  │       │              ▼                │          │ │
│  │  │       │  ┌──────────────────────┐   │          │ │
│  │  │       │  │ ROS2 Topic (선택적)   │   │          │ │
│  │  │       │  │ RealSense 드라이버만   │   │          │ │
│  │  │       └─▶│ ROS2 Node (최소화)    │   │          │ │
│  │  │          └──────────────────────┘   │          │ │
│  │  └──────────────────────────────────────┘          │ │
│  │              │                                      │ │
│  │              ▼                                      │ │
│  │  ┌──────────────────────────────────────┐          │ │
│  │  │  InputOrchestrator                    │          │ │
│  │  │  - Sensor 데이터 수집 및 관리          │          │ │
│  │  └───────┬──────────────────────────────┘          │ │
│  └──────────┼───────────────────────────────────────┘ │
│             │                                            │
│             ▼                                            │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  2. Fuser System                                      │ │
│  │  ┌──────────────────────────────────────┐            │ │
│  │  │  Fuser                                │            │ │
│  │  │  - Multiple Sensor Inputs →          │            │ │
│  │  │    Single Formatted Prompt            │            │ │
│  │  │  - Mode별 System Prompt 적용          │            │ │
│  │  └───────┬──────────────────────────────┘            │ │
│  └──────────┼───────────────────────────────────────────┘ │
│             │                                            │
│             ▼                                            │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  3. LLM System (AGT-LLM)                             │ │
│  │  ┌──────────────────────────────────────┐            │ │
│  │  │  LLM (Mode별)                        │            │ │
│  │  │  - conversation: Ollama (llama3.1:8b)│            │ │
│  │  │  - autonomous_navigation: OpenAI/Gemini│          │ │
│  │  │  - Function Calling (Actions)        │            │ │
│  │  │  - Response → Actions                 │            │ │
│  │  └───────┬──────────────────────────────┘            │ │
│  └──────────┼───────────────────────────────────────────┘ │
│             │                                            │
│             ▼                                            │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  4. Action Connectors (NAV-ACTION / AGT-ACTION)      │ │
│  │  ┌──────────────────────────────────────────────┐   │ │
│  │  │ NavigateGPSConnector (autonomous_navigation) │   │ │
│  │  │  - goal_to_xy() (vendor 포팅)                │   │ │
│  │  │  - PriorityPD.step() (vendor 포팅)          │   │ │
│  │  │  - LinearPath (vendor 포팅)                  │   │ │
│  │  ├──────────────────────────────────────────────┤   │ │
│  │  │ NavigateDWAConnector (autonomous_navigation)  │   │ │
│  │  │  - 코스트 계산 (vendor 포팅)                 │   │ │
│  │  │  - 최소 코스트 셀 선택 (vendor 포팅)         │   │ │
│  │  ├──────────────────────────────────────────────┤   │ │
│  │  │ GoSimpleAction (conversation)                │   │ │
│  │  │ StopSimpleAction (conversation)              │   │ │
│  │  │ LabChatAction (conversation, RAG)            │   │ │
│  │  │ GeneralChatAction (conversation)             │   │ │
│  │  └──────┬───────────────────────────────────────┘   │ │
│  │         │                                             │ │
│  │         ▼                                             │ │
│  │  ┌──────────────────────────────────────┐            │ │
│  │  │  Providers                           │            │ │
│  │  │  ┌──────────┐  ┌──────────┐         │            │ │
│  │  │  │GPS       │  │Go2       │         │            │ │
│  │  │  │Provider  │  │Control   │         │            │ │
│  │  │  │RAG       │  │Provider  │         │            │ │
│  │  │  │Provider  │  │          │         │            │ │
│  │  │  └────┬─────┘  └─────┬────┘         │            │ │
│  │  │       │              │               │            │ │
│  │  │       │              ▼               │            │ │
│  │  │       │  ┌──────────────────────┐  │            │ │
│  │  │       │  │ ROS2 Node (선택적)    │  │            │ │
│  │  │       │  │ Go2 제어 브리지만      │  │            │ │
│  │  │       └─▶│ (최소화)              │  │            │ │
│  │  │          └──────────────────────┘  │            │ │
│  │  └──────────────────────────────────────┘            │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  5. Backgrounds (SYS-BG)                               │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │ │
│  │  │RealSense │  │Go2       │  │ChromaDB  │          │ │
│  │  │Background│  │Control   │  │Background│          │ │
│  │  │          │  │Background│  │(RAG)     │          │ │
│  │  └────┬─────┘  └─────┬────┘  └─────┬────┘          │ │
│  │       │              │             │                │ │
│  │       └──────┬───────┴─────────────┘                │ │
│  │              │                                         │ │
│  │              ▼                                         │ │
│  │  ┌──────────────────────────────────────┐              │ │
│  │  │  ROS2 Node Launcher (최소화)         │              │ │
│  │  │  - RealSense 드라이버만              │              │ │
│  │  │  - Go2 제어 브리지만                  │              │ │
│  │  └───────┬──────────────────────────────┘              │ │
│  │          │                                             │ │
│  │          ▼                                             │ │
│  │  ┌──────────────────────────────────────┐              │ │
│  │  │  ROS2 Nodes (src/ros2_ws/src/)       │              │ │
│  │  │  - realsense2_camera (드라이버)      │              │ │
│  │  │  - unitree_go2_bridge (제어)         │              │ │
│  │  └──────────────────────────────────────┘              │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**데이터 흐름**:
1. **Input Sensors** → Provider → InputOrchestrator
2. **Fuser** → Multiple Sensor Inputs → Single Formatted Prompt
3. **LLM** → Prompt → Function Calling → Actions
4. **Action Connectors** → Provider → ROS2 Node / 직접 API
5. **Backgrounds** → Provider 초기화 및 ROS2 Node 실행

**핵심 차이점**:
- Multi Mode 중심 설계 (conversation / autonomous_navigation)
- 연산 로직이 OM1 Action Connector 내부에 있음
- ROS2 노드는 하드웨어 직접 제어만 (RealSense 드라이버, Go2 제어)
- Provider는 하드웨어 추상화 + 연산 로직 포함 (BEV, PointCloud)
- LLM System이 데이터 흐름의 핵심 역할

---

## 2. Input Sensors (PER-INPUT)

### 2.1 RealSense Sensor

**구조**:
```
src/inputs/plugins/realsense_camera.py
src/providers/realsense_camera_provider.py
src/backgrounds/plugins/realsense_camera.py
```

**데이터 흐름**:
```
ROS2 Node (realsense2_camera) - 하드웨어 드라이버만
    ↓ /camera/color/image_raw, /camera/depth/image_rect_raw
RealsenseProvider (ROS2 subscriber)
    ↓ latest_image_data (property)
RealsenseSensor
    ↓ formatted_latest_buffer()
InputOrchestrator → Fuser → LLM
```

**구현 계획**:
1. `RealsenseProvider`: ROS2 토픽 구독 (`/camera/color/image_raw`, `/camera/depth/image_rect_raw`)
2. `RealsenseSensor`: Provider에서 이미지 데이터 읽기 → 텍스트 변환
3. `RealsenseBackground`: `ros2 launch realsense2_camera rs_launch.py ...` 실행

### 2.2 PointCloud Sensor

**구조**:
```
src/inputs/plugins/pointcloud.py
src/providers/pointcloud_provider.py
```

**데이터 흐름**:
```
RealsenseProvider (이미지 데이터)
    ↓
PointCloudProvider
    ↓ (연산 로직: vendor 포팅)
    - RealSense depth → PointCloud 변환
    - CUDA 가속 (vendor/distmap_def.py 참고)
    ↓ latest_pointcloud_data (property)
PointCloudSensor
    ↓ formatted_latest_buffer()
InputOrchestrator → Fuser → LLM
```

**구현 계획**:
1. `PointCloudProvider`: 
   - RealSense Provider에서 depth 이미지 읽기
   - **연산 로직 포팅**: vendor `pointcloud_xyzrgb` 로직을 OM1으로 이동
   - CUDA 가속 지원
2. `PointCloudSensor`: Provider에서 PointCloud 데이터 읽기 → 텍스트 변환
3. **ROS2 노드 불필요**: 연산 로직이 OM1 Provider 내부에 있음

### 2.3 BEV Occupancy Grid Sensor

**구조**:
```
src/inputs/plugins/bev_occupancy_grid.py
src/providers/bev_occupancy_grid_provider.py
```

**데이터 흐름**:
```
PointCloudProvider (PointCloud 데이터)
    ↓
BEVProvider
    ↓ (연산 로직: vendor 포팅)
    - PointCloud → BEV Occupancy Grid 변환
    - CUDA 가속 (vendor/bev_cuda 참고)
    ↓ latest_occupancy_grid (property)
BEVSensor
    ↓ formatted_latest_buffer()
InputOrchestrator → Fuser → LLM
```

**구현 계획**:
1. `BEVProvider`:
   - PointCloud Provider에서 데이터 읽기
   - **연산 로직 포팅**: vendor `bev_cuda` 로직을 OM1으로 이동
   - CUDA 가속 지원
2. `BEVSensor`: Provider에서 Occupancy Grid 읽기 → 텍스트 변환
3. **ROS2 노드 불필요**: 연산 로직이 OM1 Provider 내부에 있음

### 2.4 Segmentation Sensor

**구조**:
```
src/inputs/plugins/segmentation.py
src/providers/segmentation_provider.py
```

**데이터 흐름**:
```
RealsenseProvider (이미지 데이터)
    ↓
SegmentationProvider
    ↓ (연산 로직: vendor 포팅)
    - TensorRT 모델 실행
    - Segmentation mask 생성
    ↓ latest_segmentation (property)
SegmentationSensor
    ↓ formatted_latest_buffer()
InputOrchestrator → Fuser → LLM
```

**구현 계획**:
1. `SegmentationProvider`:
   - RealSense Provider에서 이미지 읽기
   - **연산 로직 포팅**: vendor TensorRT 모델 실행 로직을 OM1으로 이동
2. `SegmentationSensor`: Provider에서 segmentation mask 읽기 → 텍스트 변환
3. **ROS2 노드 불필요**: 연산 로직이 OM1 Provider 내부에 있음

---

## 5. Action Connectors (NAV-ACTION)

### 5.1 Navigate GPS Action

**구조**:
```
src/actions/navigate_gps/
├── connector/
│   └── ros2.py              # GPS Navigation Connector (연산 로직 포함)
└── interface.py              # NavigateGPSInput interface
src/providers/gps_provider.py
src/providers/go2_control_provider.py
```

**데이터 흐름**:
```
LLM → NavigateGPSConnector
    ↓ (비즈니스 로직: LLM 입력 파싱)
    - "go to lat=37.xxx, lon=127.xxx" → GPS 좌표 추출
    ↓ (연산 로직: vendor 포팅)
    - goal_to_xy() (vendor/controller.py 포팅)
    - PriorityPD.step() (vendor/controller.py 포팅)
    - LinearPath (vendor/nav_utils.py 포팅)
    ↓ (제어 명령 생성)
    - vx, vy, vyaw 계산
    ↓
Go2ControlProvider
    ↓ (직접 API 호출 또는 ROS2 토픽)
    - Unitree Go2 SportClient API
    - 또는 /cmd 토픽 발행 (unitree_go2_bridge가 구독)
    ↓
Unitree Go2 Robot
```

**구현 계획**:
1. `GPSProvider`:
   - GPS 시리얼 포트 직접 읽기 (`/dev/gps`, 115200 baud)
   - UBX/NMEA 파싱 (vendor `pyubx2`, `pynmeagps` 사용)
   - NTRIP 클라이언트 (RTK 보정)
   - **ROS2 노드 불필요**: 시리얼 포트 직접 접근
2. `NavigateGPSConnector`:
   - LLM 입력 파싱: "go to lat=37.xxx, lon=127.xxx" → GPS 좌표 추출
   - **연산 로직 포팅**:
     - `goal_to_xy()`: vendor `controller.py`에서 포팅
     - `PriorityPD.step()`: vendor `controller.py`에서 포팅
     - `LinearPath`: vendor `nav_utils.py`에서 포팅
     - `compute_yaw_offset()`: vendor `controller.py`에서 포팅
   - 경로 파일 로드/생성
   - Provider 호출하여 제어 명령 전송
3. `Go2ControlProvider`:
   - Unitree Go2 SportClient API 직접 호출 (선택 1)
   - 또는 ROS2 토픽 `/cmd` 발행 (선택 2, `unitree_go2_bridge`가 구독)

**코드 예시**:
```python
# src/actions/navigate_gps/connector/ros2.py
from actions.base import ActionConnector, ActionConfig
from providers.gps_provider import GPSProvider
from providers.go2_control_provider import Go2ControlProvider
from vendor_algorithms.gps_nav.controller import PriorityPD, goal_to_xy
from vendor_algorithms.gps_nav.nav_utils import LinearPath

class NavigateGPSConnector(ActionConnector[ActionConfig, NavigateGPSInput]):
    def __init__(self, config: ActionConfig):
        super().__init__(config)
        self.gps_provider = GPSProvider()
        self.go2_provider = Go2ControlProvider()
        
        # 연산 로직 (vendor 포팅)
        self.controller = PriorityPD(
            kp_x=0.6, kd_x=0.05,
            kp_y=0.6, kd_y=0.05,
            kp_yaw=2.0, kd_yaw=0.2
        )
        self.path_follower = LinearPath(waypoints=[], reach_tol=2.0)
    
    async def connect(self, output_interface: NavigateGPSInput):
        # LLM 입력 파싱
        goal_lat, goal_lon = self._parse_gps_coords(output_interface)
        
        # 현재 위치 읽기
        curr_lat, curr_lon = self.gps_provider.get_latlon()
        curr_x, curr_y, curr_yaw = self.gps_provider.get_odom_xy_yaw()
        
        # 연산: GPS 좌표 → 로봇 기준 (x, y) 변환 (vendor 포팅)
        dx, dy = goal_to_xy(curr_lat, curr_lon, goal_lat, goal_lon, heading_deg)
        
        # 연산: 경로 추종 제어 (vendor 포팅)
        vx, vy, vyaw = self.controller.step(dx, dy)
        
        # 제어 명령 전송
        self.go2_provider.send_velocity(vx, vy, vyaw)
```

### 5.2 Navigate DWA Action

**구조**:
```
src/actions/navigate_dwa/
├── connector/
│   └── ros2.py              # DWA Navigation Connector (연산 로직 포함)
└── interface.py              # NavigateDWAInput interface
src/providers/bev_occupancy_grid_provider.py
src/providers/go2_control_provider.py
```

**데이터 흐름**:
```
LLM → NavigateDWAConnector
    ↓ (비즈니스 로직: LLM 입력 파싱)
    - "move forward 2 meters" → 목표 위치 계산
    ↓ (연산 로직: vendor 포팅)
    - CUDA 거리 맵 생성 (vendor/distmap_def.py 포팅)
    - 코스트 계산: (x-dx)^2 + (y-dy)^2 + penalty * (1 - d/margin)^2
    - 전방 창에서 최소 코스트 셀 선택
    - 속도 명령 생성 (vx, vyaw)
    ↓
Go2ControlProvider
    ↓ (직접 API 호출 또는 ROS2 토픽)
    - Unitree Go2 SportClient API
    - 또는 /cmd 토픽 발행
    ↓
Unitree Go2 Robot
```

**구현 계획**:
1. `BEVOccupancyGridProvider`:
   - PointCloud Provider에서 데이터 읽기
   - **연산 로직 포함**: PointCloud → BEV Occupancy Grid 변환 (CUDA 가속)
2. `NavigateDWAConnector`:
   - LLM 입력 파싱: "move forward 2 meters" → 목표 위치 계산
   - **연산 로직 포팅**:
     - CUDA 거리 맵 생성: vendor `distmap_def.py`에서 포팅
     - 코스트 계산: vendor `dwa_node_success.py`에서 포팅
     - 최소 코스트 셀 선택: vendor `dwa_node_success.py`에서 포팅
     - 속도 명령 생성: vendor `dwa_node_success.py`에서 포팅
   - Provider 호출하여 제어 명령 전송
3. **ROS2 노드 불필요**: 연산 로직이 OM1 Action Connector 내부에 있음

**코드 예시**:
```python
# src/actions/navigate_dwa/connector/ros2.py
from actions.base import ActionConnector, ActionConfig
from providers.bev_occupancy_grid_provider import BEVOccupancyGridProvider
from providers.go2_control_provider import Go2ControlProvider
from vendor_algorithms.dwa_nav.dwa_algorithm import compute_cost, select_best_cell
from vendor_algorithms.dwa_nav.distmap_def import build_dist_map_bfs_cuda

class NavigateDWAConnector(ActionConnector[ActionConfig, NavigateDWAInput]):
    def __init__(self, config: ActionConfig):
        super().__init__(config)
        self.bev_provider = BEVOccupancyGridProvider()
        self.go2_provider = Go2ControlProvider()
        
        # DWA 파라미터
        self.penalty = config.penalty  # 13.0
        self.margin = config.margin    # 1.2
        self.v_max = config.v_max      # 0.9
        self.w_max = config.w_max      # 0.75
    
    async def connect(self, output_interface: NavigateDWAInput):
        # LLM 입력 파싱
        dx, dy = self._parse_target(output_interface)
        
        # BEV Occupancy Grid 읽기
        occupancy_grid = self.bev_provider.get_latest()
        
        # 연산: CUDA 거리 맵 생성 (vendor 포팅)
        dist_map = build_dist_map_bfs_cuda(occupancy_grid)
        
        # 연산: 코스트 계산 및 최소 코스트 셀 선택 (vendor 포팅)
        best_cell = select_best_cell(dx, dy, dist_map, self.penalty, self.margin)
        
        # 연산: 속도 명령 생성 (vendor 포팅)
        vx, vyaw = self._compute_velocity(best_cell, self.v_max, self.w_max)
        
        # 제어 명령 전송
        self.go2_provider.send_velocity(vx, 0, vyaw)
```

### 5.3 Navigate Go2 Control Action

**구조**:
```
src/actions/navigate_go2/
├── connector/
│   └── ros2.py              # Go2 Control Connector
└── interface.py              # NavigateGo2Input interface
src/providers/go2_control_provider.py
```

**데이터 흐름**:
```
LLM → NavigateGo2Connector
    ↓ (비즈니스 로직: 속도 명령 파싱)
    - "move forward", "turn left" → 속도 명령
    ↓
Go2ControlProvider
    ↓ (직접 API 호출 또는 ROS2 토픽)
    - Unitree Go2 SportClient API
    - 또는 /cmd 토픽 발행 (unitree_go2_bridge가 구독)
    ↓
Unitree Go2 Robot
```

**구현 계획**:
1. `Go2ControlProvider`:
   - Unitree Go2 SportClient API 직접 호출 (선택 1)
   - 또는 ROS2 토픽 `/cmd` 발행 (선택 2, `unitree_go2_bridge`가 구독)
2. `NavigateGo2Connector`:
   - LLM 입력 파싱: "move forward", "turn left" → 속도 명령
   - Provider 호출하여 로봇 제어

---

## 6. Providers (SYS-PROVIDER)

### 6.0 RealSense Camera Provider

**구조**:
```
src/providers/realsense_camera_provider.py
```

**역할**:
- ROS Workspace 접근에 대한 추상화
- ROS2 토픽 구독 (`/camera/color/image_raw`, `/camera/depth/image_rect_raw`) 또는 pyrealsense2 SDK 직접 사용
- PointCloud/BEV Provider의 입력 소스

**구현 계획**:
```python
# src/providers/realsense_camera_provider.py
@singleton
class RealsenseCameraProvider:
    """RealSense 카메라 데이터 제공 (ROS Workspace 접근 추상화)"""
    def __init__(self, use_ros2: bool = True):
        self.use_ros2 = use_ros2
        
        if use_ros2:
            # ROS2 토픽 구독 (realsense2_camera 노드가 데이터 발행)
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            
            rclpy.init()
            self.node = Node("realsense_camera_provider")
            self.color_sub = self.node.create_subscription(
                Image, "/camera/color/image_raw", self._color_callback, 10
            )
            self.depth_sub = self.node.create_subscription(
                Image, "/camera/depth/image_rect_raw", self._depth_callback, 10
            )
            
            self.latest_color_image = None
            self.latest_depth_image = None
        else:
            # pyrealsense2 SDK 직접 사용
            import pyrealsense2 as rs
            
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(self.config)
            
            self.latest_color_image = None
            self.latest_depth_image = None
            
            # 백그라운드 스레드로 프레임 읽기
            self.running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
    
    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """최신 color, depth 이미지 반환"""
        return self.latest_color_image, self.latest_depth_image
```

### 6.1 GPS Provider

**구조**:
```
src/providers/gps_provider.py
```

**구현 계획**:
```python
# src/providers/gps_provider.py
@singleton
class GPSProvider:
    """GPS 시리얼 포트 직접 읽기 (ROS2 노드 불필요)"""
    def __init__(self, serial_port: str = "/dev/gps", baud: int = 115200):
        self.serial = serial.Serial(serial_port, baud)
        self.ubx_reader = UBXReader(self.serial)
        self.nmea_parser = NMEAParser()
        
        # NTRIP 클라이언트 (RTK 보정)
        self.ntrip_client = None  # 선택적
        
        self.latest_lat = None
        self.latest_lon = None
        self.latest_rtk_quality = None
        
        # 백그라운드 스레드로 시리얼 포트 읽기
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
    
    def _read_loop(self):
        """시리얼 포트에서 UBX/NMEA 메시지 읽기"""
        while True:
            try:
                msg = self.ubx_reader.read()
                if isinstance(msg, UBXMessage):
                    if msg.identity == "NAV-PVT":
                        self.latest_lat = msg.lat / 1e7
                        self.latest_lon = msg.lon / 1e7
                        self.latest_rtk_quality = msg.carrSoln
            except Exception as e:
                logging.error(f"GPS read error: {e}")
    
    def get_latlon(self) -> Tuple[float, float]:
        """현재 GPS 좌표 반환"""
        return self.latest_lat, self.latest_lon
    
    def get_odom_xy_yaw(self) -> Tuple[float, float, float]:
        """Odom Provider에서 x, y, yaw 읽기 (통합 필요)"""
        # OdomProvider와 통합 필요
        pass
```

### 6.2 BEV Occupancy Grid Provider

**구조**:
```
src/providers/bev_occupancy_grid_provider.py
```

**구현 계획**:
```python
# src/providers/bev_occupancy_grid_provider.py
@singleton
class BEVOccupancyGridProvider:
    """BEV Occupancy Grid 생성 (연산 로직 포함, ROS2 노드 불필요)"""
    def __init__(self, resolution: float = 0.05, width: int = 80, height: int = 80):
        self.resolution = resolution
        self.width = width
        self.height = height
        
        # PointCloud Provider 의존
        self.pointcloud_provider = PointCloudProvider()
        
        # 연산 로직 (vendor 포팅)
        from vendor_algorithms.bev_cuda.bev_generator import generate_bev_grid
        
        self.latest_occupancy_grid = None
    
    def update(self):
        """PointCloud → BEV Occupancy Grid 변환 (CUDA 가속)"""
        pointcloud = self.pointcloud_provider.get_latest()
        if pointcloud is None:
            return
        
        # 연산 로직 (vendor 포팅)
        self.latest_occupancy_grid = generate_bev_grid(
            pointcloud, self.resolution, self.width, self.height
        )
    
    def get_latest(self) -> Optional[np.ndarray]:
        """최신 Occupancy Grid 반환"""
        self.update()
        return self.latest_occupancy_grid
```

### 6.3 PointCloud Provider

**구조**:
```
src/providers/pointcloud_provider.py
```

**구현 계획**:
```python
# src/providers/pointcloud_provider.py
@singleton
class PointCloudProvider:
    """PointCloud 생성 (연산 로직 포함, ROS2 노드 불필요)"""
    def __init__(self):
        # RealSense Provider 의존
        self.realsense_provider = RealsenseCameraProvider()
        
        # 연산 로직 (vendor 포팅)
        from vendor_algorithms.pointcloud_xyzrgb.pointcloud_generator import generate_pointcloud
        
        self.latest_pointcloud = None
    
    def update(self):
        """RealSense depth → PointCloud 변환 (CUDA 가속)"""
        color_img, depth_img = self.realsense_provider.get_latest()
        if color_img is None or depth_img is None:
            return
        
        # 연산 로직 (vendor 포팅)
        self.latest_pointcloud = generate_pointcloud(color_img, depth_img)
    
    def get_latest(self) -> Optional[np.ndarray]:
        """최신 PointCloud 반환"""
        self.update()
        return self.latest_pointcloud
```

### 6.4 Go2 Control Provider

**구조**:
```
src/providers/go2_control_provider.py
```

**구현 계획**:
```python
# src/providers/go2_control_provider.py
@singleton
class Go2ControlProvider:
    """Unitree Go2 제어 (직접 API 또는 ROS2 토픽)"""
    def __init__(self, use_ros2: bool = False):
        self.use_ros2 = use_ros2
        
        if use_ros2:
            # ROS2 토픽 발행 (unitree_go2_bridge가 구독)
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import Twist
            
            rclpy.init()
            self.node = Node("go2_control_provider")
            self.cmd_pub = self.node.create_publisher(Twist, "/cmd", 10)
        else:
            # 직접 SportClient API 호출
            from unitree.unitree_sdk2py.go2.sport.sport_client import SportClient
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(10.0)
            self.sport_client.Init()
    
    def send_velocity(self, vx: float, vy: float, vyaw: float):
        """속도 명령 전송"""
        if self.use_ros2:
            msg = Twist()
            msg.linear.x = vx
            msg.linear.y = vy
            msg.angular.z = vyaw
            self.cmd_pub.publish(msg)
        else:
            self.sport_client.Move(vx, vy, vyaw)
```

### 6.5 ASR Provider

**구조**:
```
src/providers/asr_provider.py (기존 OM1 Provider 사용)
```

**역할**:
- 마이크 입력 처리 (음성 → 텍스트)
- PyAudio로 마이크 직접 접근 (ROS2 노드 불필요)
- 오디오 팬텀 등 USB 오디오 인터페이스 지원

**구현 계획**:
```python
# src/providers/asr_provider.py (기존 OM1 Provider)
@singleton
class ASRProvider:
    """
    Audio Speech Recognition Provider
    - PyAudio로 마이크 직접 접근
    - USB 오디오 인터페이스 (오디오 팬텀) 지원
    - ROS2 노드 불필요
    """
    def __init__(
        self,
        device_id: Optional[int] = None,
        microphone_name: Optional[str] = None,  # "오디오 팬텀" 등
        rate: Optional[int] = None,
        chunk: Optional[int] = None,
        language_code: Optional[str] = None,
    ):
        from om1_speech import AudioInputStream
        
        self.audio_stream: AudioInputStream = AudioInputStream(
            device=device_id,
            device_name=microphone_name,
            rate=rate,
            chunk=chunk,
            language_code=language_code,
        )
```

### 6.6 TTS Provider

**구조**:
```
src/providers/riva_tts_provider.py (기존 OM1 Provider 사용)
```

**역할**:
- 텍스트 → 음성 변환 및 스피커 출력
- PyAudio로 스피커 직접 출력 (ROS2 노드 불필요)
- AUX 스피커 등 시스템 기본 오디오 출력 사용

**구현 계획**:
```python
# src/providers/riva_tts_provider.py (기존 OM1 Provider)
@singleton
class RivaTTSProvider:
    """
    Text-to-Speech Provider
    - PyAudio로 스피커 직접 출력
    - 시스템 기본 스피커 또는 AUX 스피커 사용
    - ROS2 노드 불필요
    """
    def __init__(self, url: str, api_key: Optional[str] = None):
        from om1_speech import AudioOutputStream
        
        self._audio_stream: AudioOutputStream = AudioOutputStream(
            url=url,
            headers={"x-api-key": api_key} if api_key else None,
        )
    
    def add_pending_message(self, text: str):
        """TTS 텍스트 추가"""
        self._audio_stream.add_request({"text": text})
```

---

## 7. Backgrounds (SYS-BG)

**역할**:
- **Provider 초기화**: Background는 해당 Provider의 초기화를 담당
- **ROS2 노드 실행**: 하드웨어 직접 제어가 필요한 경우 ROS2 노드 실행
- **리소스 관리**: Provider와 ROS2 노드의 생명주기 관리

### 7.0 GPS Background

**구조**:
```
src/backgrounds/plugins/gps.py
```

**역할**:
- GPS Provider 초기화
- GPS는 시리얼 직접 읽기로 ROS2 노드 불필요

**구현 계획**:
```python
# src/backgrounds/plugins/gps.py
class Gps(Background[GpsConfig]):
    """GPS Provider 초기화"""
    def __init__(self, config: GpsConfig):
        super().__init__(config)
        
        # Provider 초기화
        port = self.config.serial_port or "/dev/ttyACM0"
        self.gps_provider = GPSProvider(serial_port=port, baud=115200)
        logging.info(f"GPS Provider initialized with port: {port}")
        
        # ROS2 노드 불필요 (시리얼 직접 읽기)
```

### 7.1 RealSense Background

**구조**:
```
src/backgrounds/plugins/realsense_camera.py
```

**역할**:
- RealSenseCameraProvider 초기화
- ROS2 노드 실행 (하드웨어 드라이버만)

**구현 계획**:
```python
# src/backgrounds/plugins/realsense_camera.py
class RealsenseCamera(Background[RealsenseCameraConfig]):
    """RealSense 카메라 Provider 초기화 및 ROS2 드라이버 실행"""
    def __init__(self, config: RealsenseCameraConfig):
        super().__init__(config)
        
        # Provider 초기화
        self.realsense_provider = RealsenseCameraProvider(use_ros2=config.use_ros2)
        self.realsense_provider.start()
        
        if config.use_ros2:
        # ROS2 노드 실행 (하드웨어 드라이버만)
        launch_args = [
            f"depth_module.depth_profile:={config.depth_profile}",
            f"rgb_camera.color_profile:={config.color_profile}",
            f"enable_depth:={config.enable_depth}",
            f"enable_color:={config.enable_color}",
        ]
        self._launch_ros2_launch(
            "realsense2_camera",
            "rs_launch.py",
            launch_args
        )
```

### 7.2 Go2 Control Background

**구조**:
```
src/backgrounds/plugins/go2_control.py
```

**역할**:
- Go2ControlProvider 초기화
- ROS2 노드 실행 (선택적, 하드웨어 제어 브리지만)

**구현 계획**:
```python
# src/backgrounds/plugins/go2_control.py
class Go2Control(Background[Go2ControlConfig]):
    """Unitree Go2 Provider 초기화 및 ROS2 브리지 실행 (선택적)"""
    def __init__(self, config: Go2ControlConfig):
        super().__init__(config)
        
        # Provider 초기화
        self.go2_provider = Go2ControlProvider(use_ros2=config.use_ros2)
        
        if config.use_ros2:
            # ROS2 노드 실행 (하드웨어 제어 브리지만)
            self._launch_ros2_node(
                "unitree_go2_bridge",
                "dwa2go2_node",
                []
            )
```

### 7.3 ASR/TTS Background (선택적)

**역할**:
- ASRProvider, TTSProvider 초기화
- 오디오 하드웨어는 OS 레벨 지원으로 Background 선택적
- Provider가 자체 초기화 가능 (권장)

**구현 계획**:
```python
# src/backgrounds/plugins/asr_tts.py (선택적)
class ASRTTS(Background[ASRTTSConfig]):
    """ASR/TTS Provider 초기화 (선택적)"""
    def __init__(self, config: ASRTTSConfig):
        super().__init__(config)
        
        # Provider 초기화
        self.asr_provider = ASRProvider(
            microphone_name=config.microphone_name,
            language_code=config.language_code,
        )
        self.asr_provider.start()
        
        self.tts_provider = RivaTTSProvider(
            url=config.tts_url,
            api_key=config.api_key,
        )
        self.tts_provider.start()
        
        # ROS2 노드 불필요 (PyAudio 직접 접근)
```

---

## 3. Fuser System

### 3.1 Fuser 역할

**기능**:
- Multiple Sensor Inputs를 Single Formatted Prompt로 변환
- Mode별 System Prompt 적용
- Sensor 데이터 포맷팅 및 통합

**데이터 흐름**:
```
InputOrchestrator (Multiple Sensors)
    ↓
Fuser
    ↓ (fuse(inputs))
    - Sensor 1: "Camera sees: ..."
    - Sensor 2: "GPS location: ..."
    - Sensor 3: "BEV obstacles: ..."
    ↓
Single Formatted Prompt
    ↓
LLM System
```

**Mode별 System Prompt**:
- `conversation`: 대화 중심 프롬프트
- `autonomous_navigation`: 자율 주행 중심 프롬프트

### 3.2 Fuser 구현

**구조**:
```
src/fuser/__init__.py
```

**구현 계획**:
- 기존 Fuser 코드 유지 (수정 불필요)
- Config에서 Mode별 System Prompt 정의
- Mode 전환 시 System Prompt 자동 변경

---

## 4. LLM System (AGT-LLM)

### 4.1 LLM 역할

**기능**:
- Fuser에서 받은 Prompt 처리
- Function Calling (Actions)
- Mode별 LLM 모델 선택

**데이터 흐름**:
```
Fuser (Formatted Prompt)
    ↓
LLM System
    ↓ (ask(prompt))
    - Function Calling (Actions)
    - Response Generation
    ↓
Action Connectors
```

### 4.2 Mode별 LLM 구성

**conversation Mode**:
- LLM: Ollama (llama3.1:8b)
- Actions: go_simple, stop_simple, set_goal_simple, lab_chat, general_chat
- 특징: 간단한 명령 처리, 대화 중심

**autonomous_navigation Mode**:
- LLM: OpenAI/Gemini
- Actions: navigate_gps, navigate_dwa, navigate_go2
- 특징: 복잡한 경로 추종 명령 처리

### 4.3 LLM 구현

**구조**:
```
src/llm/plugins/ollama_llm.py
src/llm/plugins/openai_llm.py
src/llm/plugins/gemini_llm.py
```

**구현 계획**:
- 기존 LLM 플러그인 유지
- Mode별 LLM 자동 선택
- Function Schema 자동 생성 (Actions 기반)

---


## 8. LLMagent 통합 설계

### 8.1 LLMagent 구조 분석

**핵심 컴포넌트**:

1. **LLMAgentNode** (`agent_llm.py`):
   - LangChain ReAct 에이전트
   - Tools: `go`, `stop`, `set_goal`, `lab_chat`, `general_chat`
   - ROS2 토픽:
     - 구독: `/user_question` (STT 입력), `/fsm/state` (FSM 상태)
     - 발행: `/llm/selected_tool` (RobotCommand)

2. **FSMNode** (`fsm_node.py`):
   - FSM 상태 관리
   - 상태: `base` (START/ING/END), `nav` (0/1), `speak` (0/1/2), `monitor` (0/1)
   - ROS2 토픽:
     - 구독: `/llm/selected_tool`
     - 발행: `/fsm/state`, `/fsm/nav_cmd`, `/fsm/tts`

3. **RAG 기능** (`chat_backends.py`):
   - ChromaDB 벡터 스토어
   - 연구소 관련 질문 RAG 답변 (`lab_chat`)
   - 일반 대화 (`general_chat`)

### 8.2 LLMagent vs Autonomous_Navigation 비교

| 특징 | LLMagent | Autonomous_Navigation |
|------|----------|----------------------|
| 목적 | 대화 중심 (RAG, 일반 대화) | 자율 주행 중심 (GPS, DWA, BEV) |
| 제어 | 간단한 명령 (go, stop, set_goal) | 복잡한 경로 추종 및 장애물 회피 |
| 상태 관리 | FSM 기반 (base, nav, speak, monitor) | 연산 로직 중심 |
| LLM | Ollama (llama3.1:8b) | OpenAI/Gemini (복잡한 명령 처리) |
| 통신 | ROS2 토픽 기반 | ROS2 토픽 + OM1 통합 예정 |

### 8.3 LLMagent → OM1 포팅 계획

#### 8.3.1 LLMAgentNode → OM1 LLM System

**구조**:
```
src/llm/plugins/ollama_llm.py (기존 또는 확장)
src/actions/conversation/
├── connector/
│   └── rag.py              # RAG 기반 대화 Connector
└── interface.py
```

**포팅 계획**:
1. LangChain ReAct 에이전트 → OM1 LLM System으로 통합
2. Tools (`go`, `stop`, `set_goal`, `lab_chat`, `general_chat`) → OM1 Actions로 변환
3. FSM 상태 관리 → OM1 Multi Mode 상태 관리로 통합

#### 8.3.2 FSMNode → OM1 Multi Mode Manager

**구조**:
```
src/runtime/multi_mode/manager.py (기존 확장)
```

**포팅 계획**:
1. FSM 상태 (`base`, `nav`, `speak`, `monitor`) → OM1 Mode 상태로 매핑
2. FSM 전이 로직 → OM1 Mode 전환 규칙으로 변환
3. ROS2 토픽 기반 통신 → OM1 내부 통신으로 변경

#### 8.3.3 RAG 기능 → OM1 Provider

**구조**:
```
src/providers/rag_provider.py
src/providers/chromadb_provider.py
```

**포팅 계획**:
1. ChromaDB 벡터 스토어 → `ChromaDBProvider`로 추상화
2. RAG 체인 → `RAGProvider`로 추상화
3. 연구소 문서 로딩 → Background로 이동

#### 8.3.4 Tools → OM1 Actions

**구조**:
```
src/actions/go_simple/          # 간단한 이동 (LLMagent용)
src/actions/stop_simple/        # 정지 (LLMagent용)
src/actions/set_goal_simple/    # 목표 설정 (LLMagent용)
src/actions/lab_chat/           # 연구소 대화 (RAG)
src/actions/general_chat/       # 일반 대화
```

**포팅 계획**:
1. `go` → `GoSimpleAction` (간단한 이동, LLMagent용)
2. `stop` → `StopSimpleAction` (정지, LLMagent용)
3. `set_goal` → `SetGoalSimpleAction` (목표 설정, LLMagent용)
4. `lab_chat` → `LabChatAction` (RAG 기반 대화)
5. `general_chat` → `GeneralChatAction` (일반 대화)

---

## 9. Multi Mode 구성

### 9.1 Mode 정의

**Mode 1: "conversation" (LLMagent 기반)**
- 목적: 대화 및 간단한 로봇 제어
- 특징:
  - RAG 기반 연구소 질문 답변
  - 일반 대화
  - 간단한 이동 명령 (go, stop, set_goal)
- 구성:
  - LLM: Ollama (llama3.1:8b)
  - Inputs: ASR (STT)
  - Actions: go_simple, stop_simple, set_goal_simple, lab_chat, general_chat
  - Backgrounds: RAG 벡터 스토어 (ChromaDB)

**Mode 2: "autonomous_navigation" (Autonomous_Navigation 기반)**
- 목적: 자율 주행 및 복잡한 경로 추종
- 특징:
  - GPS 기반 전역 경로 계획
  - DWA 기반 장애물 회피
  - BEV Occupancy Grid 기반 인지
- 구성:
  - LLM: OpenAI/Gemini (복잡한 명령 처리)
  - Inputs: RealSense, PointCloud, BEV, GPS, Odom
  - Actions: navigate_gps, navigate_dwa, navigate_go2
  - Backgrounds: RealSense 드라이버, Go2 제어

### 9.2 Mode 전환 규칙

**INPUT_TRIGGERED 전환**:
- 사용자 입력 키워드 기반
  - "자율 주행 모드로 전환", "GPS 모드로 전환" → `autonomous_navigation`
  - "대화 모드로 전환", "대화 시작" → `conversation`

**CONTEXT_AWARE 전환**:
- 복잡한 경로 추종 요청 → `autonomous_navigation`
  - 예: "GPS 좌표로 이동", "장애물 회피하며 이동"
- 대화 요청 → `conversation`
  - 예: "연구소에 대해 알려줘", "일반 대화"

**MANUAL 전환**:
- 사용자가 명시적으로 모드 전환 요청

### 9.3 Mode 전환 시 고려사항

1. **상태 보존**: Mode 전환 시 필요한 상태 정보 보존
2. **리소스 관리**: Mode별 리소스 (Provider, Background) 초기화/정리
3. **부드러운 전환**: 현재 작업 완료 후 전환 또는 즉시 전환 선택 가능

---

## 10. 연산 로직 포팅 계획

### 10.1 GPS Navigation 알고리즘 포팅

**Vendor 소스**:
- `vendor/gps_nav/gps_nav/controller.py`: `PriorityPD`, `goal_to_xy`, `compute_yaw_offset`
- `vendor/gps_nav/gps_nav/nav_utils.py`: `LinearPath`, `haversine_xy`, `normalize`

**포팅 대상**:
- `src/actions/navigate_gps/connector/ros2.py`: Action Connector 내부
- 또는 `src/utils/navigation/`: 공통 유틸리티로 분리

**포팅 단계**:
1. `PriorityPD` 클래스 포팅
2. `goal_to_xy()` 함수 포팅
3. `compute_yaw_offset()` 함수 포팅
4. `LinearPath` 클래스 포팅
5. `haversine_xy()`, `normalize()` 함수 포팅

### 10.2 DWA 알고리즘 포팅

**Vendor 소스**:
- `vendor/dwa_nav/dwa_nav/dwa_node_success.py`: 코스트 계산, 셀 선택, 속도 생성
- `vendor/dwa_nav/dwa_nav/distmap_def.py`: CUDA 거리 맵 생성

**포팅 대상**:
- `src/actions/navigate_dwa/connector/ros2.py`: Action Connector 내부
- 또는 `src/utils/navigation/`: 공통 유틸리티로 분리

**포팅 단계**:
1. CUDA 거리 맵 생성 함수 포팅 (`build_dist_map_bfs_cuda`, `build_dist_map_bf_cuda`)
2. 코스트 계산 함수 포팅
3. 최소 코스트 셀 선택 함수 포팅
4. 속도 명령 생성 함수 포팅

### 10.3 BEV Occupancy Grid 생성 포팅

**Vendor 소스**:
- `vendor/bev_cuda/`: BEV Occupancy Grid 생성 로직

**포팅 대상**:
- `src/providers/bev_occupancy_grid_provider.py`: Provider 내부

**포팅 단계**:
1. PointCloud → BEV 변환 로직 포팅
2. CUDA 가속 코드 포팅

### 10.4 PointCloud 생성 포팅

**Vendor 소스**:
- `vendor/pointcloud_xyzrgb/`: PointCloud 생성 로직

**포팅 대상**:
- `src/providers/pointcloud_provider.py`: Provider 내부

**포팅 단계**:
1. RealSense depth → PointCloud 변환 로직 포팅
2. CUDA 가속 코드 포팅

---

## 11. Config 파일 구조

### Multi Mode Config 예시

```json5
{
  "version": "v1.0.1",
  "name": "kist_integrated_system",
  "default_mode": "conversation",
  "allow_manual_switching": true,
  "mode_memory_enabled": true,
  
  "api_key": "openmind_free",
  "unitree_ethernet": "en0",
  
  "modes": {
    "conversation": {
      "name": "conversation",
      "description": "LLMagent 기반 대화 모드",
      
      "agent_inputs": [
        {
          "type": "GoogleASR",
          "config": {
            "language_code": "ko-KR"
          }
        }
      ],
      
      "cortex_llm": {
        "type": "OllamaLLM",
        "config": {
          "model": "llama3.1:8b",
          "agent_name": "ConversationBot",
          "history_length": 10
        }
      },
      
      "agent_actions": [
        {
          "name": "go_simple",
          "llm_label": "move_forward_simple",
          "connector": "ros2",
          "config": {
            "default_distance": 1.0
          }
        },
        {
          "name": "stop_simple",
          "llm_label": "stop_robot",
          "connector": "ros2",
          "config": {}
        },
        {
          "name": "set_goal_simple",
          "llm_label": "set_goal_location",
          "connector": "ros2",
          "config": {
            "available_goals": ["L0", "L1", "L2"]
          }
        },
        {
          "name": "lab_chat",
          "llm_label": "answer_lab_question",
          "connector": "rag",
          "config": {
            "rag_provider": "ChromaDBProvider",
            "top_k": 4
          }
        },
        {
          "name": "general_chat",
          "llm_label": "answer_general_question",
          "connector": "ollama",
          "config": {
            "model": "exaone3.5:7.8b"
          }
        }
      ],
      
      "backgrounds": [
        {
          "type": "ChromaDB",
          "config": {
            "db_path": "vendor/LLMagent/_rag_chroma_db",
            "embedding_model": "intfloat/multilingual-e5-base"
          }
        }
      ]
    },
    
    "autonomous_navigation": {
      "name": "autonomous_navigation",
      "description": "Autonomous_Navigation 기반 자율 주행 모드",
      
      "agent_inputs": [
        {
          "type": "RealsenseCamera",
          "config": {
            "camera_index": 0
          }
        },
        {
          "type": "PointCloud",
          "config": {}
        },
        {
          "type": "BEVOccupancyGrid",
          "config": {
            "resolution": 0.05,
            "width": 80,
            "height": 80
          }
        },
        {
          "type": "GPS",
          "config": {
            "serial_port": "/dev/gps",
            "baud": 115200
          }
        }
      ],
      
      "cortex_llm": {
        "type": "OpenAILLM",
        "config": {
          "agent_name": "NavBot",
          "history_length": 10
        }
      },
      
      "agent_actions": [
        {
          "name": "navigate_gps",
          "llm_label": "navigate_to_gps_location",
          "connector": "ros2",
          "config": {
            "gps_serial_port": "/dev/gps",
            "gps_baud": 115200,
            "kp_x": 0.6,
            "kd_x": 0.05,
            "kp_y": 0.6,
            "kd_y": 0.05,
            "kp_yaw": 2.0,
            "kd_yaw": 0.2
          }
        },
        {
          "name": "navigate_dwa",
          "llm_label": "navigate_with_obstacle_avoidance",
          "connector": "ros2",
          "config": {
            "penalty": 13.0,
            "margin": 1.2,
            "v_max": 0.9,
            "w_max": 0.75,
            "ahead_m": 2.0,
            "half_width_m": 1.2
          }
        },
        {
          "name": "navigate_go2",
          "llm_label": "control_go2_robot",
          "connector": "ros2",
          "config": {
            "use_ros2": false
          }
        }
      ],
      
      "backgrounds": [
        {
          "type": "RealsenseCamera",
          "config": {
            "depth_profile": "640x480x30",
            "color_profile": "640x480x30",
            "enable_depth": true,
            "enable_color": true
          }
        },
        {
          "type": "Go2Control",
          "config": {
            "use_ros2": false
          }
        }
      ]
    }
  },
  
  "transition_rules": [
    {
      "from_mode": "conversation",
      "to_mode": "autonomous_navigation",
      "type": "input_triggered",
      "triggers": [
        "자율 주행 모드",
        "GPS 모드",
        "자율 주행 시작"
      ],
      "cooldown_seconds": 5.0
    },
    {
      "from_mode": "autonomous_navigation",
      "to_mode": "conversation",
      "type": "input_triggered",
      "triggers": [
        "대화 모드",
        "대화 시작",
        "대화 모드로 전환"
      ],
      "cooldown_seconds": 5.0
    },
    {
      "from_mode": "*",
      "to_mode": "autonomous_navigation",
      "type": "context_aware",
      "conditions": [
        {
          "input_contains": ["GPS", "좌표", "경로 추종", "장애물 회피"]
        }
      ]
    },
    {
      "from_mode": "*",
      "to_mode": "conversation",
      "type": "context_aware",
      "conditions": [
        {
          "input_contains": ["연구소", "대화", "알려줘", "질문"]
        }
      ]
    }
  ]
}
```


---

## 12. ROS2 Workspace 구조 (최소화)

### 제안 폴더 구조

```
src/
├── ros2_ws/                          # ROS2 workspace (최소한만)
│   ├── src/
│   │   ├── realsense2_camera/       # RealSense 드라이버 (하드웨어 직접 제어)
│   │   └── unitree_go2_bridge/      # Unitree Go2 제어 브리지 (하드웨어 직접 제어)
│   ├── install/
│   ├── build/
│   └── log/
│
└── vendor_algorithms/                # Vendor 연산 로직 (OM1으로 포팅할 소스)
    ├── gps_nav/
    │   ├── controller.py            # PriorityPD, goal_to_xy 등
    │   └── nav_utils.py             # LinearPath, haversine 등
    ├── dwa_nav/
    │   ├── dwa_algorithm.py         # DWA 코스트 계산, 셀 선택
    │   └── distmap_def.py           # CUDA 거리 맵 생성
    ├── pointcloud_xyzrgb/
    │   └── pointcloud_generator.py  # PointCloud 생성 로직
    └── bev_cuda/
        └── bev_generator.py        # BEV Occupancy Grid 생성 로직
```

**이유**:
- `ros2_ws`: 하드웨어 직접 제어만 포함 (RealSense 드라이버, Go2 제어)
- `vendor_algorithms`: 연산 로직 소스 코드 (OM1으로 포팅할 참고용)
- ROS2 의존성 최소화

---

## 13. 구현 단계별 계획

### Phase 1: Vendor 코드 분석 및 포팅 준비
1. Autonomous_Navigation 연산 로직 코드 분석
2. LLMagent 구조 및 FSM 분석
3. `vendor_algorithms/` 디렉토리 생성 및 소스 코드 복사
4. 의존성 분석 (CUDA, pyubx2, pynmeagps, langchain, chromadb 등)

### Phase 2: Provider 구현 (SYS-PROVIDER)
1. `RealsenseCameraProvider` 구현 (ROS Workspace 접근 추상화)
2. `GPSProvider` 구현 (시리얼 포트 직접 읽기)
3. `PointCloudProvider` 구현 (연산 로직 포함)
4. `BEVOccupancyGridProvider` 구현 (연산 로직 포함)
5. `SegmentationProvider` 구현 (TensorRT 모델 실행)
6. `Go2ControlProvider` 구현 (직접 API 또는 ROS2 토픽)
7. `ASRProvider` 확인/확장 (기존 OM1 Provider 사용)
8. `TTSProvider` 확인/확장 (기존 OM1 Provider 사용)

### Phase 3: 연산 로직 포팅
1. GPS Navigation 알고리즘 포팅 (`PriorityPD`, `goal_to_xy`, `LinearPath`)
2. DWA 알고리즘 포팅 (코스트 계산, 셀 선택, CUDA 거리 맵)
3. BEV Occupancy Grid 생성 포팅
4. PointCloud 생성 포팅

### Phase 4: Action Connector 구현 (NAV-ACTION)
1. `NavigateGPSConnector` 구현 (포팅된 알고리즘 사용)
2. `NavigateDWAConnector` 구현 (포팅된 알고리즘 사용)
3. `NavigateGo2Connector` 구현

### Phase 5: Sensor 구현 (PER-INPUT)
1. `RealsenseCameraSensor` 구현
2. `PointCloudSensor` 구현
3. `BEVOccupancyGridSensor` 구현
4. `SegmentationSensor` 구현

### Phase 6: Background 구현 (SYS-BG)
1. `GpsBackground` 구현 (GPS Provider 초기화)
2. `RealsenseCameraBackground` 구현 (ROS2 드라이버만)
3. `Go2ControlBackground` 구현 (ROS2 브리지만, 선택적)
4. `ASRTTSBackground` 구현 (선택적, Provider 자체 초기화 가능)

### Phase 7: ROS2 Workspace 최소화
1. `src/ros2_ws/` 디렉토리 생성
2. RealSense 드라이버 패키지만 복사
3. Unitree Go2 브리지 패키지만 복사
4. `colcon build` 테스트

### Phase 8: LLMagent 통합
1. `RAGProvider` 구현 (ChromaDB 벡터 스토어)
2. `ChromaDBProvider` 구현
3. `GoSimpleAction`, `StopSimpleAction`, `SetGoalSimpleAction` 구현
4. `LabChatAction`, `GeneralChatAction` 구현
5. FSM 상태 관리 → OM1 Multi Mode Manager 통합

### Phase 9: Multi Mode 구성
1. Multi Mode Config 파일 작성
2. Mode 전환 규칙 구현
3. Mode별 리소스 관리 구현
4. Mode 전환 테스트

### Phase 10: 통합 테스트
1. 단위 테스트 (각 Provider/Sensor/Action)
2. Mode별 통합 테스트
3. Mode 전환 테스트
4. 전체 플로우 테스트
5. 실제 하드웨어 테스트

---

## 14. 기술적 고려사항

### ROS2 초기화
- `rclpy.init()`는 한 번만 호출 (singleton pattern)
- Provider는 ROS2 Node로 동작하지 않을 수 있음 (GPS Provider는 시리얼 직접 읽기)
- Background에서 ROS2 노드 프로세스 관리 (최소화)

### CUDA 의존성
- PointCloud, BEV, DWA 거리 맵은 CUDA 필요
- GPU 환경 체크 및 fallback 로직
- CUDA 코드는 OM1 Provider/Action Connector 내부에 포함

### GPS 시리얼 통신
- GPS Provider는 시리얼 포트 직접 접근 (ROS2 노드 불필요)
- UBX/NMEA 파싱 로직 포함
- NTRIP 클라이언트 구현 (RTK 보정)

### Unitree Go2 제어
- 직접 SportClient API 호출 (선택 1, ROS2 불필요)
- 또는 ROS2 토픽 `/cmd` 발행 (선택 2, `unitree_go2_bridge`가 구독)
- Config에서 선택 가능

### 연산 로직 포팅
- Vendor 코드를 OM1으로 포팅 시 의존성 최소화
- 공통 유틸리티는 `src/utils/navigation/`에 배치
- CUDA 코드는 Provider/Action Connector 내부에 포함

### LLMagent 통합
- LangChain ReAct 에이전트 → OM1 LLM System 통합
- FSM 상태 관리 → OM1 Multi Mode Manager 통합
- RAG 기능 → OM1 Provider로 추상화
- Tools → OM1 Actions로 변환

### Multi Mode 전환
- Mode 전환 시 상태 보존
- Mode별 리소스 초기화/정리
- 부드러운 전환 (현재 작업 완료 후 또는 즉시 전환)

---

## 15. 파일 구조 요약

```
src/
├── ros2_ws/                          # ROS2 workspace (최소화)
│   ├── src/
│   │   ├── realsense2_camera/       # RealSense 드라이버만
│   │   └── unitree_go2_bridge/      # Go2 제어 브리지만
│   ├── install/
│   ├── build/
│   └── log/
│
├── vendor_algorithms/                # Vendor 연산 로직 (참고용)
│   ├── gps_nav/
│   ├── dwa_nav/
│   ├── pointcloud_xyzrgb/
│   └── bev_cuda/
│
├── inputs/plugins/
│   ├── realsense_camera.py           # PER-001
│   ├── pointcloud.py                 # PER-002
│   ├── bev_occupancy_grid.py        # PER-003
│   └── segmentation.py               # PER-004
│
├── actions/
│   ├── navigate_gps/
│   │   ├── connector/ros2.py         # NAV-001 (연산 로직 포함)
│   │   └── interface.py
│   ├── navigate_dwa/
│   │   ├── connector/ros2.py         # NAV-002 (연산 로직 포함)
│   │   └── interface.py
│   └── navigate_go2/
│       ├── connector/ros2.py         # NAV-003
│       └── interface.py
│
├── providers/
│   ├── realsense_camera_provider.py # SYS-PROVIDER-000 (ROS Workspace 추상화)
│   ├── gps_provider.py              # SYS-PROVIDER-001 (시리얼 직접 읽기)
│   ├── pointcloud_provider.py       # SYS-PROVIDER-002 (연산 로직 포함)
│   ├── bev_occupancy_grid_provider.py # SYS-PROVIDER-003 (연산 로직 포함)
│   ├── segmentation_provider.py     # SYS-PROVIDER-004 (TensorRT)
│   ├── go2_control_provider.py       # SYS-PROVIDER-005 (직접 API 또는 ROS2)
│   ├── asr_provider.py              # SYS-PROVIDER-006 (마이크 입력, PyAudio)
│   └── riva_tts_provider.py         # SYS-PROVIDER-007 (스피커 출력, PyAudio)
│
├── utils/navigation/                 # 공통 유틸리티 (선택적)
│   ├── gps_utils.py                 # goal_to_xy, PriorityPD 등
│   └── dwa_utils.py                 # DWA 알고리즘 등
│
├── actions/
│   ├── go_simple/                   # LLMagent용 간단한 이동
│   │   ├── connector/ros2.py
│   │   └── interface.py
│   ├── stop_simple/                  # LLMagent용 정지
│   │   ├── connector/ros2.py
│   │   └── interface.py
│   ├── set_goal_simple/              # LLMagent용 목표 설정
│   │   ├── connector/ros2.py
│   │   └── interface.py
│   ├── lab_chat/                    # 연구소 대화 (RAG)
│   │   ├── connector/rag.py
│   │   └── interface.py
│   └── general_chat/                # 일반 대화
│       ├── connector/ollama.py
│       └── interface.py
│
├── providers/
│   ├── rag_provider.py              # SYS-PROVIDER-008 (RAG 체인)
│   └── chromadb_provider.py         # SYS-PROVIDER-009 (ChromaDB 벡터 스토어)
│
└── backgrounds/plugins/
    ├── gps.py                       # SYS-BG-000 (GPS Provider 초기화)
    ├── realsense_camera.py          # SYS-BG-001 (ROS2 드라이버만)
    ├── go2_control.py               # SYS-BG-002 (ROS2 브리지만, 선택적)
    ├── asr_tts.py                   # SYS-BG-003 (ASR/TTS Provider 초기화, 선택적)
    └── chromadb.py                 # SYS-BG-004 (ChromaDB 초기화)
```

---

## 16. 모듈 매핑 요약

### Perception Inputs (PER-INPUT)

| Vendor Node | OM1 Module | Requirement ID | 파일 위치 | ROS2 노드 필요 여부 |
|------------|-----------|----------------|----------|-------------------|
| RealSense Camera | `RealsenseCameraSensor` | PER-001 | `src/inputs/plugins/realsense_camera.py` | ✅ (드라이버만) |
| Segmentation | `SegmentationSensor` | PER-002 | `src/inputs/plugins/segmentation.py` | ❌ |
| PointCloud | `PointCloudSensor` | PER-003 | `src/inputs/plugins/pointcloud.py` | ❌ |
| BEV Occupancy Grid | `BEVOccupancyGridSensor` | PER-004 | `src/inputs/plugins/bev_occupancy_grid.py` | ❌ |

### Navigation Actions (NAV-ACTION)

| Vendor Node | OM1 Module | Requirement ID | 파일 위치 | ROS2 노드 필요 여부 |
|------------|-----------|----------------|----------|-------------------|
| GPS Global Planner | `NavigateGPSConnector` | NAV-001 | `src/actions/navigate_gps/connector/ros2.py` | ❌ (연산 로직 포팅) |
| DWA Local Planner | `NavigateDWAConnector` | NAV-002 | `src/actions/navigate_dwa/connector/ros2.py` | ❌ (연산 로직 포팅) |
| Go2 Control Bridge | `NavigateGo2Connector` | NAV-003 | `src/actions/navigate_go2/connector/ros2.py` | ⚠️ (선택적) |

### Providers (SYS-PROVIDER)

| Vendor Node | OM1 Provider | Requirement ID | 파일 위치 | ROS2 노드 필요 여부 |
|------------|-------------|----------------|----------|-------------------|
| RealSense Camera | `RealsenseCameraProvider` | SYS-PROVIDER-000 | `src/providers/realsense_camera_provider.py` | ⚠️ (선택적, ROS2 토픽 또는 SDK 직접) |
| GPS 시리얼 읽기 | `GPSProvider` | SYS-PROVIDER-001 | `src/providers/gps_provider.py` | ❌ (시리얼 직접 읽기) |
| PointCloud 생성 | `PointCloudProvider` | SYS-PROVIDER-002 | `src/providers/pointcloud_provider.py` | ❌ (연산 로직 포함) |
| BEV Occupancy Grid | `BEVOccupancyGridProvider` | SYS-PROVIDER-003 | `src/providers/bev_occupancy_grid_provider.py` | ❌ (연산 로직 포함) |
| Segmentation | `SegmentationProvider` | SYS-PROVIDER-004 | `src/providers/segmentation_provider.py` | ❌ (TensorRT 직접 실행) |
| Go2 Control | `Go2ControlProvider` | SYS-PROVIDER-005 | `src/providers/go2_control_provider.py` | ⚠️ (선택적, 직접 API 가능) |
| ASR (마이크 입력) | `ASRProvider` | SYS-PROVIDER-006 | `src/providers/asr_provider.py` | ❌ (PyAudio 직접 접근) |
| TTS (스피커 출력) | `RivaTTSProvider` | SYS-PROVIDER-007 | `src/providers/riva_tts_provider.py` | ❌ (PyAudio 직접 출력) |

### Backgrounds (SYS-BG)

| Vendor Node | OM1 Background | Requirement ID | 파일 위치 | ROS2 노드 필요 여부 |
|------------|---------------|----------------|----------|-------------------|
| GPS | `GpsBackground` | SYS-BG-000 | `src/backgrounds/plugins/gps.py` | ❌ (Provider 초기화만) |
| RealSense Camera | `RealsenseCameraBackground` | SYS-BG-001 | `src/backgrounds/plugins/realsense_camera.py` | ✅ (드라이버만) |
| Go2 Control | `Go2ControlBackground` | SYS-BG-002 | `src/backgrounds/plugins/go2_control.py` | ⚠️ (선택적) |
| ASR/TTS | `ASRTTSBackground` | SYS-BG-003 | `src/backgrounds/plugins/asr_tts.py` | ❌ (선택적, Provider 자체 초기화 가능) |

---

## 17. 데이터 플로우 상세

### Perception Pipeline (연산 로직 OM1 내부)

```
RealSense Camera (ROS2 Node - 드라이버만)
    ↓ /camera/color/image_raw, /camera/depth/image_rect_raw
RealsenseProvider (ROS2 subscriber)
    ↓ latest_image_data
    ↓
PointCloudProvider (OM1 내부 연산)
    ↓ (연산 로직: vendor 포팅)
    - RealSense depth → PointCloud 변환 (CUDA)
    ↓ latest_pointcloud
    ↓
BEVProvider (OM1 내부 연산)
    ↓ (연산 로직: vendor 포팅)
    - PointCloud → BEV Occupancy Grid 변환 (CUDA)
    ↓ latest_occupancy_grid
    ↓
BEVSensor
    ↓ formatted_latest_buffer() → "Obstacles detected: ..."
    ↓
InputOrchestrator → Fuser → LLM
```

### Navigation Pipeline (연산 로직 OM1 내부)

```
LLM → NavigateGPSConnector
    ↓ (비즈니스 로직: LLM 입력 파싱)
    - "go to lat=37.xxx, lon=127.xxx" → GPS 좌표 추출
    ↓ (연산 로직: vendor 포팅, OM1 내부)
    - goal_to_xy() (GPS 좌표 → 로봇 기준 변환)
    - PriorityPD.step() (경로 추종 제어)
    - LinearPath (경로 추종)
    ↓ (제어 명령 생성)
    - vx, vy, vyaw 계산
    ↓
Go2ControlProvider
    ↓ (직접 API 호출 또는 ROS2 토픽)
    - Unitree Go2 SportClient API (선택 1)
    - 또는 /cmd 토픽 발행 (선택 2, unitree_go2_bridge가 구독)
    ↓
Unitree Go2 Robot
```

```
LLM → NavigateDWAConnector
    ↓ (비즈니스 로직: LLM 입력 파싱)
    - "move forward 2 meters" → 목표 위치 계산
    ↓ (연산 로직: vendor 포팅, OM1 내부)
    - CUDA 거리 맵 생성 (build_dist_map_bfs_cuda)
    - 코스트 계산: (x-dx)^2 + (y-dy)^2 + penalty * (1 - d/margin)^2
    - 전방 창에서 최소 코스트 셀 선택
    - 속도 명령 생성 (vx, vyaw)
    ↓
Go2ControlProvider
    ↓ (직접 API 호출 또는 ROS2 토픽)
    - Unitree Go2 SportClient API (선택 1)
    - 또는 /cmd 토픽 발행 (선택 2)
    ↓
Unitree Go2 Robot
```

---

## 18. 모듈 매핑 요약 (LLMagent 포함)

### LLMagent → OM1 매핑

| LLMagent Component | OM1 Module | Requirement ID | 파일 위치 |
|-------------------|-----------|----------------|----------|
| LLMAgentNode | `OllamaLLM` (확장) | AGT-LLM-001 | `src/llm/plugins/ollama_llm.py` |
| go Tool | `GoSimpleAction` | NAV-004 | `src/actions/go_simple/connector/ros2.py` |
| stop Tool | `StopSimpleAction` | NAV-005 | `src/actions/stop_simple/connector/ros2.py` |
| set_goal Tool | `SetGoalSimpleAction` | NAV-006 | `src/actions/set_goal_simple/connector/ros2.py` |
| lab_chat Tool | `LabChatAction` | AGT-007 | `src/actions/lab_chat/connector/rag.py` |
| general_chat Tool | `GeneralChatAction` | AGT-008 | `src/actions/general_chat/connector/ollama.py` |
| FSMNode | `ModeManager` (확장) | SYS-CORE-002 | `src/runtime/multi_mode/manager.py` |
| RAG Chain | `RAGProvider` | SYS-PROVIDER-008 | `src/providers/rag_provider.py` |
| ChromaDB | `ChromaDBProvider` | SYS-PROVIDER-009 | `src/providers/chromadb_provider.py` |

---

## 19. 다음 단계

1. **Vendor 코드 분석 및 포팅 준비**
   - `vendor_algorithms/` 디렉토리 생성
   - Autonomous_Navigation 연산 로직 코드 분석
   - LLMagent 구조 및 FSM 분석
   - 의존성 정리

2. **Provider 구현 시작**
   - `GPSProvider` (시리얼 직접 읽기)
   - `PointCloudProvider` (연산 로직 포함)
   - `BEVOccupancyGridProvider` (연산 로직 포함)
   - `RAGProvider`, `ChromaDBProvider` (LLMagent용)

3. **연산 로직 포팅**
   - GPS Navigation 알고리즘 포팅
   - DWA 알고리즘 포팅
   - BEV/PointCloud 생성 포팅

4. **Action Connector 구현**
   - Autonomous_Navigation Actions (포팅된 알고리즘 사용)
   - LLMagent Actions (간단한 제어, 대화)

5. **Multi Mode 구성**
   - Multi Mode Config 파일 작성
   - Mode 전환 규칙 구현
   - Mode별 리소스 관리

6. **ROS2 Workspace 최소화**
   - RealSense 드라이버만 포함
   - Unitree Go2 브리지만 포함 (선택적)

7. **통합 테스트**
   - Mode별 통합 테스트
   - Mode 전환 테스트
   - 전체 플로우 테스트
   - 실제 하드웨어 검증
