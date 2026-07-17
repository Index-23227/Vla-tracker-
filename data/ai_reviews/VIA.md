# VIA: Visual Interface Agent for Robot Control

> **한 줄 요약**: 로봇 제어를 별도의 fine-tuning 없이 "컴퓨터 사용(computer use)" 형태의 agentic 과제로 재정의하여, 기성 frontier agent(Claude Code, Codex)가 브라우저 기반 3D 인터페이스를 MCP 도구로 조작해 로봇을 zero-shot으로 제어하는 프레임워크. 최강 구성(CC-Fable)이 LIBERO-Goal 3개 과제에서 96.7% 성공률 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 주류 접근인 VLA(Vision-Language-Action)는 기성 foundation model(FM)을 로봇 데이터로 fine-tuning하여 저수준 action을 직접 출력한다. 그러나 로봇 데이터·compute 부족으로 VLA는 frontier FM보다 수십~수백 배 작아, 물리 추론·장기 계획 능력이 제한된다.
- Fine-tuning 자체가 base FM의 추론·멀티모달 이해 능력을 훼손(catastrophic forgetting)할 수 있다.
- 또 다른 축인 Code-as-Policies(CaP)는 FM이 perception API·수작업 skill primitive를 호출하는 코드를 작성하지만, FM이 장면을 직접 지각하지 못하고 인간이 설계한 추상화에 병목이 걸린다.

### 핵심 질문
- **적절한 시각 인터페이스와 인간 친화적 도구만 주어지면, 최신 agent가 로봇 특화 학습 없이 로봇을 제어할 수 있는가?**
- 컴퓨터 사용 agent가 소프트웨어를 화면 캡처·클릭으로 조작하는 능력이, 그대로 로봇 제어로 전이되는가?

---

## 2. 방법론 심층 분석

### 2.1 핵심 아이디어
VIA는 로봇 제어를 **관찰-행동(observe-act) 루프의 agentic 과제**로 재구성한다. 기성 agent가 브라우저 기반 3D 로봇 제어 UI를 인간처럼 조작한다: 스크린샷을 찍고, 관찰한 내용을 추론하고, MCP 도구로 명령을 내리고, 결과를 관찰하고, 조정한다. 로봇 특화 fine-tuning도, 특권적 시뮬레이터 상태 접근도 없다.

### 2.2 시각 인터페이스
- 복수의 캘리브레이션된 RGB-D 카메라로 재구성한 **3D point cloud**가 주 작업 공간. 가상 카메라의 orbit/pan/zoom으로 임의 시점 관찰 가능(Blender 같은 3D 디자인 소프트웨어와 유사 → 모델의 3D 추론 능력 전이 기대).
- 좌측에 third-person(전체 장면)·wrist(근접) 카메라 원본 피드 2개.
- agent는 UI 전체를 스크린샷으로만 지각하며, point cloud의 원시 좌표 벡터는 읽지 않는다.
- **SPHINX**(Sundaresan et al., 2025)의 teleoperation 인터페이스에서 waypoint 방식 제어를 차용.

### 2.3 제어 방식
파란 target gripper를 클릭/드래그/키보드로 조작해 6-DoF 목표 pose + 개폐 상태(waypoint)를 설정한 뒤 `execute_waypoint`를 호출하면, 단순 PI 컨트롤러가 선형 보간으로 실제 gripper를 목표 pose로 이동시킨다.

---

## 3. MCP 도구 설계

Table 1의 도구는 4개 범주로 구성되며, **minimalism(각 도구는 대응하는 인간 조작을 최소 추상화로 래핑)**과 **agent ergonomics(agent는 연속 애니메이션이 아닌 이산 스크린샷으로 관찰하므로 정확한 각도·오프셋을 입력받는 직접적 대안 제공)** 두 원칙으로 설계.

| 범주 | 도구 |
|------|------|
| Observation | `screenshot`, `hover`, `gripper_get_pose`, `gripper_show_rotation_gizmo`, `camera_get_pose` |
| Action | `gripper_teleport_via_click`, `gripper_drag`, `gripper_translate`, `gripper_advance_or_retreat`, `gripper_rotate`, `gripper_toggle`, `gripper_reset` |
| Camera | `camera_orbit_via_key`, `camera_pan_via_key`, `camera_zoom`, `camera_reset` |
| Execution | `execute_waypoint`, `end_episode` |

- 대부분의 도구는 UI/환경 변화를 반영한 스크린샷을 반환해 폐루프 확인이 가능.
- Action 도구는 `execute_waypoint` 호출 전까지 실제 로봇에 영향을 주지 않음(안전한 계획·재설정).
- 좌표계: robot frame(미터), 방향은 approach(손가락이 뻗는 방향)·opening(집게가 열리는 축) 두 단위벡터로 보고. **CaP의 최저 추상화 계층보다 더 낮은 수준**(perception primitive 없이 일반 6-DoF waypoint만 명령)에서 동작.

---

## 4. Agentic 루프
- 관찰→사고→도구 호출(새 관찰 반환)의 폐루프를 종료 조건까지 반복. 이 구조가 오류 복구·재계획을 가능케 함.
- context·history 관리를 agent 자체에 위임하여 장기 계획 능력을 그대로 활용.
- 인터페이스 기본·waypoint 시퀀스 개념·개방루프보다 폐루프 검증 선호 등 일반 가이드를 담은 짧은 system prompt만 제공.

---

## 5. 실험 설정
- **6개 태블탑 조작 과제**: Stack(robosuite), Turn on stove·Open drawer·Put bowl on plate(LIBERO-Goal), Rainbow(7개 블록 무지개 배열, 저자 제작, 장기 계획 테스트), T-block(BuilderBench 재구현, 약 8mm 오차만 허용하는 정밀 과제). 모두 robosuite에서 실행.
- **Agent**: Claude Code(Opus 4.8 = CC-Opus, Fable 5 = CC-Fable), Codex(GPT-5.5 = Codex-5.5, GPT-5.6-Sol). 모두 xhigh reasoning effort.
- **프롬프트 2종**: minimal(목표·성공 조건만 → zero-shot 측정), detailed(예시 waypoint 리스트 = 텍스트 데모 추가).
- 각 구성 과제당 10 seed, 에피소드당 최대 1시간 wall-clock. cross-episode 메모리 비활성화.

---

## 6. 주요 결과

### 6.1 성공률 (Table 2, minimal prompt)
| Task | CC-Opus | CC-Fable | Codex-5.5 | Codex-5.6-Sol |
|------|--------|---------|-----------|---------------|
| Stack | 100% | 100% | 100% | 100% |
| Turn on stove | 100% | 100% | 90% | 90% |
| Open drawer | 70% | 90% | 30% | 30% |
| Put bowl on plate | 60% | 100% | 40% | 80% |
| Rainbow | 80% | 100% | 60% | 50% |
| T-block | 10% | 40% | 40% | 20% |
| **Overall** | **70%** | **88%** | **60%** | **62%** |

- **LIBERO-Goal 3개 과제(Turn on stove, Open drawer, Put bowl on plate)에서 CC-Fable은 minimal prompt만으로 96.7% 성공(29/30 seed 통과)**, 유일한 실패도 서랍을 몇 cm만 더 당겼으면 성공. Rainbow에서 CC-Fable 100%.
- CC 계열이 Codex 계열보다 전반적으로 우수. T-block은 모든 agent에 여전히 난제(10~40%, 정밀도 한계).

### 6.2 텍스트 데모의 효과
- detailed prompt(예시 waypoint 리스트)로 CC-Opus는 LIBERO-Goal 3과제에서 77% → 100%로 향상. 반면 Codex-5.5는 이득 없음. 실물 teleoperation 없이 텍스트만으로 "데모"를 만들 수 있음을 시사(learning via reflection 가능성).

### 6.3 모델 스케일링
- CC 계열에서 Opus 4.8 → Fable 5 업그레이드 시 여유가 있는 모든 과제의 성공률 상승(overall 70% → 88%). "모델이 강해질수록 프롬프트 엔지니어링이 덜 필요"한 추세 반영.

---

## 7. 비용 및 효율 (Table 3, 4)
- 성공 에피소드당 평균 MCP 도구 호출: CC-Opus 68, CC-Fable 67, Codex-5.5 64, Codex-5.6-Sol 84회.
- 성공 에피소드당 평균 API 비용: **CC-Opus $9.5, CC-Fable $15.1, Codex-5.5 $7.2, Codex-5.6-Sol $4.1**. Codex가 비용 우위(더 공격적인 context auto-compaction). CC-Fable은 2배 토큰 단가로 최고가.

---

## 8. 확장 방향
- **Automatic Tool Improvement**: 제어 agent가 동시에 강력한 코딩 agent이므로, 새 도구의 docstring을 읽고 테스트한 뒤 기능·문서를 스스로 개선하도록 활용.
- **Learning via Reflection**: 텍스트/멀티모달 reflection으로 가중치 갱신 없이 정책 개선. 태스크별 프롬프트 최적화로 성능을 크게 높일 여지(높은 천장).

---

## 9. 강점
- **로봇 특화 학습·데이터 불필요**: frontier FM의 일반 지각·추론·계획·도구 사용 능력을 그대로 로봇 제어에 전이. FM 발전이 곧바로 로봇 성능 향상으로 이어짐("foundation model scaling의 배에 올라탐").
- **폐루프 오류 복구·재계획**: agentic 루프가 관찰 기반 재계획을 내장.
- **인간 친화적·과제 불문 인터페이스**: perception primitive 없이 일반 kinematic 제어만 노출 → CaP의 primitive 설계 민감성 회피.

---

## 10. 한계
- **최고 사양 frontier 모델 필요** → inference가 느리고 비쌈(에피소드당 수 달러~수십 달러).
- 느린 inference로 **quasi-static 과제에 국한**(공 받기 같은 동적 과제 부적합).
- T-block 등 정밀 과제(≈8mm 허용 오차)는 여전히 저조 → 단순 PI 컨트롤러의 한계.
- 현재 시뮬레이션(robosuite) 평가 중심, 실물 로봇은 향후 과제.

---

## 11. 의의 및 전망
VIA는 로봇 제어를 "또 하나의 경제적 가치를 지닌 agentic 과제"로 자리매김한다. 기존 VLA 패러다임(fine-tuning으로 축소 모델 생성)과 정반대로, **fine-tuning 없이 최대·최강 FM을 그대로 활용**한다. computer-use가 최신 모델 릴리스의 핵심이 되는 흐름에서, VIA는 그 일반 능력 향상을 로봇 제어로 수확하기 좋은 위치에 있다. 향후 실물 로봇 확장, 더 정교한 컨트롤러(정밀도·충돌 회피), 그리고 느린 agent가 빠른 정책 학습용 데모를 생성하는 활용이 유망하다.

---

## 12. 총평
VIA는 "당신의 코딩/컴퓨터 사용 agent는 사실상 로봇 제어 agent"라는 도발적 주장을, 잘 설계된 3D 시각 인터페이스와 최소한의 MCP 도구 집합으로 실증한다. LIBERO-Goal에서의 96.7%(zero-shot, minimal prompt)와 7블록 무지개 과제 100%는, 적절한 인터페이스만 있으면 frontier agent의 일반 능력이 로봇 제어로 직접 전이됨을 보여주는 설득력 있는 증거다. 비용·속도·정밀도라는 실용적 제약이 남아 있으나, FM 발전에 자동으로 편승한다는 점에서 로봇 학습의 대안적 패러다임으로 주목할 가치가 크다.

<!-- VERIFIED: pdf -->
