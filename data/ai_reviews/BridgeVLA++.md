# BridgeVLA++: A Data-Efficient, Generalizable, and Memory-Augmented Vision-Language-Action Framework for 3D Manipulation

> **한 줄 요약**: PaliGemma의 입출력을 2D heatmap 공간에 정렬한 3D VLA인 BridgeVLA에 temporal/spatial 통합 메모리(269.77M, +9.2%)를 얹어, RLBench 18-task 93.7%, COLOSSEUM 65.2%, GemBench 51.1%, MemoryBench 99.7%, 그리고 dual-arm RMBench 96.0%를 동시에 달성한 memory-augmented 3D 조작 프레임워크.

- arXiv: 2608.05042 (cs.RO, 2026-08-05), IEEE TPAMI 투고본
- 소속: CASIA NLPR / UCAS / FiveAges / ByteDance Seed
- NeurIPS 2025 conference paper(BridgeVLA)의 저널 확장판

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 대부분의 VLA는 2D 이미지 위에서 동작하며 대량의 로봇 데이터를 요구함. 반면 3D 정책은 기하 구조를 활용해 sample efficiency가 높지만 VLM prior를 못 쓴다.
- 기존 3D VLA는 action을 **token sequence로 autoregressive 예측**하여, 3D 관측과 action 사이의 공간적 대응(spatial correspondence)을 버린다. 또한 3D 입력을 VLM에 넣는 순간 2D 사전학습과의 modality gap이 생겨 prior 전이가 막힌다.
- 세 번째 한계는 **메모리 부재**: 대부분 정책이 현재 관측만으로 action을 예측하므로, (a) 과거 상호작용에 따라 정답이 달라지는 태스크, (b) 초기에 보였던 기하가 로봇 팔/물체에 가려지는 상황에서 실패한다.

### 핵심 질문 (논문의 Q1~Q5)
- 사전학습 VLM의 semantic generalization과 3D 정책의 geometric efficiency를 하나의 모델에서 결합할 수 있는가?
- 메모리를 붙이면서도 원래의 data efficiency와 OOD generalization을 **깎지 않을** 수 있는가?

---

## 2. 방법론 심층 분석

### 2.1 BridgeVLA (base): input-output alignment
- 백본은 **PaliGemma** = SigLIP vision encoder + Gemma transformer (2.92B).
- 핵심 아이디어: 사전학습과 downstream을 **같은 2D visual localization 공간**에서 수행한다.
  - Pre-training: 이미지 + "찾을 물체" 텍스트 → 물체 중심에 truncated Gaussian을 렌더링해 정규화한 GT heatmap을 cross-entropy(식 8)로 학습.
  - Fine-tuning: point cloud를 top/front/right **3개 orthographic view**로 렌더 → 같은 VLM이 view별 translational heatmap 예측.
- 출력 image token을 원래 patch 위치대로 2D grid로 재배열한 뒤 **convex upsampling**(고정 bilinear 대신 공간 가변 보간 가중치 예측)으로 입력 해상도 heatmap 복원.

### 2.2 3D action 복원
- workspace 안에서 후보 3D 좌표를 균일 샘플링 → 3 view에 투영해 heatmap 값을 합산(식 9), argmax를 다음 keyframe의 end-effector translation으로 선택(식 10).
- **proprioception(관절 상태/EE pose)을 VLM forward에 넣지 않는다.** 사전학습의 image+language 입력 포맷을 그대로 유지해 distribution shift를 줄이기 위함.
- Rotation/gripper/collision: 3개 view의 global feature와 heatmap 최대점 위치의 local feature를 concat → **3-layer MLP**. Rotation은 discretized Euler가 아니라 **연속 6D 표현** + Gram-Schmidt 직교화. Gripper/collision은 2-class softmax.
- **Coarse-to-fine**: 1차 pass로 전체 workspace에서 coarse translation → 그 지점 중심 cuboid로 point cloud를 crop/확대 → 동일 백본으로 2차 pass. 두 pass는 파라미터를 공유하고 투영 범위만 다르다.

### 2.3 BridgeVLA++의 통합 spatio-temporal memory
- **Temporal memory (~168M)**: 초기 관측의 anchor view, 최근 neighboring keyframe, 그리고 adaptive하게 선택된 sub-goal keyframe의 **인코딩된 image token**을 슬롯 버퍼에 유지. "다음에 무엇을 할지(what to do next)"를 결정. token을 저장하므로 과거 관측을 재인코딩하지 않는다.
- **Spatial memory (~84M)**: 초기 관측의 컬러 point cloud를 보관했다가 매 decision step마다 (coarse waypoint에 따라 crop 범위가 달라지므로) 재렌더/재인코딩. 로봇 팔이나 물체에 가려진 영역의 기하를 복원해 "정확히 어디에 작용할지(where exactly to act)"를 결정.
- **Adaptive sub-goal selection (~18M)**: 현재 keyframe을 sub-goal로 유지할지 binary 분류. `L_check` (BCE)로 지도학습, 전체 목적함수 `L = L_base + λ_check · L_check`.
- 메모리 주입 후에도 token grid shape가 보존되므로 convex upsampling / action head는 **수정 없이** 재사용된다.

### 2.4 Bimanual 확장
- 메모리는 arm-specific이 아니라 scene-level이므로, convex upsampling + MLP head만 팔별로 복제하고 VLM 백본·temporal/spatial memory·selection 모듈은 공유. Coarse에서 팔별 waypoint, fine에서 팔별 zoom crop.

---

## 3. 데이터 전략

- **2D heatmap pre-training**: object grounding용 detection 데이터에서 box center를 Gaussian heatmap으로 변환. 논문은 keypoint/segmentation 등 "공간 타깃으로 변환 가능한 모든 VL 데이터"로 확장 가능하다고 주장(scalability 논거).
- **Fine-tuning demo 수**: RLBench 100 demos/task, COLOSSEUM은 unperturbed RLBench 20 tasks × 100 demos, GemBench 16-task train split, **RMBench 50 demos/task**, MemoryBench 100 demos/task.
- **실기**: Franka 13 tasks × 10 demos/task (참조 행은 3 demos/task), Dobot CR5A는 language instruction당 10 demos.
- 학습 시 메모리 샘플은 해당 expert demo의 선행 관측에서 구성하며, random rigid-body augmentation을 현재 관측·메모리 관측·GT action에 **일관되게** 적용해 기하 정합성을 유지.

---

## 4. 시스템/학습 세부사항

- 파인튜닝은 2-phase: 초기 freeze epoch 동안 PaliGemma 백본 동결, 이후 해제.
- 학습 자원: RLBench/COLOSSEUM/GemBench는 공용 설정, RMBench는 **task당 1 모델**을 8× H20에서, MemoryBench는 8× A100.
- 평가는 run당 GPU 1장 (RMBench/GemBench/MemoryBench/COLOSSEUM은 A100, RLBench는 H20).
- **추가 파라미터 269.77M = 2.92B 백본 대비 +9.2%**.
- **추론 지연**: RTX 4090 단일 GPU에서 BridgeVLA 0.35 s/step, BridgeVLA++ 0.57 s/step (약 1.75 Hz). 저자들은 keyframe 기반 제어에서 관측 전송·모션 실행 시간이 지배적이라 이 차이가 미미하다고 논증.
- RMBench 슬롯 예산 K=12.

---

## 5. 실험 설계 및 평가 프로토콜

- **RLBench** (Table I): 18 tasks, CoppeliaSim + Franka Panda, RGB-D 4대. 5 seed × 25 episode. Avg SR와 Avg Rank 동시 보고.
- **COLOSSEUM** (Table III): 12 perturbation 축 + 원본 RLBench variation + All Perturb. = **14 설정**. RVT-2/BridgeVLA/BridgeVLA++는 저자들이 직접 학습·평가(3회 반복).
- **GemBench** (Table XI): L1 placement / L2 novel rigid / L3 novel articulated / L4 long-horizon. 5 seed, variation당 20 trial. 저자 모델은 **keyframe만, demo augmentation 없이** 학습(불리한 조건 명시).
- **RMBench** (Table II): RoboTwin 2.0 위에 구축된 dual-arm episodic-memory 벤치마크 9 tasks. M(1) 단기 / M(n) 장기 그룹. task당 100 episode 단일 평가, task별 best checkpoint 선택.
- **MemoryBench** (Table XII): single-arm memory 3 tasks(9 variants), 25-step budget, 5 seed.
- **실기**: Franka Research 3 + ZED 2i (13 tasks, 6개 generalization 설정), Dobot CR5A (memory 3 + memory-free 2 tasks, 5개 설정).

---

## 6. 실험 결과 심층 분석

### RLBench 18 tasks (Table I, Avg SR %)
| Method | Avg SR | Avg Rank |
|---|---|---|
| PerAct | 49.4 | 11.33 |
| RVT | 62.9 | 9.08 |
| Act3D | 65.0 | 9.17 |
| 3D Diffuser Actor | 81.3 | 6.19 |
| RVT-2 | 81.4 | 5.97 |
| SAM2Act | 86.8 ±0.5 | 5.47 |
| **BridgeVLA** | **90.5 ±1.1** | 4.75 |
| **BridgeVLA++** | **93.7 ±0.6** | **3.64** |

- BridgeVLA가 이전 SOTA SAM2Act를 +3.7pt 상회. 특히 정밀 태스크(Stack Cups 88.8)에서 dense heatmap의 이점이 뚜렷.
- BridgeVLA의 주 실패 모드는 **occlusion**(fine-localization 단계에서 로봇 팔이 타깃을 가림) → 메모리 도입 후 Sort Shape +16.8pt(55.2→72.0), Place Cups +18.4pt(58.4→76.8), Stack Cups 88.8→98.4.

### COLOSSEUM (Table III, 14 설정 평균)
| Method | Avg SR | All Perturb. |
|---|---|---|
| PerAct | 27.9 | 7.2 |
| RVT | 35.4 | 6.4 |
| RVT-2 | 56.7 | 15.6 |
| **BridgeVLA** | **64.0** | 18.7 |
| **BridgeVLA++** | **65.2** | **38.9** |

- BridgeVLA는 RVT-2 대비 +7pt 이상. 흥미로운 점은 **All Perturb.에서 BridgeVLA++가 18.7 → 38.9로 2배 이상** 뛴 것 — 모든 교란이 동시에 걸린 최악 조건에서 메모리가 특히 효과적.
- 다만 개별 축에서는 BridgeVLA++가 BridgeVLA보다 낮은 항목도 존재(Table Texture 75.7→71.5, Background 71.3→69.2, Camera Pose 73.8→68.7). 평균 우위는 Distractor(51.8→61.6)와 All Perturb.가 견인.

### GemBench (Table XI)
| Method | Avg | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| 3D-LOTUS | 45.7 | 94.3 | 49.9 | 38.1 | 0.3 |
| 3D-LOTUS++ | 48.0 | 68.7 | 64.5 | 41.5 | 17.4 |
| **BridgeVLA** | **50.0** | 91.1 | 65.0 | **43.8** | 0.0 |
| **BridgeVLA++** | **51.1** | 88.6 | **68.9** | 38.5 | 8.2 |

- BridgeVLA는 L4(long-horizon)에서 **0.0%** — heatmap 기반 keyframe 정책의 명확한 약점. BridgeVLA++는 메모리로 8.2까지 올렸지만 3D-LOTUS++(17.4)에 여전히 못 미친다.
- 반대로 L2(novel rigid object)에서는 68.9로 최고. VLM semantic prior 전이의 효과.

### RMBench dual-arm (Table II)
| Method | Overall | M(1) Avg | M(n) Avg |
|---|---|---|---|
| DP | 5.8 | 6.4 | 5.0 |
| ACT | 5.9 | 6.8 | 4.8 |
| π0.5 | 10.4 | 14.4 | 5.5 |
| X-VLA | 9.8 | 11.8 | 7.3 |
| Mem-0 | 42.0 | 52.8 | 28.5 |
| LingBot-VA | 78.2 | 80.0 | 76.0 |
| MemoryWAM | 83.0 | 84.2 | 81.5 |
| **BridgeVLA** | **18.9** | 19.0 | 18.8 |
| **BridgeVLA++** | **96.0** | **95.2** | **97.0** |

- 이 표가 논문에서 가장 극적인 결과다. **메모리 없는 base가 18.9%로 붕괴**하고, 메모리를 붙이면 96.0%. MemoryWAM 대비 +13.0pt, Mem-0 대비 +54.0pt, 9개 중 8개 태스크에서 best/tied-best.
- Battery Try(과거 시도 추적이 필요한 trial-and-error 정렬)에서 96% vs 최강 baseline 41%.
- 주의: BridgeVLA base는 Observe & Pick Up에서 75(최고 수준)인데 나머지 8개는 0~11 — 즉 순수 지각 태스크는 되지만 episodic 태스크는 전혀 못 푼다는 것이 깔끔하게 드러난다.

### MemoryBench single-arm (Table XII)
| Method | Avg | Reopen Drawer | Put Block Back | Rearrange Block |
|---|---|---|---|---|
| RVT-2 | 54.0 | 60.0 | 50.0 | 52.0 |
| SAM2Act | 55.0 | 48.0 | 35.0 | 82.0 |
| SAM2Act+ | 94.3 | 84.0 | 100.0 | 99.0 |
| **BridgeVLA** | **11.3** | 29.6 | 2.8 | 1.6 |
| **BridgeVLA++** | **99.7 ±0.3** | **100.0** | 99.8 | **99.2** |

### 실기 (Table IV, Fig. 4)
- Franka basic (13 tasks × 10 trial, 10 demos/task): SpatialVLA(50 demos) 28.5, π0.5 20.0, ACT 21.5, RVT-2 90.0, **BridgeVLA 96.9**. **3 demos만으로도 95.4%**.
- 7개 generalization 설정 평균에서 RVT-2 대비 **+32%**. Fig. 4 요약: memory-free 태스크 75.0% → BridgeVLA, memory-dependent 태스크 30.0% → BridgeVLA++ 93.3%.
- SpatialVLA도 3D 정보를 쓰지만 데이터 효율이 훨씬 낮다는 점 → "3D 정보 자체보다 관측-행동 공간 정렬 설계가 결정적"이라는 저자 주장의 근거.

---

## 7. Ablation 분석

### BridgeVLA 아키텍처 ablation (Table I, RLBench Avg)
| Variant | Avg SR |
|---|---|
| BridgeVLA (full) | 90.5 |
| w/ discretized rotation | 88.2 |
| w/ 3D position input | 56.2 |
| **w/o heatmap decoding** | **31.4** |

- **heatmap decoding 제거 시 90.5 → 31.4 (-59.1pt)**. 이 논문의 핵심 주장(입출력 정렬)을 가장 강하게 뒷받침하는 수치. Put in Drawer / Stack Blocks / Stack Cups는 0.0%로 완전 붕괴.
- 연속 6D rotation → discretized Euler로 바꾸면 88.2. Euler 분해의 gimbal-lock/특이점 문제 때문이라고 설명(roll/yaw).
- VLM forward에 3D position을 직접 넣으면 56.2로 급락 → 2D 사전학습과의 modality gap이 실재함을 확인.

### 메모리 ablation
| Variant | RLBench | RMBench |
|---|---|---|
| BridgeVLA++ | 93.7 | 96.0 |
| w/o spatial memory (S) | 92.0 | — |
| w/o temporal memory (T) | 91.9 | 60.8 |

- **Spatial memory 제거**: RLBench 93.7→92.0. 특히 Place Wine 95.2→78.4로 occlusion 민감 태스크에서 손실이 집중.
- **Temporal memory 제거**: RMBench 96.0 → 60.8로 붕괴. 반면 RLBench는 91.9로 소폭만 하락 → T는 episodic 태스크 전용 기여가 압도적. 다만 T가 RLBench에서도 이득(93.7 vs 91.9, 특히 Insert Peg 82.4→99.2)을 준다는 점을 저자들이 강조.
- 요약하면 **S = "어디에" (occlusion 복원), T = "무엇을" (episodic 문맥)** 이라는 역할 분리가 실험적으로 확인된다.

---

## 8. 강점

1. **입출력 정렬이라는 단일 아이디어의 설명력**: heatmap decoding ablation(-59.1pt)이 이 프레임의 정당성을 거의 단독으로 입증한다.
2. **극단적 data efficiency**: 실기에서 **3 demos/task로 95.4%**. 50 demos를 쓴 SpatialVLA(28.5%)와 대비하면 한 자릿수 배가 아니라 질적 차이.
3. **메모리 추가가 기존 능력을 훼손하지 않음**: RLBench 90.5→93.7, COLOSSEUM 64.0→65.2, GemBench 50.0→51.1로 모두 상승. 보통 메모리 확장에서 흔한 trade-off가 관측되지 않는다.
4. **비용 대비 효과**: +9.2% 파라미터, +0.22 s/step으로 RMBench 18.9→96.0. 
5. **평가 폭**: 시뮬 5개 벤치마크 + 2개 실기 embodiment(Franka, Dobot) + 단완/양완 모두 커버.
6. **정직한 비교 설정 공개**: GemBench에서 자기 모델을 keyframe-only, demo augmentation 없이 학습했다고 명시.

## 9. 한계 및 비판적 검토

1. **Long-horizon 취약**: GemBench L4에서 BridgeVLA 0.0%, BridgeVLA++ 8.2%로 3D-LOTUS++(17.4)에 열세. keyframe + heatmap 패러다임이 긴 태스크 분해에 약하다는 구조적 한계.
2. **RMBench 프로토콜의 관대함**: task당 1 모델을 따로 학습하고 **각 학습 run의 best checkpoint를 선택**한다고 명시. 96.0%는 multi-task 단일 정책 수치가 아니며, best-checkpoint 선택은 낙관 편향을 준다. baseline과 동일 조건인지 검증 필요.
3. **COLOSSEUM 개별 축 퇴행**: 평균은 올랐지만 Table Texture/Background/Camera Pose/Light Color 등에서 base보다 낮다. 메모리가 초기 관측을 참조하므로 시점·외형 변화에 오히려 민감해질 수 있다는 해석이 가능한데 논문은 이를 깊이 다루지 않는다.
4. **제어 주파수**: 0.57 s/step ≈ 1.75 Hz. keyframe 기반이라 정당화되지만 동적/접촉 풍부한 태스크에는 적용 불가.
5. **오픈소스 여부 불명확**: 프로젝트 페이지(bridgevla-plus.github.io)만 제공되며 본문에 코드/체크포인트 공개 약속이 명시되지 않는다.
6. **Spatial memory의 가정**: "초기 관측이 덜 가려져 있다"는 전제에 의존. 초기부터 타깃이 가려진 경우나 장면이 크게 변하는 경우 stale geometry가 오히려 해가 될 수 있다.
7. **PaliGemma 의존**: 백본 교체 시 heatmap pre-training을 다시 해야 하며, 다른 VLM으로의 일반성은 검증되지 않았다.

## 10. 관련 연구와의 위치

- **RVT / RVT-2** (multi-view orthographic projection, coarse-to-fine): BridgeVLA는 이 렌더링 파이프라인을 그대로 계승하되 transformer를 **사전학습된 VLM**으로 교체하고, 그 정렬을 위해 2D heatmap 사전학습 단계를 추가했다. RLBench 81.4→90.5가 그 순수 이득에 가깝다.
- **SAM2Act / SAM2Act+**: 메모리 뱅크로 RLBench·MemoryBench를 공략한 직전 SOTA. BridgeVLA++는 MemoryBench 94.3→99.7로 추월.
- **3D Diffuser Actor**: diffusion으로 3D trajectory 생성. BridgeVLA는 diffusion 없이 heatmap argmax만으로 이를 상회(81.3 → 90.5).
- **SpatialVLA / π0.5 / X-VLA**: 2D·3D VLA 계열. 실기와 RMBench 모두에서 크게 밀림 — 대규모 사전학습 VLA가 저데이터 3D 정밀 조작에서는 약하다는 것을 다시 보여준다.
- **MemoryWAM / LingBot-VA / Mem-0**: RMBench의 메모리 특화 baseline. BridgeVLA++가 최고치.

## 11. 재현 및 실무 적용 관점

- **필요 자원**: 학습 8× H20 또는 8× A100. 추론은 RTX 4090 1장으로 충분(0.57 s/step) — 배포 문턱은 낮은 편.
- **센서 요구**: 캘리브레이션된 RGB-D 1대 이상. proprioception 불필요.
- **적용 적합 시나리오**: keyframe으로 분해 가능한 pick-and-place/삽입/서랍 조작, 데모 수집이 비싼 산업 환경(3~10 demos), 그리고 "이전에 무엇을 했는지" 추적이 필요한 순차 태스크.
- **부적합**: 고주파 제어, 유연물/접촉 풍부 조작, 매우 긴 horizon의 자유형 태스크.
- **재현 리스크**: RMBench의 per-task 학습 + best-checkpoint 프로토콜, GemBench의 keyframe-only 설정 등 세부 조건이 결과에 크게 영향을 주므로 그대로 따라야 한다.

## 12. 총평

BridgeVLA++는 "새로운 손실함수/새 백본"이 아니라 **표현 공간을 통일한다**는 한 가지 설계 원칙을 끝까지 밀어붙인 논문이고, heatmap decoding ablation(-59.1pt)이 그 원칙의 값을 명료하게 증명한다. 확장부인 spatio-temporal memory는 흔한 "성능 조금 올리는 모듈"이 아니라 RMBench 18.9→96.0이라는 **능력의 유무를 가르는** 변화이며, 그것을 +9.2% 파라미터로 달성하면서 기존 벤치마크 성능까지 동반 상승시킨 점이 인상적이다. 다만 GemBench L4 8.2%가 보여주듯 long-horizon 조합 태스크는 여전히 미해결이고, RMBench의 per-task/best-checkpoint 프로토콜은 수치를 액면 그대로 받기 어렵게 만든다. 3D keyframe 조작 영역에서 현재 가장 강력한 참조 구현으로 보되, 벤치마크 간 프로토콜 차이를 함께 읽어야 하는 논문이다.

<!-- VERIFIED: pdf -->
