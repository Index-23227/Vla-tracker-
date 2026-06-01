# VILAS: VLA-Integrated Low-cost Architecture with Soft Grasping 세미나 리뷰

> **한 줄 요약**: Fairino FR5(저가 6-DoF cobot, ~$4,000) + Jodell RG52-50 평행 그리퍼 + **PEBA 3D 프린트 kirigami soft 확장부**(<$100) + 듀얼 RealSense + ZMQ 모듈러 스택(~$8,000 총 비용)을 GELLO leader arm으로 텔레오퍼레이션하여 **100 episodes × 1,200 frames** 포도 파지 데이터를 수집하고, 동일 데이터로 **π₀ / π₀.₅ / GR00T N1.6** 셋을 fine-tune 하여 *단일 파지 70~84% / 다중 파지 22~58%* 성능을 비교한 **시스템·하드웨어·정책 통합 페이퍼**.

---

## 1. 배경 및 동기

### VLA 연구의 hardware-access 격차

- 최근 VLA(Vision-Language-Action) 연구는 대형 dual-arm humanoid(Optimus, GR1, Apollo) 또는 고가 산업용 매니퓰레이터(Franka Research 3, UR5e)에 집중되어 있다.
- 결과적으로 *재현성과 확장성*이 떨어지며, soft/contact-rich 조작 연구는 expensive force/tactile sensor 셋업에 의존한다.
- 농업·식품가공·생물의료 등 fragile object를 다뤄야 하는 도메인은 *force feedback 없이도* 안전한 파지가 가능한 platform이 필요하다.

### 핵심 질문

- **(a)** 공개된 *pretrained* VLA(π₀, π₀.₅, GR00T N1.6)를 *저가 ($8K급) 협동 로봇 셋업*에 fine-tune했을 때 실용적 성능이 나오는가?
- **(b)** Force/tactile sensing 없이 *형태(morphology) 자체*로 compliance를 구현하는 kirigami 변형이 fragile object 파지에 충분한가?

### VILAS의 답

- 동일 platform·동일 100-episode 데모로 세 VLA를 통제 비교 → **GR00T N1.6이 multi-grasp 58%로 최강**, π₀.₅가 single-grasp 84%로 최강. π₀는 양 metric 모두 열위.
- **<$100 PEBA filament FDM 프린트** kirigami extension만으로 force sensor 없이 포도를 안전하게 파지 가능.

📌 저자: Zijian An, Hadi Khezam, Bill Cai, Ran Yang, Shijie Geng, Yiming Feng, Yue Zheng, Lifeng Zhou — Drexel University · Virginia Tech Seafood Agricultural Research Center · Amazon Store Foundation AI. arXiv:2605.02037v2 (2026-05-22).

---

## 2. 하드웨어 플랫폼 심층 분석

### 2.1 구성 요소와 비용

| 구성 요소 | 모델 | 사양 | 비용 (USD) |
|---|---|---|---|
| Arm | **Fairino FR5** | 6-DoF, 922 mm reach, 5 kg payload, ±0.02 mm repeatability | ~$4,000 |
| Gripper | **Jodell RG52-50** | 평행 2지, 52 mm stroke, 2–50 N force | ~$1,500 |
| Teleop leader | **GELLO** | 7-DoF leader arm | ~$500 |
| Overhead camera | Intel **RealSense D455** | 86° FoV | ~$500 |
| Wrist camera | Intel **RealSense D405** | 7–50 cm 근접 stereo | ~$500 |
| Soft extension | **Kirigami PEBA 3D print** | 자체 제작, <45분/유닛 | <$100 |
| **총 합** | | | **~$8,000** |

비교 기준: Franka Research 3 단독 ~$30,000+, ALOHA 셋업 ~$32,000. VILAS는 ~1/4 비용으로 VLA 연구를 가능하게 한다.

### 2.2 통신 아키텍처

- **ZMQ-based abstraction layer**: perception · policy · control 노드 분리, message-passing 비동기 통신.
- 물리 채널: FR5에 Ethernet, 그리퍼·GELLO·카메라에 USB.
- 모듈 분리 덕에 정책(π₀/π₀.₅/GR00T)을 *플러그인 방식*으로 교체 가능.

### 2.3 Kirigami Soft Gripper 확장부

| 항목 | 값/설명 |
|---|---|
| 컨셉 | Kirigami(절개 패턴) 기반 shell 확장부, 압축 시 **예측 가능한 변형** 유도 |
| 재료 | **PEBA filament** (Pebax 계열 thermoplastic elastomer) |
| 프린터 | Bambu Lab H2D FDM |
| 출력 조건 | 노즐 250°C, 베드 50°C |
| 제작 시간 | **~45분/유닛** |
| 비용 | **<$100/유닛** |
| 기능 | active force sensing 없이 morphology만으로 compliance 제공 → fragile object(포도) safe contact |

이는 sensor-free *mechanical intelligence*의 한 예로, sensing 부담을 hardware design으로 옮긴 접근이다.

---

## 3. 비교한 세 VLA 모델

저자들은 *새로운 VLA를 제안하지 않는다*. 대신 **공개 사전학습 체크포인트**를 동일 데이터에 fine-tune해 통제 비교한다.

| 모델 | 출처 | Action Head | Action Horizon | Inference Latency |
|---|---|---|---|---|
| **π₀** | Black et al. 2024 (Physical Intelligence) | Flow-matching expert + VL backbone | 50 steps | **73.8 ms** |
| **π₀.₅** | Pluribus Intelligence 2025 | **Mixture-of-Experts** flow-matching | 50 steps | **82.8 ms** |
| **GR00T N1.6** | NVIDIA 2025 | **Diffusion Transformer** head | 16 steps | **63.6 ms** |

세 모델 모두 *identical* 100-episode 포도 데이터로 fine-tune되어 backbone·action head·horizon의 효과를 isolating할 수 있다.

> ❓ **예상 질문**: action horizon 50 vs 16의 차이가 multi-grasp 성능에 영향을 주는가?
> **답변**: GR00T N1.6의 16-step horizon은 더 짧은 closed-loop replanning을 의미하며, 이는 multi-grasp 시 객체 위치 변화에 적응할 기회를 늘려 58% multi-grasp 성공률의 한 원인일 수 있다. 반면 π₀/π₀.₅의 50-step chunk는 single-grasp 한 번에는 정밀하나 grasp 간 재계획이 부족해 22~36%에 머문다 — 다만 저자들이 직접 이 ablation을 수행하지는 않았다.

---

## 4. 데이터 수집

| 항목 | 값 |
|---|---|
| 수집 방식 | **GELLO leader arm**을 이용한 텔레오퍼레이션 |
| Episode 수 | **100** |
| Frame 수 / episode | **1,200** |
| 총 frame 수 | ~120,000 |
| 캡처 데이터 | 듀얼 카메라 RGB (D455 overhead + D405 wrist) + 7-D joint state (6 arm + 1 gripper) |
| 주 task | **포도(grape) grasping** |
| 일반화 평가 | 체리(cherry) grasping (zero-shot, fine-tune 없이) |

---

## 5. 실험 결과 — Table 1 (PDF 기반)

### 5.1 포도 파지 성공률

| 모델 | **Single Grasp** | **Multi-Grasp (≥2)** | Mean Inference Latency |
|---|---|---|---|
| π₀ | 70% | 22% | 73.8 ms |
| π₀.₅ | **84%** | 36% | 82.8 ms |
| **GR00T N1.6** | 82% | **58%** | **63.6 ms** |

**해석**:
- **Single grasp 챔피언**: π₀.₅ (84%) — flow-matching MoE가 single high-precision grasp에 강점.
- **Multi-grasp 챔피언**: GR00T N1.6 (58%) — 짧은 16-step horizon으로 매 grasp 후 재계획이 잘 작동.
- **π₀**: 두 metric 모두 열위. π₀.₅ 대비 약 14%p 낮음 — 후속 pi_0.5의 일반화 강화가 fine-tune 효율에도 영향.

### 5.2 Cherry Grasping (zero-shot 일반화)

- 포도 학습 정책을 *fine-tuning 없이* 체리에 적용.
- 정성적 평가만 — "reasonable grasping performance" 언급. **정량 % 미보고.**

---

## 6. 어블레이션 / 비교 실험

본 논문은 **명시적 ablation table을 제공하지 않는다**. 다음 비교는 부재하다:

- **Kirigami soft vs rigid 그리퍼**: 압축 손상률 직접 비교 없음.
- **PEBA 재료 vs 다른 elastomer**: 단일 재료만 검증.
- **데이터 양 (100 vs 50 vs 200 episodes)**: scaling 곡선 없음.
- **camera ablation (overhead-only vs wrist-only vs dual)**: 없음.
- **action horizon 통제 비교** (위 §3 질문 참조): 없음.

이는 시스템 페이퍼로서의 한계로, "kirigami가 정말 필요한가"라는 가장 critical한 질문에 정량 답변이 없다.

---

## 7. 명시된 한계

저자들이 본문에서 직접 언급하는 한계:

1. **데모 데이터 다양성 부족**: 객체 layout 변화에 대한 다양성이 충분치 않음 → multi-grasp 성능 제약.
2. **π₀/π₀.₅ 그리퍼 actuation 불일치**: premature release, oscillation 관측.
3. **GR00T N1.6 spatial bias**: 이전 grasp 위치 근처를 우선 탐색하는 경향 → workspace 전체 exploration이 부족.
4. **Multi-grasp 성능이 object placement density에 민감**.

### 리뷰어 관점 추가 한계

- **Cherry generalization 정량값 부재**: zero-shot 주장의 근거 약함.
- **Force/torque 측정 부재**: kirigami가 *얼마나* 부드러운지에 대한 정량 metric 없음.
- **Damage rate 미보고**: "safe grasping" 주장의 핵심인 *포도 손상률 %*가 누락.
- **Failure mode 분석 부재**: π₀가 왜 약한지에 대한 systematic analysis 없음.
- **단일 task 중심**: insertion, pouring, tool-use 등 다른 contact-rich task 검증 없음.

---

## 8. 비교 — 다른 저비용 VLA 플랫폼

| 플랫폼 | 비용 | Compliance | Pretrained VLA 비교 |
|---|---|---|---|
| ALOHA / Mobile ALOHA | ~$32,000 | rigid | ACT, RDT 평가 |
| LeRobot SO-100/101 | ~$300 (단일 팔) | rigid | π₀, GR00T 평가 |
| Franka Research 3 + Robotiq | ~$32,000 | rigid (force sensor 옵션) | 다양 |
| **VILAS** | **~$8,000** | **morphological (kirigami)** | **π₀ / π₀.₅ / GR00T N1.6 통제 비교** |

VILAS의 차별성: (a) ~$8K 가격대에서 산업급 정밀도(±0.02 mm), (b) sensor-free compliance, (c) 셋 이상의 SOTA VLA를 동일 조건에서 비교했다는 점.

---

## 9. 종합 평가

| 항목 | 평가 |
|---|---|
| **Novelty** | ★★★☆☆ — Kirigami soft 그리퍼 + 저가 cobot + 공개 VLA 셋 비교의 *통합*은 신규. 개별 요소는 기존 연구 활용 |
| **Engineering rigor** | ★★★★☆ — ZMQ 모듈러, 1,200 frame/episode 표준화, dual-camera 동기 |
| **Experimental rigor** | ★★☆☆☆ — Table 1 외 ablation/baseline 부재. 통계적 유의성 분석 없음. cherry 정량값 누락 |
| **Practical impact** | ★★★★☆ — $8K 가격대에서 즉시 reproducible, 농업/식품가공 도메인 즉시 응용 가능 |
| **Reproducibility** | ★★★☆☆ — "공개 예정"이나 publication 시점 release, 현재 GitHub URL 없음 |

### 핵심 기여 (PDF 기반 재확인)

1. **저비용 통합 플랫폼**: 총 ~$8,000으로 산업급 정밀도(±0.02 mm) VLA 연구 가능.
2. **Morphological compliance**: <$100 PEBA 3D 프린트 kirigami 확장부로 force sensor 없이 fragile object 파지.
3. **세 VLA 통제 비교**: π₀ / π₀.₅ / GR00T N1.6을 *동일 데이터*로 fine-tune한 첫 보고 중 하나.
4. **명확한 정량 결과**: single grasp 70~84%, multi-grasp 22~58%, latency 63.6~82.8 ms 모두 Table 1에 보고.
5. **오픈소스 약속**: hardware spec, SW stack, assembly tutorial을 acceptance 후 공개 예정.

---

## 10. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|---|---|
| 1 | Kirigami 없이 rigid 그리퍼만으로 같은 데이터로 학습하면 포도 손상률은? | **본문에 비교 없음** — 가장 critical한 ablation 부재 |
| 2 | π₀가 두 metric 모두 열위인 이유는? | 저자들은 actuation 불일치(premature release, oscillation) 언급. Backbone 자체 한계인지 fine-tune 데이터 양(100 ep)이 부족했는지는 미분리 |
| 3 | GR00T N1.6의 16-step horizon이 multi-grasp 58%의 원인인가? | 가능성 높으나 직접 ablation 없음. 동일 horizon으로 통제한 비교 부재 |
| 4 | Cherry zero-shot 일반화 성공률은? | **정량값 없음** — "reasonable" 언급만 |
| 5 | 포도 손상률 (squeezed/burst rate) 정량 metric은? | **없음** — "safe grasping" 주장의 근거가 약함 |
| 6 | $8,000은 정말 진입 장벽이 낮은 가격인가? | 학부생/취미 수준은 아니지만, $30K급 연구 셋업에 비해 1/4 수준. 농업 도메인 도입 가능성은 높음 |
| 7 | π₀.₅의 MoE가 single grasp 84%에 기여한 만큼 multi-grasp는 왜 36%로 떨어지는가? | MoE expert가 single-grasp pattern에 over-specialize, 다양한 후속 grasp pose에 적응 부족 — 가설 |
| 8 | 100 episodes는 충분한가? | Scaling 실험 없음. GR00T 같은 large pretrained 모델은 100 ep으로도 OK라는 게 결과로 보여지나, π₀의 70% single-grasp는 데이터 부족 가능성 시사 |
| 9 | RealSense D405 wrist camera는 7~50cm 범위인데, multi-grasp 시 카메라 시야 밖 객체는? | 본문 미분석. spatial bias(GR00T)의 한 원인일 수 있음 |
| 10 | Code release 시점이 acceptance라면 현재 검증 어떻게? | 현 시점 GitHub URL 없음. YAML `code_url=null`, `open_source=true` (저자 의도 기준) |

---

## 11. YAML 점검 (PDF 검증 후)

- `organization` → "Drexel University; Virginia Tech Seafood Agricultural Research Center; Amazon Store Foundation AI" (PDF에서 확인).
- `open_source` → `true`로 갱신 (논문 본문에서 "hardware specifications, software stack, ... assembly and deployment tutorials upon acceptance" 명시).
- `code_url` → `null` 유지 (현 시점 GitHub URL 미공개).
- `venue` → "arXiv preprint" 명시.
- `architecture.action_head` → π₀ flow-matching / π₀.₅ MoE flow-matching / GR00T N1.6 diffusion transformer로 세분화.
- `architecture.action_head_category=hybrid` 유지 — 세 backbone이 flow-matching과 diffusion을 혼합. 단일 카테고리로 환원 불가.
- `architecture.parameters="varies"` 유지 — 세 backbone의 파라미터 수 상이.
- `benchmarks: {}` 유지 — LIBERO/CALVIN/SimplerEnv 등 표준 벤치마크 없음. 포도/체리 grasping은 proprietary task로 표준 키에 부합하지 않음.
- `tags=[low-cost, hardware-platform, soft-grasping, real-world]` 유지 — 본 논문 정체성 정확히 반영.

<!-- VERIFIED: pdf -->
