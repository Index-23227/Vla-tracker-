# CoRE-VLA: Towards Scalable and Robust Vision-Language-Action Modeling via Conditional Routing of Experts

**arXiv**: 2607.03693 (2026-07-04) · **소속**: Zhejiang University, Shanghai Innovation Institute, Fudan University, Nanjing University, Jilin University (교신: Jingjing Gong, Xipeng Qiu)

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
실제 로봇 배치 환경에서 센서 구성은 이질적이다. 어떤 플랫폼은 depth/tactile/force 센서를 갖추고 있고, 어떤 플랫폼은 설계상 없으며, 있더라도 운용 중 고장날 수 있다. 그러나 기존 VLA 정책(OpenVLA, pi0, GR00T N1 등)은 두 가지 구조적 문제를 안고 있다:

1. **센서 결합(sensor coupling)**: 고정된 관측 인터페이스와 공유 dense 연산으로 학습되어, 보조 센서와 함께 학습하면 그 센서에 과의존(over-reliance)하고 센서 부재 시 급격히 열화된다.
2. **Dense action generator의 태스크 간섭**: 이질적 태스크와 long-horizon subgoal에 동일한 연산 경로를 강제하면 gradient conflict, negative transfer가 발생한다 (multi-task learning 문헌의 알려진 문제).

### 핵심 질문
"보조 센서가 있으면 활용하고, 없으면 RGB+언어+proprioception만으로 신뢰성 있게 동작하는 **단일 통합 정책**을 어떻게 만들 것인가?" 저자들은 이를 아키텍처+학습의 결합 문제로 명시적으로 정식화한 최초의 연구라고 주장한다.

## 2. 방법론 심층 분석

### 2.1 전체 아키텍처
- **VLM 백본**: Qwen3-VL-4B-Instruct가 RGB 관측(224×224)과 언어 지시를 인코딩.
- **보조 모달리티 인코더**: Conv Projector(kernel 28×28, stride 28)가 depth를 인코딩. GT depth가 아니라 Depth Anything V2로 RGB에서 추정한 **pseudo-depth**를 사용 — 실제 배치의 노이즈를 모사하고 depth 센서 없는 데이터 수집도 가능하게 함.
- **Action generator**: Flow-matching Action DiT (DiT-B). LIBERO 설정은 16층/hidden 1024, RoboCasa 설정은 32층/hidden 2560. 일부 층을 CoRE 블록으로 교체 (LIBERO: 층 2,4,6,8 / RoboCasa: 층 8~22 짝수층).

### 2.2 Intent-conditioned routing
Task-intent 임베딩 g = AvgPool(C_text) — VLM이 인코딩한 **텍스트 토큰만** 평균 풀링. VLM attention을 거쳤으므로 시각 문맥에 grounding된 지시 중심 표현이다. 이 g가 토큰 선택과 expert 라우팅 양쪽의 조건으로 쓰인다.

### 2.3 Action-side 토큰 희소 선택
각 action-side 토큰 h_i에 대해 p_i = σ(w_sel^T [h_i; g])를 계산, Top-K(capacity ratio ρ=0.5)만 선택. 선택된 토큰만 cross-attention과 expert 연산을 거치고, 나머지는 bypass 후 scatter-merge. 연산량이 ρ에 의해 상한됨 (F_CoRE/F_dense ≈ ρ + ε).

### 2.4 General / Modality-specialized experts + 가용성 마스킹
Expert 집합을 E_gen(공유 조작 패턴)과 E_mod(모달리티 의존 연산)로 분리. 모달리티 지시자 z_mod ~ Bernoulli(1−p_drop), p_drop=0.2로 학습 중 depth 분기를 확률적으로 끄고(**modality dropout**), z_mod=0이면 E_mod의 라우팅 logit을 −∞로 마스킹. 추론 시 z_mod는 실제 센서 가용성에 따라 설정. Top-1 token-choice routing이며, 연속 게이트 p_i·π_i를 통해 이산 선택 경로에도 gradient가 흐른다.

### 2.5 학습 목적함수
Flow matching (x_τ = (1−τ)ε + τa, v = a−ε) + 선택 질량을 ρ에 정렬하는 selection regularizer + 표준 MoE load-balancing loss. L = L_act + 0.01·L_sel + 0.01·L_moe. 추론은 4-step Euler 적분.

## 3. 데이터 전략

- **LIBERO**: 4개 suite 전체를 단일 정책으로 joint 학습 (libero_all mix).
- **RoboCasa GR1 Tabletop**: 24개 태스크 × 1000 demo (GR00T-X Embodiment Sim).
- **실기 로봇**: AgileX ALOHA 듀얼암, 머리 1 + 손목 2 카메라 (Orbbec Dabai DC1 RGB-D). Vegetables-Picking 44 demo, Clothes-Folding 1.3K demo (leader-follower 원격조작). Fabric-Folding은 zero-shot.
- 학습 depth는 전부 DA-V2 pseudo-depth. 실기 배치에서는 물리 depth 카메라 또는 no-depth 두 설정을 fine-tuning 없이 평가 — 학습/배치 간 모달리티 shift 강건성 검증.

## 4. 시스템/학습 세부사항

| 항목 | LIBERO | RoboCasa GR1 |
|---|---|---|
| Action DiT | 16층, hidden 1024 | 32층, hidden 2560 |
| Action/State dim | 7 / 7 | 29 / 58 |
| Action horizon | 8 | 16 |
| General / Mod experts | 12 / 4 | 32 / 8 |
| CoRE 층 | 2,4,6,8 | 8,10,...,22 |
| 학습 스텝 | 100K | 200K |
| GPU | 8×H200 | 8×H200 |

공통: batch 128, AdamW, cosine LR (action model 1e-4, VLM 1e-5), full fine-tuning, ρ=0.5, p_drop=0.2, 추론 diffusion 4 step, seed 42. Dense DiT를 먼저 학습한 뒤 FFN 복제로 general expert를 만들고 modality expert와 depth 인코더를 삽입해 **continued training**으로 전환 — RGB 사전학습 정책의 사후 모달리티 확장이라는 주장의 근거.

## 5. 실험 설계 및 평가 프로토콜

- **LIBERO**: 4 suite × 10 task, 태스크당 50 rollout.
- **RoboCasa GR1 Tabletop**: 24 task, 5개 독립 그룹 × 100 rollout으로 mean±std 보고 — 시뮬 평가 분산까지 보고하는 드물게 성실한 프로토콜.
- **실기**: 태스크당 20 rollout, 100점 만점 subgoal 점수(Score)와 전 subgoal 완수율(SR) 병행 보고. 전 rollout의 개별 점수를 부록에 공개(Tables 9-11).

## 6. 실험 결과 심층 분석

### LIBERO (Table 1)
| 모델 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| pi0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GR00T-N1.7 | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| **CoRE-VLA** | **99.0** | **99.2** | **98.8** | **97.6** | **98.7** |

전 suite 최고 성능이며 특히 LIBERO-Long에서 격차가 크다(97.6 vs 차순위 94.5). Long-horizon subgoal마다 다른 expert 경로가 할당되어 간섭이 줄었다는 해석. 다만 LIBERO는 이미 97% 안팎으로 포화 상태라 절대 격차는 작다.

### RoboCasa GR1 Tabletop (Table 2)
Diffusion Policy 40.4, GR00T-N1.5 48.0, GR00T-N1.6 47.6 대비 **56.5±0.4**로 +8.5pt 우위. 24개 태스크 전반에서 고르게 강하나, tiered-shelf 계열(26.2~26.6)은 모든 모델이 낮아 여전히 난제.

### 실기 (Table 3)
세 태스크 모두에서 pi0.5(JAX, 사전학습 체크포인트에서 fine-tune) 및 dense DiT baseline을 상회. depth 없이 추론해도 dense baseline을 크게 이기고(예: Vegetables SR 65 vs 35), 물리 depth 카메라를 켜면 추가 향상(Fabric-Folding SR 80, Score 87.5). 대규모 로봇 사전학습 없이 pi0.5를 이겼다는 점이 인상적이나, pi0.5의 사전학습 이점이 소규모 demo fine-tune 체제에서 충분히 발휘됐는지는 논쟁 여지가 있다.

## 7. Ablation 분석 (LIBERO-Long, Table 12)

12행 ablation이 이 논문의 백미다:
- **Task-intent routing 단독** (RGB만, 8 experts): 94.4 → 95.2. 보조 모달리티 없이도 routing 자체가 이득.
- **Naive depth 추가** (dropout/mod-expert 없음): depth on 95.6, depth off **91.2** — 과의존의 정량적 증거 (−4.4pt).
- **Modality dropout만**: depth off 96.2로 회복하지만 depth on은 94.4로 오히려 하락 — dropout은 강건성은 주지만 depth 활용 능력은 못 줌.
- **Mod-expert만 (dropout 없음)**: depth off 92.8 — expert 용량만으로는 불충분.
- **Routing 조건 변경** g = AvgPool(C_vl) (전체 vision-language 토큰 풀링): 94.2로 full model 대비 −3.4pt — **텍스트 중심 intent 신호가 결정적**이라는 흥미로운 발견.
- **Full CoRE-VLA**: depth on 97.6 / off 97.0 — 격차 단 0.6pt.

각 구성요소의 역할(depth=기하 단서, dropout=결손 강건성, mod-expert=모달리티 전용 용량, intent routing=태스크 적응)이 깔끔하게 분해된다.

## 8. 관련 연구 비교

- **MoE-VLA 선행연구**: ChatVLA/ChatVLA-2는 reasoning-control 분리를 위한 MoE, ForceVLA는 force-aware MoE. CoRE-VLA의 차별점은 **task intent + 센서 가용성**이라는 이중 조건 라우팅과, 가용성 마스킹을 통한 결손 모달리티 대응.
- **Mixture-of-Depths**와 유사한 토큰 수준 희소화(선택 안 된 토큰 bypass)를 action DiT에 적용한 점도 특징 — expert 희소화와 토큰 희소화를 동시에 수행.
- **pi0/pi0.5** 대비: 동일한 flow-matching 계열이지만 action generator 내부를 조건부 희소 연산으로 재구성.
- **GR00T N1.x** 대비: 동급 시뮬 벤치마크에서 우위이나, GR00T는 cross-embodiment 사전학습 모델이라는 체급 차이 고려 필요.

## 9. 한계 및 미해결 문제

1. **보조 모달리티가 depth 하나뿐**: 프레임워크는 tactile/force도 지원한다고 주장하나 실험은 전부 depth. 모달리티 2개 이상 동시 확장 시 expert 수/라우팅 안정성은 미검증.
2. **Pseudo-depth 의존**: 학습이 DA-V2 depth 기반이므로 "depth 활용"의 상한이 monocular 추정 품질에 묶임. GT/물리 depth로 학습했을 때와의 비교 없음.
3. **효율 주장의 실측 부재**: 부록 D의 FLOPs 분석은 이론적 상한(ρ 기반)이며 실제 wall-clock latency/throughput 수치가 없다. VLM(4B)+DA-V2가 병목일 가능성.
4. **사전학습 없음**: cross-embodiment 대규모 사전학습 미수행 — 스케일링 시 routing이 유지되는지는 future work.
5. **코드/체크포인트 미공개** (논문 내 저장소 링크 없음).
6. LIBERO 포화(98.7%)로 인해 시뮬 상 개선 폭의 의미가 제한적.

## 10. 총평

"센서 이질성·결손 하의 통합 VLA 배치"라는 실용적 문제를 아키텍처 수준(가용성 조건 라우팅)에서 정면으로 다룬 잘 설계된 논문. 특히 Table 12의 12행 ablation은 naive depth 추가의 과의존 문제(−4.4pt)와 각 구성요소의 역할을 정량적으로 분리해 보여주는 모범적 실험이다. LIBERO 98.7 / RoboCasa GR1 56.5는 각각 SOTA급이며, 실기 20 rollout 전수 공개 등 평가 투명성도 높다. 반면 depth 단일 모달리티 실험, 효율성 실측 부재, 코드 미공개가 아쉽다. MoE 기반 action generator 설계의 좋은 레퍼런스.

## 11. 🔥 예상 날카로운 질문 모음

1. Modality dropout p=0.2로 depth-off 체제 노출이 20%뿐인데, dropout 확률에 대한 민감도 분석은 왜 없는가?
2. g = AvgPool(C_text)가 AvgPool(C_vl)보다 3.4pt 우위라는데, learned query나 attention pooling 같은 더 강한 intent 추출기는 시도했는가?
3. Top-1 token-choice routing인데 top-2 이상에서는 어떤가? Expert 수(12+4 vs 32+8)는 어떻게 정했는가?
4. Dense→CoRE 전환(FFN 복제) 대신 처음부터 CoRE로 학습하면 성능이 어떻게 달라지는가?
5. 실기에서 DA-V2 pseudo-depth로 학습하고 물리 depth 카메라로 추론하는 domain gap은 왜 문제가 안 되는가? (Orbbec depth와 DA-V2 depth의 분포 차이)
6. ρ=0.5면 이론상 최대 절반 절감인데 실측 latency는? 4-step 추론에서 routing 오버헤드 ε의 실제 크기는?
7. LIBERO-Long ablation의 dense baseline(94.4)이 이미 OpenVLA-OFT Long(94.5)과 동급인데, 백본(Qwen3-VL-4B) 효과와 CoRE 효과를 어떻게 분리하는가?
8. 태스크 수가 늘어날 때(예: 수백 태스크) expert 수를 어떻게 스케일하는가? Routing collapse는 load-balancing loss만으로 충분한가?

## 12. 세미나 토론 포인트

- **"가용성 조건 연산"의 일반화**: z_mod를 이진이 아닌 센서별 다중 지시자로 확장하면 (depth, tactile, force) 조합 2^k 체제를 단일 정책이 커버할 수 있는가? Expert 수가 조합적으로 늘지 않는가?
- **토큰 희소화 vs expert 희소화의 기여 분리**: Table 12는 expert 축 위주 ablation이라, ρ=1.0(전 토큰 선택) 대비 ρ=0.5의 성능/효율 trade-off 곡선이 궁금한 지점.
- **Routing 시각화(부록 C)의 해석**: 시간에 따라 expert 사용이 변한다는 것이 subgoal 분해의 증거인지, 단순히 diffusion step/토큰 통계의 부산물인지 — subgoal 경계와 routing 전환의 정렬을 정량화할 방법 논의.
- VLA 커뮤니티 관점에서, "backbone 스케일링"이 아닌 "action generator의 조건부 연산화"가 다음 스케일링 축이 될 수 있는지.

<!-- VERIFIED: pdf -->
