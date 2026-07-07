# FurnitureVLA: Learning Long-Horizon Bimanual Furniture Assembly with Vision-Language-Action Model

> **한 줄 요약**: MERL/Oxford/UNC 연구로, **실물 크기(real-scale) 양팔 가구 조립**을 VLA로 체계적으로 다룬 첫 연구. π0.5를 backbone으로 action 차원을 14→15로 확장해 **연속 progress 신호를 action과 공동 예측**하고, post-retreat 지점에서 subtask 경계를 정의해 자동 subtask 전환을 구현. 시뮬레이션 평균 성공률을 monolithic finetuning의 48%에서 **80%**로 끌어올렸고 (design factor 연구로 추가 +21%p), 실제 dual Kinova Gen3에서 최난도 IVAR 의자 조립을 검증 (가장 어려운 태스크에서 시뮬 대비 16%p 하락에 그침).

- **arXiv**: [2607.01212](https://arxiv.org/abs/2607.01212) (v1, 2026-07-01)
- **저자**: Chenyang Ma, Yue Yang, Radu Corcodel, Siddarth Jain, Andrew Wu, Chiori Hori†, Diego Romeres
- **소속**: Mitsubishi Electric Research Laboratories (MERL); University of Oxford; UNC Chapel Hill

---

## 1. 배경 및 동기

가구 조립은 로보틱스의 오랜 난제다: (1) 부품들을 엄격한 순서로 조작해야 하는 **long-horizon** 특성, (2) mating part 간 **tight한 기하학적 정렬** 요구, (3) 초기 실수가 연쇄 실패로 전파되는 구조. 기존 연구(FurnitureBench, JUICER, IKEA env 등)는 대부분 **toy-scale 축소 모형 + 단일 팔**에 머물렀다. 실물 크기에서는 크고 무거운 부품 때문에 **양팔 협조(bimanual coordination)**가 필수이고, reachability 한계와 kinematic singularity를 조립 내내 회피해야 한다.

한편 현재 VLA들은 short-horizon, 완화된 정밀도 요구의 태스크에 최적화되어 있어, 긴 시연 전체를 monolithic하게 finetuning하면 distribution drift와 compounding error로 무너진다. 본 논문은 이 간극 — **"real-scale bimanual assembly를 위한 generalist VLA 정책을 어떻게 학습하는가"** — 를 정면으로 다룬다.

## 2. 태스크 및 시스템 설계

**태스크**: 실제 IKEA 3종 — LACK 사이드 테이블 (4 subtask, 12 skill, 650 steps), KALLAX 선반 (4 subtask, 14 skill, 850 steps), IVAR 의자 (7 subtask, 25 skill, **1550 steps ≈ 155초**). 나사 조임은 자석(NdFeB, 나사 구멍에 에폭시 부착)으로 대체하고, 부품 초기 위치는 nominal pose에서 ±3cm / ±5° 랜덤화.

**하드웨어**: 듀얼 Kinova Gen3 7-DoF (왼팔 Robotiq Hand-E, 오른팔 2F-85), front/rear D435 + 양쪽 wrist 카메라 4-view. 대형 부품과 양팔 상호작용이 정면 시야를 자주 가리므로 **rear 카메라 추가**가 핵심 설계.

**시뮬레이션 파이프라인**: Isaac Gym에서 FurnitureBench 코드베이스를 확장. motion planning으로 expert 시연 생성 (single-arm은 EE pose 궤적, bimanual은 물체 궤적을 계획하고 양팔을 rigid attachment로 구속). Isaac Gym이 runtime weld를 지원하지 않아 자석 부착은 **pose-reset**으로 구현 (허용오차 진입 시 매 tick 상대 pose를 kinematic하게 유지). IVAR는 이 제약 때문에 2-stage로 분리 시뮬레이션.

**성공 판정**: base part 대비 상대 pose 오차 — 소형 부품 ε=1cm / 대형 2cm, 회전 δ=0.998 (축당 ≈4.0°). 모든 부품이 기준을 만족해야 전체 조립 성공.

**VR 텔레오퍼레이션**: Meta Quest 3 + Quest2ROS (72Hz), 단일 조작자가 양팔 제어. 3가지 설계 원칙 — (1) 병진/회전 **분리 제어** (index/middle trigger), (2) 90° 단위 **사전 정의 grasp preset** (버튼으로 orientation snap), (3) 양팔 **동기화 미러 모드** (대형 부품 공동 lift/rotate용).

## 3. 방법론 심층 분석: Progress-Enhanced VLA

**핵심 아이디어 1 — 연속 progress의 공동 예측**: 태스크를 subtask G=(g₁,…,g_K)로 분해하고, 14차원 bimanual action aₜ에 스칼라 progress pₜ를 붙여 ãₜ = [aₜ⊤, pₜ]⊤ ∈ R¹⁵로 확장. 각 subtask는 N_k개의 action primitive (pick up/place/retreat)로 구성되며, primitive들이 progress를 균등 milestone으로 이산화하고 각 구간 안에서는 시간에 대해 **선형 보간** — pₜ = i/N_k + (1/N_k)·(t−sᵢ)/(sᵢ₊₁−sᵢ). 결과적으로 subtask마다 0→1로 단조 증가하는 매끄러운 신호. flow matching으로 π_θ(ãₜ:ₜ₊H₋₁ | oₜ, g_k)를 finetuning.

**핵심 아이디어 2 — Post-retreat subtask 경계**: 경계를 조립 완료 직후(contact-rich)가 아니라 **retreat 이후의 contact-free 상태**에 둔다. 접촉 직후 상태는 작은 실행 오차에 극도로 민감해 다음 subtask의 초기 상태 분포를 넓히지만, post-retreat 상태는 접촉/힘 제약이 없어 오차 증폭이 적고 초기 분포가 좁아져 cross-subtask distribution shift가 감소한다.

**핵심 아이디어 3 — 추론 시 자동 전환**: p̂ₜ ≥ τ_p(=0.95)이면 전환 후보. 고립된 spike를 걸러내는 경량 필터 (연속 2회 high signal, 또는 최근 Δ≥3 이내 이력이 있는 재점화) 통과 시 다음 subtask로 진행하고 progress와 action buffer를 리셋. 외부 stage estimator나 고비용 reasoning 없이 subtask 전환을 policy 내부에서 해결하는 것이 SeqVLA·Long-VLA 류 대비 차별점.

## 4. 학습 셋업

| 항목 | 값 |
|---|---|
| Backbone | π0.5 (pi05_base), full finetune (LoRA 미사용) |
| 파라미터 | ~2.6B (PaliGemma-2B 2.3B + Gemma-300M action expert + <1M projection) |
| 해상도 | 224→**448** 업스케일 (SigLIP So400m/14, 32×32 token grid, pos-emb bicubic 보간) |
| 입력 | 4 RGB view (front/rear/wrist×2), proprio는 discretize해 프롬프트에 인라인, history 없음 |
| Action | chunk H=50, 15차원 (14 robot + 1 progress) |
| 학습 | 8× L40S, 40K steps, batch 64, LR 2.5e-5 cosine (30K decay, 1K warmup), AdamW |
| 추론 | L40S 1장, flow-matching 10 denoising steps, 10Hz 제어 |
| 데이터 | 시뮬: 가구당 500 demo (3종 통합 학습) / 실물: IVAR 100 demo (VR teleop, DROID식 no-op 필터) |

## 5. 시뮬레이션 실험 결과

**Table I (가구당 100 rollouts, 전체 조립 성공률)**:

| Method | LACK | KALLAX | IVAR | Average |
|---|---|---|---|---|
| π0.5 (zero-shot) | 0.00 | 0.00 | 0.00 | 0.00 |
| π0.5 (monolithic finetuned) | 0.91 | 0.11 | 0.41 | 0.48 |
| **FurnitureVLA** | **0.98** | **0.85** | **0.56** | **0.80** |

zero-shot π0.5는 전 태스크 0% — 사전학습 분포에서 완전히 벗어난 태스크임을 확인. monolithic finetuning 대비 최대 이득은 KALLAX (+74%p): 크고 무거운 부품이 long-horizon drift로 인한 singularity에 특히 취약하기 때문. IVAR는 5번째 subtask (양팔로 의자 프레임을 잡고 들어 좌측 프레임에 부착)가 최대 병목.

**Design factor 연구 (Table II)**: temporal ensembling λ=−0.1이 전 가구에서 최선 (최근 예측에 ~70% 가중; progress 신호는 ensembling에서 제외하고 항상 최신 예측 사용). action horizon은 가구별로 10 또는 25가 최적 (KALLAX는 무거운 부품 때문에 긴 horizon 25 선호). **rear 카메라 제거 시 평균 0.80→0.47로 급락** — occlusion 완화에 결정적. 해상도는 448이 일관되게 최고 (224: 0.60 → 448: 0.80). 기본 설정(n/a ensembling 등) 대비 design factor 최적화로 **평균 +21%p**.

**Ablation (Table III)**: demo 25%→50%에서 최대 이득 (0.50→0.68), 50%→100%는 0.68→0.80. **Discrete progress (subtask당 상수 (2k−1)/2K)는 전 가구 0% 완전 실패** — 부품이 "거의 조립됨"과 "조립됨" 상태가 시각적으로 유사해 이산 전환을 감지하지 못하고 progress가 멈춰버림. 연속 progress 설계의 정당성을 강하게 뒷받침.

## 6. 실세계 실험 결과

최난도 **IVAR 의자**로 검증 (100 demo 수집, 15 rollouts). 자석 조립임에도 정밀도 요구가 높음: 1cm 편차는 snap 되지만 1.5cm 또는 10° 기울기면 부품이 떨어져 나감. Hand-E gripper 개구 5cm vs 부품 두께 최대 3.7cm의 협소한 clearance, 최대 8개 자석 다점 정렬.

**Table IV**:

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
|---|---|---|---|---|---|---|---|
| Full Assembly SR | 0.80 | 0.73 | 0.60 | 0.53 | 0.47 | 0.47 | **0.40** |
| Per-Part SR | 0.80 | 0.80 | 0.73 | 0.80 | 0.67 | 0.87 | 0.80 |

Per-part 성공률(subtask별 독립 평가, 각 15 rollouts)이 full-assembly보다 일관되게 높음 → 실패는 단일 파국 모드가 아니라 **subtask 간 누적**에서 발생. S3–S4는 왼팔이 카메라에서 멀어져 시각 정보가 부족, S5는 다점 정렬의 최고 정밀도 요구가 병목. 흥미로운 **emergent 행동**: 접촉이 불충분하면 gripper를 다시 열어 re-grasp, 자석 정렬을 위한 미세 교정 동작 — teleop 시연에 담긴 교정 행동을 policy가 학습한 것으로 해석. 초록의 "hardest task에서 16%p 하락"은 시뮬 대비 실세계 성능 저하를 의미.

## 7. 강점

1. **문제 설정의 신규성**: real-scale + bimanual + 1550 steps는 CALVIN/LIBERO/RoboCasa 등 기존 벤치마크의 horizon을 크게 초과. 태스크 형식화·시뮬 파이프라인·VR teleop·정책까지 풀스택 시스템.
2. **단순하고 효과적인 progress 메커니즘**: action 차원 1개 추가만으로 외부 stage estimator/failure detector 없이 subtask 전환 해결. discrete 대비 ablation이 설계 선택을 명확히 정당화.
3. **post-retreat 경계**라는 분포 이론적 관점의 subtask 분할 — 실용적이면서 원리가 분명.
4. **design factor 연구의 실용 가치**: rear 카메라(+33%p), 448 해상도, λ=−0.1 ensembling 등 재현 가능한 구체적 지침. Openpi Comet 계열의 "perception/control 설계가 성능을 좌우한다"는 관찰을 조립 도메인에서 정량화.
5. 시뮬과 실물 모두에서 검증, per-part 평가로 실패 원인을 분해 분석.

## 8. 약점·한계

1. **나사 조임 우회**: 자석 대체는 저자도 인정한 한계. 실제 IKEA 조립의 핵심 난제(공구 사용, 나사산 정렬)는 미해결.
2. **일반화 범위**: 가구 3종, 실물 검증은 IVAR 1종 15 rollouts — 통계적 신뢰구간이 넓음. 새로운 가구/부품 배치로의 일반화는 미검증 (초기 랜덤화도 ±3cm/±5°로 좁음).
3. **subtask 분해·primitive 라벨이 수동 정의** — progress 라벨은 primitive 경계 timestep에 의존하므로 스케일업 시 자동화 필요.
4. 고정 base 듀얼암이라 robot workspace 내 가구로 제한 (모바일 플랫폼 필요).
5. 코드/데이터 공개 여부 불명 (3D 모델·텍스처 출처는 명시했으나 파이프라인 자체 공개는 미언급).
6. 실세계 zero-shot이나 sim-to-real transfer는 다루지 않음 — 실물은 별도 100 demo finetuning.

## 9. 관련 연구와의 비교

- **FurnitureBench (RSS 2023)**: toy-scale 단일 팔 벤치마크. 본 연구는 이를 real-scale bimanual로 확장하고 성공 판정 기준도 계승·강화.
- **Long-VLA / SeqVLA / LiLo-VLA**: 장기 horizon을 위한 phase 분해 계열이지만 외부 스위칭이나 별도 모듈에 의존. FurnitureVLA는 progress를 action space에 내장해 단일 forward pass로 전환.
- **ECoT / Hi Robot**: reasoning으로 다음 subtask를 출력하는 계열 — 표현력은 높으나 계산 비용이 큼. progress 스칼라는 사실상 무비용.
- **ACT**: temporal ensembling을 차용하되 λ 튜닝이 조립 성공에 미치는 영향을 체계 분석.
- **VT-Refine, Fabrica**: 양팔 조립 계열이나 정밀 residual/planning 중심 — generalist VLA finetuning 접근과 상보적.

## 10. 재현성 평가

시뮬 환경(Isaac Gym + FurnitureBench 확장), IKEA 시리얼 ID·3D 모델·텍스처 출처, 하이퍼파라미터 전체(Table V), 성공 판정 임계값, teleop 버튼 매핑까지 부록에 상세 기술 — 문서화 수준은 높음. 다만 **코드·데이터셋 공개 링크가 논문에 없어** 완전 재현에는 상당한 재구현 노력 필요. π0.5 base checkpoint 접근성에도 의존.

## 11. 결론

FurnitureVLA는 "VLA가 실물 크기 가구를 양팔로 조립할 수 있는가"라는 질문에 대해 처음으로 체계적인 긍정적 증거를 제시한다. 기술적 기여(연속 progress 공동 예측 + post-retreat 경계)는 단순하지만 ablation으로 필요성이 명확히 입증되었고, 시스템 기여(시뮬 파이프라인 + VR teleop)와 design factor 연구는 후속 연구의 실용적 발판이 된다. long-horizon VLA 연구가 "더 긴 컨텍스트"가 아니라 **"전환 가능한 좋은 경계 + 진행도 인식"**으로 풀릴 수 있음을 보여주는 사례. 나사 조임·일반화·공개라는 숙제가 남지만, VLA의 실세계 고난도 배치를 향한 의미 있는 진전이다.

## 12. Discussion Seeds (세미나 토론용 질문)

1. progress 신호를 action 차원에 넣는 것과 별도 head로 예측하는 것 — flow matching의 denoising 과정에서 progress가 action과 얽히는 것이 득인가 실인가?
2. post-retreat 경계 아이디어는 조립 외 도메인(요리, 정리정돈)에서도 "contact-free 안정 상태"로 일반화 가능한가?
3. discrete progress의 완전 실패(0%)는 supervision 신호 문제인가, 추론 시 bin 분류 문제인가? 예컨대 discrete + temporal smoothing이면 살아날까?
4. rear 카메라 제거 시 −33%p인데, 이를 depth나 wrist 카메라 개선으로 대체할 수 없었던 이유는? active perception (카메라를 든 세 번째 팔)이 대안이 될까?
5. 자석→나사로 가려면 무엇이 필요한가 — force/torque 피드백, 촉각(VT-Refine), 아니면 hybrid position-force 제어?
6. subtask당 500 demo 규모에서 25%→50% 이득이 최대였는데, real-scale 조립에서 data scaling law는 어떤 형태일까?
7. IVAR S5 (양팔 프레임 부착)가 시뮬·실물 공통 병목 — 이런 다점 정렬 subtask만 residual RL (From Imitation to Refinement)로 보강하는 hybrid는?
8. 155초 horizon에서 progress reset이 누적 오차를 실제로 얼마나 차단하는가 — subtask 전환 시점의 오차 분포 분석이 있으면 좋았을 부분.

---

**Overall Score: 8.0 / 10** — 신규 문제 설정 + 풀스택 시스템 + 명확한 ablation. 나사 우회와 좁은 실물 검증 범위, 미공개 코드가 감점 요인.

<!-- VERIFIED: pdf -->
