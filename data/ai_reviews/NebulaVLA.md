# NebulaVLA: A Dual-Frequency Vision-Language-Action Model With Guide Action for Robotic Manipulation

> **한 줄 요약**: Qwen3-VL 기반 System 2(~10 Hz)와 DINOv2+Q-Former+DiT 기반 System 1(~20 Hz)을 비동기로 분리하고, 언어 키워드로 그리퍼·손 동작을 통일한 GESTURE-7 표현으로 System 2를 SFT→GRPO 학습한 뒤, 이전 청크의 미실행 액션을 마스크로 고정해 다음 청크를 "outpainting"하는 Guide Action으로 청크 경계 jitter를 줄인 ZTE의 VLA. LIBERO-Plus 85.5%, 실기(AgiBot A2) 92.08%/92.5%, 지연 42 ms.

- **arXiv**: 2608.16503v1 (2026-08-17, cs.RO)
- **소속**: ZTE Corporation
- **코드**: 공개 정보 없음

---

## 1. 배경 및 동기

저자들은 현재 VLA의 병목을 세 가지로 정리한다.

1. **효율-성능 트레이드오프**: 단일 동기식(monolithic synchronous) 구조는 느린 의미 추론과 빠른 모터 제어를 한 주기로 묶어버린다.
2. **임베디먼트 표현 부족**: 연속 운동학을 이산 토큰으로 매핑하는 방식(FAST 등)은 큰 어휘와 대량 데이터가 필요하고, 이종 로봇 간 행동 의미를 통일하기 어렵다.
3. **실행 부드러움**: 비동기 추론과 액션 청크 생성의 시간 불일치로 청크 경계에서 jitter와 불연속이 생긴다.

HiRT, Figure Helix처럼 fast-slow 분리를 시도한 선례 위에서, NebulaVLA는 세 병목을 각각 이중 주파수 구조, GESTURE-7, Guide Action으로 대응한다.

## 2. 아키텍처 개요

- **System 2 (S2, ~10 Hz)**: Qwen3-VL 백본. 언어 지시와 희소한 시각 입력을 받아 학습 가능한 토큰으로 hidden feature를 뽑아 "추상 액션 가이던스"로 넘긴다.
- **System 1 (S1, ~20 Hz)**: InternVLA-M1에서 영감을 받은 설계. 시각 입력과 end-effector 상태로 contrastive fine-tuning한 DINOv2가 고주파 시각 토큰을 만들고, Q-Former가 S2 가이던스와 이를 압축, Diffusion Transformer(DiT)가 노이즈에서 액션 청크를 denoise한다.
- 두 시스템은 명시적 데이터 흐름으로만 연결되어 서로 다른 연산 장치에 분산 배치가 가능하다.

📌 [Figure 1 삽입] — S2(Qwen3-VL, 10 Hz)와 S1(DINOv2 + Q-Former + DiT, 20 Hz) 구조도

## 3. GESTURE-7 표현

End-effector 상태를 `[x, y, z, r, p, w, gesture]` 7차원으로 표현한다.

- 위치는 **정수 cm**, 자세(roll/pitch/yaw)는 **정수 degree**로 양자화 — 기계 공차를 감안해 학습 부담과 환각을 줄이려는 의도.
- `gesture`는 자연어 키워드(`grasp`, `pinch`, `fist`, `thumb_bent_finger` 등). 평행 그리퍼든 다지 손이든 같은 동작이면 같은 키워드를 공유한다.
- 키포인트는 다지 손은 가운데 손가락 뿌리, 그리퍼는 그리퍼 중심.
- 어휘는 고정이 아니라 과제에 맞게 확장 가능.

이 표현이 텍스트이므로 VLM 출력과 GRPO 보상 설계가 자연스럽게 연결된다는 것이 핵심 논리다.

## 4. Guide Action

비동기 실행에서는 새 청크가 도착하는 순간 궤적이 "hard switch"되며 기계적 jitter가 생긴다. Guide Action은 이를 이미지 outpainting처럼 다룬다.

- 다음 청크 생성 시, 이전 청크 중 아직 실행되지 않은 액션(예: a4–a7)을 prior로 조건화: p_θ(A^{t-1}_full | A^t_full, A_prior, τ).
- **매 denoising step 후** 알려진 구간을 강제로 재삽입: A^{t-1}_full = M ⊙ A^{(t-1)}_prior + (1−M) ⊙ A^{t-1}_noisy, M[1..k]=1.
- guide 포인트 개수는 추론 지연에 맞춰 가변이며, 학습 시 0~3개로 무작위 샘플링.
- 학습 손실은 M=0인 새로 생성되는 구간에만 적용된다 (Eq. 3).

RTC나 spline 등 사후 평활화와 달리 생성 과정 내부에서 연속성을 강제한다는 점을 차별점으로 내세운다.

## 5. 3단계 학습

| Stage | 대상 | 내용 |
|---|---|---|
| I | S2 SFT | Qwen3-VL 원본 가중치에서 시작. 5개 과제: GESTURE-7 pose 회귀, 2D 손 키포인트 회귀, 2 Hz 2D waypoint 예측, 2 Hz 3D GESTURE-7 궤적 예측, CoT(서브태스크 추론 + 궤적) 생성 |
| II | S2 GRPO | 같은 과제에 대해 GESTURE-7 거리(위치+자세+제스처 불일치, 가중 1.0씩), 키포인트 가우시안 보상, waypoint(점 거리 + DTW), 궤적 보상(K∈{2,3,4}), 형식 보상(가중 0.5) |
| III | S1+S2 공동 SFT | 두 시스템 가중치를 함께 최적화, 각 시각 인코더는 동결. 마스크된 noise-prediction 손실 |

저자들은 대규모 cross-embodiment 사전학습이 목표 과제에 대한 이득이 제한적이고 망각을 유발할 수 있다며, 추가 사전학습 없이 과제 데이터로 바로 fine-tuning한다고 밝힌다. 학습은 8×H800 단일 노드, 배치 96.

> 참고: 기여 목록에서는 Stage III를 "weighted flow-matching loss"라고 부르지만 Eq. 3은 ε-예측(noise prediction) L2 손실로 적혀 있다. 본 트래커에서는 DiT denoiser라는 점에서 `diffusion`으로 분류했다.

## 6. 실험 설정

- **시뮬레이션**: LIBERO-Plus — 7개 perturbation 차원(21 하위 차원), 10,030개 과제, 5단계 난이도. 지표는 성공률.
- **실기**: AgiBot A2 양팔 로봇, 스테레오 RGB, 정책은 RTX 4070에서 구동. Task 1 Pick-and-Place, Task 2 Packaging Line Material Feeding.
- **베이스라인**: InternVLA-M1, π0.5, GR00T N1.5, OpenVLA-OFT. 공식 LIBERO-Plus 수치가 있으면 인용, 없으면 LeRobot으로 재현.
- **부드러움 지표**: 관절별 평균 절대 jerk(3차 차분 / Δt³), 오른팔 관절 7–13.

## 7. 주요 결과

**LIBERO-Plus (Table 2, NebulaVLA(ALL))**

| Camera | Robot | Lang. | Light | Bkg. | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|
| 91.0 | 58.5 | 85.2 | 95.9 | 95.9 | 96.9 | 81.8 | **85.5** |

본문 기준 전체 성공률 비교: NebulaVLA 85.5 > InternVLA-M1 81.3 > OpenVLA-OFT+ 79.6 > GR00T N1.5 59.0 > π0.5 58.0. 본문은 Camera 차원에서 InternVLA-M1보다 약간 낮다고 인정한다.

**실기 (Table 1)**

| | InternVLA-M1 | NebulaVLA-Homo | NebulaVLA-Heter |
|---|---|---|---|
| Task 1 Pick-and-Place SR | 77.91% | 87.14% | **92.08%** |
| Task 2 Feeding SR | 72.5% | 90% | **92.5%** |
| 평균 추론 지연 (ms) | 81 | 115 | **42** |

동일 모델을 동기식(homogeneous)으로 돌리면 115 ms → 이종 주파수로 42 ms. 초록의 "~2.7× 가속"은 이 115/42 비율과 일치한다.

## 8. 어블레이션

**학습 단계 (Table 2)**

| 설정 | Camera | Robot | Lang. | Light | Bkg. | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|---|
| w/o SFT, w/o RL | 93.5 | 57.9 | 80.5 | 97.8 | 94.9 | 92.6 | 73.0 | 83.8 |
| w/o RL | 87.8 | 56.3 | 82.8 | 93.3 | 93.2 | 94.2 | 79.2 | 83.0 |
| ALL | 91.0 | 58.5 | 85.2 | 95.9 | 95.9 | 96.9 | 81.8 | 85.5 |

- S2 SFT는 Layout을 +6.2 올리지만, **SFT만 하면 Total이 오히려 83.8 → 83.0으로 떨어진다**. RL까지 해야 85.5.
- Camera는 SFT/RL 모두에서 하락(93.5 → 87.8 → 91.0). 저자들은 GESTURE-7이 시각 특징과 공간 좌표 사이에 고정 매핑을 학습시키기 때문이라고 해석한다.

**Guide Action (Table 3, 평균 jerk ×10⁻³, 실기 10회)**

| | J7 | J8 | J9 | J10 | J11 | J12 | J13 |
|---|---|---|---|---|---|---|---|
| Native Async | 6.18 | 6.47 | 6.55 | 7.66 | 9.69 | 4.95 | 3.37 |
| Guide Action | 5.27 | 5.08 | 5.47 | 5.59 | 8.43 | 2.91 | 1.85 |

모든 관절에서 감소, J12 −41.2%, J13 −45.1%, 평균 −25.6%.

## 9. Related Work 상의 위치

- **Fast-slow VLA**: HiRT, Figure Helix, InternVLA-M1 계열과 같은 계보. 차별점은 S2를 텍스트 기반 GESTURE-7로 SFT+GRPO 한다는 것.
- **액션 표현**: FAST/Mind-to-Hand(이산 토큰), KineVLA/PUMA(연속 회귀)와 대비해 "언어 키워드 + 정수 양자화 포즈"의 하이브리드.
- **청크 평활화**: RTC, ABPolicy(B-spline), SmoothVLA(jerk 보상)와 달리 denoising 내부의 inpainting식 제약. 개념적으로는 RTC의 inpainting 계열과 매우 가깝다.

## 10. 강점

- 이중 주파수 분리의 효과를 **같은 모델의 Homo vs Heter 비교**로 실기에서 보여 줘, 지연·성공률 개선이 아키텍처 때문임을 비교적 깨끗하게 분리했다.
- Guide Action은 구현이 단순(마스크 재삽입)하면서 관절별 jerk 수치로 효과를 제시했다.
- GESTURE-7은 텍스트 출력이라 GRPO 보상(거리, DTW, 형식) 설계가 직관적이며, 다지 손/그리퍼 통일이라는 실용적 목적이 뚜렷하다.
- 단계별 어블레이션에서 SFT 단독의 역효과와 Camera 하락을 숨기지 않고 보고했다.

## 11. 약점 및 한계

- **표준 LIBERO 수치가 없다.** LIBERO-Plus만 보고하고, 과제 suite별 수치(Spatial/Object/Goal/Long)는 그림(Fig. 3b)에만 있다.
- LIBERO-Plus Robot 차원 58.5로 여전히 낮으며, 7개 차원 평균과 Total이 다른데 Total 산출 방식(샘플 가중)이 명시되지 않았다.
- 실기 평가는 InternVLA-M1 한 개 베이스라인, 과제 두 개, 시행 횟수 명시 부족(92.08% 같은 비정수 비율이 나오는 근거 불명).
- 파라미터 수, 학습 데이터 규모, S2/S1의 정확한 크기가 기재되지 않았다.
- Stage III 손실을 "flow-matching"과 "noise prediction"으로 혼용 서술.
- GESTURE-7의 cross-embodiment 이점을 주장하지만 정량적 교차 임베디먼트 실험은 없다.
- 코드·가중치 미공개.

## 12. 총평

NebulaVLA는 fast-slow VLA에 (i) 텍스트 기반 end-effector 표현 GESTURE-7과 GRPO, (ii) inpainting식 Guide Action을 결합한 산업계 시스템 논문이다. 자체 정책을 3단계로 직접 학습했고, LIBERO-Plus 85.5%와 실기 42 ms/92%대 성공률이라는 정량 결과를 낸다. 다만 표준 벤치마크 부재, 소수 과제의 실기 평가, 모델 규모 미기재 등으로 재현성과 비교 가능성은 제한적이다. "비동기 분리 + 청크 경계 inpainting"이 실기 지연과 부드러움에서 실제로 효과가 있다는 사례 보고로 읽는 것이 적절하다.

**한 문장 요약**: 느린 VLM과 빠른 DiT를 비동기로 떼어 내고 이전 청크를 마스크로 붙잡아 다음 청크를 그리면, 지연은 115→42 ms로 줄고 관절 jerk는 평균 25.6% 줄어든다 — 다만 증거는 LIBERO-Plus와 두 개의 실기 과제에 한정된다.

<!-- VERIFIED: pdf -->
