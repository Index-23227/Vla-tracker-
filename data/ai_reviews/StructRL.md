# StructRL: Structured Action-Space Exploration for Flow-Based VLAs

> **한 줄 요약**: flow-matching VLA의 RL 미세조정에서 탐색 노이즈를 denoising chain 내부가 아니라 **실행되는 action 공간**에 직접 넣고(AR(1) 시간 상관 + position/rotation/gripper 그룹별 스케일), **last-step replay**로 마지막 Euler step만 재생해 정책 경사를 flow decoder에 전달하는 방법. GR00T N1.5 LIBERO 평균 52.5 → 99.0, π0.5 ManiSkill OOD 평균 70.3, 실제 Franka pick-banana 84%(가우시안 baseline 56%).

- **arXiv**: 2608.15139v1 (2026-08-15, cs.RO)
- **소속**: Fudan University, Singapore Management University, Shanghai AI Laboratory
- **프로젝트 페이지**: https://flyfaerss.github.io/structrl/
- **백본**: π0, π0.5, GR00T N1.5 (piRL few-shot SFT 체크포인트에서 시작)

---

## 1. 배경 및 동기

π0, π0.5, GR00T 같은 flow 기반 VLA는 대규모 behavior cloning으로 학습되므로 오프라인 데모의 품질과 커버리지에 묶인다. 이를 넘어서기 위해 πRL(Flow-Noise, Flow-SDE), Flow-CPS, π-StepNFT 같은 온라인 RL 방법이 등장했는데, 이들은 공통적으로 **denoising chain 내부에 등방성·시간 독립 노이즈**를 넣어 확률적 정책을 만든다.

저자들의 문제의식은 로봇 탐색 노이즈가 구조를 가져야 한다는 것이다.
1. **시간적 매끄러움**: 한 action chunk 안에서 jitter가 심하면 실행이 불안정하거나 위험하다.
2. **그룹별 스케일**: position, rotation, gripper는 범위와 안전 민감도가 다르다.

## 2. 핵심 관찰: Structured Noise Dilution

구조화된 노이즈를 단순히 in-chain에 넣으면 되지 않느냐는 질문에 대해, 저자들은 **남은 denoising step이 섭동된 샘플을 SFT action manifold 쪽으로 다시 끌어당겨 구조를 지운다**는 것을 보인다. 이를 *Structured Noise Dilution*이라 부른다.

GR00T N1.5 SFT 모델, LIBERO-Long에서의 측정(Table 1, σ=0.2):

| 설정 | Action-space (pos/rot/grip) | In-chain (pos/rot/grip) |
|---|---|---|
| SFT lag-1 상관 ρ̂ | 0.78 / 0.72 / 0.57 | 0.78 / 0.72 / 0.57 |
| ρ_inj = 0.95 주입 | 0.70 / 0.64 / 0.63 | 0.79 / 0.72 / 0.58 |
| SFT 표준편차 σ̂ | 0.10 / 0.08 / 0.21 | 0.10 / 0.08 / 0.21 |
| R_approach=(1.0,0.5,0.05) | 0.23 / 0.13 / 0.19 | 0.11 / 0.08 / 0.19 |

부록 Table A1에서 ρ_inj를 0→0.95로 바꿨을 때 action-space 출력 상관은 범위 0.590(pos)만큼 움직이지만 in-chain은 0.004에 불과하다. σ=0.5(Table A2)에서도 같은 패턴이 유지된다. 즉 in-chain 노이즈의 "구조"는 대부분 SFT 모델의 것이지 주입한 것이 아니다.

## 3. 방법 개요

StructRL은 세 가지 결합된 선택으로 구성된다.
1. **결정론적 ODE decoder**: x₁ ~ N(0,I)에서 T Euler step으로 clean action x₀ 생성.
2. **action 공간 구조화 노이즈**: a = x₀ + Δ₀.
3. **last-step replay**: rollout 시 마지막 직전 denoising 상태 x_{t_n}을 저장하고, 업데이트 시 F_θ^{n→0}(x_{t_n})만 재생해 Δ₀^θ = a − x̂₀를 복원, 그 로그확률을 정책 likelihood로 사용.

이렇게 하면 likelihood가 중간 denoising 전이가 아니라 **실제로 실행된 action**에 대해 정의된다. 시뮬레이션에서는 PPO, 실제 로봇에서는 AWAC(expectile value)와 결합한다.

## 4. 구조화 노이즈 분포

- **그룹 인지 스케일**: decoder의 마지막 hidden h_θ를 입력으로 하는 경량 MLP f_φ가 σ_{c,d} = σ_base · clip(f_φ(h_θ), [α_g, β_g])를 예측.
- **AR(1) 시간 상관**: ε_{c+1} = ρ ε_c + √(1−ρ²) ζ_c. 최종 Δ₀ = σ ⊙ ε.
- **로그확률**: AR(1) 밀도 + 대각 스케일 change-of-variables 항 (Eq. 11, 부록 C 유도).
- **정규화**: dimension-entropy(스케일 붕괴 방지), budget(전체 크기를 σ_base 근처로), temporal smoothness 세 항 (부록 D).

기본 ρ(pos, rot, grip) = (0.8, 0.6, 0.7), 클립 범위 pos (0.1, 2.0), rot (0.1, 1.5), grip (0.1, 3.0).

## 5. 학습 설정

- 시뮬레이션: LIBERO, ManiSkill, CALVIN. 4×A800. 64 병렬 환경, PPO.
- LIBERO 하이퍼파라미터(Table G1): GR00T/π0 200 epoch, π0.5 400 epoch, action chunk 5(Long은 10), denoise step 3~5, σ 0.05~0.2.
- 실제 로봇: Franka FR3, π0.5 + 비동기 AWAC, T=1(저장 latent가 정책 독립적인 x₁이 되어 무제한 재사용 가능). 1개 rollout GPU + 4개 learner GPU. critic은 캐시된 2048차원 PaliGemma feature + 19차원 proprio + GRU로 chunk 인코딩.

## 6. LIBERO 결과 (Table 2)

| 모델 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| GR00T N1.5 few-shot SFT | 41.4 | 58.6 | 48.2 | 61.9 | 52.5 |
| + πRL (Flow-SDE+PPO) | 96.6 | 100 | 93.8 | 95.6 | 96.5 |
| + Baseline (action-space Gaussian) | 96.2 | 99.4 | 91.8 | 95.6 | 95.8 |
| **+ StructRL** | **99.2** | **100** | **97.6** | **99.0** | **99.0** |
| π0 + StructRL | 98.8 | 99.8 | 98.4 | 91.8 | 97.2 |
| π0.5 + StructRL | 99.6 | 100 | 98.8 | 90.2 | 97.2 |

π0.5에서는 πRL(97.9)이 StructRL(97.2)보다 약간 높다. 저자들도 LIBERO는 포화된 in-distribution sanity check로 취급한다.

## 7. ManiSkill OOD 및 CALVIN

ManiSkill (Table 3):

| 모델 | IND | Vision | Semantic | Execution | OOD Avg |
|---|---|---|---|---|---|
| π0 + πRL | 78.8 | 61.1 | 25.4 | 31.5 | 39.3 |
| π0 + Baseline | 84.1 | 79.4 | 63.3 | 59.1 | 67.3 |
| π0 + StructRL | 87.8 | 80.0 | 64.2 | 56.2 | 66.8 |
| π0.5 + πRL | 90.9 | 68.0 | 34.5 | 45.4 | 49.3 |
| π0.5 + π-StepNFT | 85.4 | 76.9 | 56.6 | 45.1 | 59.5 |
| π0.5 + Baseline | 85.9 | 76.2 | 62.2 | 59.4 | 65.9 |
| **π0.5 + StructRL** | **90.9** | 78.3 | **68.4** | **64.2** | **70.3** |

핵심 해석: OOD 이득의 대부분은 **주입 위치 이동**(Flow-SDE → endpoint Gaussian: π0 +28.0, π0.5 +16.6)에서 나오며, AR/그룹 구조는 주로 **학습 효율**을 개선한다. π0에서는 StructRL의 OOD 평균이 Gaussian Baseline보다 오히려 0.5 낮다.

CALVIN (부록 Table F1, π0.5): StructRL 평균 길이 4.775 (SFT 3.838, Flow-SDE 4.717, Flow-Noise 4.652, Baseline 4.749), Len5 0.880. 다만 ABC→D 등 split이 명시되지 않았다.

## 8. Ablation (GR00T N1.5, LIBERO-Long, 그림 3)

- σ_base: 중간값(0.1)이 가장 안정적; 작으면 탐색 부족, 크면 불안정.
- 주입 방식: StructRL은 약 40 step에 90% 도달, Gaussian Baseline은 약 80 step. 최종값은 포화 후 비슷.
- Train–inference 일관성: Last-Last, 2ndlast-2ndlast 같은 일관 설정이 불일치 설정보다 확연히 좋음.
- denoising step T ∈ {2,3,4,5}: 수렴 속도·최종 성능 비슷 → last-step replay만으로 전체 flow field에 충분한 신호.

(이 결과들은 그림으로만 제시되므로 수치는 텍스트에 언급된 것만 인용.)

## 9. 실제 로봇 실험

Franka FR3, 두 과제 모두 10개 데모 SFT로 초기화(초기 성공률 0%), sparse binary reward, 소수의 SpaceMouse 개입.
- **pick-banana**: 60분 온라인 RL 후 StructRL 84% (21/25) vs Gaussian 56% (14/25).
- **plug-charger-in**: StructRL 약 30분, Gaussian 약 35분에 안정 성공 도달.

## 10. 강점

- in-chain 노이즈의 구조 소실을 정량적으로(Table 1, A1, A2) 보여주는 명확한 진단.
- last-step replay는 chain-level likelihood 없이 action-level likelihood를 얻는 간결한 트릭이며, PPO와 AWAC 양쪽에 그대로 들어간다.
- 세 가지 서로 다른 flow VLA 백본에서 일관된 검증, 그리고 실제 로봇 온라인 RL까지 연결.
- "위치가 OOD 이득의 주원인, 구조는 효율"이라고 스스로 기여를 분해해 과장하지 않는다.

## 11. 약점 및 한계

- 구조화 노이즈 자체의 최종 성능 이득은 작고, π0 ManiSkill OOD와 π0.5 LIBERO에서는 비교 대상보다 낮다.
- 실제 로봇은 과제 2개, 25 trial 수준이며 plug-charger-in은 수렴 시간만 보고.
- 학습 효율 주장의 핵심 근거(90% 도달 step)가 그림 기반.
- CALVIN split 미기재, 부록에만 위치.
- AR(1)·그룹 분할은 delta EEF action 표현을 전제로 하므로 joint-space 또는 휴머노이드 action에 대한 일반화는 미검증.

## 12. 총평

StructRL은 flow VLA RL에서 "노이즈를 어디에 넣느냐"가 "어떤 노이즈를 넣느냐"보다 먼저라는 점을 설득력 있게 보여준 논문이다. endpoint 주입 + last-step replay라는 프레임워크가 실질적 기여이며, AR(1)·그룹 스케일은 그 위의 효율 개선이다. 자체적으로 π0/π0.5/GR00T 가중치를 RL로 갱신한 정책을 내놓고 LIBERO·ManiSkill·CALVIN·실제 로봇에서 정량 결과를 보고하므로 트래커 대상 정책으로 분류된다.

**한 문장 요약**: denoising chain을 결정론적으로 두고 탐색은 실행 action에 직접 하라 — 그러면 OOD 일반화가 크게 오르고, 노이즈에 구조를 주면 더 빨리 배운다.

<!-- VERIFIED: pdf -->
