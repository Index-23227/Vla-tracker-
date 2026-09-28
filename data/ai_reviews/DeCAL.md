# DeCAL — Physically-Grounded Dexterous VLA via Contact-Aware Latent Co-Imagination

- arXiv: 2609.09119 (v1 2026-09-08)
- 소속: Peking University, Beijing Academy of Artificial Intelligence (Yankai Fu, Ning Chen, Junkai Zhao, Shanghang Zhang 외)

## 1. 한 줄 요약

DeCAL은 이해·상상(생성)·행동 세 전문가로 이루어진 Mixture-of-Transformers 기반 시각-촉각-언어-행동 모델로, 접촉 인지 게이트로 촉각을 적응적으로 주입하고 미래 시각·촉각 잠재를 함께 예측해 물리 동역학을 내재화한다. 양팔 UR5 + 22-DoF 다지 손 6개 접촉 과제에서 평균 성공률 71%, 진행률 83.4%로 모든 기준선을 앞선다.

## 2. 문제 설정

다지 손 조작은 심한 가림과 복잡한 접촉 동역학 때문에 시각 중심 VLA가 취약하다. 기존 촉각 활용 연구는 촉각을 보조 신호로 균질하게 결합해 접촉이 없는 구간에서는 오히려 시각 표현을 교란하고, 반응형이라 앞으로의 접촉 전이를 추론하지 못한다.

## 3. 핵심 아이디어

- **Adaptive Visuo-Tactile Fusion**: 매 트랜스포머 블록에서 국소 촉각 토큰을 key/value로 하는 교차 어텐션을 잔차로 더하되, 전역 촉각 토큰에서 나온 게이트 σ로 가중해 접촉 구간에서만 촉각이 강하게 작동하도록 한다.
- **Visuo-Tactile Latent Co-Imagination**: 생성 전문가가 미래 Cosmos-VAE 시각 잠재와 촉각 잠재를 병렬 디코딩으로 함께 예측해 "암묵적 물리 세계 지식"을 제공한다.
- **Factorized Flow Matching**: 팔과 손 행동을 독립 노이즈·별도 토큰으로 생성하되 같은 트랜스포머로 공동 처리한다.

## 4. 아키텍처

- 이해 전문가: Qwen3-VL. 텍스트 + 3개 RGB 시점(헤드, 좌/우 손목) + 촉각 교차 어텐션.
- 생성 전문가: Qwen3 백본. 32×32 VAE 잠재를 8×8 커널 conv로 4×4로 압축, 이해 전문가 KV 캐시를 활용한 병렬 디코딩. 6-DoF 힘을 인코딩해 촉각 잠재를 만들고 학습 시 raw 촉각 이미지·deform map·6-DoF 힘으로 재구성 감독.
- 행동 전문가: Qwen3 백본의 flow matching, 액션 청크 길이 50.
- 블록 단위 어텐션 마스크로 이해 → 생성 → 행동 단방향 정보 흐름.
- 촉각 인코더: 손가락별 공유 ResNet → 손가락 간 트랜스포머 → 국소 토큰 + 전역 토큰.

## 5. 학습과 추론

InternVLA-A1-3B 사전학습 가중치에서 시작해 L_total = λ_v·L_visual + λ_t·L_tactile + L_action으로 학습한다. H100 ×8, AdamW 5e-5(2,000 step warmup, 100k step에 걸쳐 5e-6까지 감소), 과거 15프레임(30Hz 기준 약 0.5초) 히스토리 사용. 과제당 100개 원격조작 시연(MetaGlove Pro + VIVE 트래커). 추론 시에는 재구성 디코더를 생략하고 RTX 4090 단일 GPU에서 청크당 0.27초, 시스템 30Hz.

## 6. 주요 결과 — 실세계 6개 과제 (Table 1, 과제당 20회, SR %)

| 방법 | Wipe Vase | Erase WB | Assemble | Twist Cap | Pipetting | Screw Bulb |
|---|---|---|---|---|---|---|
| GR00T N1.6 | 30 | 50 | 30 | 10 | 60 | 10 |
| InternVLA-A1 | 65 | 50 | 15 | 5 | 35 | 30 |
| ViTacFormer | 80 | 45 | 15 | 65 | 55 | 25 |
| DECO | 90 | 60 | 35 | 70 | 45 | 35 |
| InternVLA-A1t | 75 | 45 | 45 | 25 | 20 | 25 |
| **DeCAL** | **100** | **80** | **65** | **80** | **60** | **40** |

평균 SR 71%, PSR 83.4%(초록). Pipetting은 GR00T N1.6과 동률(60%)이며, PSR에서는 Pipetting(ViTacFormer와 86.3 동률), Screw Bulb(InternVLA-A1 53.8 > DeCAL 48.8)처럼 최고가 아닌 경우도 있다. 촉각을 VLM에 단순 주입한 InternVLA-A1t는 Twist Cap·Pipetting에서 원본보다 나빠져 "순진한 촉각 주입은 시각 분포를 교란한다"는 주장을 뒷받침한다.

## 7. 일반화와 생성 품질

- OOD(Twist Cap, Table 6 최종 단계): 배경 0.60 / 혼잡 0.70 / 조명 0.65 / 새 물체 0.75, 단계 평균 0.81(DECO 0.70, ViTacFormer 0.69). 새 물체에서 DECO(0.35) 대비 격차가 가장 크다. Fig. 9는 조명을 70%로 그려 표와 5pt 차이가 있다.
- 단계별 평균(Table 5): DeCAL 0.91 vs ViTacFormer 0.79, DECO 0.78.
- 미래 프레임 생성(Table 3): Assemble Parts Cos 0.946 / LPIPS 0.222, Twist Cap 0.927 / 0.245로 InternVLA-A1(0.913/0.243, 0.908/0.257)보다 우수.

## 8. 어블레이션 (Table 2, Assemble Parts / Twist Cap SR %)

| 구성 | Assemble | Twist Cap |
|---|---|---|
| w/o Factorized FM | 25 | 35 |
| w/o 시각·촉각 생성 | 20 | 30 |
| w/o 시각 생성 | 35 | 45 |
| w/o 촉각 생성 | 55 | 60 |
| w/o 촉각 게이트 | 50 | 70 |
| Full | 65 | 80 |

생성(상상) 브랜치를 모두 제거할 때 가장 크게 떨어지고, 팔-손 분해 flow matching도 기여가 크다.

## 9. 설계 해석

DeCAL은 InternVLA-A1류 "이해-생성-행동 MoT" 틀에 촉각을 일급 모달리티로 넣은 확장이다. 시각 미래 예측이 촉각 미래 예측보다 기여가 크다는 어블레이션은 촉각이 "현재 접촉 판단"에, 시각 상상이 "장기 계획"에 주로 기여함을 시사한다. 게이트 값이 접촉 순간에만 커지는 시각화와 t-SNE 군집은 적응 융합이 의도대로 작동한다는 정성적 근거다.

## 10. 강점

- 촉각 게이팅·시각/촉각 공동 상상·팔-손 분해를 하나의 모델에 통합하고 각 요소를 분리 검증.
- 범용 VLA(GR00T N1.6, InternVLA-A1)와 촉각 전문 정책(ViTacFormer, DECO), 촉각 단순 주입 변형까지 포함한 공정한 비교 설계.
- 청크당 0.27초로 실시간성을 유지.
- 단계별 성공률과 OOD 결과를 부록 표로 상세히 제공.

## 11. 한계

- 모든 결과가 과제당 20회 실세계 시행이며 시드·분산 보고가 없다.
- 시뮬레이션 벤치마크가 없어 다른 VLA와의 직접 순위 비교가 어렵다.
- OOD·어블레이션은 1–2개 과제(주로 Twist Cap)로 제한되며, Fig. 9와 Table 6 사이 수치 불일치가 있다.
- Sharpa 고해상도 시각 촉각 센서라는 특수 하드웨어에 의존하고, 대규모 시각-촉각 사전학습은 없다(저자 명시).
- Screw Light Bulb는 40%로 여전히 낮다.

## 12. VLA-Tracker 관점 평가

InternVLA-A1-3B에서 초기화해 새 모듈과 함께 전체 학습한 자체 정책이므로 수록 대상이다. `real_world` 블록에는 Table 1의 과제별 SR과 초록에 명시된 평균 71.0을 넣었고, PSR(평균 83.4), 기준선 SR, 단계별 평균(Table 5), OOD(Table 6), 어블레이션(Table 2), 생성 품질(Table 3)은 별도 블록으로 분리했다. 촉각 기반 다지 손 VLA 계열에서 강한 기준점이 될 모델이다.

<!-- VERIFIED: pdf -->
