# MIRTH: Mutual-Information Reasoning with Temporal Hubs for Vision-Language-Action Agents

> **한 줄 요약**: Frozen OpenVLA 백본 위에 (1) K=4 multi-rate EMA workspace hub + w=4 short-horizon attention hub의 이중 스케일 temporal memory, (2) InfoNCE 기반 mutual-information 목적함수로 학습되는 latent reasoning token, (3) autoregressive scalar 디코딩을 대체하는 parallel vector-wise action decoding을 결합하여 LIBERO 평균 98.1% (Long 95.3%)와 64.4 Hz 제어 처리량, 그리고 12.1%의 emergent error recovery를 달성한 8B(학습 482M) VLA.

---

## 1. 배경 및 동기

### 기존 단일 프레임 VLA의 세 가지 구조적 한계
- **Temporal myopia**: OpenVLA, RT-1 류의 single-frame 모델은 현재 관측만 조건으로 행동을 디코딩 → 물체가 가려지거나(occlusion) 이동하면 상태 추적 실패, motion trend를 활용 못함
- **Reasoning gap**: 고수준 언어 목표와 저수준 모터 명령 사이의 간극. Explicit 텍스트 CoT는 annotation 비용이 크고, 다양한 언어 서술이 동일한 물리 동작에 대응하는 many-to-one 문제 발생
- **Inference 비효율**: scalar-wise 양자화 토큰의 autoregressive 생성은 하나의 pose마다 긴 토큰 체인을 요구 → 실시간 고주파 제어 불가

### 핵심 질문
- **컨텍스트 윈도우를 늘리지 않고 임의 길이의 히스토리를 고정 길이 프롬프트로 압축할 수 있는가?**
- **텍스트 supervision 없이 latent 공간에서 reasoning을 유도할 수 있는가?**

📌 [Figure 1 삽입] — occlusion 상황에서 단일 프레임 모델의 action drift vs MIRTH의 memory hub 기반 recovery

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

MIRTH = **Frozen OpenVLA 백본**(DINOv2+SigLIP 비전 인코더 730.91M frozen, LLM은 LoRA rank 32) + 세 모듈:

1. **Temporal Memory Hubs** (46.86M): workspace(장기) + short-horizon(단기)
2. **Latent Reasoning Tokens** (52.96M): MI 목적함수로 정렬되는 m개의 학습 가능 토큰
3. **Parallel Action Decoder** (285.74M): 2-layer projection head, 학습 파라미터의 59.24%

총 8.02B 중 482.34M(6.01%)만 학습.

### 2.2 Dual-Scale Temporal Memory Hubs

| Hub | 메커니즘 | 역할 |
|------|------|------|
| Workspace (장기) | K=4개의 EMA 메모리 맵, decay율 β≈{0.01, 0.031, 0.097, 0.3} 로그 간격 + 1/2차 motion 통계(µ, σ²) + per-patch mixture weight | 느리게 변하는 scene layout, 과거 물체 상태 보존 |
| Short-horizon (단기) | 최근 w=4 프레임 큐에 대한 temporal attention (recency bias γ=1.1, τ=1.0) | 고주파 motion trend 추적 (~0.4초 @ 10Hz) |

두 hub는 per-patch sigmoid gate로 융합. 통합 방식은 **prefix**(메모리 토큰을 시퀀스 앞에 부착, 95.3% / 64.4Hz)와 **infusion**(현재 프레임 patch embedding에 곱셈/덧셈 변조, 93.1% / 70.0Hz) 두 변형 — 본 실험은 prefix 채택.

> ❓ **예상 질문**: 히스토리 길이가 늘어나면 계산량이 늘지 않는가?
> **답변**: EMA는 incremental update이므로 히스토리 길이와 무관하게 메모리 크기와 LM 입력 토큰 수가 고정. Video Transformer의 dense spatio-temporal attention(quadratic scaling)과의 핵심 차별점.

### 2.3 Latent Reasoning with Mutual Information

- 입력 시퀀스: `X_t = [Z̃_t(융합 시각/proprio); L_t(언어); T_reas(reasoning 토큰); T_act(action 토큰)]` — reasoning 토큰이 조건(perception+instruction)과 효과(action) **사이**에 위치
- Hidden state에서 reasoning(r), action(a), context(x) 표현을 pooling → shared contrastive space로 projection
- **InfoNCE 이중 손실**: L_ra(reasoning→action 예측성) + L_rx(reasoning→context 정박), MI의 lower bound
- 최종 손실: `L = L_l1 + λ_mi·L_MI` (λ_mi=0.001)

> ❓ **예상 질문**: 왜 explicit CoT 텍스트 대신 latent MI인가?
> **답변**: (1) dense 텍스트 annotation 비용 제거, (2) 언어의 many-to-one 모호성 회피, (3) t-SNE 분석에서 task-ID supervision 없이도 reasoning embedding이 task semantic별 클러스터로 자발적 조직화됨을 확인. Trade-off는 human readability 상실(저자들도 Limitations에서 인정).

### 2.4 Parallel Action Decoding

- Scalar-wise(N=T×F 토큰) 대신 **timestep당 1토큰**(N=T), 시퀀스 길이 1/N_F로 감소
- Appendix F의 4가지 디코딩 패러다임 비교 결과, **global concatenated vector-wise** 방식(flatten 후 global projection)이 가장 빠른 수렴 → 최종 채택
- Chunk size 10 (수집 궤적 샘플링 레이트와 동일): 성공률 94~95%와 62Hz의 균형점 (chunk 30이면 68Hz지만 88%로 하락)
- Full causal attention 채택 (hybrid attention은 FlashAttention-2 비호환으로 학습 속도 저하, 성능 차이는 미미)

---

## 3. 실험 설정

| 항목 | 내용 |
|------|------|
| 시뮬레이션 | LIBERO 4개 suite (Spatial/Object/Goal/Long, 각 10 tasks), suite당 500 에피소드 평균 |
| 실제 로봇 | LeRobot 단일 팔 (작업 반경 ~55cm, wrist + overhead 카메라), 20 tasks × 50 demos = 1000 궤적, task당 30 trials |
| 학습 | 2× RTX Pro 6000, global batch 64, 모델당 ~5일, LoRA rank 32 |
| 추론 | RTX 5090 1장 |
| 전처리 | idle frame 필터링, 224×224 resize, vertical flip augmentation |

LeRobot 태스크는 난이도 5단계: Basic pick-place → Mechanism(서랍) → Scene Rearrange → Category Reasoning → Semantic Recipe("아침 식사 준비" 등 추상 지시).

---

## 4. 주요 결과

### LIBERO (Table 1, 500 episodes/suite)

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| **MIRTH** | **98.2** | **100.0** | **98.8** | **95.3** | **98.1** |

- OpenVLA-OFT 대비 +1.0%p, 특히 장기 컨텍스트가 필요한 LIBERO-Long에서 95.3%
- LeRobot 실기: 5개 태스크 그룹 전반에서 최상위 성공률·처리량, 난이도가 오를수록 baseline 대비 격차 확대

---

## 5. Ablation 분석 (LIBERO-Long)

| 변형 | 성공률 |
|------|--------|
| **MIRTH (Full)** | **95.3%** |
| w/o workspace hub | 94.0% (−1.3) |
| w/o short-term hub | 94.4% (−0.9) |
| w/o 양쪽 hub (single-frame) | 93.2% (−2.1) |
| w/o MI loss (토큰만 유지) | 94.5% (−0.8) |
| w/o reasoning tokens | 93.9% (−1.4) |

- 두 hub 제거 시 하락폭(2.1%)이 개별 제거 합과 유사 → 상보적 역할
- MI 제약 없는 latent 토큰은 의미 있는 intent를 포착 못함 → 목적함수가 핵심

---

## 6. Temporal Grounding 검증 (RQ3)

- **Linear probing**: frozen 표현 위 경량 regressor로 로봇 상태/속도 예측. MAE — 상태: OpenVLA 0.32 vs MIRTH 0.11, 속도: 0.15 vs 0.04. 단일 프레임 모델은 정적 layout은 포착하나 motion의 고차 미분을 인코딩 못함
- **Frame shuffling**: short-horizon hub 내 프레임 순서를 무작위 셔플 시 95.3% → 88.0% (−7.3%p) → 메모리를 bag-of-frames가 아닌 **인과적 시간 구조**로 활용함을 입증

## 7. Latent Reasoning 분석 (RQ4)

- **t-SNE**: 20개 태스크의 reasoning embedding이 task semantic별 클러스터로 자발적 분리 (container 열기 vs pick-place 등)
- **Error recovery** (LeRobot 실패 시나리오): OpenVLA 5.2%, MIRTH w/o reasoning 8.7%, **MIRTH full 12.1%** — reasoning 토큰이 내부 계획과 피드백의 불일치를 감지하는 discrepancy checker로 기능, re-grasping 유도

---

## 8. 관련 연구와의 위치

| 접근 | 대표 | MIRTH의 차별점 |
|------|------|----------------|
| Single-frame VLA | OpenVLA, RT-2 | 고정 길이 메모리로 temporal myopia 극복 |
| Video Transformer | TRecViT 등 | dense attention의 quadratic cost 회피 (EMA + 고정 토큰) |
| Explicit reasoning | SayCan, RT-Trajectory, CoT-VLA | 텍스트 supervision 불필요한 latent MI reasoning |
| Diffusion policy | Diffusion Policy, Octo | 통합 token 인터페이스 유지하며 parallel decoding으로 속도 확보 |

## 9. 강점

1. **파라미터 효율**: 8.02B 중 6%만 학습 (LoRA + 경량 모듈)
2. **속도-성능 양립**: 64.4 Hz 처리량으로 OpenVLA급 이상의 속도 + SOTA 성공률
3. **철저한 분석**: linear probing, frame shuffling, t-SNE, recovery rate 등 메커니즘 검증 실험이 풍부
4. **재현성**: 코드/데이터 공개(github.com/kiva12138/mirth), 오픈소스 LeRobot 플랫폼 사용, 하이퍼파라미터 상세 공개

## 10. 한계 및 비판적 검토

1. **해석 불가능성**: latent reasoning은 텍스트 CoT와 달리 실행 전 감사(audit) 불가 (저자 인정)
2. **Embodiment 범위**: 고정형 단일 팔에 한정 — bimanual/mobile 미검증
3. **고정 메모리 용량**: 수백 스텝 전 상태가 필요한 극단적 장기 태스크에서는 catastrophic forgetting 가능
4. **LIBERO 포화**: OpenVLA-OFT 97.1% 대비 +1.0%p는 벤치마크 포화 구간의 개선 — 실질 기여는 LeRobot 실기와 recovery 분석에서 더 뚜렷
5. **본문-부록 불일치**: 본문 3.3은 independent parallel decoding을 기술하나 Appendix F는 global concatenated 방식을 최종 채택했다고 명시. 저자 스스로 accepted version의 기호 오류를 각주로 언급 — 수식 표기 신뢰도에 주의 필요
6. **Recovery rate 12.1%**: baseline 대비 2배 이상이지만 절대값은 여전히 낮음

## 11. 세미나 토론 포인트

1. EMA 기반 workspace memory는 결국 지수적 망각을 내포 — Retrieval 기반 external memory와의 결합이 더 나은가?
2. MI 목적함수의 λ_mi=0.001이라는 매우 작은 가중치 — reasoning 효과가 실제로 MI loss에서 오는가, 아니면 토큰 추가 자체의 capacity 효과인가? (ablation의 w/o MI loss −0.8%p가 부분적 답)
3. Frame shuffling에 대한 민감도(−7.3%p)는 강건성 관점에서 오히려 취약점일 수 있는가? (카메라 프레임 드랍/지연 상황)
4. Parallel decoding과 diffusion/flow-matching head의 속도-정밀도 비교는 공정하게 이루어졌는가?

## 12. 결론

MIRTH는 단일 프레임 VLA의 세 가지 고질적 한계(temporal myopia, reasoning gap, autoregressive 병목)를 하나의 통합 프레임워크로 공략한다. 개별 기법(EMA memory, InfoNCE, parallel decoding)은 각 분야에서 알려진 것이지만, 이를 frozen VLA 백본 위에 파라미터 효율적으로 조립하고 linear probing·shuffling·recovery 분석으로 각 모듈의 기여를 기계적으로 검증한 점이 돋보인다. LIBERO 98.1%는 포화 구간의 수치이나, LeRobot 실기에서의 추상 지시 수행과 emergent error recovery는 latent reasoning 접근의 실용적 가치를 보여준다. 완전 공개된 코드·데이터와 저비용 오픈소스 플랫폼 기반이라는 점에서 후속 연구의 좋은 출발점이 될 것이다.

---

**arXiv**: [2606.31167](https://arxiv.org/abs/2606.31167) | **Code**: [github.com/kiva12138/mirth](https://github.com/kiva12138/mirth) | **Affiliation**: Ritsumeikan University, Zhejiang University

<!-- VERIFIED: pdf -->
