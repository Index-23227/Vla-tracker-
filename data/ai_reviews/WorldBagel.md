# WorldBagel: Uncovering the Power of Unified Multimodal Models for Vision-Language-Action-World Modeling

> **한 줄 요약**: 통합 멀티모달 모델 BAGEL(two-tower UND/GEN 전문가 구조)을 Vision-Language-Action-World(VLAW) 모델로 확장한 프레임워크. (1) 연속 행동을 다중 주파수 sin/cos 토큰으로 사상하는 **Fourier Feature Action Tokenizer/Decoder(FFAT/FFAD)** — Lipschitz 안정성·단사성·복원 오차 유계를 수학적으로 보장, (2) 시각 관측·언어·행동·미래 프레임을 하나의 autoregressive 시퀀스로 배열하는 **sequence plan** 기반 interleaved VLAW 학습, (3) LLM식 smoothed dataset mixture + priority plan 샘플링을 제안. LIBERO 4-suite 평균 **98.0%**(RynnVLA-002 97.4, OpenVLA-OFT 97.1 상회), 세계 모델링에서도 LIBERO action-conditioned FVD 373.1(vs RynnVLA-002 389.5), 행동 분포 이동(노이즈/스케일링/시간 섭동)에서 일관되게 더 높은 PSNR·낮은 LPIPS를 달성. Georgia Tech, ECCV 2026.

---

## 1. 배경 및 동기

- **World model과 VLA의 수렴**: 세계 모델(Dreamer, DINO-WM 등)은 환경 동역학을 학습하되 언어·대규모 멀티모달 지식과의 결합이 얕고, VLA(RT-2, OpenVLA, π0.5)는 행동 예측·과제 성능에 집중하며 환경 동역학의 명시적 생성 모델링은 부차적이다. 저자들은 두 방향의 자연스러운 수렴점으로 **VLAW(Vision-Language-Action-World)** — 하나의 모델이 지각, 언어 grounding, 행동 예측, 미래 관측 생성을 공동 수행 — 를 제시한다.
- **통합(unification)의 힘**: BAGEL 등 최신 통합 멀티모달 모델은 생성(GEN)·이해(UND) 전문가를 분리한 two-tower 구조로 멀티모달 벤치마크 SOTA를 달성했으나, 이를 세계 모델로 어떻게 적응·평가할지는 미개척 영역이다. 본 논문은 "통합 자체가 VLAW 학습에 이득이 되는가"를 체계적으로 검증한다.
- 형식화: 시점 t에서 관측 o_t, 지시 l, 연속 행동 a_t ∈ R^d가 주어질 때 결합 분포 p(o_{t+1}, a_t | o_{≤t}, a_{<t}, l)을 단일 멀티모달 아키텍처로 공동 모델링 — 고전 세계 모델처럼 잠재 동역학과 정책을 분리 학습하지 않는다.

## 2. 방법론 심층 분석

### 2.1 Two-Tower 백본과 두 분기 (Sec 3.1)

- BAGEL의 two-tower 설계를 계승: 모든 모달리티(시각·언어·행동)가 공유 토큰 임베딩 공간에 임베딩되어 공통 멀티모달 self-attention 백본을 통과하되, 각 transformer 블록의 QKV 사영은 학습 목표에 따라 **UND 전문가** 또는 **GEN 전문가**로 라우팅.
- **VLA 분기**: ViT로 인코딩된 시각 상태 토큰 + 언어 토큰을 UND 전문가가 처리, 그 위에 FFAD가 연속 행동을 예측.
- **World Modeling 분기**: VAE로 인코딩된 시각 잠재 + 언어 + FFAT로 토큰화된 행동을 조건으로 GEN 전문가가 flow matching으로 다음 관측 V_{t+1}을 생성, VAE 디코더로 복원.
- 별도의 action expert는 도입하지 않음 — 행동 모델링은 미세조정된 토크나이저/디코더로 해결. 학습 손실: L = L_action + 0.1·L_vision (Fourier 공간 ℓ2 회귀 + flow matching).

### 2.2 Fourier Feature Action Decoder (FFAD, Sec 3.2)

- 표준 회귀는 행동을 독립 실수 출력으로 취급하고, bin 이산화·FAST는 이산화 아티팩트와 BPE 학습의 데이터 의존성 문제가 있음. FFAD는 행동 a_t를 다중 주파수 Fourier 특징 φ(a_t) = [sin(2^0πa), cos(2^0πa), ..., sin(2^Kπa), cos(2^Kπa)]로 확장(K=32)하고, **Fourier 공간에서 ℓ2 회귀**로 φ̂(a_t)를 예측.
- **역사상(inverse mapping)**: 예측된 (ŝ_k, ĉ_k)를 단위원에 정규화 → atan2로 위상 복원 → k=0 저주파를 anchor로 고주파 위상을 unwrap → 주파수 전체 등가중 평균으로 â 복원. 반복 최적화 불필요, 국소 Lipschitz 연속.
- **구현**: BAGEL 위 33M 경량 어댑터. `<|action_pred|>` 특수 토큰을 horizon H만큼 append(다중 스텝 autoregressive 행동 생성), 각 anchor 은닉 상태를 d_r=1024로 사영 → 2-layer Transformer encoder(4 heads, FFN 4d_r) → 2-block residual MLP head.

### 2.3 FFAT와 sequence plan (Sec 3.3-3.4)

- **FFAT**: 세계 모델 분기에서 행동은 예측 대상이 아니라 조건 신호 — 동일한 Fourier 표현을 선형 사영 W_f로 공유 토큰 공간에 매핑. 추론 시 FFAD가 생성한 행동을 연속 공간으로 복원 후 FFAT로 재인코딩하여 세계 모델에 공급, 예측-조건화 일관성 확보.
- **Sequence plan**: 일반화 형태 S = [V^{1:M}_{t-K+1:t}, L, Â^{1:H}_t, V̂^{1:M}_{t+1:t+H}] — 다중 뷰(M)·다중 스텝(K) 관측과 H-step 행동 rollout, 미래 프레임을 단일 autoregressive 시퀀스에 배열. ˆ표시 토큰에만 손실 적용. Appendix B: 정책 우선 Plan A와 행동-관측 interleave Plan C 등 복수 템플릿에서 샘플링.
- 테스트 시: 다중 뷰 관측 + 지시를 조건으로 H-step 행동 시퀀스를 autoregressive 생성, **open-loop 실행**; 미래 프레임 생성은 세계 모델링용으로 병행 제공.

### 2.4 LLM식 데이터 샘플링 (Sec 3.5)

- 이질적 데이터셋 크기 불균형에 대응해 **smoothed mixture sampling** p_i ∝ n_i^α (α ∈ (0,1)) 채택.
- **Priority sequence-plan sampling**: task → demonstration → training pair 계층 샘플링 후, policy-focused plan과 joint VLAW plan에 우선순위 가중치 부여 — w_joint=2 > w_policy=1로 행동 조건부 동역학 학습을 더 자주 감독.

## 3. 데이터 전략

- 세 벤치마크의 VLAW 궤적으로 SFT: **LIBERO**(시뮬레이션 장기 언어조건 조작, 4 suites), **Language Table**(실세계 탁상 블록 밀기, 인간 시연 + 자연어), **Franka**(NVIDIA IsaacSim Franka Emika Panda, 다중 객체 pick-and-place/재배치) — 시뮬 장기 조작, 실세계 언어 grounding, 물리 기반 제어를 포괄하는 상보적 구성.
- 멀티태스크 설정: **전체 태스크를 하나의 체크포인트로 학습**.
- 사전학습 BAGEL 체크포인트에서 초기화, UND/GEN 전문가 전체 + action head + Fourier 사영 + 생성 비전 디코더 갱신, 비전·언어 토크나이저는 동결.

## 4. 실험 설계

- **지표**: VLA는 성공률(SR), 세계 모델링은 FVD/PSNR/SSIM/LPIPS (action-conditioned vs action-free 양쪽).
- **학습**: AdamW lr 2e-5, weight decay 0.01, cosine decay, global batch 32, 80K steps, 8×H200, **5 seeds 평균**. K=32 (FFAT/FFAD).
- **베이스라인**: VLAW 최근접 비교 대상은 RynnVLA-002(정책과 세계 모델에 별도 가중치 사용). VLA 비교로 OpenVLA-OFT, π0.5, SmolVLA, ThinkAct, DiT Policy, Octo, Diffusion Policy 등. 동시기 미공개 연구 BagelVLA는 코드/모델 미공개로 재현 불가하여 제외.
- 3개 연구 축: (I) 멀티태스크 성능, (II) 행동 표현 품질, (III) 분포 이동 하 안정성.

## 5. 주요 결과

| Model | Multi-task | WM | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|---|
| Diffusion Policy | ✗ | ✗ | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| π0.5 | ✓ | ✗ | 91.2 | 87.5 | 94.3 | 74.1 | 86.8 |
| SmolVLA | ✗ | ✗ | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 |
| OpenVLA-OFT | ✓ | ✗ | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| RynnVLA-002 | ✗ | ✓ | 99.0 | 99.8 | 96.4 | 94.4 | 97.4 |
| **WorldBagel** | ✓ | ✓ | **99.2** | **99.9** | 97.5 | **95.3** | **98.0** |

- **LIBERO 평균 98.0%** — 최강 베이스라인 RynnVLA-002 대비 +0.6%p. RynnVLA-002는 정책/세계 모델 가중치를 분리하는 반면 WorldBagel은 완전 통합으로 우위 — 통합 모델링이 상보적 감독 신호를 제공함을 시사.
- **세계 모델링 (Table 3)**: LIBERO action-conditioned FVD 389.5→373.1, PSNR 21.74→23.88, SSIM 76.92→82.41, LPIPS 21.02→16.33. Language Table(FVD 418.7→392.2), Franka(462.4→421.9)에서도 일관된 개선. action-conditioned가 action-free를 전 데이터셋에서 상회 — 제어 신호의 명시적 조건화가 동역학 학습에 필수적임을 확인.

## 6. Ablation 분석

- **행동 디코딩 전략 (Table 4a)**: Regression(A-MSE 0.042 / SR 96.3) → Bin 256(0.037 / 95.6) → FAST(0.035 / 96.9) → **FFAD(0.028 / 98.0)**. Fourier 공간 예측이 이산화·직접 회귀 모두 상회.
- **Fourier band 수 (4b, 4c)**: FFAD는 K=8→32에서 A-MSE 0.034→0.028, SR 96.8→98.0 개선, K=64는 이득 없음(97.6). FFAT도 K=8→32에서 PSNR 23.21→23.88, 이후 포화 — K=32가 용량-안정성 균형점.
- **표현 구조 (4d)**: 고정 행동 임베딩에 linear probe로 LIBERO task 분류 — regression 49.8% / bin 52.1% / FAST 54.2% / **FFAD 68.7%**. Fourier 표현이 task 수준 구조를 훨씬 잘 인코딩.
- **분포 이동 안정성 (4e-4g)**: Gaussian 행동 노이즈(σ=0.05), 스케일링([0.8,1.2]), 시간 섭동(±1 step)에서 WorldBagel PSNR 23.64/23.57/23.41 vs RynnVLA-002 21.38/21.91/20.94 — RynnVLA-002는 시각 이력에 의존해 기존 궤적을 재생하는 경향, WorldBagel은 행동 조건 동역학을 실제로 포착.
- **고유값 스펙트럼 (4h)**: effective rank 33.7 vs 16.3, 지배 고유값 비율 λ1/Σλ 0.43 vs 0.67 — 분산이 더 많은 방향으로 분포된 풍부한 표현.

## 7. 관련 연구와의 위치

- **RynnVLA-002 (최근접)**: 동일하게 VLA+세계 모델 통합을 지향하나 Chameleon 기반 + bin 이산화 + 분리 가중치. WorldBagel은 BAGEL 기반 + Fourier 토크나이저 + 완전 공유 백본으로 SR·세계 모델링·강건성 모두 우위.
- **BagelVLA (동시기)**: 같은 BAGEL을 쓰지만 action-conditioned 세계 모델링과 행동 토크나이저가 없음(Table 1 기준 ✗). 미공개로 정량 비교 제외.
- **FAST/π0.5 계열**: BPE 기반 FAST 토크나이저의 데이터 의존성·불안정성을 지적하며, 학습 불필요·결정론적·이론 보장이 있는 Fourier 토크나이저로 대체.
- **고전 세계 모델(Dreamer, DINO-WM)**: 언어 grounding 부재 + 잠재 동역학/정책 분리 학습이라는 한계를 VLAW 통합 형식화로 극복.

## 8. 강점

1. **이론적 뒷받침이 있는 행동 토크나이저**: Lipschitz 안정성(Thm 1), 거의 모든 곳에서의 단사성(Thm 2), 위상 복원 오차 유계 |â−a| ≤ Cε(Thm 3), Stone-Weierstrass 기반 근사 능력(Thm 4)까지 — 행동 표현 설계에 드물게 수학적 정당화를 제공.
2. **3중 검증 체계**: 성능(SR)뿐 아니라 표현 품질(linear probe 68.7%), 강건성(3종 섭동 + 고유값 스펙트럼)으로 "통합이 이득"이라는 주장을 다각도로 입증.
3. **5 seeds 평균 + 멀티태스크 단일 체크포인트**: 평가 프로토콜이 상대적으로 엄격하고, 태스크별 특화 없이 98.0% 달성.
4. **세계 모델링과 정책의 진정한 공유**: RynnVLA-002의 분리 가중치 대비 완전 공유 백본으로 우위를 보여 "unification 자체의 이득"이라는 핵심 주장에 직접 증거 제공.

## 9. 약점 및 한계

1. **LIBERO 포화 구간에서의 미세 격차**: 97~98% 구간에서 +0.6%p는 5 seeds 평균이라 해도 분산·유의성 보고가 없어 결정적이라 보기 어려움.
2. **실물 로봇 부재**: Language Table은 실세계 데이터셋이지만 평가는 세계 모델링 지표 중심 — 실기 정책 배포(closed-loop 실제 로봇 SR)는 없음. open-loop H-step 실행의 실환경 강건성 미검증.
3. **베이스라인 비대칭**: Table 2의 여러 베이스라인은 학습 데이터·에피소드 수·파인튜닝 설정이 상이한 인용 수치. RynnVLA-002 외 통제 비교는 제한적.
4. **BAGEL 규모·추론 비용 미보고**: 파라미터 수, 제어 주파수, 추론 지연이 명시되지 않아 실용 배포 관점의 평가 불가. 8×H200 학습은 상당한 자원.
5. **코드 미공개**: "acceptance 이후 공개 예정" — 현재 재현 불가.
6. **unification 이득의 원인 분석 부족**: 상보적 감독 신호라는 해석은 그럴듯하나, world modeling loss를 끈 순수 VLA-only ablation(0.1·L_vision 제거)이 명시적으로 제시되지 않음.

## 10. 실용적 시사점

- **Fourier 토크나이저는 FAST/bin의 실용적 대체재**: 학습 불필요, 도메인·제어 범위에 무관하게 결정론적, 33M 어댑터로 기존 통합 모델에 부착 가능 — action MSE와 SR을 동시 개선.
- **통합 멀티모달 체크포인트(BAGEL류)는 VLAW의 강력한 초기화**: 별도 action expert 없이 SFT + 경량 어댑터만으로 SOTA급 정책 획득.
- **sequence plan 샘플링**은 정책 학습과 세계 모델링의 감독 균형을 맞추는 일반 레시피(w_joint > w_policy) — 이질적 멀티모달 로봇 데이터 학습에 이식 가능.
- 행동 조건 세계 모델의 강건성 평가(노이즈/스케일링/시간 섭동 + effective rank)는 세계 모델 벤치마킹의 유용한 프로토콜.

## 11. 예상 질문과 답변

| # | 질문 | 답변 |
|---|---|---|
| 1 | RynnVLA-002 대비 +0.6%p가 유의미한가? | SR 격차는 작지만, 세계 모델링(FVD −16.4, LPIPS −4.69)과 섭동 강건성(PSNR +2.3~2.5), effective rank(33.7 vs 16.3)에서 격차가 훨씬 크다 — 논문의 기여는 SR 자체보다 통합 모델링의 다면적 이득. |
| 2 | Fourier 표현이 왜 회귀보다 나은가? | 다중 주파수 대역에 정보를 분산시켜 고주파 신호 학습을 촉진(NeRF/Fourier features 문헌과 동일 논리)하며, Thm 4에 의해 선형 예측기로도 비선형 행동-상태 관계를 근사 가능. |
| 3 | 위상 unwrap이 잘못되면? | k=0 저주파 추정을 anchor로 2π 정수배를 더해 편차를 최소화 — a ∈ [−1,1] 유계 가정 하에 Thm 3이 오차 유계 보장. 단 ε가 크면 anchor 자체가 틀릴 수 있어 실패 모드 분석은 없음. |
| 4 | BagelVLA와의 차이는? | 동일 BAGEL 기반이나 BagelVLA는 action-conditioned 생성과 행동 토크나이저가 없음(Table 1). 동시기 미공개 연구라 정량 비교는 제외했다고 명시. |
| 5 | open-loop 실행의 한계는? | H-step rollout을 재계획 없이 실행 — LIBERO에서는 충분하나 실환경 교란에는 취약할 수 있다. closed-loop 재계획과의 비교는 없음. |
| 6 | world modeling loss가 정책에 실제로 도움이 되나? | Table 2에서 세계 모델 포함 모델(RynnVLA-002, WorldBagel)이 상위권이고 FFAD ablation은 통합 학습 하에서 수행되었으나, L_vision 제거 단독 ablation이 없어 인과는 간접 증거에 그침. |
| 7 | K=32가 모든 로봇에 일반적인가? | LIBERO 검증셋 기준 최적점이며, 행동 차원·제어 범위가 다른 도메인에서는 재탐색 필요. 다만 Fourier 인코딩 자체는 데이터 비의존적이라 이식은 용이. |
| 8 | 왜 action expert를 추가하지 않았나? | 통합 토큰 기반 모델링(Gato류) 철학 유지 + 경량 어댑터(33M)로 충분함을 보임 — BAGEL의 멀티모달 추론 능력을 보존하면서 행동 특화 학습을 격리. |

## 12. 결론

WorldBagel은 "통합 멀티모달 모델은 좋은 VLAW 모델인가"라는 질문에 체계적으로 답한 논문이다. BAGEL의 two-tower UND/GEN 구조 위에 이론 보장을 갖춘 Fourier 행동 토크나이저(FFAT/FFAD), sequence plan 기반 interleaved 학습, LLM식 우선순위 데이터 샘플링을 얹어, LIBERO 98.0%로 RynnVLA-002와 OpenVLA-OFT를 넘어서는 동시에 세계 모델링 fidelity와 행동 섭동 강건성에서 더 큰 격차를 보였다. linear probe(68.7%)와 고유값 스펙트럼(effective rank 33.7)으로 표현 품질까지 검증한 3중 평가 체계가 방법론적으로 돋보인다. 다만 SR 격차의 통계적 유의성, 실물 로봇 closed-loop 검증, world modeling loss의 단독 기여 ablation, 모델 규모·추론 비용 보고가 빠져 있고 코드도 미공개다. Fourier 기반 행동 표현과 통합 백본 SFT라는 레시피는 RynnVLA-002·BagelVLA와 함께 "VLA+세계 모델 통합" 흐름의 대표 사례로, 후속 실기 검증과 스케일 분석이 자연스러운 다음 과제다.

<!-- VERIFIED: pdf -->
