# PearlVLA: Progressive Embodied Action-Plan Refinement in Latent Space

> **한 줄 요약**: OpenVLA-7B 위에 12개 meta-query 토큰을 4개 visual anchor + 8개 latent plan 토큰으로 분리하고, frozen UWM 잠재 world model을 K=4회 닫힌 루프로 질의하여 plan 토큰을 잔차 갱신(coarse-to-fine) → parallel regression head로 H=8 행동 청크 디코딩. 위에 CRG-PRL(group-relative process-reward RL)을 얹어 LIBERO 98.7%로 SOTA 달성.

---

## 1. 배경 및 동기

### VLA 정책의 trade-off
- **직접 디코딩(OpenVLA-OFT 등)**: VLM 표현에서 행동을 곧장 회귀 → 저지연이지만 latent deliberation 부재. 단일 forward pass라 잘못된 plan을 자체 수정할 기회가 없음.
- **명시적 추론(CoT-VLA, Hume, ECoT)**: 텍스트 chain, 픽셀 subgoal, 후보 action 평가 등 → planning 품질은 좋아지지만 추가 지연·계산 비용 큼.
- **World Model 활용(VPP, TriVLA, MIND)**: 미래 정보를 정책 컨텍스트로만 소비 → 현재 plan에 대한 closed-loop 피드백은 없음.

### 핵심 질문
- **명시적 텍스트/픽셀 추론 없이, VLM의 latent space 안에서 효율적이고 점진적인 action-plan refinement가 가능한가?**
- **frozen latent WM의 미래 feedback을 현재 plan에 다시 써넣어 self-correction을 수행할 수 있는가?**

📌 [Figure 1 삽입] — (a) Direct decoding (b) Text/visual reasoning (c) WM scoring (d) PearlVLA: progressive latent refinement

---

## 2. 방법론 심층 분석

### 2.1 전체 파이프라인
1. OpenVLA-style 다중모달 시퀀스 `[BOS][vision][proprio][text][meta_query]` 처리 (prefix-LM attention).
2. 12개 meta-query 토큰의 마지막 레이어 표현을 `z_meta = [z_vis, z̃_0]`로 분리: **4개 read-only visual grounding** + **8개 writable plan 토큰**.
3. plan 토큰에 작은 Gaussian noise (`t* = 50`, `sqrt(1-α_bar) ≈ 0.089`) 주입 → `z_0`. **diffusion noise-prediction 목적이 아님**, 단순히 local robustness/CRG-PRL stochasticity 용도.
4. **K=4 closed-loop refinement** 수행 후 query-transformer regression head로 H=8 action chunk를 병렬 회귀.

### 2.2 Plan-Conditioned World Query
- 시각 토큰은 루프 전에 한 번 `q_anchor = P_vis(z_vis)`로 projection.
- 매 라운드 `q_k = q_anchor + β_k * P_plan(z_k)` (β_k: 학습가능 round-specific 스칼라).
- frozen UWM(300M DiT-based)이 action-free observation rollout(10-step DDIM)로 `o_k = WM(q_k; ε_o)` 생성. UWM의 action 채널은 사용 안 함, observation prediction path만 호출.

### 2.3 Future-Guided RefineNet
- `s_k = FutureEncoder(o_k)`: 미래 latent의 visual token들을 transformer encoder로 요약.
- `δ_k = RefineNet(z_k, s_k, e_k)`, `z_{k+1} = z_k + w_k δ_k`.
- RefineNet: 4-layer weight-shared transformer (AdaLN self-attention + gated cross-attention to future summary + gated MLP). Cross-attention gate와 modulation을 zero-init → identity update 근처에서 시작.
- **modulation 신호의 출처가 plan 토큰 자신이 아니라 (future summary + round embedding e_k)**: modulation path가 업데이트 대상에 완전 coupling되는 것을 방지.

### 2.4 두 개의 분리된 control variable
| 변수 | 작용 측 | 역할 |
|---|---|---|
| β_k | WM condition side | q_k가 q_anchor에서 멀어지는 정도 제한 (pretrained WM condition 분포 안에 머무름) |
| w_k | VLM latent side | 각 잔차가 plan을 얼마나 크게 바꿀지 통제. 기본 `w_k = sqrt(1-α_bar_{t_k})` with `t_k ∈ [100, 80, 60, 40] → w_k ≈ [0.169, 0.138, 0.106, 0.072]` (감소형 → coarse-to-fine 리듬) |

### 2.5 학습 목적 (Supervised)
```
L_SFT(n) = ω(n) · λ_mse · MSE(â, a*) + λ_dct · L_dct + λ_align · (1/K) Σ_{k=0..K-1} MSE(q_k, q_wm*)
ω(n) = min(1, n/T_warm)   # action-MSE에만 적용되는 linear warmup
```
- `q_wm*`: frozen UWM observation encoder가 demonstration state에서 만든 teacher condition.
- **Input-side alignment**가 핵심: pixel-level supervision은 image decoder를 거쳐야 해 noisy, output-side latent alignment는 frozen rollout 전체를 통과해야 함. 반면 input-side는 `VLM latent → projector → composer` 경로를 latent WM의 condition manifold에 직접 묶음.
- Warmup 의도: 초기에 action gradient가 plan을 망가뜨리지 않도록, world-query 경로가 먼저 WM condition manifold 근처에 자리잡게 함.

### 2.6 Causal Refinement-Grouped Process-Reward RL (CRG-PRL)
- supervised는 z_K가 최종적으로 demo action을 디코딩하면 만족 → **중간 edit들을 랭킹하지 않음**.
- γ=0 inner MDP: 상태 `x_k=(z_vis, z_k, o_k, k)`, 행동 `a_k=δ_k`, 전이 `z_{k+1}=z_k+w_k a_k`.
- **post-edit imagined future**에 process reward 부여: `a_k → z_{k+1} → q_{k+1} → õ_{k+1} → O_{k+1}^{imgs} → r_k = Score(O_{k+1}^{imgs}, ℓ)`. Score는 frozen **Robometer-4B**, H_judge=3 ordered imagined frame을 평가.
- **same-state local branching**: 한 base path에서 매 라운드마다 M=8 single-round residual branch를 `a_k^{(i)} = μ_θ(x_k^base) + σ_k ε_k^{(i)}`로 샘플 → 그룹 내 표준화 advantage `Ã_k^{(i)} = (r_k^{(i)} - r̄_k)/(std + ε)`. critic 없이 state difficulty와 round-별 reward scale 모두 상쇄.
- PPO-clipped objective + 세 가지 guardrail: (1) frozen supervised policy로의 KL, (2) write-back magnitude penalty, (3) base path의 final-action BC anchor.
- 학습률: RefineNet 1e-5, FutureEncoder 5e-6. VLM backbone, latent WM, projectors, composer, write-back schedule, action head 모두 **frozen**. inference 시에는 deterministic mean 사용 → latency 동일.

📌 [Figure 2 삽입] — VLM meta-query → visual anchor + noised latent plan → K rounds of {plan-conditioned query → frozen WM → RefineNet residual} → parallel H-step action chunk decoding.

---

## 3. 데이터 전략

| 데이터 | 용도 | 규모 |
|---|---|---|
| RLDS-formatted modified LIBERO (Spatial/Object/Goal/Long 4 suites) | 정책 학습/평가 | suite당 10 tasks, 총 500 expert demonstrations |
| LIBERO-90 action-free video | UWM post-training | 50K steps (4 평가 suite와 disjoint, scene-similar) |
| RoboCasa kitchen (24 tasks) action-free video | RoboCasa few-shot용 별도 UWM post-training | 100K steps |
| LIBERO-Plus | 강건성 평가 | 카메라/로봇/언어/조명/배경/노이즈/레이아웃 perturbation |

- proprio history는 transformer-based proprio encoder로 토큰화 후 multimodal sequence에 삽입.
- meta-query 토큰: 12개 학습가능, prefix-LM attention.

---

## 4. 핵심 실험 결과

### 4.1 LIBERO main (suite-specific protocol, 성공률 %)

| Group | Model | Spatial | Object | Goal | Long | Avg |
|---|---|---:|---:|---:|---:|---:|
| Flow/Diffusion | FLOWER | 97.5 | 99.1 | 96.1 | 94.9 | 96.9 |
| Flow/Diffusion | VLANeXt | 99.0 | 99.2 | 96.6 | 94.6 | 97.4 |
| Flow/Diffusion | π0.5 | 97.0 | 99.0 | 98.0 | 96.0 | 97.5 |
| Regression | OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| Regression | WorldVLA | 85.6 | 89.0 | 82.6 | 59.0 | 79.1 |
| Regression | CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| Regression | NORA | 92.2 | 95.4 | 89.4 | 74.6 | 87.9 |
| Regression | UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| Regression | π0-Fast | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| Regression | π0 (reg) | 97.8 | 98.2 | 94.6 | 90.2 | 95.2 |
| Regression | OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| **Regression** | **PearlVLA (supervised)** | **99.2** | **99.6** | **98.2** | **96.8** | **98.5** |
| **Regression** | **PearlVLA + CRG-PRL** | **99.4** | **99.8** | **98.4** | **97.2** | **98.7** |

- OpenVLA-OFT 대비 supervised만으로 **97.1 → 98.5 (+1.4)**, CRG-PRL 추가로 **98.7**. 동일 OpenVLA-7B backbone + parallel chunk decoding이라 잔여 gap은 inserted latent refinement에 귀속.
- regression head를 유지한 채 flow-matching/diffusion VLA group(FLOWER, VLANeXt, π0.5)까지 넘어선 점이 핵심 메시지.

### 4.2 LIBERO-Plus 강건성 (perturbation suites 평균, %)

| Model | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| π0-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| PearlVLA (sup.) | **65.9** | **40.9** | **81.2** | **93.8** | **93.9** | **79.9** | **78.9** | **76.3** |

- OFT 대비 +6.7 (69.6 → 76.3). 특히 **Robot state(+9.0)**, **Camera viewpoint(+9.5)** perturbation에서 큰 폭 개선 → latent refinement가 distributional shift에 둔감.

### 4.3 Refinement Depth ablation (K)

| K | Spatial | Object | Goal | Long | Avg |
|---:|---:|---:|---:|---:|---:|
| 0 | 98.0 | 98.6 | 97.2 | 93.4 | 96.8 |
| 1 | 98.6 | 99.2 | 97.2 | 95.2 | 97.6 |
| 2 | 99.4 | 99.2 | 97.4 | 95.6 | 97.9 |
| 4 | 99.2 | 99.6 | 98.2 | 96.8 | 98.5 |
| 4 + CRG-PRL | 99.4 | 99.8 | 98.4 | 97.2 | 98.7 |

- **96.8 → 98.5**가 supervised refinement만으로 얻은 이득. CRG-PRL은 그 위에 추가로 +0.2.
- K=0은 "RefineNet 업데이트가 빠진 PearlVLA"이지 OpenVLA-OFT 자체가 아님(주의).

### 4.4 Action chunk horizon 분석

| H | K | Spatial | Object | Goal | Long | Avg |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4 | 99.2 | 99.6 | 98.2 | 96.8 | 98.5 |
| 20 | 4 | 97.8 | 99.0 | 96.8 | 93.4 | 96.8 |
| 8 | 0 | 98.0 | 98.6 | 97.2 | 93.4 | 96.8 |
| 20 | 0 | 95.4 | 96.2 | 95.2 | 87.6 | 93.6 |

- H=8→20에서 drop이 K=4: **-1.7**, K=0: **-3.2**. LIBERO-Long의 경우 K=0은 93.4→87.6, K=4는 93.4→93.4로 유지 → **long-horizon open-loop drift를 latent refinement가 강하게 완화**.
- chunk 길이를 H=20로 늘리면 effective throughput 27.5 Hz → 68.5 Hz (Appendix C.2).

---

## 5. 기여 요약 (저자 주장)

1. VLM **latent space 내부에서 deliberation을 수행**하는 VLA 프레임워크 + parallel continuous action chunk decoding 양립.
2. **plan-conditioned world query**로 closed-loop future feedback. pixel reconstruction 없음, action-space rollout 없음.
3. **anchored residual update scheme**: anchor query로 WM condition drift 차단, scheduled write-back으로 plan drift 차단.
4. **CRG-PRL**: imagined future 기반 group-relative process-reward RL로 refinement trajectory를 critic-free하게 최적화.
5. LIBERO 98.7% — flow-matching/diffusion VLA들도 넘어선 새로운 regression-head 기반 SOTA.

---

## 6. 강점 및 차별점

- **Parallel decoding + iterative deliberation 양립**: flow/diffusion이 iterative inference cost를 행동 디코딩 쪽에 쓰는 반면, PearlVLA는 그 compute를 plan 자체에 투자.
- **frozen WM을 정책 forward pass 내부**로 끌어들이는 첫 closed-loop 형태. VPP/TriVLA가 WM을 context로 "소비"하는 것과 대비.
- **Critic-free RL**: same-state branching + group-relative advantage가 state difficulty와 round-별 reward drift를 모두 상쇄.
- 두 control variable(β_k, w_k)을 명확히 분리 → 학습 안정성과 해석성 확보.
- inference latency는 deterministic mean을 쓰므로 supervised model과 동일.

---

## 7. 한계 및 약점

- **고정 refinement depth K=4**: 쉬운 reactive step에도 4회 돌리고, 매우 ambiguous한 step에도 4회로 멈춤. 저자도 conclusion에서 adaptive depth를 future work로 명시.
- **WM의 품질에 결정적으로 의존**: UWM이 frozen이라 prediction 품질이 정책 상한을 결정. action-free video data 다양성·규모가 부족하면 latent feedback이 misleading.
- **단일 벤치마크 중심 평가**: 본 실험이 LIBERO + LIBERO-Plus + RoboCasa(appendix) 위주. CALVIN/SimplerEnv/실로봇 등 다른 표준 벤치마크 부재.
- **real-robot 결과 부재**: 시뮬레이션만 검증. sim-to-real transfer 미평가.
- **CRG-PRL stage compute cost**: M=8 branch × per-round reward evaluation × 매 라운드 → 학습 시간 측면의 overhead 정량적 보고 부족.
- **저자 정보의 짧은 affiliation**: 1저자 두 명, 코드 미공개(논문 시점). 재현 어려움.
- **flow-matching VLA들과 진정한 controlled 비교 불가**: 저자 표 자체에서도 backbone, data mixture, action 표현이 달라 "context로만 읽으라"고 인정.

---

## 8. 베이스라인과의 위치

- **OpenVLA-OFT (직계 비교)**: 동일 OpenVLA-7B backbone, 동일 parallel chunk decoding. 차이는 plan refinement loop의 유무. 97.1 → 98.5(+1.4) 차이가 latent refinement의 순효과.
- **CoT-VLA / Hume / ECoT (명시적 reasoning)**: 텍스트/픽셀/action search로 deliberation 비용 큼. PearlVLA는 latent 안에서 끝내므로 추론 latency가 직접 디코딩 수준 유지.
- **VPP / TriVLA / MIND (WM consumer)**: WM을 정책 입력으로 쓰지만 closed-loop write-back 없음. PearlVLA의 핵심 차별점.
- **WMPO / VLA-RFT (RL with WM)**: WM을 RL 신호 생성기로 사용. PearlVLA의 CRG-PRL도 비슷한 계보지만 **refinement 내부의 process reward**를 같은 state 안에서 비교한다는 점이 specific.
- **Diffusion-of-Thought류와의 구분**: diffusion 추론은 고정 conditioning에서 iterative denoising. PearlVLA는 라운드마다 **conditioning(world query)이 plan 갱신으로 바뀜** → "feedback refinement"가 본질이라 diffusion-style reasoning과 다른 카테고리.

---

## 9. 실험 디테일 점검

- backbone fusion: SigLIP + DINOv2 시각 인코더 + 3-layer GELU MLP projector + LLaMA-2 7B.
- 12 meta-query → 4 visual anchor + 8 latent plan.
- proprio: transformer-based encoder.
- noise schedule: `t*=50`, `sqrt(1-α_bar_50) ≈ 0.089`.
- write-back: `timestep_list=[100,80,60,40]`, `w_k=[0.169, 0.138, 0.106, 0.072]`.
- WM: UWM 300M DiT, 10-step DDIM rollout, observation-only path.
- RefineNet: 4-layer weight-shared transformer, gates와 modulation outputs zero-init.
- action head: AdaLN-conditioned query-transformer, learnable per-horizon queries, zero-init final projection, H=8.
- loss: action-MSE warmup + DCT auxiliary loss + input-side teacher alignment loss.
- CRG-PRL: PPO clipping, KL to frozen SFT policy, edit-magnitude penalty, final-action BC anchor on base path. RefineNet lr=1e-5, FutureEncoder lr=5e-6.

---

## 10. 잠재적 영향과 후속 연구 방향

- **"VLM latent space에서 reasoning"이라는 LLM 쪽 연구(Coconut, Looped Transformer, CoT2)를 embodied control로 옮긴** 첫 본격 시도. 이 연결고리는 향후 VLA 연구의 표준 reference가 될 가능성.
- **frozen latent WM의 forward-pass 활용**이라는 패턴은 다른 분야(예: planning agent, multimodal reasoning)로 일반화 여지가 큼.
- **adaptive refinement depth** (uncertainty- 또는 future-consistency-guided): 짧은 reactive step에선 K=0~1, 모호한 long-horizon step에선 K≥4로 동적 할당.
- **larger / more diverse WM**: action-free video 데이터를 대규모로 늘리면 latent feedback이 풍부해져 long-tail 상황에도 확장 가능.
- **Process reward beyond Robometer-4B**: domain-specific reward model로 sim-to-real transfer 강건성 향상.

---

## 11. 핵심 인사이트 정리

1. **Deliberation의 장소 선택이 중요**: 텍스트도, 픽셀도, action space도 아닌 **latent space**가 효율과 표현력의 sweet spot.
2. **Closed-loop > one-way feed**: 미래 정보를 정책에 "공급"하는 것보다 **plan에 써넣고 다시 미래를 갱신**하는 것이 본질적으로 더 강력.
3. **Anchor + residual write-back**의 분리가 학습 안정성의 핵심. 둘 중 하나만으로는 drift 폭증.
4. **same-state branching**은 critic-free RL이 VLA 같은 sparse/imagined reward 환경에서 작동하게 만드는 실용적 기법.
5. **regression head로도 SOTA 가능**: flow/diffusion head가 필수가 아니라는 반증 케이스. saved compute를 어디에 투자할지의 설계 문제.

---

## 12. 예상 Q&A

| # | 질문 | 답변 |
|---|---|---|
| 1 | K=0 변형이 왜 OpenVLA-OFT(97.1)와 다른가(96.8)? | K=0은 PearlVLA의 구조(meta-query split, noise, action head 등)는 유지하고 RefineNet 업데이트만 끈 변형. OFT의 정확한 head/training과 일치하지 않으므로 직접 비교 대상이 아님. 저자도 본문에서 명시. |
| 2 | UWM이 frozen인데 WM 자체의 한계가 정책 성능을 막지 않는가? | 맞다. β_k anchor와 input-side teacher alignment로 query를 WM condition manifold 안에 묶어 misuse를 방지하지만, WM이 못 보는 분포에 대한 feedback은 여전히 제한적. LIBERO-90으로 post-trained된 UWM이라 LIBERO-Plus에서도 효과가 유지된 점이 다행. |
| 3 | β_k와 w_k 둘 다 학습되는가 아니면 schedule인가? | β_k는 **per-round 학습가능 스칼라 (composer)**. w_k는 noise schedule에서 결정된 **고정 schedule** (sqrt(1-α_bar_{t_k}), t_k=[100,80,60,40]). 즉 conditioning side는 학습, write-back side는 schedule. |
| 4 | 왜 noise를 주입하면서 diffusion objective는 안 쓰는가? | noise는 (a) 시작점 근방의 local robustness와 (b) CRG-PRL의 stochasticity 용도. diffusion noise-prediction loss로 z̃_0를 복원하려는 게 아님. plan 자체는 deterministic refinement target. |
| 5 | input-side alignment loss는 왜 output-side가 아니라 input-side인가? | output-side는 frozen rollout 전 경로를 미분해 통과해야 하고 gradient noisy. input-side는 `VLM latent → projector → composer` 경로만 직접 묶어 stable. 본문 3.5에서 명시. |
| 6 | M=8 branch가 너무 적지 않나? | DeepSeekMath/GSPO 등에서 그룹 기반 RL의 표준이 M=4~16. M=8은 분산 감소와 compute 사이의 standard trade-off. ablation은 본 논문에 없음. |
| 7 | CRG-PRL이 SFT 대비 +0.2 (98.5→98.7)에 불과한데 의미 있나? | LIBERO ceiling 근처라 절대 값은 작지만, **RefineNet trajectory가 SFT 후에도 최적화 가능한 객체임을 증명**한 것이 paper-level 기여. LIBERO-Plus나 long-horizon에서 효과가 더 클 여지. |
| 8 | inference latency는 OpenVLA-OFT 대비 얼마나 느려지나? | K=4 라운드마다 frozen UWM 1회(10-step DDIM)+RefineNet 1회. 정량 latency 수치는 본문 미보고, 대신 chunk H=20 + K=4가 27.5→68.5 Hz throughput으로 보고됨. |
| 9 | real robot 결과가 없는데 왜 안 했나? | 본 논문은 LIBERO/LIBERO-Plus/RoboCasa(시뮬레이션) 위주. real-robot transfer는 future work. WM이 LIBERO-90 video로 post-trained되었기 때문에 real-world 분포로 직접 확장은 추가 video pretraining이 필요. |
| 10 | regression head로 SOTA가 flow/diffusion head를 넘는다는 일반화 가능한 교훈은? | "head 종류"가 아니라 "saved compute를 어디 쓰는가"가 본질. parallel regression head + iterative latent refinement가 iterative flow-matching 디코딩보다 효율적인 compute 배분일 수 있다는 가설을 지지. |
| 11 | refinement loop가 long-horizon에서 더 효과적인 이유? | H=20 ablation이 직접적 증거. open-loop drift가 누적되는 long chunk에서 latent refinement가 chunk-level action coherence를 유지 → LIBERO-Long에서 K=0의 87.6 vs K=4의 93.4. |
| 12 | CRG-PRL의 guardrail (KL/edit penalty/BC anchor) 중 가장 중요한 것? | 본 논문에서 ablation은 명시되지 않음. 일반적으로 KL term이 supervised operator로부터의 drift 통제에 가장 critical. edit penalty는 write-back magnitude blow-up 방지, BC anchor는 final action 품질 보존. |

<!-- VERIFIED: pdf -->
