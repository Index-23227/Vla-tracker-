# LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies

> **한 줄 요약**: 픽셀 공간에서 미래 비디오를 생성하는 대신, **DINOv3 latent space의 latent action model (LAM) 디코더를 그대로 Latent World Model (LaWM)로 재활용**하여 단일 forward pass로 "latent visual subgoal" 한 장만 예측하고 이를 Alternate-DiT action expert에 조건으로 주입. 2.3B 파라미터로 **LIBERO 98.6% / RoboTwin 2.0 clean 92.64% / 실세계 평균 90.0%**를 달성하면서 **action-chunk당 187ms (LingBot-VA 대비 24x 저지연)** 달성.

**Score: 8.0 / 10**

---

## 1. 배경 및 동기

- VLA(OpenVLA, π0/π0.5, GR00T-N1.6 등)는 vision-language pretraining 덕에 강한 semantic grounding을 갖지만, **행동이 장면을 어떻게 바꾸는지에 대한 명시적 예측(foresight)이 없음**.
- World-Action Model(WAM)은 미래 관측을 예측해서 정책에 조건으로 주입 → 시간 동역학(dynamics) 정보 제공. 그러나 두 가지 비용 문제:
  1. **픽셀 합성에 모델 용량 낭비** (5B WAN backbone 등).
  2. **반복적(iterative) 생성으로 인한 지연** — LingBot-VA는 정책 한 번 추론에 **4,482 ms** 소요 (vs π0.5 220 ms).
- 핵심 가설: chunk-level 제어에 필요한 미래 정보는 **"장면 변화의 컴팩트한 latent 기술"** 이면 충분하고, 픽셀까지 복원할 필요 없음.

---

## 2. 방법론 심층 분석

### 2.1 Problem Formulation

표준 WAM은 다음과 같이 인수분해됨:

$$p(a_{1:T}, o_T \mid o, l) = \underbrace{p(o_T \mid o, l)}_{\text{Future Prediction}} \cdot \underbrace{p(a_{1:T} \mid o, o_T)}_{\text{IDM}}$$

LaWAM은 픽셀 $o_T$ 대신 **frozen visual encoder** $f_\psi$ (DINOv3)의 feature $u_T = f_\psi(o_T)$ 만 다룸:

$$p(a_{1:T}, \hat{u}_T, \hat{z} \mid o, l) = \underbrace{p_\theta(\hat{z} \mid o, l)}_{\text{Policy Prior}} \cdot \underbrace{p_\omega(\hat{u}_T \mid u, \hat{z})}_{\text{LaWM}} \cdot \underbrace{p_\eta(a_{1:T} \mid o, l, u, \hat{u}_T)}_{\text{Action Expert}}$$

### 2.2 Stage 1: Latent World Model (LaWM)

- **입력**: 현재 관측 $o$, $\tau$ 시간 후 horizon 관측 $o_T$. DINOv3로 encode하여 $(u, u_T)$.
- **Inverse Dynamics**: posterior encoder $q_\phi(z \mid u, u_T)$가 latent action $z$ 추론.
- **Forward decoder = LaWM**: $\tilde{u}_T = \text{LaWM}_\omega(u, z)$ — 230M 파라미터.
- **Auxiliary embodied head**: $g(s, z) \to s_T$ (end-effector state 예측). latent action이 **시각 외형 변화가 아니라 실제 신체 운동**을 인코딩하도록 유도.
- **Loss**:

$$\mathcal{L}_{\text{LAM}} = \underbrace{\|\tilde{u}_T - u_T\|_2^2}_{\mathcal{L}_{\text{wm}}} + \underbrace{\|g(s,z) - s_T\|_2^2}_{\mathcal{L}_{\text{aux}}} + \beta \, D_{\text{KL}}(q_\phi(z \mid u, u_T) \,\|\, \mathcal{N}(0,I))$$

- 학습 후 auxiliary head 폐기, **decoder만 LaWM으로 보존**. (선행 latent-action 기반 VLA들은 decoder를 학습 보조용으로 버리는데 — LaWAM의 핵심 차별점은 decoder를 **정책 facing 모듈로 재활용**한다는 것.)
- KL 정규화 덕에 stage-2의 policy prior가 같은 분포에서 $z$를 샘플할 수 있음.

### 2.3 Stage 2: Latent World Action Model (LaWAM)

- 배포 시 미래 $u_T$를 모르므로 IDM encoder 사용 불가 → **policy prior $p_\theta(\hat{z} \mid o, l)$** 학습.
- 추론 흐름: $o, l \to \hat{z} \to \hat{u}_T = \text{LaWM}_\omega(u, \hat{z}) \to a_{1:T}$.
- **Action Expert: Alternate-DiT** (GR00T-N1.6에서 차용).
  - 한 stream: VLM backbone (Qwen3-VL-2B의 **first 16 layers**)에서 오는 semantic context.
  - 다른 stream: $(u, \hat{u}_T)$ latent dynamics context.
  - Alternate DiT Block 안에서 **Self Attn → Inverse Dynamics Attn → Self Attn → Semantic Attn** 순으로 두 stream을 번갈아 가며 융합.
- **Knowledge Insulation (KI)**: action-expert gradient가 pretrained LaWM 가중치로 흐르지 않도록 차단. ablation에서 결정적.
- **Loss**:

$$\mathcal{L}_{\text{LaWAM}} = \lambda_{\text{distill}}\, \mathbb{E}[\|\hat{z} - z\|_2^2] + \lambda_{\text{wm}} \|\hat{u}_T - u_T\|_2^2 + \mathcal{L}_{\text{act}}$$

여기서 $\mathcal{L}_{\text{act}}$는 conditional flow matching loss.

### 2.4 학습 데이터 및 추론 디테일

- LaWM 사전학습: **로봇 비디오 3,000h + 1인칭 인간 비디오 1,500h** (cross-embodiment dynamics prior). 정책 stage에서는 인간 비디오 직접 사용 안 함.
- Mixed-frequency 로봇 데이터 → **physical-time encoding** (각 데이터셋의 native control frequency 유지).
- 추론: A100, 10 denoising step, **187 ms / action chunk**. LaWM은 **single forward pass** (반복 디퓨전 없음).

---

## 3. 실험 결과 심층 분석

### 3.1 LIBERO (Table 1, 50 trials/task)

| Method | Size | Latency | Long | Goal | Object | Spatial | **Avg** |
|---|---|---|---|---|---|---|---|
| OpenVLA-OFT | 7B | — | 94.5 | 97.9 | 98.4 | 97.6 | 97.1 |
| π0.5 | 3.5B | 220 | 92.4 | 98.0 | 98.2 | 98.8 | 96.9 |
| GR00T-N1.6 | 3.3B | 259 | 94.4 | 97.5 | 98.5 | 97.7 | 97.0 |
| UniVLA | 7B | — | 92.0 | 95.6 | 96.8 | 96.5 | 95.2 |
| VLA-JEPA | 3B | — | 95.8 | 97.2 | 99.6 | 96.2 | 97.2 |
| Motus (pixel WAM) | 8B | 3,231 | 97.6 | 96.6 | 99.8 | 96.8 | 97.7 |
| Cosmos-Policy | 2.1B | 1,413 | 97.6 | 98.2 | 100.0 | 98.1 | 98.5 |
| LingBot-VA | 5.5B | 4,482 | 98.5 | 97.2 | 99.6 | 98.5 | 98.5 |
| Fast-WAM | 6B | 486 | 95.2 | 97.0 | 100.0 | 98.2 | 97.6 |
| **LaWAM** | **2.3B** | **187** | **97.0** | **98.4** | 99.6 | **99.4** | **98.6** |

- 가장 작은 모델로 **최고 평균(98.6%)** + 가장 낮은 latency. Spatial 99.4는 single best.

### 3.2 RoboTwin 2.0 (Table 2, 50 bimanual tasks, 100 trials/task)

| Method | Clean | Random |
|---|---|---|
| Fast-WAM | 91.98 | 90.52 |
| GigaWorld-Policy | 86.36 | 85.04 |
| LingBot-VA | 91.50 | 90.92 |
| π0.5 | 82.74 | 76.76 |
| Motus | 88.66 | 87.02 |
| **LaWAM** | **92.64** | 89.80 |

- Clean에서 SOTA, randomized에서도 LingBot-VA(90.92)에 1.1%p 차로 근접하면서 24x 빠름.

### 3.3 실세계 (Table 3, 30 trials/task, 2 platforms)

| Method | Pick&Place | Drawer | Towel Fold | **Avg** |
|---|---|---|---|---|
| π0.5 | 86.7 | 80.0 | 83.3 | 83.3 |
| GR00T-N1.6 | 83.3 | 76.7 | 46.7 | 68.9 |
| Fast-WAM | 56.7 | 63.3 | 70.0 | 63.3 |
| LingBot-VA | 76.7 | 83.3 | **0.0** | 53.3 |
| **LaWAM** | **93.3** | **86.7** | **90.0** | **90.0** |

- **3개 과제 모두 1위**. 특히 towel folding에서 LingBot-VA가 0% (높은 latency 때문에 천이 움직이는 동안 멈춤) — **저지연이 동적 환경에서 결정적**임을 시연.

### 3.4 LaWM Dynamics 분석 (Fig. 5)

- 한 source 비디오에서 추출한 **동일 latent action trajectory**를 4개의 unseen 환경/embodiment에 적용 → 각 환경별로 그럴듯한 latent rollout이 생성됨 (panels (d), (e)는 pi.website의 unseen 스크린샷).
- → **latent action = embodiment-agnostic**, **LaWM = embodiment-specific grounding**.
- "latent action만 정책 인터페이스로 쓰는 것이 왜 부족한가"를 시각적으로 증명: latent action은 LaWM이 현재 embodiment에 grounding한 후에야 유용.

### 3.5 Component Ablations (Fig. 6, LIBERO)

| Variant | 효과 |
|---|---|
| Full LaWAM | baseline |
| w/o pretrain | LaWM 사전학습 제거 — 중간 정도 하락 |
| w/o distill | latent-action distillation 제거 — 큰 하락 |
| w/o KI & distill | KI까지 제거 — 추가 하락 |
| **w/o WM** | LaWM 자체 제거 — **가장 큰 하락 (특히 LIBERO-Long)** |

- 결론: **LaWM의 subgoal conditioning이 성능의 주된 원천**, distillation은 policy prior가 LaWM과 호환되는 latent action을 안정적으로 만들기 위해 필요, KI는 LaWM 보호용.

---

## 4. 강점

1. **Pixel-free WAM**: LingBot-VA, Cosmos-Policy, Motus 대비 1/10~1/20 latency를 유지하면서 동등 이상 성능.
2. **LAM decoder 재활용**: 기존 latent-action VLA(LAPA, UniVLA, VLA-JEPA)가 버리던 decoder를 정책 facing 모듈로 살려냈음 — Garrido et al.이 동시기에 한 관찰을 실제 정책에 통합한 첫 시스템 중 하나.
3. **2.3B로 SOTA**: 7B OpenVLA-OFT, 8B Motus, 5.5B LingBot-VA를 LIBERO/RoboTwin에서 동시에 능가.
4. **실세계 강건성**: towel folding 90.0% (LingBot-VA 0%) — 동적 manipulation에서 저지연의 실용 가치 입증.
5. **Cross-embodiment latent action 시각화**: latent action의 의미를 정성적으로 입증하는 정직한 Fig. 5.
6. **체계적 ablation**: KI / distill / pretrain / WM 4가지 축을 명확히 분해.

---

## 5. 한계 및 미해결 문제

1. **카메라 모션에 약함** (저자 인정): egocentric 영상에서 abrupt shake나 큰 시점 변화가 dominant하면 LaWM이 coherent latent action space를 학습하지 못함 → **humanoid / mobile robot에 그대로 적용 어려움**.
2. **Deformable object 데이터 부족**: towel folding 같은 미세 변형은 training mix에서 희소 → LaWM이 신뢰성 있게 모델링하기 어려움 (실세계 towel 90%지만 추가 일반화 보장 없음).
3. **DINOv3 의존**: frozen visual encoder의 표현력에 모든 dynamics modeling이 종속. DINOv3가 잡지 못하는 미세 텍스처/접촉 정보는 LaWM이 다룰 수 없음.
4. **Open-source 미공개** (현 시점): 코드/체크포인트 미배포 → 독립 재현 불가.
5. **Distill loss 의존**: w/o distill에서 큰 폭 하락 → policy prior가 LAM posterior에 강하게 결합되어야 함. 새 embodiment로 일반화할 때 distill 단계 재수행이 필요할 가능성.
6. **Single subgoal per chunk**: 1 chunk = 1 latent subgoal. Long-horizon task에서 chunk 경계를 넘는 sub-task switching은 명시적 메커니즘 없음.

---

## 6. 동시기 연구와의 비교

| 시스템 | Latent space | Decoder 재활용? | Test-time iter? | Latency |
|---|---|---|---|---|
| **LaWAM** | DINOv3 frozen | **Yes (LaWM)** | No (1-step) | **187 ms** |
| Fast-WAM | Pixel (Wan 5B) | — | No (training-only video) | 190 ms |
| LingBot-VA | Pixel | — | Yes | 4,482 ms |
| Motus | Pixel + optical flow | — | Yes | 3,231 ms |
| Cosmos-Policy | Pixel (Cosmos) | — | Yes | 1,413 ms |
| LDA-1B (concurrent) | DINO | jointly denoise | Yes (diffusion) | — |
| LAPA / UniVLA / VLA-JEPA | latent action only | No (decoder discarded) | N/A | — |
| π0.7 | Pixel subgoal (separate) | — | Yes | — |

- LaWAM의 unique 위치: **DINO latent + decoder retained + non-iterative**. LDA-1B와는 latent space는 공유하나 LDA는 future state + action을 **공동 디퓨전**, LaWAM은 **single forward pass + flow-matching action**.

---

## 7. 재현 / 검증을 위해 필요한 것

- **공개되지 않은 항목**: (i) LaWM 230M 체크포인트, (ii) policy prior 가중치, (iii) auxiliary state predictor $g$, (iv) RoboTwin clean/random split, (v) Knowledge Insulation 구현 세부.
- 논문에서 검증 가능: LIBERO/RoboTwin 2.0 evaluation protocol은 표준이므로 동일 모델 사이즈(2.3B)로 재구현 시 비교 기준점은 존재.

---

## 8. 향후 연구 방향

1. **Moving camera robust LaWM**: ego-motion factorization (예: relative pose conditioning)을 latent action 분해에 추가.
2. **Multi-subgoal per chunk**: long-horizon task의 subtask boundary 자동 탐지 → 다중 subgoal 시퀀스.
3. **Deformable dynamics 강화**: cloth/fluid 전용 sim/real co-training으로 LaWM 데이터 mix 보강.
4. **Latent space scaling**: DINOv3 → DINOv4 / SigLIP-2 등으로 frozen encoder 교체했을 때의 scaling law.
5. **Online LaWM 업데이트**: 배포 후 실패 trajectory로 LaWM을 in-context fine-tune (KI를 일시 해제).
6. **Latent action → high-level planning**: 추출된 latent action sequence를 LLM/VLM에 expose하여 자연어 reasoning 가능성.

---

## 9. 실용적 함의

- **Edge/실시간 manipulation**에 즉시 가치: 187 ms는 약 5.3 Hz chunk-level 제어 (chunk 내 action은 high frequency interpolation 가능).
- **Pixel WAM 라인업의 효율성 압박**: Cosmos-Policy/LingBot-VA 같은 대형 video tower 기반 WAM은 이제 "동일 성능 + 24x latency" 부담을 정당화해야 함.
- **Latent action 학파에 새 표준**: LAPA/UniVLA 이후로 "decoder를 버리지 말라"는 명제를 baseline으로 만듦.

---

## 10. 총평 및 점수 (8.0 / 10)

| 항목 | 평가 |
|---|---|
| Novelty | ★★★★☆ — "decoder 재활용"은 동시기 관찰이지만 정책에 통합한 첫 본격 시스템 중 하나 |
| Technical depth | ★★★★☆ — two-stage 학습/KI/Alternate-DiT가 깔끔하게 정의됨 |
| Experimental rigor | ★★★★★ — LIBERO + RoboTwin 2.0 + 실세계 3과제 + dynamics 시각화 + 4축 ablation |
| Practical impact | ★★★★★ — 24x latency 감소, 2.3B로 SOTA, 실세계 90% |
| Writing | ★★★★☆ — 수식과 figure가 잘 정렬됨, 일부 appendix 참조 빈번 |
| Reproducibility | ★★☆☆☆ — 코드 비공개 |

**최종 8.0**: 픽셀 합성을 latent subgoal로 대체하는 명료한 thesis와 강력한 실험. 코드 공개와 mobile/humanoid 확장이 다음 step.

---

## 11. 핵심 수식 요약

1. **WAM 표준 인수분해**:
   $p(a_{1:T}, o_T \mid o, l) = p(o_T \mid o, l) \cdot p(a_{1:T} \mid o, o_T)$

2. **LaWAM 인수분해** (latent space):
   $p(a_{1:T}, \hat{u}_T, \hat{z} \mid o, l) = p_\theta(\hat{z} \mid o, l) \cdot p_\omega(\hat{u}_T \mid u, \hat{z}) \cdot p_\eta(a_{1:T} \mid o, l, u, \hat{u}_T)$

3. **Stage 1 loss**:
   $\mathcal{L}_{\text{LAM}} = \|\tilde{u}_T - u_T\|_2^2 + \|g(s,z)-s_T\|_2^2 + \beta\, D_{\text{KL}}(q_\phi \,\|\, \mathcal{N}(0,I))$

4. **Stage 2 loss**:
   $\mathcal{L}_{\text{LaWAM}} = \lambda_{\text{distill}} \|\hat{z}-z\|_2^2 + \lambda_{\text{wm}} \|\hat{u}_T - u_T\|_2^2 + \mathcal{L}_{\text{act}}$

5. **추론 흐름 (1 forward pass for subgoal)**:
   $o, l \;\to\; \hat{z} = p_\theta(o,l) \;\to\; \hat{u}_T = \text{LaWM}_\omega(u, \hat{z}) \;\to\; a_{1:T} = p_\eta(o, l, u, \hat{u}_T)$

---

## 12. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | LDA-1B도 DINO latent에서 dynamics를 모델링한다. 본질적 차이는? | LDA-1B는 미래 visual state + action chunk를 **공동 디퓨전**(반복적), LaWAM은 LaWM 1-step + flow-matching action. LaWAM이 non-iterative. |
| 2 | Fast-WAM은 test-time video를 끄고도 190ms로 비슷하다. LaWAM이 정말 우월한가? | (i) LaWAM은 test-time에 **명시적 future feature**를 expose (Fast-WAM은 expose 안 함), (ii) RoboTwin clean 92.64 vs 91.98, 실세계 90.0 vs 63.3 — 실세계 격차가 큼. |
| 3 | w/o WM ablation이 가장 크게 하락 — 그럼 latent action distillation은 사실상 LaWM 호환 토큰 학습용 보조? | 정확. distill loss는 policy prior를 LAM posterior 분포에 매칭시켜 LaWM이 의미 있는 subgoal을 디코딩하게 함. LaWM 없으면 distill만으론 부족. |
| 4 | DINOv3 동결인데 frozen encoder가 잡지 못하는 contact/force 정보는? | 인정되는 한계. auxiliary state predictor $g(s, z) \to s_T$가 end-effector 차원에서 일부 보완하지만 force/tactile은 미해결. |
| 5 | "단일 subgoal per chunk"가 long-horizon에 충분한가? LIBERO-Long 97.0이 정말 LaWM 덕? | w/o WM ablation에서 Long이 가장 크게 떨어짐 (Fig. 6) → Long task에서 LaWM 기여가 가장 큼. 다만 chunk 경계 넘는 subtask switching은 chunk-time 인코딩에 암묵적으로 의존. |
| 6 | KI 없이 학습하면 어떻게 되나? action gradient가 LaWM에 흘러서 dynamics가 망가지나? | w/o KI & distill 변형이 추가 하락 → KI 없으면 action loss가 LaWM 가중치를 task-specific 단축경로로 변형, dynamics prior 소실. |
| 7 | Cross-embodiment Fig. 5에서 (d), (e)는 pi.website 스크린샷 — 데이터 누설 위험? | LaWM 학습 데이터에 pi.website 포함 안 됨 (open-source 데이터셋만 사용한다고 명시). open-loop **시각화**이지 정량 평가 아님 — 의미 있는 sanity check. |
| 8 | 187 ms latency는 A100 기준. 소비자 GPU(예: 4090)에서는? | 논문은 미보고. LaWM이 230M으로 작아 대부분 cost는 2.3B 전체 forward → consumer GPU에서도 sub-second 가능성 있음. |
| 9 | 실세계 towel fold에서 LingBot-VA가 0%인 게 정말 latency 탓인가? | 저자 설명: 4.5s/inference 동안 천이 계속 움직여서 받아쓰는 frame이 stale → "delayed actions become mismatched". 정성적으로 plausible하나 ablation 부재. |
| 10 | 5B WAN 대비 230M LaWM이 95% 적다는 주장 — task 다양성이 늘면 230M으로 부족하지 않나? | 가능한 future bottleneck. RoboTwin 50 task, 실세계 3 task에서는 충분. Open X-Embodiment 전체로 확장 시 scaling 연구 필요. |

<!-- VERIFIED: pdf -->
