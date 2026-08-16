# JEPA-WAM: Learning Vision-Language-Action Policies with Joint-Embedding World Modeling

> **한 줄 요약**: 사전학습된 **V-JEPA 2.1 표현공간 안에서** latent world action model을 구성하고, **하나의 shared predictor**가 latent transition 예측과 continuous action 생성을 동시에 담당하게 만든 VLA. 절대적 미래 상태 대신 현재-미래를 **함께 인코딩한 joint target**을 patch 단위로 예측하여 시간적 관계를 학습한다. LIBERO 96.7%, LIBERO-Plus **79.2%** (robot-policy pretraining 없는 방법 중 최고), RoboTwin 2.0 Clean 79.9 / Random 36.9. 동일한 transition supervision을 사전학습 π₀.₅에 이식하면 LIBERO-Plus 84.5 → **86.3%**로 전체 1위.

---

## 1. 배경 및 동기

### 문제 정의
- VLA policy는 action prediction objective만으로 학습되므로 **state transition을 암묵적으로만** 모델링 → distribution shift에서 취약
- World Action Model(WAM)은 미래 상태를 명시적으로 모델링해 이를 보완하나, **video-generation 기반 WAM은 iterative future prediction 때문에 배포 비용이 큼**
- 그래서 등장한 것이 **latent WAM**: 미래 프레임을 생성하지 않고 표현공간에서 예측

### Latent WAM이 답해야 할 두 가지 질문
1. **무엇(what)을 예측 타깃으로 삼을 것인가?**
   - 기존: 사전학습된 video generator의 중간 feature 재활용 → generation을 위해 최적화된 feature라 state change를 명시적으로 표현하지 않음
   - 기존: 미래 관측을 소수의 latent token / subgoal로 압축 → **fine-grained spatial structure 손실**
2. **어떻게(how) predictive supervision을 action generation에 통합할 것인가?**
   - 예측된 미래 표현을 action module의 추가 context로 제공 → action module이 **중복된 미래 상태 정보에 노출**
   - 별도의 latent dynamics module / 별도 objective를 policy 옆에 둠 → action이 실제로 생성되는 표현에 **약하게만 영향**

📌 [Figure 2] — (a) WAMs, (b) Dedicated Latent-Dynamics VLAs, (c) VLAs with Dedicated Latent Dynamics, (d) Ours(shared predictor)의 패러다임 비교

---

## 2. 방법론 심층 분석

### 2.1 V-JEPA 표현공간과 현재 시각 표현
동결된 V-JEPA 2.1 인코더 `E_J`가 각 뷰를 독립적으로 처리하고, 결과 토큰을 **고정된 카메라 순서로 concat**:

$$Z_t = \text{Concat}_{v \in V} E_J(O_t^v) \in \mathbb{R}^{N_{vis} \times d_J}$$

- global pooling 없이 **dense patch-level** 표현 유지 (24×24 grid, 1024-d)
- 이 patch 조직과 카메라 순서가 이후 예측 타깃에서도 그대로 보존되는 것이 핵심 설계

### 2.2 Joint Current-Future Target — 이 논문의 심장
각 뷰에 대해 현재 `O_t^v`와 δ 스텝 뒤 `O_{t+δ}^v`를 **시간축으로 stack하여 함께 인코딩**:

$$Y_{t,t+\delta} = \text{Concat}_{v \in V}\ \text{sg}\Big[ E_J\big(\text{Stack}_{time}(O_t^v, O_{t+\delta}^v)\big) \Big]$$

- **왜 이게 작동하는가 (구조적 이유)**: V-JEPA 2.1의 video tokenizer는 **2 프레임을 하나의 temporal tubelet**으로 묶는다. 따라서 2-프레임 입력이 단일 이미지와 **동일한 spatial token grid(24×24)** 를 만들어내고, `Y_{t,t+δ}`와 `Z_t`가 카메라/공간 토큰 순서를 공유하게 된다. 이것이 patch-wise 정렬을 "설계상 공짜로" 얻는 지점이다.
- **future-only 타깃과의 차이**: `E_J(O_{t+δ}^v)`는 미래를 *고립적으로* 표현하지만, joint target은 두 시점 끝점을 모두 인코더에 노출시켜 **어느 영역이 안정적이고 어느 영역이 변하는지, 국소 객체·공간 관계가 어떻게 달라지는지**라는 *관계*를 표현한다.
- 유일한 미래를 재구성할 필요가 없다 → multimodality 문제 회피

> ❓ **예상 질문**: 미래를 정확히 못 맞히면 world model이라 할 수 있나?
> **답변**: 이 논문은 의도적으로 "unique future 재구성"을 포기하고 **task-shared visual temporal structure**를 학습한다. 저자들도 Limitations에서 이것이 language에 largely independent한 supervision임을 인정한다.

### 2.3 Shared Predictor (Qwen2.5-0.5B)
$$(Q_t^{wm},\ C_t) = F_\theta\big(P_{vis}(Z_t),\ \ell,\ P_{act}\big)$$

- `Q_t^{wm}`: **visual token 위치의 hidden state** → joint target 예측용. `Z_t`의 카메라/공간 순서를 보존하므로 patch-level correspondence 성립
- `C_t`: **64개 dedicated action placeholder**의 hidden state → action expert 조건화용. 앞선 visual + task context를 집약
- 한 번의 forward pass에서 둘 다 산출 → transition supervision이 **action이 읽어가는 바로 그 backbone**을 직접 갱신
- Visual projector `P_vis`: 1024 → 896 → 896 (GELU 2-layer)
- Prediction head `G_φ`: token-wise MLP 896 → 2048 → 1024 (GELU)

### 2.4 손실 함수
Transition loss — patch 단위 cosine distance:

$$\mathcal{L}_{wm} = \frac{1}{B N_{vis}}\sum_{b}\sum_{n}\Big(1 - \cos\big(\hat{Y}^{(b)}_{t,t+\delta,n},\ Y^{(b)}_{t,t+\delta,n}\big)\Big)$$

Action loss — conditional flow matching (velocity prediction):

$$\mathcal{L}_{act} = \mathbb{E}_{\epsilon,\tau}\Big[\big\|A_\psi(a_\tau, \tau, s_t, C_t) - (a - \epsilon)\big\|_2^2\Big],\quad a_\tau = (1-\tau)\epsilon + \tau a$$

전체: $\mathcal{L} = \mathcal{L}_{act} + \lambda_{wm}\mathcal{L}_{wm}$, **λ_wm = 0.5**. flow time τ는 Beta(α=1.5, β=1.0) 스케줄.
RoboTwin 2.0에서는 velocity 대신 **x-prediction**(clean trajectory 직접 예측)을 사용 — 긴 bimanual action chunk에서 더 안정적이며, JiT(Li & He 2025)의 주장과 궤를 같이한다.

### 2.5 사전학습 VLA로의 이식 (π₀.₅ + JEPA Obj.)
- VLM prefix에 **64개 learnable future token** 추가 → hidden states `R_t ∈ R^{64×2048}`
- 8×8 coarse grid로 reshape → LayerNorm + MLP(2048→2048→1408) → **bilinear upsample to 24×24**
- 동결된 V-JEPA 2.1 **ViT-G** joint target과 patch-wise 정렬
- λ_wm은 첫 1K step 동안 0.1까지 linear warm-up
- **결정적 설계**: future token은 원래 image/language prefix에 attend할 수 있지만, **action token은 이 새 query들에 attend하지 못하도록 마스킹** → 원래 action pathway를 전혀 건드리지 않고 backbone만 shaping. 추론 시 예측된 미래 표현이 action expert에 명시적으로 주어지지 않는다.

### 2.6 배포
현재 관측 + 언어 지시만 필요. **target encoder, joint target 인코딩, transition prediction head 모두 제거**. Gaussian noise에서 시작해 **4-step Euler integration**으로 action chunk 생성.

---

## 3. 데이터 전략

| 환경 | 카메라 | Action | H | δ |
|------|--------|--------|---|---|
| LIBERO / LIBERO-Plus | primary + wrist (2뷰) | 7-dim | 8 | 31 |
| RoboTwin 2.0 | external + wrist ×2 (3뷰) | 14-dim bimanual | 50 | 50 |
| Real-world (AgileX CobotMagic) | global + wrist ×2 | 14-dim bimanual | — | — |

- 궤적 끝 근처에서 미래 관측은 **마지막 가용 프레임으로 clipping**
- Vision-language 초기화: LLaVA v1.5로 Prismatic 스타일 single-stage finetune (2 epoch, lr 2e-5, wd 0.1, batch 128, BF16) — 별도 projector-alignment 단계 없음
- Policy 학습: V-JEPA / visual projector / base Qwen 동결, **LoRA(rank 32, scale 64, dropout 0.1) + prediction head + action expert만** 최적화
- 최적화: AdamW, peak lr 2e-4 → cosine decay to 1e-5, 3% warmup, weight decay 0, grad clip 1.0, BF16, FSDP 8 GPU, global batch 128, **60K steps**
- Real-world: task당 **100 demonstration**

---

## 4. 실험 결과

### 4.1 LIBERO (in-distribution) — Table 1

| Method | Params(B) | Spatial | Object | Goal | Long | **Avg** |
|--------|-----------|---------|--------|------|------|---------|
| Diffusion Policy | – | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| ResVLA | 2 | 96.0 | 100.0 | 97.4 | 92.8 | 96.6 |
| Fast-WAM | 5 | 98.2 | 100.0 | 97.0 | 95.2 | **97.6** |
| **JEPA-WAM** | **0.5** | 95.6 | 99.4 | 97.2 | 94.6 | **96.7** |
| *— robot-policy pretraining 사용 —* | | | | | | |
| π₀.₅ | 3 | 98.6 | 98.2 | 98.4 | 92.4 | 96.9 |
| VLA-JEPA | 2 | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 |
| Motus | 5 | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| **π₀.₅ + JEPA Obj.** | 3 | 99.0 | 98.0 | 97.6 | 96.4 | **97.8** |

**핵심 관찰**: 0.5B backbone으로 5B Fast-WAM에 0.9%p 차이. ID 성능은 이미 포화 구간이라 변별력이 낮다 — 진짜 이야기는 아래.

### 4.2 LIBERO-Plus (OOD, fine-tuning 없음) — Table 2

| Method | Params | PT | Camera | Robot | Language | Light | Back. | Noise | Layout | **Avg** |
|--------|--------|----|--------|-------|----------|-------|-------|-------|--------|---------|
| VLA-Adapter | 0.5 | ✗ | 36.2 | 37.9 | 74.6 | 70.6 | 76.1 | 58.0 | 69.7 | 60.4 |
| RoVLA | 2 | ✗ | 58.4 | 36.3 | 92.9 | 95.6 | 95.0 | 80.9 | 73.0 | 76.0 |
| ResVLA | 2 | ✗ | 49.8 | 59.9 | 88.5 | 90.5 | 94.9 | 76.8 | 79.0 | 77.1 |
| **JEPA-WAM** | **0.5** | ✗ | **79.2** | 59.2 | 68.2 | 93.3 | 94.6 | 83.6 | 76.1 | **79.2** |
| VLA-JEPA | 2 | ✓ | 63.3 | 67.1 | 85.4 | 95.6 | 93.6 | 66.3 | 85.1 | 79.5 |
| PokeVLA | 0.5 | ✓ | 84.7 | 46.1 | 84.8 | 94.6 | 82.6 | 89.8 | 77.2 | 80.0 |
| ABot-M0 | 4 | ✓ | 60.4 | 67.9 | 86.4 | 96.2 | 91.6 | 86.4 | 82.6 | 81.6 |
| Cosmos-Policy | 2 | ✓ | 75.8 | 63.3 | 81.7 | 96.5 | 88.9 | 92.7 | 82.2 | 83.0 |
| π₀.₅ | 3 | ✓ | 69.4 | 75.3 | 82.6 | 96.7 | 96.8 | 84.3 | 86.2 | 84.5 |
| Being-H0.7 | 3 | ✓ | 82.0 | 59.0 | 82.8 | 97.8 | 90.0 | 93.5 | 88.5 | 84.8 |
| **π₀.₅ + JEPA Obj.** | 3 | ✓ | 66.0 | 82.0 | 86.5 | 96.8 | 96.0 | 88.3 | 88.3 | **86.3** |

**핵심 관찰**:
- JEPA-WAM은 **0.5B로 79.2%** — pretraining 없는 그룹 1위이며, 2B VLA-JEPA(79.5, PT 사용)와 사실상 동급
- **Camera 79.2**는 pretraining 없는 방법 중 압도적(차순위 58.4) — V-JEPA 표현공간의 시점 강건성이 직접적으로 드러나는 지점
- **약점은 Language 68.2** — 표 전체에서 최하위권. joint target이 language-independent한 visual temporal structure만 학습한다는 방법론적 특성이 그대로 나타난다 (저자들도 Limitations에서 인정)
- π₀.₅ 이식판은 Robot 75.3→82.0, Layout 86.2→88.3로 개선되며 전체 1위

### 4.3 RoboTwin 2.0 — Table 3 (AVG는 20 task 전체)

| Method | PT | Clean AVG | Random AVG |
|--------|----|-----------|------------|
| DP | ✗ | 48.0 | 1.6 |
| ACT | ✗ | 51.8 | 4.0 |
| DP3 | ✗ | 73.9 | 8.3 |
| **JEPA-WAM** | ✗ | **79.9** | **36.9** |
| RDT-1B | ✓ | 56.0 | 24.1 |
| π₀ | ✓ | 62.5 | 23.9 |
| π₀.₅ | ✓ | 75.4 | 37.2 |
| **π₀.₅ + JEPA Obj.** | ✓ | **84.6** | **37.5** |

**압권은 Random 36.9** — pretraining 없이 DP3(8.3) 대비 **4.4배**, 사전학습 π₀.₅(37.2)와 사실상 동률. Clean-only 학습 후 domain randomization으로 직행한 조건임을 감안하면 transition supervision의 효과가 가장 극적으로 드러나는 지표다.

### 4.4 Real-world (AgileX CobotMagic 5 task) — Figure 5

| Method | ID | OOD |
|--------|----|----|
| π₀ | 51.8 | 22.5 |
| JEPA-WAM | 59.8 | 54.2 |
| π₀.₅ | 77.5 | 72.5 |
| π₀.₅ + JEPA Obj. | **90.3** | **84.7** |

π₀는 ID→OOD에서 **29.3%p 붕괴**하는 반면 JEPA-WAM은 5.6%p만 하락. π₀.₅ 이식판은 ID +12.8%p, OOD +12.2%p.

### 4.5 추론 효율 — Appendix C.3, Table 12

| Method | Median Latency (ms) | Frequency (Hz) |
|--------|---------------------|----------------|
| π₀.₅ | 54.05 | 18.50 |
| π₀.₅ + JEPA Obj. | 55.12 | 18.14 |
| **JEPA-WAM** | **85.00** | **11.76** |
| ABot-M0 | 125.23 | 7.99 |

π₀.₅ 이식판의 오버헤드가 **1.07 ms(2%)** 에 불과하다는 점이 실용적으로 중요하다 — future token은 prefix에 남지만 action token이 마스킹되어 있어 실질 비용이 거의 없다.

---

## 5. Ablation 분석 (Table 4, LIBERO-Plus)

| # | Variant | Cam. | Rob. | Lang. | Lit. | Back. | Noi. | Lay. | **Avg** |
|---|---------|------|------|-------|------|-------|------|------|---------|
| a | DINO+SigLIP | 60.0 | 61.9 | 74.1 | 88.7 | 88.0 | 64.2 | 75.7 | 73.2 |
| b | V-JEPA only (transition 없음) | 78.7 | 40.9 | 70.9 | 96.7 | 84.1 | 88.3 | 79.3 | 77.0 |
| c | Future only 타깃 | 75.1 | 47.1 | 69.6 | 96.0 | 93.4 | 81.5 | 78.4 | 77.3 |
| d | iREPA-style conv align. | 68.9 | 45.5 | 69.2 | 90.9 | 89.1 | 81.5 | 77.7 | 74.7 |
| e | Lower-16 layer align. | 77.5 | 41.6 | 75.0 | 95.5 | 86.3 | 82.2 | 77.2 | 76.5 |
| f | Full hidden (placeholder 제거) | 62.5 | 49.9 | 70.1 | 89.3 | 88.6 | 75.6 | 76.0 | 73.1 |
| — | **JEPA-WAM** | 79.2 | 59.2 | 68.2 | 93.3 | 94.6 | 83.6 | 76.1 | **79.2** |

**해석 (중요)**:
1. **(a→b) +3.8%p**: V-JEPA 표현공간 자체가 OOD의 상당 부분을 설명. transition prediction을 완전히 끈 상태에서도 DINO+SigLIP보다 3.8%p 우위 — 즉 **79.2의 절반 이상은 표현공간 선택의 공로**다.
2. **(b→JEPA-WAM) +2.2%p**: transition supervision의 순수 기여. 주로 **Robot 40.9→59.2 (+18.3%p)** 와 **Background 84.1→94.6 (+10.5%p)** 에서 나온다.
3. **(c) future-only 77.3 vs joint 79.2**: joint target 설계가 +1.9%p. 미래 상태 자체가 아니라 *시간적 관계*를 타깃으로 삼는 것이 유효.
4. **(d) 74.7**: iREPA-style convolution이 이웃 feature를 국소적으로 섞어 spatial correspondence를 약화 → **-4.5%p**. patch-level 구조 보존이 이 방법의 필수 조건임을 보여주는 가장 강한 증거.
5. **(f) 73.1**: action placeholder를 없애고 전체 last hidden state로 조건화하면 **-6.1%p** — transition objective와 action objective가 같은 표현을 공유하면 **간섭**이 발생. "shared backbone + separate readout"이라는 설계가 우연이 아님.
6. **(e) 76.5**: 중간 레이어 supervision보다 최종 shared predictor를 직접 지도하는 편이 우월.

---

## 6. 강점

- **설계의 내적 일관성**: V-JEPA tubelet=2 → 2프레임 joint 인코딩이 단일 이미지와 같은 grid를 만든다는 관찰이, joint target·patch-wise alignment·shared ordering을 한꺼번에 성립시킨다. 억지로 붙인 모듈이 아니라 표현공간의 성질에서 자연스럽게 도출된 설계.
- **파라미터 효율**: 0.5B backbone으로 2-5B 모델과 경쟁. Table 1/2의 Params 열을 함께 읽으면 인상이 크게 달라진다.
- **배포 비용 제로**: target branch/prediction head가 추론에서 완전히 제거되어 latent WAM의 원래 동기(video-gen WAM의 비용 회피)를 끝까지 지킨다.
- **이식 가능성 입증**: 방법이 스스로의 아키텍처에 갇히지 않고 사전학습 π₀.₅에 2% 오버헤드로 이식되어 SOTA를 만든다 — 이 논문에서 가장 재사용 가치가 높은 결과.
- **Ablation의 정직함**: (b) V-JEPA only를 명시해 표현공간 기여분을 스스로 분리해 보여준다.

---

## 7. 약점 및 한계

- **Language 카테고리 68.2는 명백한 회귀**: LIBERO-Plus 표에서 거의 최하위. joint target이 언어 조건과 무관한 visual temporal structure를 학습하기 때문이며, 저자들도 Limitations에서 "동일한 관측이 지시에 따라 크게 다른 transition으로 이어질 때 덜 표현적"이라고 인정한다. 즉 이 방법은 **language grounding을 개선하지 않으며 오히려 희생할 수 있다**.
- **표현공간 기여와 방법 기여의 비율**: (a) 73.2 → (b) 77.0 → 79.2에서 transition prediction의 순수 기여는 2.2%p. 제목이 강조하는 "world modeling"보다 "V-JEPA를 backbone으로 쓴 것"의 몫이 더 크다는 해석도 가능하다.
- **δ의 벤치마크 의존성**: LIBERO δ=31, RoboTwin δ=50이 벤치마크별로 수동 지정되며, δ 민감도 분석이 본문에 없다.
- **π₀.₅ 이식판의 Camera 회귀**: 69.4 → 66.0으로 오히려 하락. 전체 평균은 올랐지만 카테고리별로는 trade-off가 존재한다.
- **코드 미공개 상태**: 초록은 "project page is available on GitHub"라 하나 논문 본문에 URL이 없다.
- **RoboTwin에서만 x-prediction 사용**: parameterization을 벤치마크별로 바꾼 것이 공정 비교를 다소 흐린다.

---

## 8. 관련 연구와의 위치

| 접근 | 대표 연구 | JEPA-WAM과의 차이 |
|------|-----------|-------------------|
| Video-gen WAM | WorldVLA, Fast-WAM | 명시적 미래 프레임 생성 → 배포 비용. JEPA-WAM은 latent만 |
| Generator feature 재활용 | Cosmos-Policy, Light-WAM | generation용으로 최적화된 feature. JEPA-WAM은 state change 전용 타깃 |
| 미래를 latent token으로 압축 | Being-H0.7, LaWAM | compact 표현이 spatial 구조 손실. JEPA-WAM은 dense patch 유지 |
| 별도 JEPA world model로 supervise | **VLA-JEPA** (Sun et al. 2026) | 별도 WM이 policy 표현을 감독. JEPA-WAM은 **shared predictor**로 통합 |
| V-JEPA history를 policy 입력으로 | **JEPA-VLA** (Miao et al. 2026) | V-JEPA를 추가 입력으로만 사용. JEPA-WAM은 V-JEPA 공간 *안에서* WAM 구성 |

이름이 비슷한 VLA-JEPA / JEPA-VLA와의 구별이 이 논문 포지셔닝의 핵심이며, Figure 2가 그 구별을 시각화한다.

---

## 9. 재현성 체크리스트

| 항목 | 상태 |
|------|------|
| 아키텍처 하이퍼파라미터 | ✅ 상세 (projector 1024→896→896, head 896→2048→1024, LoRA r=32/α=64/p=0.1, DiT-L 16층, placeholder 64개, future token 32개) |
| 최적화 설정 | ✅ AdamW, lr 2e-4→1e-5, warmup 3%, wd 0, clip 1.0, BF16, FSDP 8GPU, batch 128, 60K steps |
| λ_wm | ✅ 0.5 (JEPA-WAM), 0.1 with 1K-step warmup (π₀.₅ 이식) |
| δ 오프셋 | ✅ LIBERO 31, RoboTwin 50 |
| 추론 설정 | ✅ 4-step Euler |
| 코드 | ⚠️ URL 미기재 |
| Seed / 분산 | ❌ 다중 seed 결과 없음 |
| 실로봇 per-rollout | ✅ Appendix E 언급 |

---

## 10. 후속 연구 방향

- **Language-conditioned transition target**: 저자 스스로 지목한 방향. Language 68.2 회귀를 해결하려면 joint target에 instruction 조건을 넣거나 multimodal transition target이 필요
- **δ 자동 선택**: 태스크 길이·속도에 따라 적응적으로 offset을 정하는 메커니즘
- **더 큰 backbone으로의 스케일링**: 0.5B에서의 결과가 7B급에서도 유지되는지
- **Shared predictor 간섭의 이론적 분석**: (f) Full-hidden에서 -6.1%p가 나온 이유를 gradient conflict 관점에서 정량화
- **다른 JEPA 계열 인코더**: V-JEPA 2.1 외 I-JEPA / 다른 dense video SSL 표현으로 일반화되는지

---

## 11. 실용적 함의

- **소규모 랩 관점**: 0.5B backbone + LoRA + 8GPU 60K step으로 LIBERO-Plus 79.2를 얻는다는 것은 pretraining 예산이 없는 팀에게 현실적인 레시피다.
- **기존 VLA 보유 팀에게 가장 유용한 것은 §3.4의 이식 레시피**: 아키텍처를 바꾸지 않고 future token 64개 + 정렬 손실만 추가해 OOD를 끌어올리며 추론 오버헤드가 2%. 자체 π₀ 계열 정책이 있다면 즉시 시도해볼 만하다.
- **주의점**: language 일반화가 중요한 애플리케이션(자유 형식 지시, 지시 재해석)에서는 이 supervision이 오히려 불리할 수 있다. Table 2의 Language 열을 반드시 확인하고 채택할 것.
- **평가 관점**: LIBERO ID 점수만 보면 이 논문의 기여가 보이지 않는다. LIBERO-Plus / RoboTwin Random / real-world OOD 세 축을 함께 봐야 한다.

---

## 12. 결론

JEPA-WAM은 latent WAM 설계에서 흔히 분리되던 두 축 — **무엇을 예측할 것인가**와 **그것을 어떻게 policy에 연결할 것인가** — 를 하나의 답으로 묶은 작업이다. 답은 각각 "현재-미래를 함께 인코딩한 spatially structured joint target"과 "transition 예측과 action readout을 공유하되 readout은 분리하는 shared predictor"이며, ablation (c)(d)(f)가 이 두 선택이 각각 +1.9, +4.5, +6.1%p를 설명한다는 것을 보여준다.

결과는 견고하다: 0.5B로 LIBERO-Plus 79.2(비-pretraining 1위), RoboTwin Random 36.9(사전학습 π₀.₅와 동급), 실로봇 OOD에서 π₀ 대비 +31.7%p. 그리고 가장 실용적인 결과는 아마도 자체 아키텍처가 아니라 **이식 실험**일 것이다 — 사전학습 π₀.₅에 2% 추론 오버헤드로 LIBERO-Plus 86.3, RoboTwin Clean 84.6, 실로봇 ID 90.3을 만들며, 이는 transition supervision이 robot-policy pretraining과 **경쟁이 아니라 보완** 관계임을 보여준다.

다만 (b) V-JEPA only가 이미 77.0에 도달한다는 사실은 이 논문의 성능 서사에서 표현공간 선택의 몫이 상당함을 인정하게 만들며, Language 카테고리의 회귀는 "language-agnostic한 시간 구조 학습"이라는 설계의 대가를 정직하게 드러낸다. 저자들이 지목한 language-conditioned transition target이 이 마지막 구멍을 메운다면, joint-embedding world modeling은 VLA 학습의 표준 보조 objective가 될 가능성이 충분하다.

<!-- VERIFIED: pdf -->
