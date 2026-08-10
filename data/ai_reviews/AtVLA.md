# AtVLA — Look Where It Matters: Adaptive Visual Refinement for Vision-Language-Action Models

> **한 줄 요약**: VLA의 시각 인코더도 ViT의 high-norm attention artifact를 앓고 있으며, embodied post-training이 그걸 **악화시킨다**는 진단에서 출발한다. 해법은 두 개. (a) SigLIP에 학습 가능한 **register token 4개**를 꽂아 넘치는 global 정보를 흡수시키고 patch attention을 정화하며, 그 register 출력을 버리지 않고 policy에 **embodied spatial context로 재활용**. (b) action expert가 K=4개 청크를 뽑아 **불확실할 때만** attention rollout으로 관심 영역을 잘라 고해상도로 재인코딩해 KV 캐시 뒤에 덧붙인다. LIBERO 평균 94.2 → **98.4%**, 실기 46.5 → **69.0%**, 추가 연산은 π₀의 1.4~1.6배.

- arXiv: 2608.02197 (v1, 2026-08-03, cs.RO)
- 저자: Jin Cui\*, Yanbin Hu\*, Xinyue Long, Linkai Li, Boran Zhao(교신), Pengju Ren — 시안교통대학교 (인간-기계 혼합 증강지능 국가중점연구실 / 인공지능·로보틱스 연구소, 소프트웨어공학부)
- 코드: 논문에 공개 링크 없음

---

## 1. 배경 및 동기

VLA는 대규모 vision-language 사전학습에서 풍부한 semantic prior를 상속받는다. 그런데 로봇 조작은 "이게 머그컵이다"를 아는 것으로 부족하다 — **목표 위치 파악, 기하 추론, 정밀 접촉**을 위해 공간적으로 충실한(spatially faithful) 표현이 필요하다.

저자들은 조작 실패를 체계적으로 뜯어보고 두 개의 시각 병목을 짚는다.

1. **모델이 목표 물체를 잘못 고르고 task 유형을 혼동한다.** VLM backbone의 의미 이해력을 생각하면 이상한 결함이다. attention 분포를 열어보니, VLA 시각 인코더가 Darcet et al.(2023)이 일반 ViT에서 보고한 **high-norm artifact**를 그대로 갖고 있었다. 배경 패치가 global 정보의 저장소로 재활용되면서 국소 공간 정보를 밀어내고 dense attention map을 오염시킨다 (Fig. 1).
2. **위치를 맞게 잡아도 정밀 조작이 보장되지 않는다.** 미세 동작에는 대상·접촉부 주변의 국소 기하가 필요한데, 저해상도 3인칭 관측에서 그 영역은 패치 몇 개뿐이다. 손목 카메라는 근접 뷰를 주지만 시점·자세 변화에 극도로 민감하고(그리퍼-물체 상대 위치가 요동쳐 관련 영역이 예측 불가하게 이동), 명시적 vision expert나 3D 표현은 센싱·캘리브레이션·모델링 복잡도를 더한다.

---

## 2. 핵심 질문

> "embodied post-training은 사전학습 인코더의 시각 표현을 **어떻게 재편하는가**, 그리고 그 표현은 미세 조작에 충분히 신뢰할 만한가?"

이 논문의 진짜 기여는 두 번째 문장에 있는 진단이다. artifact가 단순히 ViT에서 상속된 게 아니라, **embodied post-training과 밀접하게 연결되어 있다**는 주장이다. 인코더가 로봇 데이터에서 물체 위치·깊이 순서·인스턴스 구조·국소 기하를 습득할수록, 원래의 global token(CLS) 용량이 부족해 그 정보의 일부가 **spatial patch token으로 흘러넘친다**. 유용한 embodied 지식이 잘못된 표현 채널에 저장되는 것이고, 그 대가가 attention 오염이다.

여기서 처방이 자연스럽게 나온다: **정보를 없애지 말고 제대로 된 저장 슬롯을 주자.**

---

## 3. 아키텍처 개요 (Fig. 2)

π₀ 위에 두 개의 상보적 메커니즘을 얹는다. 액션 표현·생성 방식은 π₀ 그대로 건드리지 않는다.

**베이스 정책**: PaliGemma 3B (SigLIP-So400m 시각 인코더 + Gemma-2B 언어 backbone) + 300M flow-matching action expert. 입력 224×224, patch 14 → 뷰당 256 patch token.

**흐름**:
```
원시 관측 It + 지시문 ℓ + proprioception st
      ↓
register 증강 시각 인코더 → [h_cls, R_t(4개), P_t(256개)]
      ↓
prefix C_b = [Π(R_t); Π(P_t); E_lang(ℓ); E_state(s_t)]  → LLM 1회 통과, KV 캐시
      ↓
action expert가 K=4개 청크 샘플링 → 불일치도 U_t 계산
      ↓
U_t ≤ τ : 평균 청크 실행 (CONFIDENT, 기본 경로)
U_t > τ : attention rollout → crop → 재인코딩 → prefix 확장 → 재생성 (UNCERTAIN)
```

---

## 4. Register-Enhanced Visual Prefix

CLS token 바로 뒤에 학습 가능한 register embedding **N_r = 4**개를 삽입한다:

`X_t = [x_cls; r_1,…,r_{N_r}; p_1,…,p_{N_b}]`, N_b = 16×16 = 256

원논문(Darcet et al.)은 register를 인코더 내부의 임시 작업 공간으로만 쓰고 출력은 버린다. **이 논문은 R_t를 유지하고 PaliGemma 시각 projector Π로 투영해 policy에 넘긴다.** 즉 register가 downstream 정책에게 여러 개의 global context 슬롯으로 노출되고, 동시에 16×16 patch grid 전체는 공간 grounding용으로 온전히 남는다.

- 1152차원 register 4개 = **추가 파라미터 4,608개**. 사실상 공짜다.
- 다중 카메라 뷰는 같은 register 증강 인코더로 각각 독립 인코딩 후 prefix에 concat.
- 학습은 **오직 embodied 데이터와 원래 action objective만** 사용한다. ImageNet/CC3M을 섞은 별도 시각 적응도 실험했는데, 비용만 늘고 로봇 시연에서 얻은 능력을 눈에 띄게 훼손해서 폐기했다 (Q1).

---

## 5. Uncertainty-Gated Action Prediction

prefix KV 캐시를 공유한 채 flow-matching expert가 독립적인 가우시안 초기값에서 K개 청크를 생성한다 (동일한 10-step Euler solver / noise schedule). 모든 샘플이 시각·언어 prefill을 공유하므로 **비용 차이는 action expert denoising뿐**이다.

불확실성은 **병진 차원 D_tr = {Δx, Δy, Δz}**, 그리고 **실제로 실행될 앞쪽 h개 액션**에 대해서만 표준편차를 평균낸다:

`U_t = (1/(h|D_tr|)) Σ_j Σ_d std_k[A^(k)_{j,d}]`

- 회전이 아닌 병진, 전체 horizon이 아닌 근미래로 좁힌 이유: **공간적 모호성과 가장 직결된 end-effector 운동의 불일치를 분리**하기 위해서다. 설계 선택이 문제 정의와 정확히 맞물린다.
- τ는 embodiment별로 보정한다 (LIBERO 4개 suite는 공유, Google Robot·Franka는 별도). episode-disjoint 보정 세트에서 base 경로와 crop 경로의 near-term 병진 오차 감소 Δe_t를 재고, 트리거율 30% 예산 하에 `τ* = argmax E[Δe_t · 1(U_t > τ)]  s.t. Pr(U_t > τ) ≤ 0.3`. 실무적으로는 검증 불확실성 분포의 **70th percentile**에서 시작해 조정.
- 게이트는 **action chunk replanning step마다 1회**만 평가한다 — 저수준 제어 스텝마다가 아니다.

---

## 6. Attention-Guided Local Refinement

U_t > τ일 때만 발동한다. 핵심은 **새로 계산하는 게 거의 없다**는 점이다.

1. **Rollout 집계**: action expert의 **18개 joint-attention layer 전부**에서 action-to-prefix attention을 수집. head 전체와 denoising step Q = {3, 6, 9}(초기/중기/후기)에 대해 평균 → `M̃_l = RowNorm(I + mean(M))` → `M_roll = M̃_L ⋯ M̃_1`. 이 attention은 **액션 샘플링 중 이미 계산된 것**이라 rollout 자체의 비용은 미미하다.
2. **Saliency map**: 앞쪽 h개 action token 행 → 256개 base image patch 열의 항목을 평균해 16×16으로 reshape. **register 열은 제외** — register는 global context를 담을 뿐 이미지 격자와 고정 대응이 없기 때문. (설계의 자기일관성이 좋다.)
3. **Contrastive window search**: 224×224 위에서 변 길이 {84, 112, 140}px, stride 14px(SigLIP patch 크기 정렬)로 후보 윈도우를 훑고, 내부 평균 attention이 1.25배 확장된 context ring 대비 가장 두드러진 창을 고른다.
   `b* = argmax_b [ mean_{u∈b} S(u) − mean_{u∈ρ(b)\b} S(u) ]`
   이 대조 기준은 **전체 이미지에 가까운 확산형 crop**과 **고립된 attention peak 중심의 너무 작은 창**을 동시에 억제한다.
4. **재인코딩과 prefix 확장**: 선택 박스를 각 변 10% 패딩·클리핑 후 crop → 224×224 리사이즈 → 같은 register 증강 인코더로 인코딩. **crop 쪽 register 출력은 버린다**(base register가 이미 global을 요약했으므로) — patch 256개만 유지. 원본 내 위치는 정규화 좌표를 2-layer MLP(4→256→d_LLM, GELU)로 embedding해 모든 crop token에 더한다. 새로 추가된 crop segment만 prefill하고 **base image/instruction/state의 KV 캐시는 재사용**한다.
5. 다중 뷰면 뷰별로 rollout을 계산해 localization 점수가 가장 높은 **뷰-윈도우 쌍 하나만** 고른다 → 불확실 스텝당 추가 crop 인코딩은 최대 1회.

**Invalid rollout 처리**가 꼼꼼하다. saliency에 non-finite 값이 있거나, attention 총합 < 1e-6이거나, 후보 점수가 유한하지 않거나, 좌표가 퇴화·이미지 밖이거나, 최종 crop 변이 56px 미만이면 **crop을 건너뛰고 base 경로 결과를 실행**한다. 그리고 이 규칙을 **학습과 배포에서 동일하게** 쓴다 — ground-truth fallback이 train-test 불일치를 만들지 않도록.

---

## 7. 학습 목표와 3단계 커리큘럼

π₀ 사전학습 체크포인트에서 초기화하고 원래 objective L_π₀를 전 구간 유지한다. 최종 목표:

`L = L_π₀ + λ_cp·L_cp + λ_ag·L_ag`, λ_cp = 0.1, λ_ag = 1.0

L_ag는 attention grounding 손실로, 근미래 action token이 base image patch에 할당한 attention A_p 중 주석된 task 영역 Ω(b_gt) 안쪽 비율의 음의 로그:
`L_ag = −log[ (Σ_{p∈Ω(b_gt)} A_p + ε) / (Σ_p A_p + ε) ]`

액션 손실만으로는 attention이 시각적으로 튀는 배경이나 지속적 positional sink에 몰릴 수 있으므로, **추론 시 crop을 생성하는 바로 그 attention을 명시적으로 감독**한다.

| 단계 | 스텝 | 학습 대상 | 동결 | lr |
|---|---|---|---|---|
| Stage 0-a: register warm-up | 4K | register embedding 4개만 | 전부 | 1e-4 |
| Stage 0-b: 인코더 적응 | 16K | register + SigLIP 마지막 4블록 | 나머지 전부 | 2e-5 |
| Stage 1: GT crop 정렬 | 10K | crop-position encoder + rank-32 LoRA | SigLIP·register·projector·backbone | 1e-4 |
| Stage 2: joint refinement | 40K | 모든 LoRA + register + projector + crop encoder | base 가중치 | 1e-4 (constant) |

총 **70K gradient update**, AdamW, batch size 4, RTX 6000 Ada 8장.

**Stage 2의 crop 커리큘럼**(α = rollout crop 사용 확률, p_drop = crop dropout):
- 0~40%: α=0, p_drop=0 — 전부 GT crop. 초기의 부정확한 attention이 입력을 오염시키는 것을 막고 멀티모달 융합을 안정화.
- 40~90%: α 0→1 선형, p_drop 0→0.3 선형. **rollout이 invalid면 GT로 fallback하지 않고 crop 자체를 건너뛴다** — 배포 시의 실패 처리와 일치.
- 90~100%: α=1, p_drop=0.3 — 완전히 배포와 동일한 crop 생성/실패 처리로 최적화.

crop dropout은 confident 경로(base-only 입력)에서도 정책이 강해야 하므로 필수다. 두 추론 경로 모두에 노출시키는 장치.

---

## 8. 주요 결과 — 시뮬레이션 (Table 1)

**LIBERO (4 suite 통합 단일 정책, 40 task × 50 demo = 2,000 궤적)**

| Method | Spatial | Object | Goal | Long-10 | Avg |
|---|---|---|---|---|---|
| Octo | 78.9 | 85.7 | 84.6 | 50.9 | 75.0 |
| OpenVLA | 85.0 | 88.6 | 79.2 | 53.6 | 76.6 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π₀.₅ | 97.3 | 98.8 | 96.9 | 94.2 | 96.8 |
| OFT | 96.2 | 98.3 | 96.2 | 90.7 | 95.4 |
| VLANeXt | 99.0 | 99.2 | 96.6 | 94.8 | 97.4 |
| π₀ + Registers | 98.8 | 99.0 | 98.0 | 93.1 | 97.2 |
| π₀ + Cropping | 96.5 | 98.2 | 96.7 | 93.3 | 96.2 |
| **AtVLA (Full)** | **99.3** | **99.4** | **98.3** | **96.5** | **98.4** |

**4개 suite 전부에서 1위.** 가장 큰 증분은 Long-10(85.2 → 96.5, +11.3)이다.

**SimplerEnv Google Robot** (공식 프로토콜; WidowX는 BridgeData 특화 학습·embodiment 적응이 필요해 시각 개선과 교란되므로 의도적으로 제외)

| Method | Pick Coke Can | Move Near | Open/Close Drawer |
|---|---|---|---|
| π₀ | 88.0 | 80.3 | 56.0 |
| SpatialVLA | 86.0 | 77.9 | 57.4 |
| RT-1-X | 56.7 | 31.7 | **59.7** |
| π₀ + Registers | 88.2 | 80.5 | 56.0 |
| π₀ + Cropping | 90.0 | 79.2 | 56.9 |
| **AtVLA (Full)** | **91.3** | **81.6** | 57.5\* |

Open/Close Drawer만 경쟁력 수준에 그친다. 저자의 설명이 설득력 있다: **캐비닛이 장면 대부분을 차지해서 대부분의 crop이 원본 이미지 거의 전체를 보존**하고, 손잡이나 접촉점을 해상하지 못한다. 적응형 refinement가 기여할 여지 자체가 없는 레짐이다.

**실기 벤치마크** (Franka Research 3, **단일 3인칭 RealSense D435i**, 셀당 20회 독립 실험):

- Kitchen (Move/Grab/Pick/Long/Spatial): π₀ 50/35/55/15/75 → AtVLA **80/65/80/40/75**
- Building Blocks (Stack/Edge/Spatial/Long/Grab): π₀ 55/25/55/15/35 → AtVLA **80/50/70/35/70**
- 평균 46.5% → **69.0%**

단일 뷰 depth 모호성이 지배하는 소물체 파지·적층에서 격차가 가장 크다. Building Blocks Grab 35 → 70(2배)이 대표적.

---

## 9. Ablation — register가 하는 일 (Table 3, Fig. 5, Fig. 6)

**Q: register는 artifact를 억제만 하는가, 정보를 저장하는가?**

π₀+Registers에서 학습된 register를 **제거**하면:

| Variant | Spatial | Object | Goal | Long-10 | Avg Δ |
|---|---|---|---|---|---|
| π₀ + Registers | 98.8 | 99.0 | 98.0 | 93.1 | — |
| w/o REGs | 93.2 | 92.1 | 92.2 | 81.3 | **−4.45** |

**π₀ 베이스라인보다도 낮게 떨어진다.** register가 단순한 artifact 흡수 장치가 아니라 task 관련 embodied 지식의 실제 저장소임을 보여주는, 이 논문에서 가장 강한 증거다.

**Linear probing (Fig. 5)**: CLS token / mean-pooled patch / register token에서 각각 물체 위치·깊이·표면 법선을 예측하는 독립 선형 프로브. register가 세 항목 모두 **0.70 초과**로 일관되게 최고. CLS 0.60 대비 REG 0.87~0.88 수준의 격차가 보고된다. → embodied 학습이 원래 global token 용량을 넘는 공간·기하 정보를 도입한다는 가설을 지지.

**Register 개수 (Fig. 6)**: 개수가 너무 적으면 오히려 **π₀보다 성능이 떨어질 수 있다.** artifact 흡수가 불완전해 남은 오염 패치가 embodied post-training 중 형성된 인코더-LLM 협응을 깨뜨리기 때문이라는 해석. 4개에서 성능도, 선형 프로브가 측정한 공간 정보량도 포화 → **최소 충분 용량으로 4개 채택**.

**Cropping 단독의 한계**: π₀+Cropping의 crop은 embodied 사전학습이 없는 외부 VLM이 생성한다. 정성 검사 결과 상호작용 임계 영역을 자주 놓치고, full AtVLA보다 성능이 낮다. **깨끗한 attention 신호 없이는 부정확한 crop이 오히려 정책 입력을 왜곡하고 action uncertainty를 키운다.** 두 컴포넌트가 독립적 개선이 아니라 **직렬 의존**임을 보여주는 대목이다.

의외의 부수 효과: cropping이 LIBERO Long-10과 실기 long-horizon 양쪽을 개선한다. 저자는 이를 후속 결정을 위한 추가 시각 맥락으로 해석하며, cropping이 국소 정밀도뿐 아니라 **시간적으로 확장된 계획**도 지원할 수 있다고 본다.

---

## 10. 추론 비용 (Q4)

`C_π₀ = N(C_base + C_A)`
`C_AtVLA = N(C_base + K·C_A + r·(C_crop + C_A))`

K=4, 트리거율 r ≈ 0.3, 그리고 expert 1회 통과 ≈ π₀ replanning의 10~15%, crop refinement ≈ 20~25%라는 가정 하에 **총 연산 ≈ 1.4~1.6× π₀**.

비용을 눌러주는 세 장치:
1. rollout은 액션 샘플링 중 계산된 attention을 재사용 → 거의 공짜
2. crop token 추가 시 base prefix KV 캐시 유지 → base/language 재계산 없음
3. refined 경로는 **K개 샘플을 다시 뽑지 않고** crop 인코딩 1회 + 액션 생성 1회만 수행

배포는 RTX 4090 24GB 단일 GPU. K=4 샘플링이 상수 배수로 얹히는 게 실질 비용의 대부분이라는 점은 짚어둘 만하다.

---

## 11. 핵심 설정 요약 (Table 4)

| 항목 | 값 |
|---|---|
| LIBERO 학습 데이터 | 40 task, 2,000 궤적 |
| Register token | 4개 (1152-dim, +4,608 params) |
| 입력 해상도 | 224×224 (base·crop 동일) |
| 학습 스텝 | 20K(register) + 10K(crop align) + 40K(joint) = 70K |
| LoRA rank | 32 |
| 게이팅 샘플 수 | K = 4 |
| Flow 적분 | 10-step Euler |
| Rollout denoising step | {3, 6, 9} |
| Rollout layer | 18개 joint-attention layer 전부 |
| 후보 crop 크기 | {84, 112, 140} px, stride 14 |
| Context 확장 / 패딩 | 1.25× / 각 변 10% |
| 목표 트리거율 | embodiment당 약 30% |

---

## 12. 총평

**강점**

- **진단이 처방을 낳는 구조가 깔끔하다.** "embodied 학습이 만든 공간 정보가 global token 용량을 넘쳐 patch로 샌다" → "전용 슬롯을 주고, 그 슬롯의 출력을 버리지 말고 쓴다". register 제거 실험(−4.45, π₀ 이하)과 linear probing이 이 인과 사슬을 양쪽에서 받친다.
- **거의 공짜인 개입.** 4,608개 파라미터로 LIBERO 평균 +3.0(94.2→97.2, registers only)이라는 비용 대비 효과는 이례적이다.
- **두 컴포넌트의 의존 관계를 스스로 드러낸다.** π₀+Cropping이 π₀+Registers보다 낮다는 결과를 숨기지 않고, "깨끗한 attention 없이 crop하면 오히려 해롭다"는 해석으로 연결한다. 논문의 두 축이 병렬이 아니라 직렬임을 인정하는 정직함.
- **학습-배포 일치에 대한 집착.** crop 커리큘럼(GT → rollout), invalid rollout 시 GT fallback 금지, crop dropout — 세 장치 모두 train-test discrepancy를 겨냥한다. attention을 추론 시 사용하려면 학습 때 그 attention을 감독해야 한다는 L_ag도 같은 맥락.
- **약점을 회피하지 않고 설명한다.** Open/Close Drawer 열세를 "장면 대부분이 캐비닛이라 crop이 원본과 다를 바 없다"로 메커니즘 수준에서 해명한다.

**한계와 의문**

| # | 쟁점 | 검토 |
|---|---|---|
| 1 | crop 학습에 **주석된 task-relevant region**이 필요하다 | Stage 1·2의 GT crop과 L_ag 모두 영역 주석에 의존한다. LIBERO 2,000 궤적 전체에 이 주석이 붙었다면 결코 작은 비용이 아니며, 주석 없는 새 데이터셋으로의 확장성이 논문에서 다뤄지지 않는다. 사실상 이 방법의 가장 큰 실용적 진입 장벽이다. |
| 2 | K=4 샘플링이 상수 비용으로 항상 얹힌다 | 트리거율이 0이어도 action expert를 4번 돌린다. 1.4~1.6×의 상당 부분은 crop이 아니라 게이팅 자체의 값이다. 더 싼 불확실성 추정(단일 패스 분산 예측 등)과의 비교가 없다. |
| 3 | 비용이 **추정치**다 | "expert 1회 ≈ 10~15%, crop ≈ 20~25%"는 가정이고, 본문은 "정확한 지연시간은 RTX 4090 측정으로 보고한다"고 쓰지만 실제 측정 표는 제시되지 않는다. 실시간 제어율(Hz) 수치가 없어 배포 적합성을 독립 검증하기 어렵다. |
| 4 | τ가 embodiment마다 보정 데이터를 요구한다 | episode-disjoint 보정 세트에서 base/crop 두 경로를 모두 평가해야 τ*를 얻는다. 새 로봇마다 이 절차가 필요하고, 트리거율 30%는 예산으로 **주어진** 값이지 최적화로 도출된 값이 아니다. |
| 5 | 왜 하필 병진 3차원, 왜 근미래 h개인가 | 논리는 설득력 있지만 회전 포함 버전이나 h 변화에 대한 ablation이 없다. 게이팅 신호 설계는 이 방법의 심장인데 검증이 얕다. |
| 6 | artifact와 embodied post-training의 연결이 **상관 수준** | "closely associated"라는 표현을 저자 스스로 쓴다. 사전학습 인코더와 post-training 후 인코더의 artifact 강도를 정량 비교한 통제 실험은 제시되지 않고, 정성적 attention map과 probing이 대신한다. |
| 7 | π₀ 의존 | 모든 변형이 동일 π₀ 체크포인트에서 초기화된다. register 정화가 다른 인코더(DINOv2, 다른 VLM backbone)나 autoregressive action head에서도 성립하는지는 미지수다. |

**총평**: 큰 모델이나 새 센서를 더하는 대신, **이미 계산되고 있지만 버려지던 신호(attention)와 4,608개 파라미터**로 문제를 푼다. VLANeXt(97.4)를 1.0점 앞선 LIBERO 98.4보다 실기 46.5 → 69.0이 훨씬 인상적인데, 단일 3인칭 카메라라는 가장 불리한 조건에서 얻은 결과이기 때문이다. 다만 crop 감독에 필요한 영역 주석과 측정 지연시간의 부재는, 이 방법이 주석이 준비된 벤치마크를 넘어 얼마나 옮겨갈 수 있는지에 대한 판단을 유보하게 만든다.

<!-- VERIFIED: pdf -->
