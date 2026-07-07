# DART: Domain Arithmetic — One-Shot VLA Adaptation under Environmental Shifts

**arXiv**: 2607.00666 · **Venue**: ECCV 2026 · **저자**: Taewook Kang*, Taeheon Kim*, Donghyun Shin, Jonghyun Choi† (Seoul National University) · **코드**: https://github.com/snumprlab/dart

---

## 1. 배경 및 동기

### 환경 변화(Environmental Shift) 하의 VLA 실패
대규모로 학습된 VLA 모델(π0.5, π0-FAST 등)은 학습 환경(source domain)에서는 강력한 멀티태스크 성능을 보이지만, 카메라 포즈 변경, 센서 캘리브레이션 차이, 로봇 임바디먼트 교체(Panda → UR5e) 같은 환경 변화가 생기면 **동일한 학습된 태스크**조차 실패한다. 실제로 π0.5의 LIBERO zero-shot 성능은 viewpoint shift 크기에 따라 88.3% (Small) → 63.9% (Medium) → 11.3% (Large)로 붕괴한다 (Table 1).

### 기존 적응 방법의 한계
- **Full-data fine-tuning**: 타깃 도메인에서 태스크별 시연을 대량 수집해야 함 (LIBERO 기준 1,716개 데모, Table 16) — 수집에 수 일, 학습에 수 시간 소요.
- **One-shot fine-tuning**: 데이터는 적지만 held-out 태스크로 일반화 실패 (catastrophic forgetting 포함).
- **아키텍처 특화 방법** (FLA의 vision encoder LoRA 등): 백본/배포 환경에 따라 일반성 제한.

### 핵심 질문
단 **하나의 시연(one demonstration, one task)** 으로 VLA를 새 환경에 적응시키면서, 베이스 모델의 멀티태스크 능력을 보존할 수 있는가?

---

## 2. 방법론 심층 분석

### 2.1 One-shot Fine-tuning 실패 원인 분석 (Sec. 4)
- **Update-vector** 정의: Δ_m,tgt = θ_m,tgt − θ0 (Eq. 1).
- Subspace alignment score γ (Eq. 2, Iso-C의 metric 차용): γ^(l) = ‖U_j U_jᵀ Δ_i‖_F / ‖Δ_i‖_F. SVD의 left singular vector로 두 update-vector의 부분공간 겹침을 측정.
- **발견 1** (Fig. 2b): 같은 태스크의 update-vector는 도메인이 달라도 강하게 정렬 (0.911~0.912) → one-shot 업데이트는 **태스크 방향이 지배적**이고 도메인 방향은 소량 존재.
- **발견 2** (Fig. 3a): 태스크 프로토타입 + 도메인 프로토타입 − 글로벌 프로토타입의 **가법 조합(additive composition)** 이 실제 update-vector와 가장 높은 정렬 → 태스크/도메인 방향이 선형 분해 가능.
- **발견 3** (Fig. 3b, LIBERO-Plus): 유사한 도메인 변화(viewpoint 크기별)는 유사한 update 방향을 만들고, 복합 shift(View+Noise)는 개별 shift 방향을 부분 재사용 → 도메인 지식이 **구조적/조합적**으로 조직됨.

### 2.2 Domain Vector 추출 (Sec. 5.1)
"queen = king + woman − man" 방식의 **weight-space analogy**:
- 같은 태스크 T_m의 source 데모와 target 데모로 각각 1,000 스텝 fine-tuning하여 Δ_m,src, Δ_m,tgt를 얻음.
- 도메인 벡터: δ_tgt = Δ_m,tgt − Δ_m,src (Eq. 3) — 공유된 태스크 방향이 상쇄되고 도메인 방향만 남음.

### 2.3 Subspace Filtering & Scaling (Sec. 5.2)
직접 뺄셈은 fine-tuning 노이즈와 source 아티팩트를 주입할 수 있어 두 가지 보정 도입:
- **Subspace filtering**: SVD 후 interaction matrix C = U_tgtᵀ U_src로 각 source basis의 overlap energy e_j = ‖C_:,j‖² (Eq. 4)를 계산. 정렬 점수 γ^(l)을 동적 컷오프로 사용해 (Eq. 5) 타깃 부분공간과 정렬된 source basis만 남긴 Δ̃_m,src를 뺌. 모델 머징(TIES, Iso-C)이 각 벡터의 고유 성분을 최대화하는 것과 반대로, **공통 성분을 찾아 제거**하는 것이 목적.
- **Subspace scaling**: δ̃_tgt = γ^(l) · (Δ_m,tgt − Δ̃_m,src) (Eq. 6) — 근본적으로 misaligned된 레이어(γ→0)의 노이즈 지배적 도메인 벡터를 감쇠.
- 최종 적응: θ* = θ0 + α · δ̃_tgt (Eq. 7), α=0.8 (LIBERO 한 suite에서 10 rollout 소규모 탐색으로 결정 후 전 설정 공통 사용; visual perturbation은 α=0.6, 보충 C.2).

### 2.4 알고리즘 특성
- 1D 레이어(bias, norm)는 단순 뺄셈만 적용 (Algorithm 1).
- Randomized SVD (r=256)로 15m35s → 6m33s 가속, 성능 유지 79.1 → 78.7 (Table 8).
- 아키텍처 무수정(architecture-agnostic): flow matching(π0.5)과 autoregressive(π0-FAST) 모두 적용.

---

## 3. 실험 설정

- **모델**: π0.5 (주력, flow matching), π0-FAST (일반성 검증, autoregressive FAST 토큰). openpi JAX 코드베이스, AdamW, batch 64, one-shot FT는 peak LR 5e-5 / 1,000 스텝 (Table 11).
- **베이스라인**: Zero-shot, One-shot FT, FLA (CVPR 2026, vision encoder LoRA), RETAIN (ICLR 2026, 모델 머징).
- **시뮬레이션 (LIBERO)**: 4개 suite 40개 태스크. Viewpoint shift 3단계 (Small ~24°, Medium ~57°, Large ~118°, Table 12) + Noise/Light perturbation. 5개 scene마다 태스크 1개의 시연 1개로 적응. 3회 반복 × 태스크당 50 rollout.
- **Cross-embodiment (MimicGen)**: Stack/Stack Three, Panda → UR5e, Stack 시연 각 1개. 5 seed × 50 rollout.
- **실제 로봇**: UR10e + Robotiq 2F-85, 5개 태스크 (Eggplant/Lemon/Carrot pick-and-place, Stack Cube, Press Stapler). Source viewpoint 시연 120개(태스크당 24개)로 π0.5 학습, Target viewpoint의 Stack Cube 시연 1개로 적응. 태스크당 12 rollout.

---

## 4. 실험 결과 심층 분석

### 4.1 LIBERO Novel Viewpoints, π0.5 (Table 1)
| 방법 | Small | Medium | Large | 평균 |
|---|---|---|---|---|
| Zero-shot | 88.3 | 63.9 | 11.3 | 54.5 |
| One-shot FT | 43.4 | 33.3 | 17.8 | 31.5 |
| RETAIN | 87.4 | 72.4 | 48.9 | 69.6 |
| FLA | 92.2 | 76.4 | 54.3 | 74.3 |
| **DART** | 92.0 | **80.8** | **64.4** | **79.1** |

Zero-shot 대비 +24.6pp. One-shot FT가 오히려 zero-shot보다 나빠지는(54.5→31.5) 반면 DART는 도메인 지식만 이식해 멀티태스크 능력 보존. Full-data 상한선은 92.2 (1,716 데모, Table 16).

### 4.2 복합 Visual Shift, π0.5 (Table 2)
View 80.8 / View+Noise 69.2 / View+Noise+Light 75.0, 평균 75.0 — FLA 71.5, RETAIN 68.7 대비 전 설정 우위.

### 4.3 π0-FAST 일반성 (Table 3)
Small 91.2 / Medium 80.8 / Large 66.2, 평균 79.4 (FLA·RETAIN 76.5~76.6). 흥미롭게 π0-FAST의 zero-shot(73.4)이 π0.5(54.5)보다 shift에 강건 — 저자들은 이산 분류가 암묵적 정규화로 작용한다고 추정 (보충 D.1).

### 4.4 Cross-embodiment, MimicGen (Table 4)
Panda → UR5e: 평균 Succ. 69.4% (zero-shot 62.0, One-shot FT 56.4). Stack Three에서 37.2 → 45.4. 시각 shift와 전혀 다른 물리적 도메인 갭에도 알고리즘 수정 없이 적용됨.

### 4.5 실제 로봇 UR10e (Table 5)
Stack Cube 시연 **1개**로 적응 후 5개 태스크 평균 **81.7%** (zero-shot 43.3, FLA 55.0, RETAIN 48.3, One-shot FT 51.7). Eggplant/Lemon 91.7, Press Stapler 100.0. 시연에 없던 태스크들로의 도메인 전이가 실환경에서도 성립. 참고로 base policy의 source viewpoint 성능은 평균 98.3 (Table 15).

### 4.6 Ablation & 분석
- **Ablation** (Table 6): 뺄셈만 78.1 → +filtering 78.8 → +scaling 78.5 → **둘 다 79.1**. 뺄셈 자체가 One-shot FT(31.5) 대비 가장 큰 기여.
- **도메인 벡터 머징** (Table 7/21): 3개 viewpoint 도메인 벡터를 TSV로 머징하면 단일 모델로 평균 75.7 — 도메인 벡터의 가법성 재확인, 도메인별 벡터 저장 오버헤드 절감.
- **α 강건성** (Fig. 6a): subspace alignment 덕에 넓은 α 범위에서 안정.
- **FT 스텝** (Fig. 6b): One-shot FT는 스텝이 늘수록 망각으로 악화, DART는 소폭 지속 개선; 적은 스텝에서도 강함.
- **Source 망각 없음** (Table 20): 적응 후 source domain 성능 96.9 → 93.4~94.4로 거의 유지.
- **TTA 비교** (Table 19): SCALE(ICML 2026) 73.0 vs DART 79.4 (π0-FAST).
- **모델 머징 방법과 비교** (Table 18): TIES 77.6, Iso-C 74.5, RESM 75.5 < DART 79.1 — 간섭 완화형 머징은 "공통 성분 제거"라는 analogy 목적에 부적합.
- **레이어 분석** (Fig. 13-14): 전체 레이어 적응이 최선. MLP Up_proj/Gate_proj가 source-target 간 가장 misaligned — 사실적 지식이 MLP에 저장된다는 knowledge editing 문헌과 일관.
- **태스크 불일치 강건성** (Table 22): source와 target의 적응 태스크가 달라도 feature cosine similarity 기반 유사 태스크 선택이 무작위 선택보다 일관되게 우수.

---

## 5. 관련 연구 비교

| 접근 | 대표 | 한계 | DART 차별점 |
|---|---|---|---|
| 증강 기반 fine-tuning | RoVi-Aug, LIBERO-Plus 계열 | 추가 데이터 수집 비용 | 시연 1개 |
| 아키텍처 수정/부분 적응 | FLA (vision LoRA), Adapt3R | 백본 종속 | 아키텍처 무관 |
| 모델 머징 | RETAIN, MergeVLA, TIES, Iso-C | 능력 선택적 전이 불가 | analogy로 도메인 지식만 분리 |
| Weight arithmetic analogy | Chat Vector, AdaMergeX (LLM) | 직접 뺄셈에 국한 | subspace filtering/scaling 도입 |
| Test-time adaptation | SCALE, VLS | 제한적 shift 범위 | 다양한 시각+임바디먼트 shift |

---

## 6. 강점

1. **극한의 데이터 효율**: 환경당 시연 1개로 40개 태스크 전체 적응 (full-data 1,716개 대비).
2. **분석의 설득력**: task/domain 방향의 가법 분해를 프로토타입 실험으로 먼저 검증하고 방법을 구축하는 analysis-first 구성.
3. **폭넓은 검증**: 2개 아키텍처(flow matching + autoregressive) × 시각 shift 5종 × cross-embodiment × 실로봇.
4. **훈련 최소화**: one-shot FT 2회(1,000스텝) + CPU에서 수 분의 weight arithmetic; 추론 오버헤드 0.
5. **망각 회피**: source 성능 유지 + 도메인 벡터 머징으로 다중 도메인 단일 모델 운용 가능.
6. **코드 공개** (snumprlab/dart).

---

## 7. 약점 및 한계

1. **Severe shift에서 성능 하락**: Large viewpoint에서 64.4%로 여전히 낮음 (저자 인정, Sec. 7).
2. **스칼라 α 탐색 필요**: 10 rollout 소규모지만 타깃 도메인 rollout 평가가 필요 — 진정한 hyperparameter-free는 아님.
3. **Source 데모 접근 가정**: 같은 태스크의 source 시연 필요. 보충 A.2에서 완화 방안을 논하지만, 대규모/비정형 source 데이터셋에서는 태스크 매칭이 근사적.
4. **벤치마크 프로토콜 비표준**: LIBERO 수치는 viewpoint shift 하의 값으로 표준 LIBERO 리더보드와 직접 비교 불가.
5. **도메인당 재적응 필요**: 새 타깃 도메인마다 시연 1개 + FT 2회 필요 (머징으로 일부 완화).
6. **일반 시나리오 미검증**: 언어 일반화, 새 물체, 새 태스크로의 확장은 범위 밖.

---

## 8. 재현성

- 코드 공개: https://github.com/snumprlab/dart. openpi 공식 코드베이스(JAX) 기반, 아키텍처 무수정.
- 체크포인트: π0.5 LIBERO는 openpi 공개 체크포인트(pi05_libero), π0-FAST는 pi0_fast_base에서 H100 4장으로 자체 학습.
- 하이퍼파라미터 전체 공개 (Table 11), viewpoint 회전각 명시 (Table 12), 적응 태스크 조합 명시 (Table 13).
- 실로봇: UR10e, Meta Quest 3 텔레오퍼레이션, A100 4장(소스 학습)/A6000 1장(추론).

---

## 9. 영향 및 의의

- VLA에서 weight arithmetic의 용도를 기존의 merging(능력 결합)에서 **analogy(선택적 능력 전이)** 로 확장한 첫 체계적 연구.
- "도메인 지식이 weight space의 재사용 가능한 방향"이라는 관찰은 도메인 벡터 라이브러리 구축(카메라 shift용, 임바디먼트용 등)이라는 실용적 방향을 연다.
- 배포 현장(가정 등)에서 시연 1개로 즉시 재적응 가능하다는 점은 VLA 실배포의 핵심 병목(데이터 수집)을 직접 공략.
- LLM의 Chat Vector류 연구와 로봇 정책 적응을 잇는 가교.

---

## 10. 핵심 요약 (3줄)

1. One-shot fine-tuning된 VLA의 파라미터 변화는 태스크 방향(지배적)과 도메인 방향(소량)으로 **가법 분해**된다는 것을 subspace alignment 분석으로 규명.
2. 같은 태스크의 source/target one-shot update-vector를 빼고 SVD subspace filtering·scaling으로 정제한 **도메인 벡터**를 베이스 모델에 더하는 DART 제안 — 시연 1개로 적응 완료.
3. π0.5/π0-FAST × LIBERO viewpoint/perturbation, MimicGen cross-embodiment, 실제 UR10e 전반에서 FLA·RETAIN 등 기존 적응법 능가 (LIBERO viewpoint 평균 79.1%, 실로봇 81.7%).

---

## 11. 토론 질문

1. 도메인 벡터가 태스크 수가 훨씬 많은 대규모 generalist 모델(π0.6, GR00T)에서도 동일하게 분리될까? 태스크 공간이 커지면 태스크-도메인 직교성이 유지되는가?
2. α의 per-layer 적응적 결정(γ^(l) 기반)으로 hyperparameter-free화가 가능한가? (저자들이 future work로 남김)
3. 언어 지시 분포 변화(새 인스트럭션 스타일)도 "도메인"으로 취급해 같은 analogy로 전이 가능한가?
4. Large shift(11.3% zero-shot)에서 domain vector 품질이 낮아지는 근본 원인은 one-shot FT의 실패인가, 가법 분해 가정의 붕괴인가?

---

## 12. Q&A (세미나 예상 질문)

| # | 질문 | 답변 |
|---|---|---|
| 1 | One-shot FT가 zero-shot보다 나쁜 이유는? | 업데이트가 태스크 방향에 지배되어 해당 태스크에 과적합 + 멀티태스크 능력 망각 (Fig. 2a, Fig. 6b). |
| 2 | 왜 source update-vector를 필터링하고 target은 안 하나? | target의 고유 basis가 곧 추출하려는 도메인 방향이므로 보존; source의 misaligned basis만 노이즈원 (Sec. 5.2). |
| 3 | RETAIN과의 본질적 차이는? | RETAIN은 source 모델과 FT 모델의 머징(보간)으로 망각을 줄이는 것; DART는 뺄셈으로 도메인 방향만 분리해 더함. Medium/Large에서 격차 큼 (72.4/48.9 vs 80.8/64.4). |
| 4 | 계산 비용은? | FT 1,000스텝 × 2회 + CPU SVD 수 분 (randomized SVD 시 6m33s, Table 8). 추론 시 오버헤드 없음. |
| 5 | α는 어떻게 정하나? | LIBERO 한 suite에서 10 rollout 탐색으로 0.8 결정, 전 설정(아키텍처·실로봇 포함) 공통 사용. perturbation 설정만 0.6. |
| 6 | cross-embodiment가 왜 되나? | 임바디먼트 차이도 update-vector의 도메인 방향으로 인코딩됨 — Eq. 3의 analogy가 시각/물리 shift 구분 없이 성립 (Table 4). |
| 7 | 도메인 벡터를 여러 개 합칠 수 있나? | 가능. TA/TSV 등으로 3개 viewpoint 벡터 머징 시 단일 모델로 평균 74.5~75.7 (Table 7/21). |
| 8 | source 데모가 없으면? | source 데이터셋에서 태스크를 먼저 고르고 그 태스크의 target 데모를 수집하는 프로토콜로 해결; 근사 매칭 시 유사 태스크 검색이 무작위보다 우수 (보충 A.2, Table 22). |

<!-- VERIFIED: pdf -->
