# VLM2VLA-Prune: Revisiting Parameter Redundancy in Vision-Language-Action Models — Insights from VLM-to-VLA Adaptation

> **한 줄 요약**: VLM→VLA 적응 과정에서 생기는 파라미터 발산 ΔW_rel = ‖W_VLA − W_VLM‖₂/‖W_VLM‖₂를 attention head / FFN channel 단위 구조 신호로 삼아, **fine-tuning·저랭크 보정 등 어떤 회복 절차도 없이(recovery-free)** OpenVLA와 π0.5를 12–30% 압축하는 multi-module joint pruning. LIBERO에서 π0.5-Light 93.3% (base 96.9%), OpenVLA-Light 70.4% (base 76.5%)를 유지하는 반면, 동일 recovery-free 조건에서 LLM-Pruner 1.0%, FLAP 0.2%, Wanda 7.1%로 전면 붕괴. ECCV 2026.

---

## 1. 배경 및 동기

### 문제의식: "회복이 필요한 pruning은 redundancy 식별에 실패한 것"
- VLA는 VLM의 거대한 파라미터를 그대로 상속 → 배포 부담이 크지만, **파라미터 제거에 극도로 민감**해 중간 수준 pruning에도 성공률이 절벽처럼 붕괴("prune-then-collapse")
- 기존 계열의 대응은 사후 보정:
  - **RLRC** (arXiv 2506.17639) — LLM-Pruner/Taylor 기준으로 구조적 pruning 후 SFT+RL로 회복. SFT 없이는 RL 2M step으로도 회복 불가라는 자체 보고
  - **GLUESTICK** (arXiv 2510.08464) — 가중치 차이의 SVD 주성분으로 추론 시 보정항 추가. fine-tuning은 피하지만 본질적으로 post-hoc remedy
- 저자들의 근본 질문: **"pruning 후 성능 회복이 필수라면, 제거된 파라미터가 정말 redundant했는가?"** → 회복 의존 패러다임은 vital 파라미터의 무차별 삭제("falsely killed")를 은폐한다는 주장

### 4개 가설 (검증 구조)
- **H1**: 회복이 필요한 pruning은 redundant하지 않은 파라미터를 제거한 것
- **H2**: VLM→VLA 적응 중 파라미터 차이 ΔW는 redundancy 식별에 유용한 신호를 담는다
- **H3**: ΔW 신호의 유용성은 모듈마다 다르다 (module heterogeneity)
- **H4**: 올바르게 활용하면 회복 없는 구조적 pruning이 가능하다

📌 [Figure 1/2 삽입] — Prismatic↔OpenVLA, PaLI-Gemma↔π0.5 쌍의 head/channel 단위 ΔW_rel 히트맵

---

## 2. 방법론 심층 분석

### 2.1 분석 프레임워크: pruning을 최적화가 아닌 진단 도구로
- 통상 목적: max_M S(f(·; P(W;M))) s.t. ‖M‖₀ ≤ k (Eq. 1)
- 그러나 사후 fine-tuning T(·)가 개입하면 평가 대상이 f(·; T(P(W;M)))가 되어, **남은 파라미터의 내재적 기능이 아니라 모델의 재학습 능력을 측정**하게 됨
- 따라서 회복 없는 pruning을 "controlled intervention"으로 재정의 — 회복 필요성 자체가 redundancy 식별 실패의 신호

### 2.2 ΔW_rel: 구조 신호로서의 파라미터 발산 (Eq. 2)
- ΔW_rel = ‖W_VLA − W_VLM‖₂ / ‖W_VLM‖₂, 공유 백본 subspace에서 계산
- **OpenVLA (Prismatic 기반)** 관찰:
  - Llama-2: "3단계" 수직 분포 — L0 밀집 보정(멀티모달 융합), L1–L23 안정, L24–L31 재변동(action semantic 매핑). FFN은 channel 축 "strip" 희소성 → embodied 지식이 특정 sub-channel에 국소 인코딩
  - DINOv2: 얕은 층 위주 갱신(파지·회피 등 low-level 시각 단서), SigLIP: 깊은 층 반응·FFN 갱신량 전반적으로 큼(semantic 보조)
  - Projector: fc2/fc3(LLM 입구 쪽)이 fc1보다 훨씬 크게 변동 — 단조 증가 패턴
- **π0.5 (PaLI-Gemma 기반)**: Gemma MQA에서 L0 최소 발산(텍스트 파싱 prior 유지), L1–L9 대발산(embodied attention 재조직), L10–L17 안정화. Vision FFN은 중간(L3–L15)·고층(L25–L26)에 강한 신호

### 2.3 Controlled pruning probe와 Algorithm 1
- 두 상보 전략: **Highest-diff** M^high(r) = |ΔW| 상위 r% 제거 vs **Lowest-diff** M^low(r) = 하위 r% 제거, fine-tuning 없이 직접 평가
- Algorithm 1 (Module-Aware Mask Construction): 모듈별로 ΔW의 L2 norm을 aggregation 차원(입력측 d=1, 출력측 d=0)에 맞춰 계산 → LLM FFN은 gate/up/down projection에 걸쳐 channel 정렬 누적, attention은 head 단위 평균 → 전역 threshold τ로 unit mask 생성 → FFN은 fc1/up/gate 출력과 fc2/down 입력을 동시에 마스킹하는 intra-module broadcast

### 2.4 Multi-module joint pruning (Table 12)
| 모듈 | OpenVLA Light/Mod/Agg | π0.5 Light/Mod/Agg |
|------|----------------------|--------------------|
| LLM-Attn | 0.125 (H) 고정 | 0.2 (H) 고정 |
| LLM-FFN | 0.1 / 0.2 / 0.3 (H) | 0.2 / 0.3 / 0.5 (H) |
| SigLIP-Attn | 0.125 (H) | 0.2 (H) |
| SigLIP-FFN | 0.1 (**L**) | 0.4 / 0.2 / 0.2 (H) |
| DINOv2-Attn | 0.0625 (**L**) | – |
| DINOv2-FFN | 0.1 (H) | – |
| Projector | **제거 금지 (보호)** | **보호** |

핵심: 모듈마다 H/L 기준이 뒤바뀌는 **차등 선택 논리**가 방법의 본체. 단일 전역 기준(magnitude, Taylor, Wanda류)이 VLA에서 실패하는 이유를 직접 겨냥.

---

## 3. 데이터 전략

- **평가**: LIBERO 4 sub-suite (Spatial / Object / Goal / Long), Franka Panda, 자연어 지시 기반 조작, Success Rate 지표
- **모델 쌍**: ⟨Prismatic, OpenVLA⟩ (Llama-2 7B 고전 아키텍처) + ⟨PaLI-Gemma, π0.5⟩ (모듈형 경량 아키텍처) — 두 세대의 대표 설계를 모두 커버
- **일반화 검증 (Appendix D)**: π0를 RoboTwin2.0 5개 태스크(Beat Block Hammer, Move Can Pot, Shake Bottle, Place Phone Stand, Rotate QRcode)에서 LLM FFN 10% 프루닝으로 probe
- 학습 데이터 없음 — 방법 자체가 training-free. Sec 4.2의 LoRA 회복 실험만 예외(반증 목적)

---

## 4. 시스템/학습 세부사항

| 항목 | OpenVLA | π0.5 |
|------|---------|------|
| Baseline | 7.5B / 14.9GB / LIBERO avg 76.5% | 3.6B / 7.3GB / avg 96.9% |
| Ours-Light | 6.6B / 13.0GB | 3.0B / 6.1GB |
| Ours-Moderate | 6.2B / 12.4GB | 2.8B / 5.6GB |
| Ours-Aggressive | 5.7B / 11.3GB | 2.5B / 5.0GB |

- 하드웨어: NVIDIA A100. 진단 실험은 pruning 직후 direct inference
- 회복 대조 실험(Sec 4.2)만 LoRA + FSDP, 10k steps, LR 1e-4
- 구조 제원(Table 8): DINOv2 1024d×24L×16H, SigLIP 1152d×27L×16H, Llama-2 4096d×32L×32H (d_int 11008); Gemma 2048d×18L×8H MQA (Q 8 head, K/V 1 head, d_int 16384)
- Projector: OpenVLA 2-layer MLP (2176→4096), π0.5 단일 linear (1152→2048)

---

## 5. 실험 설계 및 평가 프로토콜

가설별 1:1 대응 설계:
- **H1 검증 (Sec 4.2)**: Llama-2 FFN에 Lowest/Highest/Random × {20,50,80%} pruning → LoRA 전/후 SR 비교. "회복이 pruning 품질을 은폐하는가"
- **H2 검증 (Sec 4.3)**: OpenVLA·π0.5 전 모듈에 H vs L 대조 pruning → ΔW 신호의 유효성
- **H3 검증 (Sec 4.4)**: Projector 취약성, SigLIP 극한(ratio 1.0 = 완전 제거) robustness → 신호 유용성의 모듈 의존성
- **H4 검증 (Sec 4.5)**: Light/Moderate/Aggressive 조인트 스킴 vs LLM-Pruner·FLAP·Wanda — **모두 동일한 recovery-free 조건**에서 비교 (기존 논문 보고치가 아닌 재평가)
- 통제된 인과 사슬: 발산 신호 → controlled intervention → 직접 추론 평가 → cross-module/cross-model 비교

---

## 6. 실험 결과 심층 분석

### H1: 회복의 역설 (Table 1, LIBERO-Spatial, base 84.7%)
| 전략 | Ratio | Pre-FT SR | Post-FT SR |
|------|-------|-----------|------------|
| Lowest-diff | 20% | **1.5** | 86.5 |
| Lowest-diff | 80% | 0.0 | 76.4 |
| Highest-diff | 20% | **76.3** | 85.8 |
| Highest-diff | 50% | 20.5 | 84.1 |
| Random | 20% | 12.2 | 86.0 |

- **모든 구성이 LoRA 후 baseline 수준으로 회복** (0.0%에서조차) → 회복 성능은 pruning 품질과 무관, "강한 보상"이 잘못된 삭제를 은폐
- Fig. 3: 회복 수렴 step이 pruning ratio에 비례해 증가 → 높은 ratio일수록 더 깊은 구조 손상

### H2: 모듈 이질성과 감도 역전 (Table 2/3)
- **OpenVLA DINOv2 "sensitivity reversal"**: Attn은 high-diff 제거 시 붕괴(1.6%) / low-diff는 76.7% 유지, FFN은 정반대(H 82.0% vs L 0.0%) — 같은 백본 안에서도 경로 의존적
- **Llama-2**: low-diff head/channel 제거 시 전멸(0.0% / 2.7%), high-diff 제거는 84.3% / 72.0% 유지 → 안정된(변화 적은) 파라미터가 cross-modal 정렬의 핵
- **π0.5**: Gemma Attn low-diff 제거 0.0%, FFN high-diff 50% 제거해도 95.0% — 모듈형 설계에서 신호가 더 선명
- SigLIP은 어느 방향이든 둔감 → 보조적 semantic 역할과 부합

### H3: 신호의 경계 (Table 4)
- **Projector = "fragile & non-selective"**: ratio 0.3부터 H/L 모두 붕괴(0.0%/54.0%), 0.5에서 완전 소멸 → 병목 인터페이스라 무조건 보호 대상
- **SigLIP 극한 robustness**: ratio 1.0(완전 제거)에도 FFN 70.0%, Attn 47.0% 유지. DINOv2와 극명 대조 — DINOv2가 주 구조 표상, SigLIP은 보조

### H4: Recovery-free 조인트 pruning (Table 5/6)
| OpenVLA 변형 | Params | Mem | Spatial | Object | Goal | Long | Avg |
|--------------|--------|-----|---------|--------|------|------|-----|
| Baseline | 7.5B | 14.9 | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| LLM-Pruner | 6.2B | 12.4 | 23.4 | – | – | 1.0 | – |
| FLAP | 6.3B | 12.5 | 0.2 | – | – | 0.0 | – |
| Wanda (full sparse) | – | 10.2 | 0.0 | 13.4 | 0.8 | 0.0 | 7.1 |
| Wanda (sparse lang. BB) | – | 10.6 | 31.2 | 50.8 | 20.0 | 12.4 | 28.6 |
| **Ours-Light** | 6.6B | 13.0 | 78.3 | 82.5 | 74.0 | 46.8 | **70.4 (−6.1)** |
| **Ours-Moderate** | 6.2B | 12.4 | 70.5 | 74.9 | 64.7 | 39.0 | **62.3 (−14.2)** |
| **Ours-Aggressive** | 5.7B | 11.3 | 59.0 | 65.7 | 56.0 | 29.5 | **52.5 (−24.0)** |

- **동일 용량(12.4GB) 매칭 비교**: Ours-Moderate 62.3% vs LLM-Pruner 1.0% — 60%p 격차
- **π0.5 (Table 6)**: Light 93.3% (−3.6), Moderate 89.0% (−7.9), Aggressive 82.1% (−14.8); 메모리 7.3→5.0–6.1GB. Light는 base 96.9% 대비 96% 성능 유지
- **RoboTwin2.0 일반화 (Table 13, π0)**: LLM FFN 10% 프루닝 시 high-ΔW 제거 47.4% vs low-ΔW 제거 10.6% (원본 56.0%) — 다른 VLA·벤치마크에서도 ΔW 신호 유효

---

## 7. Ablation 및 진단 분석

- **H vs L 대조 자체가 전면적 ablation**: 7개 모듈 × 2 전략 × 다중 ratio (Table 2/3/4, 부록 Table 10/11에서 4개 suite 전부로 확장) — "sensitivity reversal"이 suite 불문 일관됨 확인
- **Ratio sweep**: Projector 0.2(84.5%)→0.3(0.0%)의 급락 지점 특정; Llama FFN H는 20%(72.0)→50%(20.5)→80%(0.0)의 점진 저하로 buffer 존재
- **Random 대조군**: 20%에서 12.2%로, H(76.3%)와 L(1.5%)의 중간 — ΔW 순위가 무작위 대비 유의미함을 직접 입증
- **회복 수렴 곡선 (Fig. 3)**: pruning ratio ↑ → 수렴 step ↑, "재학습 난이도"를 구조 손상의 proxy로 사용한 인과 분석
- 단, LLM-Pruner/FLAP의 Object/Goal 칸이 공란이고 π0.5 대비 baseline 부재 등 비교표의 완성도는 아쉬움

---

## 8. 관련 연구 비교

| 계열 | 대표 | 회복 필요 | 관점 |
|------|------|----------|------|
| Pruning+회복 | RLRC (SFT+RL), GLUESTICK (SVD 보정) | ✓ (본질) | "어떻게 회복하나" |
| LLM pruning 기준 | LLM-Pruner, FLAP, Wanda | LLM에선 미미, VLA에선 필수 | 단일 전역 중요도 |
| 층/토큰/양자화 | MoLe-VLA, EfficientVLA, SP-VLA, BitVLA, SQAP-VLA | 다양 | 직교적 효율화 |
| VLM-VLA 기제 분석 | VLM4VLA, Actions as Language | – | 최종 성능 비교 중심 |
| **본 논문** | **ΔW 기반 joint pruning** | **✗ (recovery-free)** | **적응 과정 자체를 redundancy 기준으로** |

- 본 저장소의 **CLP**와 좋은 대비: CLP는 CKA 유사도로 **layer 단위** 정적 제거 후 fine-tuning으로 보완, 본 논문은 ΔW로 **head/channel 단위** 제거 후 **회복 자체를 금지**. 관점상 GLUESTICK·RLRC의 전제("성능 손실은 불가피")를 정면 반박하는 position + method 하이브리드
- SAFE-Pruner·ActQuant 등 저장소 내 압축 계열과 함께 "VLA efficiency" 클러스터 형성

---

## 9. 한계 및 미해결 문제

### 구조적 한계
1. **성능 손실이 작지 않음**: "~90% 유지"는 Light 기준. OpenVLA-Moderate는 −14.2%p, Aggressive는 −24.0%p — recovery-free라는 제약의 대가가 명확. 실배포에선 "우리 방식으로 pruning 후 가벼운 회복"이 최적일 수 있는데 그 조합 실험이 없음
2. **Table 12의 H/L 배치가 사후적(post-hoc)**: 진단 실험 결과를 보고 모듈별 기준을 수동 선택 — 새 VLA에 적용하려면 모듈별 probe를 다시 돌려야 하며, 자동 선택 규칙이 없음
3. **VLM 원본 가중치 필수**: W_VLM 접근이 안 되는 (또는 다단계 적응을 거친) VLA에는 ΔW 자체를 정의할 수 없음
4. **LIBERO 편중**: 주 결과가 시뮬레이션 단일 벤치마크. RoboTwin2.0은 부록의 10% probe뿐이고 joint 스킴 미적용, real-world 실험 전무
5. **속도 지표 부재**: 파라미터/메모리 절감만 보고, 추론 latency·throughput 수치가 없음 — structured pruning이라 speedup이 기대되지만 미측정
6. **Wanda 비교의 공정성**: Wanda는 unstructured sparsity라 메모리 수치(10.2GB)와 구조적 제거의 직접 비교가 다소 애매
7. **π0.5 baseline SR 98.8%(Spatial)은 이례적으로 높음** — 평가 episode 수/seed 미공개로 분산 추정 불가

### 열린 질문
- ΔW 신호가 task 분포에 얼마나 민감한가 (LIBERO fine-tune 데이터가 곧 적응 데이터인 순환 구조)
- Attn vs FFN에서 H/L이 역전되는 기제의 이론적 설명 부재 — 경험적 관찰에 그침

---

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — "회복 의존 = redundancy 식별 실패"라는 문제 재정의와 ΔW를 pruning 기준으로 승격한 발상이 신선. 개별 기법(magnitude 기반 mask)은 고전적 |
| **Technical depth** | ★★★☆☆ — Algorithm 1은 단순하나, 4-가설 인과 검증 구조가 분석의 깊이를 보강 |
| **Experimental rigor** | ★★★★☆ — 2개 모델 쌍 × 7개 모듈 × H/L 대조 × 4 suite 전수. 다만 baseline 표 공란과 real-world 부재 |
| **Practical impact** | ★★★☆☆ — 12–30% 압축을 zero-recovery로 얻는 것은 엣지 배포에 실질적. 그러나 Moderate 이상의 손실폭과 latency 미측정이 발목 |
| **Writing quality** | ★★★★☆ — 가설 구동형 서사가 명확, position paper와 method paper의 균형 |

**강점**: (1) recovery-free라는 엄격한 평가 프로토콜로 기존 pruning 기준의 허상을 실증 (2) sensitivity reversal, SigLIP 완전 제거 가능, projector 병목 등 VLA 내부 구조에 대한 재사용 가능한 발견 (3) 동일 용량 비교에서 60%p 격차라는 압도적 마진 (4) 코드 공개.

**약점**: 모듈별 H/L 선택의 수동성, 회복 병용 시나리오 미탐구, real-world·latency 검증 부재.

**VLA-Tracker 관점**: 순수 분석이 아니라 **배포 가능한 pruned policy(Ours-Light/Moderate/Aggressive) + LIBERO 4-suite 정량 결과**를 내는 압축 방법론이므로 CLP·SAFE-Pruner와 같은 계열로 추적 가치가 충분.

---

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | ΔW가 크다 = 중요하다인가, 작다 = 중요하다인가? | 모듈마다 다르다는 것이 논문의 핵심(H3). Llama-2/Gemma는 **low-diff가 vital**(제거 시 0%), DINOv2 Attn은 high-diff가 vital, DINOv2 FFN은 low-diff가 vital. 단일 방향 해석은 성립하지 않으며, 그래서 Table 12의 차등 기준이 필요 |
| 2 | 왜 LLM에서 변화가 적은 파라미터가 더 중요한가? | 저자 해석: 적응 중 "안정화된" 파라미터가 cross-modal 정렬·일반 언어 prior를 보존하는 경로이고, 크게 변한 파라미터는 task-specific 재조정분이라 일부 제거해도 잔여 경로가 기능 유지. 다만 이론적 증명은 없고 경험적 |
| 3 | LoRA로 다 회복되면 실용적으론 아무 기준이나 써도 되는 것 아닌가? | Fig. 3에서 나쁜 pruning일수록 수렴 step이 급증 → 회복 비용이 다름. 또 회복 후 성능(76.4~86.5%)도 ratio에 따라 계단식 저하. 무엇보다 회복 인프라(GPU, 데이터) 없는 엣지 시나리오에선 recovery-free가 유일한 선택지 |
| 4 | SigLIP을 통째로 제거해도 70%면 SigLIP은 왜 두나? | ratio 1.0에서 FFN 70.0/Attn 47.0으로 "동작은 하지만 손실이 있는" 수준(base 84.7). Light 구성이 SigLIP-FFN 10%만 L-제거하는 이유. 완전 제거는 진단용 극한 실험 |
| 5 | 동일 용량 비교(12.4GB)에서 LLM-Pruner가 1.0%인 게 과장 아닌가? | LLM-Pruner는 원래 회복(fine-tuning)을 전제로 설계된 기준이므로 recovery-free 평가는 불리한 조건이 맞음. 논문의 논지는 "그 기준들은 회복 없이는 redundancy를 못 찾는다"이지 "그 방법들이 무용하다"가 아님 — 프레이밍 주의 필요 |
| 6 | Projector가 그렇게 취약하면 향후 VLA 설계에 주는 시사점은? | 병목 인터페이스에 용량 여유(redundancy)가 거의 없다는 뜻 → projector를 넉넉히 키우거나(강건성), 반대로 pruning 대상에서 항상 제외하는 설계 원칙. π0.5의 단일 linear projector(2 tensor)도 동일하게 보호됨 |
| 7 | 새 VLA(예: GR00T, CogACT)에 적용하려면? | (i) 대응 VLM 가중치 확보 (ii) 모듈별 ΔW 계산 (iii) H/L probe를 소규모로 재실행해 모듈별 기준 결정 (iv) joint 구성. probe 재실행이 필요하다는 점이 자동화의 병목 — RoboTwin2.0의 π0 실험(Table 13)이 신호 자체의 이식성은 지지 |
| 8 | OpenVLA(-6.1%p)와 π0.5(-3.6%p)의 내구성 차이 원인은? | π0.5가 모듈형·경량 설계라 ΔW 신호가 더 규칙적(Fig. 2)이고, MQA head 단위 신호의 판별력이 높음. 또 baseline margin(96.9%)이 커서 절대 손실 흡수 여지가 큼 |
| 9 | Wanda의 unstructured sparsity와 구조적 제거를 같은 표에서 비교해도 되나? | 메모리 기준 정렬로 어느 정도 통제했지만, sparse 텐서는 실제 latency 이득이 다르므로 완전히 공정하진 않음. 저자들도 Wanda 3개 변형(full/lang-BB/75%)을 제시해 스펙트럼을 보였으나 latency 열이 없어 아쉬움 |
| 10 | 이 방법으로 pruning 후 가볍게 fine-tuning하면 SOTA 압축이 되나? | 논문이 의도적으로 배제한 조합. Table 1 논리대로면 "좋은 초기 제거 + 짧은 회복"이 최소 step으로 최고 성능에 도달할 가능성이 높음 — 가장 자연스러운 후속 연구 |
| 11 | RoboTwin2.0 실험은 왜 joint 스킴이 아닌 10% FFN probe만? | 부록 D는 ΔW 신호의 cross-benchmark 유효성 확인이 목적(high 47.4 vs low 10.6). π0에 대한 모듈별 진단·Table 12급 구성 탐색은 미수행 — joint 스킴의 일반화는 미검증으로 남음 |
| 12 | "90% 성능 유지" 주장의 정확한 범위는? | π0.5-Moderate 89.0/96.9 = 91.8%, OpenVLA-Light 70.4/76.5 = 92.0%가 근거. OpenVLA-Moderate(81.4%)·Aggressive(68.6%)는 90%에 못 미치므로, abstract의 "12–30% 압축 + ~90% 유지"는 구성별로 나눠 읽어야 함 |

<!-- VERIFIED: pdf -->
