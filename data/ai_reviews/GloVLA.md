# GloVLA: Let Geometry Move and Local VLA Interact for Robust Object-Centric Manipulation in Unstructured Environments

> **한 줄 요약**: Pick-and-place 궤적을 "기하학적 transport(비례 제어기)"와 "국소 VLA 상호작용(grasp/place 전용 로컬 정책)"으로 분해. 원본 demo에서 물체·바구니 주변 구간만 잘라 두 로컬 정책을 fine-tune. LIBERO-Object 98.4%(GR00T N1.6, FullVLA 88.8), LIBERO-Plus Object 88.2%, 자체 LIBERO-Challenge 88.5%(FullVLA 20.8), 실로봇 UR10e 90.0%(FullVLA 35.6).

---

## 1. 배경 및 동기

- 단일 end-to-end VLA는 (i) 장거리 free-space transport와 (ii) 도착 후 단시간 contact-rich 상호작용을 같은 action 분포로 동시에 학습.
- Clutter, distractor, 조명·시점 변화, 장애물, 불리한 초기 그리퍼 자세 등은 policy를 학습 분포 밖 상태로 밀어내 실패를 유발.
- 기존 대응(데이터/모델 스케일, test-time 검증, interactive post-training)은 단일 정책 인터페이스를 유지. 저자들은 "VLA가 궤적 전부를 책임질 필요는 없다"는 관점을 제시.

## 2. 방법론 심층 분석

### 2.1 분해 구조
- 입력: instruction ℓ, 관측, 로봇 상태 → 목표 물체·바구니 중심 추정(시뮬레이션은 simulator state, 실로봇은 SAM 3 open-vocabulary segmentation).
- 모드 순서: transport(물체 handoff) → grasp policy π_g → transport(바구니 handoff) → place policy π_p.

### 2.2 Sphere-Conditioned Demonstration Extraction
- 추출 영역 R_j = {p : ||p − c_j|| ≤ r_j}, j∈{object, basket}.
- Grasp clip: 그리퍼가 명령·관측 모두 open인 상태로 물체 영역에 처음 진입한 시점부터, 첫 close 명령 이후 영역을 벗어나는 시점까지.
- Place clip: grasp clip 이후 물체를 쥔 상태(명령 또는 관측 중 하나 closed)로 바구니 영역 진입 ~ 첫 release 명령 + 9 step.
- 두 로컬 정책은 같은 backbone 구조지만 파라미터를 공유하지 않고, 고정 phase instruction("pick up the target object", "place it in the basket")으로 **각각 fine-tune**.

### 2.3 Transport Controller
- Handoff 위치 p^h = ĉ_j + δ_j (δ = 수평 offset ρ + 수직 높이 z).
- LIBERO 7D OSC 공간에서 clipped P-control: a_xyz = clip(S⁻¹K_p(p^h − p_ee), −1, 1), 회전은 0, 그리퍼는 approach 시 open / 운반 시 closed.
- 충돌 검사·궤적 최적화 없음(cuRobo 등으로 대체 가능하다고 언급). Handoff 집합 H_j(허용오차 ε) 진입 또는 step budget 소진 시 모드 전환.
- Sec. III-D: ε + η + ||δ|| ≤ r 조건 하에서 모든 handoff가 추출 영역 안에 들어감을 보장.

## 3. 데이터 전략

- **추가 demo 없음**: FullVLA와 동일 source demo에서 grasp/place clip만 추출.
- Action space와 success predicate 변경 없음.
- Table V: 동일 demo 예산(10/30/50)에서 데이터 효율 비교.

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|---|---|
| Backbone | π0, π0.5, GR00T N1.6, GR00T N1.7 (주 연구: GR00T N1.6) |
| Action horizon | GR00T N1.6 50, N1.7 40, π0/π0.5 10; 쿼리당 8 action 실행 |
| 학습 | 8× A100 80GB, Adam, LR 1e-4 |
| 평가 | RTX 5090 32GB |
| 실로봇 | UR10e, SAM 3 로 물체/바구니 중심 추정 |

## 5. 실험 설계 및 평가 프로토콜

- (i) 표준 LIBERO-Object 10 task, 4개 backbone. (ii) LIBERO-Plus Object suite(7 축). (iii) 자체 제안 **LIBERO-Challenge**: LIBERO-Object 기반 50개 평가 전용 장면(easy 30: 단일 perturbation / medium 10: 2–3개 / hard 10: 4–5개), perturbation = clutter, distractor, illumination, 시점·외형 변화, 접근 경로 obstruction. 장면당 50 episode.
- 50 episode = 10 episode × 5 run, mean ± std.
- 비교 대상 FullVLA: 같은 backbone을 전체 궤적으로 학습.

## 6. 실험 결과 심층 분석 (PDF Table 직접 인용)

### LIBERO-Object (Table I, 평균)
| Backbone | FullVLA | GloVLA |
|---|---|---|
| π0 | 87.4 | **98.6** (+11.2) |
| π0.5 | 97.8 | **99.6** (+1.8) |
| GR00T N1.6 | 88.8 | **98.4** (+9.6) |
| GR00T N1.7 | 96.4 | **99.6** (+3.2) |

### LIBERO-Challenge (Table II, GR00T N1.6)
| 난이도 | FullVLA | GloVLA |
|---|---|---|
| All (50 scenes) | 20.8 | **88.5** |
| Easy | 31.4 | 93.8 |
| Medium | 8.3 | 86.5 |
| Hard | 1.5 | 74.9 |
- Perturbation별: clutter 94.5, distraction 91.7, obstruction 96.2, visual shift 90.3, illumination 96.2.

### LIBERO-Plus Object (Table IV)
- GloVLA 88.2 vs GR00T N1.6 76.6, OpenVLA-OFTm 77.1, π0-FAST 72.7. Camera 77.3, Robot 70.6, Noise 92.4(OpenVLA-OFTm 94.1보다 낮음), Layout 85.9.

### 데이터 효율 (Table V, Task 1)
- Unstructured 평균: 10 demos 45.1 vs 9.1, 30 demos 99.5 vs 19.9, 50 demos 99.9 vs 48.5.

### 실로봇 UR10e (Table VI)
- Overall 90.0% vs 35.6% (90 trials). Clean table-top 95 vs 90, medium 80 vs 10, hard 70 vs 0. 평균 추론 시간 28.2 s vs 60.1 s (2.13× 감소).

## 7. Ablation 분석

- **추출 반경 (r_o, r_b)** (Fig. 6, Alphabet Soup): 시험 범위 전체에서 80% 이상 유지.
- **Handoff offset** (Fig. 7): ρ_b, z_g, z_p는 ±8λ까지 거의 완벽, ±16λ에서 붕괴. **수평 grasp offset ρ_o가 가장 민감** — ±2λ(λ=0.005 m) 내 유지, −4λ에서 40–80%, −8λ에서 0–50%.
- 실패 모드(Fig. 5): 물체 중심 추정 오류로 handoff가 빗나감, offset이 너무 커서 로컬 정책이 transport 일부를 떠안음.

## 8. 관련 연구 비교

- **Test-time 검증/복구 계열**(단일 정책 유지)과 달리 정책이 담당하는 구간 자체를 축소.
- **Hierarchical / motion-planner + learned skill** 계열과 유사하나, VLA backbone을 수정하지 않고 demo clip 추출만으로 로컬 정책을 학습하는 model-agnostic 레시피라는 점이 차별점.
- LIBERO-Plus 비교는 Object suite에 한정되어, 4-suite 전체를 평가하는 일반 VLA와 직접 비교는 불가.

## 9. 한계 및 미해결 문제

- 시뮬레이션에서 **ground-truth 물체·바구니 중심**을 사용 → perception 난이도가 제거된 상태의 robustness. 실로봇은 SAM 3에 의존.
- Pick-and-place(물체→바구니) 구조에 특화; 관절형 물체, 도구 사용, 다단계 long-horizon 태스크 일반화는 미검증. LIBERO-Object 외 suite 평가 없음.
- Transport 제어기는 충돌 회피·자세 제어 없음(회전 0).
- LIBERO-Challenge는 저자 자체 벤치마크이며 FullVLA 외 외부 baseline 비교 없음.
- Handoff offset(특히 ρ_o)에 민감.

## 10. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — planner + local skill 분해는 고전적이지만 VLA 시대에 간결한 레시피로 재정립 |
| **Technical depth** | ★★★☆☆ — clip 추출 조건과 handoff 보장 조건이 명확 |
| **Experimental rigor** | ★★★☆☆ — 4 backbone, 실로봇 90 trials; 단 Object suite·GT 중심 사용으로 범위 제한 |
| **Practical impact** | ★★★★☆ — 추가 데이터 없이 robustness·추론 시간 대폭 개선 |
| **Writing quality** | ★★★★☆ |

**강점**: 단순하고 backbone 비의존적, 강한 분포 이동에서의 큰 향상(20.8 → 88.5), 데이터 효율.
**약점**: 태스크 구조(pick-and-place)와 물체 위치 정보에 대한 강한 가정, 표준 벤치마크 커버리지 부족.

## 11. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|---|---|
| 1 | 시뮬레이션에서 GT 물체 위치를 쓰는 건 불공정하지 않나? | 맞음 — FullVLA는 위치 정보를 받지 않음. 실로봇에서 SAM 3로 대체해 90.0% 유지한 것이 부분적 반론. |
| 2 | 이게 VLA 모델인가, 파이프라인인가? | 두 로컬 VLA 정책을 clip 데이터로 직접 fine-tune하므로 학습된 정책 + 고전 제어기의 hybrid. |
| 3 | 왜 LIBERO-Object만? | 물체→바구니 pick-and-place 구조가 가정에 맞는 suite. Goal/Long/Spatial로의 확장은 미제시. |
| 4 | 가장 민감한 파라미터는? | 수평 grasp offset ρ_o (±2λ 이내만 안전). |
| 5 | 강한 backbone에서도 이득이 있나? | π0.5 97.8→99.6, N1.7 96.4→99.6 — 표준 조건에선 작지만, 분포 이동 하에서 격차가 커짐. |
| 6 | 추론 시간 감소 이유? | Free-space 운반 동안 VLA 추론을 하지 않음(60.1 s → 28.2 s). |

## 12. 재현성 및 후속 연구 제안

- 프로젝트 페이지: https://glovla-project.github.io/ (코드 공개 여부는 논문에 명시 없음).
- 재현에 필요한 하이퍼파라미터(r, δ, ε, K_p)는 Figure 2 및 본문에 요약.
- 후속 방향: (1) 학습 기반 handoff 위치 예측, (2) 충돌 인지 motion planner(cuRobo) 통합, (3) 다단계·관절형 물체 태스크로의 일반화, (4) perception 오류에 강건한 handoff 보정.

<!-- VERIFIED: pdf -->
