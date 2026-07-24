# DiMaS: Distribution Matching for Steering Vision-Language-Action Models

> **한 줄 요약**: Flow-matching VLA(SmolVLA, π0.5)의 action expert 내부 표현을 고정된 선형 방향으로 이동시키는 대신, 최적 수송(optimal transport)으로 두 표현 **분포**(source D⁻ → target D+) 사이를 수송하는 추론 시점 스티어링 래퍼. 속도·수직 변위 같은 행동 특성을 task 성공률을 유지한 채 제어한다.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 표현 스티어링(representation steering)은 LLM/VLM에서 정렬·생성 스타일 제어에 효과적이며, 대부분 **선형 표현 가설(LRH)** — 의미 특성이 활성 공간의 선형 방향으로 인코딩된다는 가정 — 에 기반한다.
- 그러나 최신 SOTA VLA는 이산 토큰이 아니라 **flow-matching action expert**로 연속 궤적을 생성한다. LLM/VLM의 선형 스티어링 레시피가 이 visuomotor 세팅에 그대로 전이되는지 불명확하다.
- 기존 VLA 해석/스티어링 연구(Häon et al. — autoregressive VLA만 대상; Buurmeijer et al. — 회귀 계수를 스티어링 벡터로 사용하나 **성공을 위해 개입을 중단해야 하는 한계**)는 flow-matching VLA를 다루지 못하거나 개입 중단이 필요하다.

### 핵심 질문
- Flow-matching VLA의 내부 표현에 개입해 예측 궤적의 **특정 행동 특성**(how, 즉 어떻게 수행하는가)을 제어할 수 있는가?
- 왜 고전적 선형 스티어링이 VLA에서 실패하는가?

📌 [Figure 1 삽입] — DiMaS 학습/추론 파이프라인 개요

---

## 2. 방법론 심층 분석

### 2.1 VLA 아키텍처와 표기
VLA f = (f_V, f_A): f_V는 VLM 백본, f_A는 action expert. 관측 x_t = (I_t, T, x_t^s)에서 f_A는 flow-matching 목적함수로 노이즈 액션 a_{0,t} ~ N(0,I)를 M단계 denoising하여 정제한다 (식 1). 개입 대상은 잔차 스트림 표현 h^l_{p,m}(x_t) — 레이어 l, 토큰 위치 p, denoising step m.

### 2.2 행동 특성(behavioral feature)
각 예측 액션 a_i = (Δx, Δy, Δz, …, gripper)에서 스칼라 특성 φ_i를 유도. 예: 속도 = √(Δx²+Δy²+Δz²). 각 표현 h_i에 φ_i를 연결한다.

### 2.3 Source/Target 분포
특성의 경험적 분위수로 tail을 잘라 분포를 구성: D⁻ = {h_i : φ_i ≤ q_τ}, D+ = {h_i : φ_i > q_{1−τ}}. 중앙값이 아닌 tail 분할로 특성 부재/존재 집단을 깨끗이 분리해 매핑을 선명하게 한다.

### 2.4 수송 맵으로서의 스티어링
스티어링 = D⁻를 D+로 옮기는 맵 T. 선형 스티어링은 T(h)=h+(μ⁺−μ⁻) 같은 고정 방향 가산 이동. DiMaS는 T를 **Kantorovich 최적 수송 맵**(식 3, W₂²)으로 인스턴스화하여 단일 방향이 아니라 분포 전체의 기하를 존중한다. 유한 표본에서는 이산 OT로 환원되어 **low-rank Sinkhorn**(POT 패키지)으로 효율적으로 푼다.

### 2.5 테스트 시점 개입
표현 h를 최근접 이웃 투영 P(h)=argmin_{z∈D⁻}‖z−h‖로 D⁻에 투영 후 T 적용. 선형 프로브 이진 분류기 g로 게이팅: 특성이 없을 때(g=1)만 개입, 이미 있으면(g=0) 원본 유지.

📌 [Figure 6 삽입] — VLM vs action expert 선형 분리도, 선형 스티어링 분포 매칭 실패 시각화

---

## 3. 핵심 기여: Interpolation "비법"과 성공률 보존

완전 수송(α=1)은 특성은 제어하나 원 표현에서 벗어나 closed-loop에서 task 실패를 유발한다. 이를 완화하기 위해 D⁻/D+ 사이를 보간 계수 α∈[0,1]로 조절 (식 4):

h ← (1−α)·h + α·(T∘P(h))  if g(h)=1,  else h.

α=0은 무개입, α=1은 완전 수송. α는 사용자에게 **정규화된 제어 강도 다이얼**을 제공하며, DiMaS에서는 성공률에도 직접 영향을 미친다는 점이 특징이다. 기본값 α=0.5.

---

## 4. 실험 설정

- **모델**: SmolVLA (256M SmolVLM 백본 + flow-matching expert, `lerobot/smolvla_libero`), π0.5 (PaliGemma-3B + 300M flow-matching expert, `lerobot/pi05-libero`). 둘 다 10 flow-matching step. SmolVLA는 매 timestep 50-chunk 예측 후 첫 액션만 사용, π0.5는 10액션을 직접 실행해 더 부드럽다.
- **벤치마크**: LIBERO — Object, Spatial, Goal(짧은 호라이즌), 10/Long(긴 호라이즌). 각 suite 10 task × 50 초기화.
- **타깃 특성**: 모션 속도, end-effector 수직 변위.
- **지표**: 성공률(SR), 특성값, paired t-test 유의성.

---

## 5. 베이스라인 비교

세 부류: (i) Mean-difference 스티어링(평균차 가산), (ii) Regression 기반 스티어링([5], VLM/FM 모두 평가), (iii) Prompt injection("더 빠르게/느리게" 재구성). (i),(ii)는 선형 방법으로 선형 분류기로 게이팅.

**결과**: 선형·프롬프트 베이스라인은 일관성 부족 — 특성을 못 바꾸거나, 증가/감소 개입이 같은 방향으로 이동. DiMaS는 양방향 제어 성공(두 모델 속도, π0.5 수직 변위). 속도 조절 시 성공률 대체로 유지, 수직 변위는 task 완료와 결부되어 하락(예: LIBERO-Object에서 수직 변위 감소 시 물체를 못 들어 실패). 감소 방향 개입이 더 유의한 shift.

📌 [Figure 2, 3 삽입] — 속도/수직변위 스티어링 (ΔSpeed/Δz vs ΔSR)

---

## 6. 일반화 분석 (4가지 세팅)

LIBERO의 axis-isolated 구조로 (1) task 다양성, (2) 분포 이동 심각도를 변화:
- **Setting 1**: 같은 task의 held-out 초기 상태.
- **Setting 2**: suite 내 모든 task의 held-out 초기 상태(통상적 VLA/스티어링 평가 세팅).
- **Setting 3**: suite 내 held-out **task** (미학습 task 전이).
- **Setting 4**: 분리된 suite 간 전이 (가장 어려움).

**발견**:
- 다수 task 집계가 도움(Setting 2 > 1): 넓은 표현 공간 구조 포착.
- Held-out/분리 task에서도 붕괴하지 않음(Setting 3–4). π0.5 속도 감소는 Setting 3까지 유의, Goal로의 교차 전이(Setting 4)도 방향은 유지(p=0.032).
- 가장 다양한 suite인 **LIBERO-Goal이 가장 약함**(push, stove 조작 등 이질적) — 유일하게 p<0.01 미달 셀 존재(SmolVLA Setting 1 p=0.09; π0.5 Setting 4 p=0.032).

📌 [Figure 4 삽입] — 평가 레벨별 속도 스티어링 box plot

---

## 7. 긴 호라이즌 결과 (LIBERO-10)

π0.5에서 속도 감소: 하향 shift Δmean=−0.024, p<0.01 (α=0.5), 성공률 96%→90%의 작은 비용. α별로 90%(α=0.5)→81%(0.6)→72%(0.7)→46%(0.8)→7%(0.9)로 강도-성공률 트레이드오프가 뚜렷. 확장된 호라이즌에서도 행동 제어 유효.

📌 [Figure 5 삽입] — LIBERO-10 속도 density

---

## 8. VLA 내부 구조 분석 (왜 선형 스티어링이 실패하는가)

- **선형 분리도**: action expert 표현은 후반 FM step에서 거의 완전 선형 분리(≈100%). FM step 0에서도 깊은 레이어는 >93%로, VLM 최대 87%보다 높음 → **VLM이 아닌 action expert의 깊은 레이어에 개입**하는 설계 근거.
- **선형 이동이 분포를 매칭하는가?**: l=0, m=8(분리도 100%)에서 고속 표현을 단위 스티어링 벡터×β(2/50/300)로 이동해 2D PCA로 시각화. 고속(red)·저속(blue) 분포의 "모양"이 달라 고정 방향 이동은 두 분포를 못 맞춘다. 즉 **행동 특성은 선형 decodable이지만 선형 steerable은 아님** → 분포 매칭(DiMaS)의 필요성.

---

## 9. DiMaS 분석 (α 및 정성 결과)

- **α ablation**: α 증가 → 평균 속도 감소, 성공률 비용 증가. π0.5는 α=0.5 이후에야 성공률 하락(더 강건), SmolVLA는 더 이르고 불규칙하게 저하. α=0.5가 유리한 트레이드오프 → 기본 운영점.
- **정성**: 수직 변위 감소 스티어링에서 steered 궤적(red)이 원본(blue)보다 일관되게 작은 수직 변위. 실패 사례(수직 모션 억제로 task 미완료)도 제시.

📌 [Figure 7, 8 삽입] — α 트레이드오프, 수직 변위 정성 궤적

---

## 10. 하이퍼파라미터 (Table A1)

전 실험에서 고정: 분위수 q⁻=0.25/q⁺=0.75, 게이트 분류기 SVM(선형, C=0.1), OT는 low-rank Sinkhorn(ε=1e−4, 최대 5000 iter, rank=min(n,m)), α=0.5, 레이어=second-to-last, FM step=전부. 유일하게 base VLA에 의존하는 선택은 레이어 ℓ(후반 레이어가 일반적으로 유효, second-to-last 사용).

---

## 11. 한계 및 향후 연구

- 현재 모든 timestep을 동일 취급 → **언제** 개입할지(예: 회전이 아닌 병진 중에만)도 학습 대상으로 삼는 선택적 스티어링.
- 청크 액션(π0.5)에서는 하나의 출력 액션이 여러 표현에 의해 형성되므로, 정확한 대응 액션 특성으로 그룹핑하는 것은 근사에 불과 — 더 정밀한 대응 모델링 필요.
- 표현에서 관측 가능한 임의 특성에 적용 가능하므로 새로운 task·embodiment·추상적 행동 특성으로 확장 가능성.

---

## 12. 총평

DiMaS는 flow-matching VLA에 특화된 최초의 **분포 매칭 기반** 추론 시점 스티어링으로, "무엇을(what)"이 아니라 "어떻게(how)" 로봇이 task를 수행하는지를 제어하는 실용적 인터페이스를 제시한다. 핵심 통찰인 "선형 decodable ≠ 선형 steerable"은 VLA 해석가능성 연구에 개념적으로 중요하며, 개입을 중단하지 않고도 성공률을 보존한다는 점에서 선행 연구(Buurmeijer et al.)를 넘어선다. 다만 DiMaS 자체는 정책이 아니라 래퍼이므로 리더보드 점수는 기저 VLA(π0.5/SmolVLA)의 성공률이며, 스티어링은 이를 대체로 유지하는 역할이다. LIBERO-Goal 같은 이질적 suite에서의 약화, 청크 대응 근사, 레이어 의존성이 남은 과제다.

**강점**: 개념적 명료성(선형 vs 분포 매칭), 두 스케일 모델(256M/3B)에서의 일반성, 성공률 보존 α 다이얼, 철저한 4-세팅 일반화 분석.
**약점**: 자체 정책 SOTA가 아닌 후처리 래퍼, Goal suite 취약, 레이어 ℓ의 모델 의존성.

---

<!-- VERIFIED: pdf -->
