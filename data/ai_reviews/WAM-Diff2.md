# WAM-Diff2: Hierarchical AR-to-Diffusion Distillation for Highly Efficient Autonomous Driving VLA

> **한 줄 요약**: 이미 학습된 **autoregressive driving VLA generalist**를 버리지 않고, 3단계 계층적 distillation(progressive block-wise adaptation → block-wise JSD distillation → 8B→2B cross-scale distillation)으로 **block-causal discrete diffusion VLA**로 "번역"하여, NAVSIM v1 88.3 PDMS / v2 88.6 EPDMS로 AR baseline과 동등한 성능을 유지하면서 decoding을 2.8x(시스템 최적화 포함 15.1x, 22.7ms→1.5ms/token) 가속하고 exposure bias까지 완화한 연구.

- **arXiv**: 2608.01035 (v2, 2026-08-07)
- **소속**: Fudan University / Yinwang Intelligent Technology
- **형식**: AAAI 2027 스타일 (copyright 2027)

---

## 1. 배경 및 동기

### 기존 driving VLA의 두 진영
- **AR generalist** (EMMA, Percept-WAM, DriveMoE, UniDriveVLA, RecogDrive): 다중 태스크(scene understanding + 2D/3D perception + planning)를 단일 네트워크에 통합. 그러나
  - 토큰 단위 순차 생성 → O(L) latency → 실시간 주행 안전성 위협
  - teacher forcing 기반 학습 → **exposure bias** → long-horizon rollout에서 drift 누적
- **Planning-centric diffusion** (DiffusionDrive, DiffusionDriveV2 등): 병렬 디코딩과 bidirectional refinement로 저지연·강건. 그러나 scratch 학습 → **단일 태스크·좁은 아키텍처**, visual-linguistic reasoning 부재

### 핵심 질문
> AR generalist에 축적된 **semantic knowledge를 폐기하지 않고** 병렬 diffusion 아키텍처의 초기값으로 쓸 수 있는가?

문제는 **paradigm gap**이다: causal vs bidirectional attention (attention pattern 불일치), next-token prediction vs iterative denoising (목적 함수 불일치). 저자들은 이를 "localized planning diffusion model"이 아니라 **universal architecture translation paradigm**으로 다룬다.

📌 [Figure 1 삽입] — latency vs 성능, latency 분해(22.7 → 8.1 → 4.6 → 1.5 ms/token), long-horizon trajectory drift 비교

---

## 2. 방법론 심층 분석

### 2.1 Discrete Diffusion VLA 정식화

**Forward (absorbing-state / masked)**:
```
q(o_t | o_{t-1}) = prod_i o_{t-1}^i Q_t,   Q_t = (1 - beta_t) I + beta_t * 1 * e_m^T
```
- `e_m`: [MASK] 토큰 one-hot, `beta_t`: noise schedule
- marginal이 closed-form (`q(o_t|o_0) = o_0 * Qbar_t`) → 랜덤 상태 샘플링만으로 병렬 학습 가능

**Reverse (parallel remasking)**: 완전 마스킹 상태에서 시작 → 전 토큰 분포를 동시 예측 → confidence 상위 subset 고정, 나머지 재마스킹 → 반복. 디코딩 복잡도가 O(L) → **O(T), T << L**.

### 2.2 Block-Causal Attention

표준 causal mask를 **block-causal**로 변경:
- 같은 디코딩 블록 내부 → **bidirectional 병렬 refinement**
- 블록 간 → causal 제약 유지

AR은 `B = 1`인 극한 케이스로 해석된다. 이 관점이 아래 커리큘럼의 출발점.

### 2.3 Stage I — Progressive Block-Wise Adaptation

`B ∈ {4, 8, 16, 32}`로 점진 확장하는 커리큘럼 학습:
```
L_SFT^B = -E_{t,B} sum_{i in B} log p_{theta_B}(y_i | x, y_\B, t)
```
**Progressive bootstrapping**: `theta_B^(0) <- theta_{B/2}^SFT`, `theta_1 = theta_AR`
→ 한 번에 global bidirectional로 점프하지 않고 attention shift를 수학적으로 안정화.

### 2.4 Stage II — Block-Wise Distillation (on-policy)

Ground-truth만으로 학습하면 iterative refinement **궤적**에 대한 지도가 없다. 그래서 `4 → 8 → 16 → 32` 순으로 **직전 블록 크기의 diffusion 모델이 teacher**가 되어, 학생이 실제로 만든 noisy state `y~_t^B` 위에서 분포를 맞춘다:
```
L_BWD^B = E sum_{i in B} JSD( p_{theta_{B/2}}(. | x, y~_t^B, t) || p_{theta_B}(. | x, y~_t^B, t) )
```
**왜 JSD인가**: forward KL의 mode-covering 성향은 multi-modal trajectory 분포를 뭉개버린다. symmetric JSD는 mode-seeking 정밀도와 semantic coverage를 동시에 유지.

**왜 AR teacher가 아니라 diffusion anchor인가**: 최적화 신호를 "paradigm-consistent"하게 유지하여, 학습 목표를 cross-paradigm 조정이 아닌 **parallel token scaling**으로 격리.

### 2.5 Stage III — Model-Wise Cross-Scale Distillation (선택적)

2B로 압축하며 잃은 high-level intelligence를 8B block-32 diffusion teacher로 복구:
```
rho_K(p_T, p_S) = |TopK(p_T) ∩ TopK(p_S)| / K
L_MWD = E sum_{i in B} JSD( p_phi32^{8B,(i)} || p_theta_B^{2B,(i)} )
```
핵심 발견: **같은 생성 paradigm을 공유하는 모델끼리 token prediction pattern이 훨씬 정렬**되어 있어 지식 전이가 쉽다 (rho_5: diffusion teacher 58.2% vs AR teacher 51.2%).

📌 [Figure 2 삽입] — 아키텍처 개요 + 3-stage 계층 distillation

---

## 3. 데이터 전략

| 태스크 | 데이터셋 |
|--------|----------|
| Scene reasoning / VQA | DriveLM, LingoQA |
| Object detection (grounding) | COCO |
| Motion planning | NAVSIM (v1/v2), Bench2Drive |

- 모든 modality(언어 토큰, 2D bbox, 미래 waypoint)를 **하나의 text tokenizer**로 통일 — task-specific projection head를 전부 제거
- 멀티뷰 카메라 입력 1920x1080, 임의 aspect ratio 대응을 위해 2D RoPE 기반 dynamic resolution 보간

---

## 4. 시스템 / 학습 세부사항

**아키텍처 (Qwen3-VL 기반)**
- 8B: SigLIP2-SO-400M encoder (27 blocks, 4096 hidden) + 36 Transformer blocks
- 2B: SigLIP2-Large encoder (24 blocks, 2048 hidden) + 28 Transformer blocks

**학습 하이퍼파라미터**
- AdamW, weight decay 0.05, cosine schedule + 10% linear warmup, global batch 128
- **32 x Ascend 910C NPU** (NVIDIA가 아님 — 중국 국산 가속기)
- AR pretraining lr 4e-5 (5 epochs) / block adaptation·distillation: backbone 4e-5, visual encoder **decoupled 2e-6** / model-wise distillation: 양쪽 통합 2e-6
- 각 단계 5 epochs

**시스템 레벨 추론 최적화**
- **FlashInfer**: block-causal 패턴 전용 attention kernel, shared memory throughput 극대화 → 1.7x
- **CUDA Graphs**: denoising step T가 고정이고 tensor shape가 static → 전체 실행 그래프 캡처, CPU launch overhead 제거 → 추가 3.1x

---

## 5. 실험 결과 심층 분석

### 5.1 Unified multi-task protocol (Table 1, 단일 frozen checkpoint, 2B)

| 모델 | DriveBench | LingoQA | COCO mAP | NAVSIM v1 | NAVSIM v2 | Decode TPS |
|------|-----------:|--------:|---------:|----------:|----------:|-----------:|
| Qwen3-VL-2B | 47.68 | 48.00 | 34.20 | – | – | 44.5 |
| Recogdrive-8B | 56.71 | 67.20 | – | 86.50 / 89.60† | – | – |
| **Ours-2B (B=1, AR baseline)** | 51.23 | 68.40 | 39.20 | 88.14 / 91.50* | 88.32 | 44.5 |
| Ours-2B (B=4) | 48.92 | 68.20 | 37.80 | 87.87 | 87.99 | 68.3 / 401.4 |
| Ours-2B (B=8) | 48.81 | 68.00 | 37.30 | 87.57 | 87.78 | 89.5 / 561.6 |
| Ours-2B (B=16) | 48.77 | 66.00 | 36.50 | 87.32 | 87.30 | 108.4 / 635.3 |
| **Ours-2B (B=32)** | 48.80 | 65.80 | 36.30 | 87.44 / 91.05* | 87.50 | **124.8 / 673.4** |

(†: RL 사용, *: score-based candidate selection)

읽는 법: planning은 B=1→32에서 **0.7~0.8 PDMS 손실**뿐인데 throughput은 2.8x(최적화 시 15.1x). 반면 DriveBench는 51.23→48.80, COCO는 39.20→36.30으로 **perception/understanding 쪽이 더 비싼 대가**를 치른다.

### 5.2 Task-specific planning (Table 2, NAVSIM)

| 방법 | v1 PDMS | v2 EPDMS |
|------|--------:|---------:|
| DiffusionDrive | 88.1 | 84.5 |
| ReCogDrive† | 90.8 | 83.6 |
| DriveVLA-W0 | 90.2 | 86.5 |
| DiffusionDriveV2† | **91.2** | 85.5 |
| Ours (B=1, AR) | 88.1 | 88.3 |
| **Ours (B=32)** | 88.3 | **88.6** |
| **Ours (B=32)\*** | 91.1 | **90.7** |

RL/RFT 없이도 diffusion 변환이 AR baseline을 **소폭 상회**(88.1→88.3), score-based selection 결합 시 91.1 PDMS / 90.7 EPDMS로 SOTA급.

### 5.3 Closed-loop (Table 3, Bench2Drive, 2B)

| 모델 | Eff. | Comf. | Succ. | DS | Multi-Ability Mean |
|------|-----:|------:|------:|---:|-------------------:|
| UniDriveVLA | 198.86 | 11.78 | 51.82 | 78.37 | 51.53 |
| Orion | 151.48 | 17.38 | 54.62 | 77.74 | 54.72 |
| Ours (B=1, AR) | 118.10 | 21.40 | 51.96 | **80.51** | **61.59** |
| Ours (B=32) | 108.20 | 23.15 | 49.55 | 78.93 | 60.58 |

Multi-Ability Mean 60.58은 기존 최고(Orion 54.72)를 크게 상회. 특히 Emergency Brake 61.67%, Traffic Sign 85.26%. 단, closed-loop에서는 AR baseline 대비 **DS -1.58, SR -2.41**로 open-loop보다 손실이 뚜렷하다.

### 5.4 Exposure bias 정량화

12,146개 paired NAVSIM 샘플에서 **per-waypoint 평균 L2 0.5935 → 0.5589 (-5.8%)**. 중요한 건 절대 감소폭이 waypoint 1의 0.002에서 waypoint 8의 0.082로 **단조 증가**한다는 점 — 이것이 "bidirectional refinement가 long-horizon 오차 누적을 억제한다"는 가장 직접적인 증거다.

---

## 6. Ablation 분석

**Stage 누적 효과 (Table 4, NAVSIM PDMS)**

| 설정 | PDMS |
|------|-----:|
| AR baseline | 88.1 |
| + Direct AR-to-diffusion adaptation | **84.1** (-4.0) |
| + Block-wise distillation | 87.7 |
| + Model-wise distillation | **88.3** |

직접 변환은 -4.0 PDMS로 붕괴. Stage II가 3.6점을 회복, Stage III가 나머지 0.6점을 메워 baseline을 넘긴다. → **계층적 단계화 자체가 논문의 실질적 기여**.

**Divergence objective (Table 5)**: Forward KL 88.2 / Reverse KL 88.0 / **JSD 88.3**. 차이는 0.3점으로 작다 — 정직하게 말하면 JSD의 우위는 marginal.

**Teacher paradigm (Tables 6, 7)**

| Teacher | rho_1 | rho_5 | rho_10 | rho_20 | PDMS |
|---------|------:|------:|-------:|-------:|-----:|
| AR Teacher | 83.6 | 51.2 | 54.3 | 46.1 | **79.9** |
| Diffusion Teacher | 84.8 | 58.2 | 60.5 | 50.7 | **88.3** |

**8.4 PDMS 격차**. 이 표가 논문에서 가장 설득력 있는 단일 증거다: cross-scale distillation에서 paradigm consistency는 선택이 아니라 전제조건.

**Denoising step (Figure 3)**: 모든 벤치마크에서 **8~16 step에서 포화**. COCO만 낮은 step에서 더 빨리 무너지는데, 좌표 포맷팅의 엄격한 제약 때문.

---

## 7. 관련 연구 비교

| 축 | 대표 연구 | WAM-Diff2의 차별점 |
|----|-----------|--------------------|
| AR driving generalist | EMMA, Percept-WAM, UniDriveVLA, RecogDrive | 이들을 **대체**하는 게 아니라 **변환 대상**으로 삼음 |
| Planning diffusion | DiffusionDrive(V2), ViLaD | 단일 태스크 scratch 학습 → 본 연구는 multi-task 능력 보존 |
| Masked/discrete diffusion LM | LLaDA, Block Diffusion (Arriola 2025) | block-causal 아이디어를 **VLA 도메인 + distillation 커리큘럼**으로 확장 |
| On-policy distillation | Agarwal 2024 (GKD), MiniLLM | AR→AR이 아닌 **AR→diffusion cross-paradigm**으로 확장, 중간 diffusion anchor 도입 |

가장 가까운 이웃은 Block Diffusion + GKD의 조합이지만, "중간 블록 크기 모델을 anchor로 두어 paradigm gap을 우회한다"는 설계는 새롭다.

---

## 8. 한계 및 미해결 문제

논문이 인정하는 것:
1. **Discrete tokenization의 양자화 artifact** — 고정밀 trajectory에 필요한 smoothness를 해칠 수 있음
2. **성능 상한이 AR teacher에 종속** — student는 초기 AR generalist의 reasoning 능력을 넘지 못함

논문이 말하지 않는 것:
3. **Multi-task 손실의 비대칭성** — planning은 -0.7이지만 DriveBench -2.43, COCO -2.9. "performance parity" 주장은 planning에만 정확하고 perception/understanding에는 과장
4. **Closed-loop 열화** — Bench2Drive에서 DS -1.58, SR -2.41. open-loop 결과보다 손실이 크다는 사실이 abstract에 반영되지 않음
5. **비용 미보고** — Stage I/II/III 각각 5 epochs × 여러 블록 크기 × 32 NPU. "low-cost pipeline"이라 주장하지만 실제 총 NPU-hour는 어디에도 없음
6. **91.1 PDMS는 score-based candidate selection 포함 수치** — 이 selection 자체의 추가 latency가 efficiency 논의에서 빠져 있음
7. **Ascend 910C 종속** — 15.1x 가속의 이식성(NVIDIA H100 등)이 검증되지 않음
8. **베이스라인 TPS 44.5의 공정성** — AR baseline에는 FlashInfer/CUDA Graph를 적용하지 않았다. AR도 동일 최적화를 받으면 15.1x는 상당히 줄어들 가능성

---

## 9. 재현성 및 실무 적용성

- **코드**: "will soon be open-sourced" — 현재 미공개, 체크포인트도 없음
- **재현 난이도**: 높음. Qwen3-VL 2B/8B 기반 AR generalist를 먼저 확보해야 하고, 3단계 × 5 epochs × 4개 블록 크기 학습 필요
- **부분 채택 가능성**: Stage I+II만으로 87.7 PDMS 도달 → 8B teacher 없이도 실용적 이득. Stage III는 명시적으로 optional
- **적용 지침**: block size B는 런타임 조절 가능한 knob이다. planning-critical 배포는 B=32, perception 정확도가 중요하면 B=4~8이 합리적 타협점

---

## 10. VLA 커뮤니티 관점에서의 의미

이 논문의 진짜 주장은 "driving VLA를 잘 만들었다"가 아니라 **"AR VLA → diffusion VLA는 재학습 없이 번역 가능한 변환이다"**이다. Manipulation 쪽에서 diffusion policy와 AR VLA가 갈라져 온 것을 생각하면 시사점이 크다:

- π0, OpenVLA 같은 AR/flow 기반 manipulation VLA에도 동일한 3단 커리큘럼이 적용 가능한지가 자연스러운 후속 질문
- 다만 driving은 출력이 waypoint 8개 수준의 짧은 시퀀스인 반면, manipulation은 action chunk + 고주파 제어 — block-causal 이득 구조가 다를 수 있음
- "paradigm consistency > capacity"라는 rho_K 발견은 **모든 cross-architecture distillation에 일반화될 만한 교훈**

---

## 11. 총평

**강점**
- 문제 정의가 명확하고 (AR 지식 재활용 vs paradigm gap), 해법이 그 정의에서 직접 도출됨
- Table 4(누적 ablation)와 Table 7(teacher paradigm)이 설계 선택을 강하게 뒷받침
- Exposure bias를 12,146 샘플 paired 비교 + waypoint별 단조 증가로 정량화한 것은 이 분야 평균 이상의 엄밀함
- Efficiency 주장을 알고리즘(2.8x)과 시스템(1.7x, 3.1x)으로 분해해서 보고

**약점**
- "performance parity"는 planning에만 성립. multi-task 손실이 abstract에서 희석됨
- JSD 우위(0.1~0.3점)가 loss 설계 서사에 비해 빈약
- AR baseline에 동일한 시스템 최적화를 주지 않은 15.1x 비교의 공정성
- 코드/체크포인트 미공개, 학습 비용 미보고

**평점: 8.5 / 10** — 아이디어의 일반성(architecture translation paradigm)과 ablation의 설득력이 뛰어나다. 다만 efficiency 비교의 공정성과 multi-task 열화에 대한 서술 정직성이 감점 요인. 코드가 실제 공개되면 driving을 넘어 VLA 전반의 표준 변환 레시피가 될 잠재력이 있다.

---

## 12. 🔥 예상 날카로운 질문 모음

| # | 질문 | 예상 답변 / 논점 |
|---|------|------------------|
| 1 | AR baseline에 FlashInfer + CUDA Graph를 똑같이 적용하면 15.1x는 얼마로 줄어드나? | 논문에 없음. AR도 CUDA Graph 이득을 상당 부분 받으므로 순수 알고리즘 이득 2.8x가 정직한 수치. 15.1x는 "diffusion + 최적화 전부" vs "AR 무최적화" 비교 |
| 2 | Multi-task 성능이 정말 "parity"인가? | 아니다. planning -0.7, DriveBench -2.43, COCO -2.9. parity는 planning 한정 주장으로 좁혀 읽어야 함 |
| 3 | 왜 중간 diffusion 모델을 teacher로 쓰나? AR teacher를 직접 쓰면? | Table 7이 답: AR teacher 79.9 vs diffusion teacher 88.3 PDMS. rho_5도 51.2 vs 58.2. paradigm 불일치 시 token 분포 정렬 자체가 안 됨 |
| 4 | Stage III를 빼면? | 87.7 PDMS (baseline 88.1보다 낮음). 논문도 optional이라 명시. 8B teacher가 없는 팀은 -0.6점을 감수 |
| 5 | JSD가 정말 필요한가? | FKL 88.2 / RKL 88.0 / JSD 88.3. 차이 0.3점. multi-modal trajectory 논거는 그럴듯하나 실증적 마진은 작음 |
| 6 | Block size B는 어떻게 고르나? | 런타임 knob. planning만 보면 B=32가 최선(124.8 TPS, -0.7 PDMS), perception 포함이면 B=4~8이 균형점. 배포 시나리오별 선택 |
| 7 | Denoising step T와 블록 크기 B의 관계는? | Figure 3에서 8~16 step 포화. B가 커도 step 수가 함께 늘면 이득이 상쇄되므로, 실효 speedup은 (B / T_effective)에 좌우 |
| 8 | Discrete tokenization의 양자화가 trajectory smoothness를 해치는데, continuous diffusion head를 쓰면? | 그러면 unified tokenizer의 장점(task head 제거, multi-task 통합)이 깨진다. 논문의 핵심 설계 trade-off이며 저자도 한계로 인정 |
| 9 | Manipulation VLA(π0, OpenVLA)에 이 3단 커리큘럼이 적용되나? | 원리적으로 가능. 다만 driving은 waypoint 8개 수준의 짧은 출력, manipulation은 고주파 action chunk — block-causal 병렬화 이득 구조가 다름. 미검증 |
| 10 | 91.1 PDMS의 score-based candidate selection 비용은? | 미보고. 후보 생성·평가가 추가 forward pass를 요구하므로 efficiency 주장과 SOTA 주장이 같은 설정에서 성립하지 않을 수 있음 |
| 11 | Ascend 910C 결과가 NVIDIA로 이식되나? | FlashInfer는 CUDA 생태계 커널이고 CUDA Graph도 NVIDIA 기술 — 실제로는 혼재된 서술. 하드웨어별 재측정 필요 |
| 12 | Closed-loop(Bench2Drive)에서 손실이 open-loop보다 큰 이유는? | closed-loop는 오차 누적 + 분포 이동에 노출. diffusion이 exposure bias를 줄인다는 주장과 상충하는 것처럼 보이며, B=32의 quantization artifact가 제어 정밀도를 깎았을 가능성이 유력한 설명 |

<!-- VERIFIED: pdf -->
