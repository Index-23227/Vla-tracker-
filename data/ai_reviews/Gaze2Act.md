# Gaze2Act: Gaze-Conditioned Vision-Language-Action Policies for Interactive Robot Manipulation

> **한 줄 요약**: GROOT N1.5 기반 VLA에 1인칭(ego) 시선(gaze)을 cross-view semantic matching으로 3인칭(exo) 로봇 시점에 정렬해 주입하는 119.5M 파라미터의 decoupled cross-attention 분기를 추가하여, 모호한 자연어 지시를 사람의 시선으로 보완하고 동적 의도 변경(dynamic intent steering)까지 가능한 humanoid 조작 정책.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 자연어는 **referential ambiguity**를 본질적으로 가짐 — "그 컵"이 어느 컵인지, "손잡이"가 어느 부분인지 지정 불가
- 기존 VLA(OpenVLA, GR00T, pi0)는 언어만으로 목표를 specify 해야 하나, 실세계의 다물체·다부위 장면에서 실패
- 사람-로봇 협업에서 의도가 **task 도중에 변할 수 있으나**, 기존 VLA는 episode 시작 시점의 언어만 사용
- 사람의 시선(gaze)은 자연스럽고 노력 없이 의도를 표현하지만, 시점 차이(ego vs exo) 때문에 로봇 정책에 직접 주입하기 어려움

### 핵심 질문
- **사람의 1인칭 시선을 로봇의 3인칭 관측에 어떻게 정렬할 것인가?**
- **그 정렬된 시선을 사전학습된 VLA(GROOT N1.5)에 어떻게 최소 침습적으로 주입할 것인가?**
- **고정된 instruction이 아니라 시선이 task 도중 바뀌어도 정책이 동적으로 따라갈 수 있는가?**

📌 [Figure 1 삽입] — Ego-Exo 시선 정렬 + GROOT N1.5 + Decoupled Cross-Attention 구조도

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

| 구성 요소 | 모델 | 파라미터 | 역할 |
|-----------|------|----------|------|
| VLA Backbone | **GROOT N1.5** (Eagle2 VLM frozen) | ~2.4B | 언어·관측 → action |
| Cross-view Grounding | DINOv3 ViT-L/16 | - | ego 시선 → exo mask & point |
| Object Encoder | DINOv3 ViT-S+/16 | 29M | 시선 영역의 시각 토큰화 |
| Gaze Injection | Decoupled Cross-Attention | +119.5M | DiT에 zero-init gate로 주입 |
| Action Head | DiT (flow matching) | (GROOT 내장) | 50-step horizon, 4-step Euler |
| **총 추가** | — | **119.5M (~4.95%)** | — |

### 2.2 Cross-View Semantic Matching

- 사람의 ego-view에서 gaze tracker가 (x, y, t) 시선점을 출력
- DINOv3 ViT-L/16으로 ego 패치와 exo 패치 사이의 dense feature similarity 계산
- 시선점 주변 ego 패치들과 가장 유사한 exo 패치를 찾아 **(object mask, gaze point)**의 두 가지 출력 생성
- 객체 단위(coarse) + 점 단위(fine)의 **coarse-to-fine** target specification

> ❓ **예상 질문**: 왜 SAM 같은 명시적 segmentation 대신 dense feature matching인가?
> **답변**: SAM은 ego→exo 도메인 갭에서 mask 일관성을 보장하지 못함. DINOv3는 self-supervised로 학습되어 시점에 robust한 dense correspondence를 제공. 또한 mask와 point를 동시에 생성하므로 별도 모델 불필요.

### 2.3 Decoupled Cross-Attention with Zero-Init

GROOT N1.5의 DiT action head 각 블록에 **별도의** cross-attention 분기 추가:
- 기존 텍스트 cross-attention: frozen
- 신규 gaze cross-attention: trainable, gate(γ) = 0으로 초기화
- 최종 출력: `h + text_xattn(h) + γ * gaze_xattn(h, gaze_tokens)`
- γ가 학습되며 점차 활성화 → 처음에는 vanilla GROOT와 동일한 동작 보장

> ❓ **예상 질문**: 왜 LoRA 같은 PEFT가 아니라 별도 cross-attention인가?
> **답변**: LoRA는 기존 가중치에 더해지는 형태라 gaze가 없는 경우의 분리가 어렵고, gate가 없으면 초기 학습 불안정. Decoupled 구조는 (a) gaze-free / gaze-conditioned 양쪽 모두 깔끔히 지원, (b) gate를 통해 학습 안정성 보장.

### 2.4 Dynamic Intent Steering

- Inference 시 시선이 변하면 새로운 (mask, gaze_point)를 즉시 cross-attention에 주입
- Action chunk 단위(50 steps)로 재계획되므로 약 0.5-1초 내 의도 변경 반영
- 평가 시 의도적으로 task 중간에 사람 시선을 다른 객체로 전환하여 성공률 측정

---

## 3. 데이터 전략

| 항목 | 값 |
|------|----|
| 플랫폼 | Unitree G1 humanoid (bimanual) |
| 태스크 | 16개 manipulation tasks |
| 평가 | 50 trials/task = 800 trials |
| ego-view | 시선 추적기 장착 |
| exo-view | 제3자 카메라 |

데이터 수집 절차는 텔레오퍼레이션이며, ego-exo 카메라가 동기화된 paired demonstration이 핵심.

> ❓ **예상 질문**: 사람 시선 데이터를 모으는 비용이 LIBERO 같은 시뮬 대비 너무 크지 않은가?
> **답변**: 맞음. 시선 트래커 + 텔레오퍼레이션 + ego-exo 동기 카메라가 필수라 sim 데이터로 대체 어려움. 본 연구는 데이터 효율성 (task당 ~50회 시연 수준 추정) 자체가 contribution.

---

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|------|----|
| Hardware | 단일 NVIDIA RTX PRO 6000 (96 GB) |
| Batch size | 80 |
| Steps | 20,000 / task group |
| Optimizer | AdamW (β₁=0.95, β₂=0.999, wd=1e-5, lr=1e-4) |
| Precision | bf16 mixed |
| Action horizon | 50 steps |
| Solver | 4-step Euler (flow matching) |

> ❓ **예상 질문**: 단일 96GB GPU로 학습 가능한 점이 의미하는 바는?
> **답변**: 추가 파라미터가 119.5M(약 5%)에 불과하고 backbone이 frozen이므로 단일 GPU 학습이 가능. 이는 다른 ego-exo VLA 연구(보통 다중 노드 필요)와의 큰 실용적 차이.

---

## 5. 실험 결과 심층 분석

### 5.1 Real-World on Unitree G1 (16 tasks × 50 trials)

| 지표 | 수치 |
|------|------|
| Object-level intent accuracy | **93.0%** |
| Object-level task success | **89.0%** |
| Part-level intent accuracy | 80.4% |
| Part-level task success | 72.4% |

- Object-level (어느 컵?) 보다 part-level (컵의 손잡이?) 에서 정확도/성공률 모두 감소 — fine-grained 시선 정렬의 본질적 어려움
- Intent → success 간 약 4-8%p drop은 시선이 정확해도 manipulation 자체의 실패가 존재함을 시사

### 5.2 Dynamic Intent Steering

| 지표 | 수치 |
|------|------|
| Dynamic intent steering 성공 | **14/30 (47%)** |

- 시선이 task 중간에 바뀌었을 때 정책이 따라가는 비율
- 47%는 비교 baseline이 없어 절대값 해석 어려우나, **고정 instruction VLA에서는 0%에 가까울 작업**이라는 점에서 의미

> ❓ **예상 질문**: Part-level 성공률 72.4%는 object-level 89%에 비해 큰 폭으로 낮은데 원인은?
> **답변**: (a) ego→exo cross-view feature matching이 작은 영역에서 부정확, (b) GROOT N1.5 자체의 fine-grained 조작 성능 한계, (c) 시선 트래커의 픽셀 노이즈가 small target에서 비례적으로 큼.

---

## 6. Ablation 분석

논문에서 보고된 핵심 ablation:
- **Gaze 제거** (vanilla GROOT N1.5): object-level success 큰 폭 하락 (paper Table 참조)
- **Coarse-only (mask만)** vs **Fine-only (point만)** vs **Coarse+Fine**: combined가 최고
- **Zero-init gate 제거**: 학습 초기 불안정, 최종 성능 하락

→ Coarse-to-fine 구조와 zero-init gating이 모두 essential

---

## 7. 관련 연구 비교

| 모델 | 의도 입력 | Backbone | Real-world | Dynamic Intent |
|------|-----------|----------|------------|----------------|
| OpenVLA | language only | LLaMA2 7B | △ | ✗ |
| GR00T N1.5 | language + state | Eagle2 | ○ | ✗ |
| RT-2 | language only | PaLI-X | ○ | ✗ |
| **Gaze2Act** | **language + gaze (ego→exo)** | **GROOT N1.5 + gaze branch** | **○ (89%/72%)** | **○ (47%)** |

### 핵심 차이
- 자연어 외 **시선 modality** 추가 → reference ambiguity 해결
- Cross-view ego-exo gap을 명시적으로 dense feature matching으로 해결
- Dynamic intent steering을 정량 평가한 최초의 VLA 연구

---

## 8. 한계 및 미해결 문제

### 방법론적 미비점
1. **시선 트래커 의존**: 전용 ego-view 카메라/시선 트래커가 필요. 일반 협업 시나리오에서 항상 가용하지 않음
2. **Part-level 성공률 72.4%**: 정밀 조작에서 cross-view matching의 정확도 한계
3. **Dynamic steering 47%**: 절반은 실패 — 시선 변경의 지연(50-step chunk) 또는 action 분포 변환의 어려움
4. **벤치마크 점수 부재**: LIBERO/CALVIN/SimplerEnv 없음 — 시선 데이터를 요구하므로 표준 benchmark와 호환되지 않음
5. **단일 platform (Unitree G1)**: Franka/UR5 등 다른 플랫폼 일반화 미검증
6. **시선 노이즈에 대한 robustness 분석 없음**: 트래커 calibration drift 시 성능?

### Attribution 문제
- 89% object-level 성공률 향상이 (a) 시선의 정보량, (b) GROOT N1.5의 강력함, (c) cross-view DINOv3의 분리가 명확하지 않음
- DINOv3 대신 SigLIP, CLIP을 썼다면? — 미보고

---

## 9. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Ego-exo gaze conditioning을 정식 VLA로 통합한 최초 |
| **Technical depth** | ★★★★☆ — Cross-view matching + decoupled XA + zero-init이 체계적 |
| **Experimental rigor** | ★★★☆☆ — 16 tasks × 50 trials는 충실하나 표준 benchmark 부재 |
| **Practical impact** | ★★★★☆ — 단일 96GB GPU로 학습 가능, 협업 시나리오 직접 적용 가능 |
| **Writing quality** | ★★★★☆ — Real-world humanoid 결과의 가시성이 강함 |

**강점**: 의도 표현 modality를 확장하여 자연어 ambiguity 문제를 정면 해결. Decoupled cross-attention + zero-init은 사전학습 backbone 보존 측면에서 PEFT 모범 사례. **약점**: 시선 트래커 필요로 적용 시나리오 제한, dynamic steering 47%는 아직 미완성, 표준 benchmark 부재로 다른 VLA와 비교 곤란.

---

## 10. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 시선 데이터 없이 inference 가능한가? | 가능. Zero-init gate가 0이면 vanilla GROOT와 동일. gate를 강제 0으로 두면 backbone-only fallback |
| 2 | Cross-view matching이 실패하면? | mask가 비거나 잘못된 객체를 지정 → object-level intent acc 93%가 곧 그 실패율의 보완값 (실패율 ~7%) |
| 3 | LIBERO 같은 표준 benchmark는 왜 없는가? | LIBERO에 ego-view + 사람 시선이 없음. 시선을 합성하면 ground-truth 시선과 다른 분포가 됨 |
| 4 | 47%의 dynamic steering은 너무 낮지 않은가? | 절대값은 낮으나 비교 baseline은 0%에 가까움. 또한 50-step chunk 단위라 지연 불가피 |
| 5 | DINOv3 대신 SAM2 + tracker 조합은? | 미실험. SAM2는 mask는 정확하나 ego-exo correspondence는 약함. DINOv3 self-supervised feature가 viewpoint robust |
| 6 | Action head를 diffusion 대신 regression으로 바꾸면? | GROOT의 DiT(flow matching)이 50-step horizon에 적합. Regression head는 temporal coherence 부족 가능 |
| 7 | 16개 태스크 × 50 trials는 평가량으로 충분한가? | 800 trials는 real-world로는 큰 편이지만, fine-grained task variance를 모두 cover하지 못함 |
| 8 | Part-level 72.4% 성공률을 올리려면? | (a) DINOv3 ViT-L 대신 ViT-G, (b) point 주변 patch attention의 spatial sharpness 향상, (c) fine-tuning step 증가 |

<!-- VERIFIED: pdf -->
