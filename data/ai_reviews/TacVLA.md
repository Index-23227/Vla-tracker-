# TacVLA: Contact-Aware Tactile Fusion for Robust Vision-Language-Action Manipulation

> **한 줄 요약**: Pi0.5 backbone에 **15×8 taxel array → MLP-인코딩된 36개 tactile token**을 추가하고, 임계 압력 기반 binary contact flag로 tactile token을 **상태 의존적으로 attention에서 끄고 켜는** Contact-aware Gating을 도입. Constraint-locked disassembly 4종 평균 **83.75%** (Tab. II), 가려진 in-box picking **70%** (vs Pi0.5 10%, Tab. II), 전면 카메라 차단 시 평균 **~30→60%대** 회복 (Fig. 6).

---

## 1. 배경 및 동기 (§I, §II)

- 기존 VLA(OpenVLA, Pi 계열)는 vision + language만으로 동작 → 가림(occlusion), 미세 정렬, 슬립처럼 **시각으로 포착 불가능한 접촉 신호**가 필요한 태스크에서 한계 (§I).
- 촉각 통합 선행 연구는 (i) 촉각을 image처럼 dense pixel로 다뤄 token 길이가 폭증하거나(VTLA, OmniVTLA, Tactile-VLA), (ii) trajectory 내내 항상 활성화하는 static fusion이 대부분이라 **non-contact phase에서 불필요한 cross-modal 간섭**이 발생 (§II.C).
- 본 논문의 핵심 질문: "**어떻게 tactile을 효율적으로 토큰화하고, 접촉 상태에 따라 동적으로 라우팅할 것인가?**" (§I 말미)

---

## 2. 방법론 (§III, Fig. 1)

### 2.1 전체 아키텍처
- 4-component: modality tokenizers, pretrained VLM backbone, action expert, contact-aware gating module (Fig. 1b).
- **Vision**: 전면 + 손목 RGB → SigLIP visual encoder.
- **Language + proprioception**: PaliGemma tokenizer로 함께 토큰화.
- **Tactile**: 15×8 taxel map → lightweight MLP encoder → **36 tactile tokens**, 2D sin-cos positional embedding 추가 (§III.A).
- 모든 modality token은 하나의 sequence로 concat되어 VLM backbone에 prefix로 들어가며, prefix 영역은 **non-causal cross-attention** 으로 자유롭게 fuse된다.
- Action은 Pi0.5 디자인의 **flow-matching action expert**가 `a_{t:t+H} ~ π_θ(· | z̃_t)` 형태로 horizon H 액션 시퀀스를 생성.

### 2.2 Contact-aware Token Gating (§III.B)
- 접촉 판정: "사전 정의 압력 threshold를 초과하는 taxel 수가 일정 개수 이상"인 binary criterion → flag `c_t ∈ {0,1}`.
- Tactile token에 두 가지를 동시 적용:
  - Attention mask `M_t^tac = c_t · 1`
  - Embedding gating `z̃_t^tac = c_t · z_t^tac`
- 토큰 **개수는 고정**(36)되어 시퀀스 토폴로지가 변하지 않으며, `c_t=0` 일 때는 tactile token이 attention 계산에서 완전히 제외되고 embedding offset/positional encoding의 잡음 기여도 제거된다 (§III.B 결미). 이는 dynamic insert/remove 방식의 비효율을 피하면서 state-dependent routing을 달성하는 설계 선택.

### 2.3 학습 (§III.C)
- 데이터: 4종 disassembly + 1종 in-box picking, **task당 50 teleop demo**, 10 Hz로 RGB·tactile·proprioception 동기화.
- **OpenPI Pi0.5 base 체크포인트**에서 **LoRA fine-tuning, 10,000 gradient step**, **tactile encoder는 frozen** 유지 (사전학습 없는 상태에서도 접촉 표현이 그대로 활용됨을 보이는 설계).
- 하드웨어: Franka Emika Panda 7-DoF + parallel gripper, 전면+손목 RGB, finger 표면 15×8 tactile array (Fig. 2).

> ❓ **예상 질문**: Gating은 학습되는가, 임계값 휴리스틱인가?
> **답변(§V)**: **휴리스틱 binary threshold**. 저자도 limitation으로 명시 — gradual하거나 학습 가능한 modality weighting은 future work.

---

## 3. 실험 결과

### 3.1 Constraint-locked Disassembly (Tab. II, 4 task × 20 trial)
| Method | Task1 | Task2 | Task3 | Task4 | Avg |
|---|---|---|---|---|---|
| 3D Diffusion + Tactile | 40 | 40 | 45 | 0 | 31.25 |
| Diffusion Policy + Tactile | 70 | 70 | 30 | 25 | 48.75 |
| Finetuned Pi0.5 (vision-only) | 80 | 80 | 65 | 30 | 63.75 |
| **TacVLA** | **100** | **90** | **70** | **75** | **83.75** |

- 가장 큰 개선은 미세 슬라이드를 요구하는 **Task4 (+45%p)** — 부분 가림과 fine-grained sliding이 결합된 시나리오에서 tactile feedback의 효과가 극명.

### 3.2 In-box Picking (Tab. II)
- 전면 카메라가 박스 내부를 못 보고 손목 카메라도 조명·가림 문제 → vision-only Pi0.5 **10%** vs **TacVLA 70%** (+60%p). TacVLA는 접촉이 물리적으로 감지된 후에야 lift 단계로 전이.

### 3.3 Robustness — Block Camera (Fig. 6)
- 전면 카메라 강제 차단 시 Pi0.5: 40/40/5/35% (4 task) → 평균 ~30%, **TacVLA**: 70/65/45/70% → 평균 60%대 (≈ 약 2× 향상).
- Task 3에서 **+40%p** 가 가장 큰 개선 — 가림 하 회전 정렬에서 촉각이 시각을 대체.

### 3.4 Robustness — Human Disturbance (Fig. 5b, §IV.D.2)
- 작업 도중 사람이 물체를 다시 박스에 넣는 시나리오. TacVLA는 unexpected 변화를 탐지해 박스로 복귀·재파지하지만, Pi0.5 baseline은 회복 실패.

### 3.5 Ablation — Contact Gating의 기여 (Tab. III, Fig. 7)
- **Pi0.5 + Tactile (no gating)** vs **TacVLA**:
  - Disassembly avg 71.25 → **83.75** (+12.5%p)
  - In-box picking 40% → **70%** (+30%p)
  - 가림 disassembly 평균 약 **−25%p** 차이 (Fig. 6)
- 특히 Task 3에서는 **gating 없는 변형(60%)이 vision-only baseline(65%)보다 오히려 낮음** — naïve concat이 시각 localization을 방해할 수 있다는 증거 (Fig. 7 정성 분석: 잦은 misalignment, repeated regrasp, stuck).

---

## 4. 한계 및 미해결 문제 (§V)

1. **이진 휴리스틱 게이팅**: pressure threshold 기반 hard gating, gradual/learnable weighting 부재.
2. **저해상도 tactile**: 15×8 array로 fine-grained contact geometry 추론에 한계.
3. **단-horizon 평가**: 평가 task가 short-horizon, contact-centric에 한정 — long-horizon으로의 일반화 미검증.
4. **시뮬 평가 부재**: 모든 결과가 real-world이며, simulator 비교는 부재.
5. **Tactile encoder frozen**: §III.C에 명시. 사전학습된 분포에서 멀어진 신호에서의 행동 미검증.
6. **Embodiment 의존**: 단일 Franka + 1개 finger 부착 센서. cross-embodiment 일반화 미검증.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLA에서 **state-dependent token routing** 이라는 새 modality fusion 축 정립 |
| **Technical depth** | ★★★☆☆ — gating은 단순 binary, 핵심 기여는 시스템 설계 |
| **Experimental rigor** | ★★★★☆ — disassembly 4종 + in-box + 가림 + 인간 외란 + ablation |
| **Practical impact** | ★★★★☆ — Pi0.5 + LoRA로 즉시 응용 가능, 모든 contact-rich 시나리오에서 큰 폭 향상 |

---

## 6. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|------|----------|
| 1 | 36 tactile token 수가 의미가 있는가? | 15×8(=120 taxel)을 MLP로 36 token에 압축 → 길이 폭증을 막고 transformer cost를 통제. 이미지로 다루는 prior(VTLA 등) 대비 핵심 대비점 (§III.A) |
| 2 | Gating이 false negative를 내면? | §V Limitation에서 직접 언급. 본 논문은 hard threshold이며, gradual weighting은 future work |
| 3 | tactile encoder를 freeze한 이유? | §III.C. 데이터가 task당 50 demo에 불과한 만큼 representation collapse를 막는 보수적 선택. 결과적으로 ablation 우위가 유지됨 |
| 4 | Force/torque 센서로 대체 가능? | 본문 §II.B는 dense distribution이 필요한 fine-grained contact를 위해 array를 채택했음을 강조. F/T는 scalar라 spatial cue가 부족 |
| 5 | naïve concat과의 결정적 차이? | Tab. III + Fig. 7. concat은 비접촉 phase에서 tactile token이 시각 localization을 흔들어 평균 −12.5%p, 일부 task는 vision-only baseline보다도 낮아짐 |

<!-- VERIFIED: pdf -->
