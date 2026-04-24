# StableIDM: Stabilizing Inverse Dynamics Model against Manipulator Truncation via Spatio-Temporal Refinement

> **한 줄 요약**: **manipulator truncation** (로봇 팔이 시야에서 잘림) 상황에서 IDM이 붕괴하는 문제를, **robot-centric masking + Directional Feature Aggregation (DFA) + Temporal Dynamics Refinement (TDR)** 의 **공간-시간 정제**로 안정화.

---

## 1. 배경 및 동기

- IDM(Inverse Dynamics Model)은 visual obs → low-level action을 매핑, **데이터 라벨링**과 **policy 실행**의 핵심.
- 그러나 manipulator가 카메라 프레임 밖으로 나가는 **truncation** 상황에서 상태 추정이 ill-posed → 제어 불안정.
- 기존 IDM은 이런 partial observability를 명시적으로 다루지 않음.

---

## 2. 방법론

### 2.1 Robot-centric masking (SAM preprocessing)
- **Segment Anything**이 로봇 영역을 분할 → 배경 잡음을 마스킹으로 억제, encoder가 robot-centric 영역에 집중하도록.
- 마스킹은 보조 입력 + loss 형태로 전달.

### 2.2 DFA — Directional Feature Aggregation
- **DINOv2**가 Embeddings + CLS tokens 산출.
- **CLS tokens → MLP**로 *axis-dependent directional weights* 유도.
- 이 방향 가중치를 feature embeddings에 적용 후 **weighted global average pooling** → direction-aware feature descriptor.
- 핵심: truncation된 팔의 정보는 본질적으로 *방향성*을 가지므로, anisotropic aggregation이 누락 영역의 구조적 정보를 추론 가능.

### 2.3 TDR — **Two-step** Temporal Dynamics Refinement (매우 중요)

논문(Sec 3)은 TDR이 DFA의 **앞뒤**를 감싸는 two-step stack임을 명시:

1. **Temporal Fusion (DFA 이전)**: 인접 프레임들을 사용해 현재 프레임의 시각 feature를 복원.
2. **Temporal Regressor (DFA 이후)**: MLP + **causal TCN** (dilation factors 1, 2, 4, 8)로 최종 action을 안정화, spike 제거.

따라서 파이프라인은 `SAM mask → DINOv2 → TDR step 1 → DFA → TDR step 2 → action` 순서.

---

## 3. 실험 결과

### Table 1 — AgiBot offline (truncation split @ 15%)

| 방법 | Light acc↑ | Light L1↓ | **Heavy acc↑** | **Heavy L1↓** |
|------|-----------:|----------:|--------------:|-------------:|
| ResNet | 0.179 | 0.189 | 0.150 | 0.679 |
| AnyPos | 0.148 | 0.172 | 0.159 | 0.538 |
| Vidar | 0.194 | 0.163 | 0.186 | 0.573 |
| **StableIDM** | **0.286** | **0.142** | **0.307** | **0.493** |

### Ablation (같은 Table 1)

- Full - DFA: Heavy acc 0.158 (-14.9pp)
- Full - TDR: Heavy acc 0.265 (거의 유지되지만 L1 0.662 → 스파이크 증가로 제어 불안정)
- Full - mask: Heavy acc 0.288, L1 0.475 (**흥미롭게도 L1은 더 낮음**)
  - 저자 해석: 마스크 없으면 정적 배경 cues에 overfit하는 "spurious shortcut" → 일시적으로 L1은 좋아 보이나 일반성 상실.

### Table 3 — Real-robot open-loop replay SR (%)

| 방법 | Pick&Place | Microwave | Sink Cleaning | Avg |
|------|-----------:|----------:|--------------:|----:|
| ResNet | 23.1 | 27.4 | 19.7 | 23.4 |
| AnyPos | 34.6 | 39.2 | 34.5 | 36.1 |
| Vidar | 46.2 | 37.9 | 29.1 | 37.7 |
| **StableIDM** | **53.8** | **42.3** | **46.2** | **47.4** |

### Table 4 — Video-plan grasp SR (%)

ResNet 11.5 / Vidar 26.9 / AnyPos 42.3 / **StableIDM 53.8**

### Table 5 — π0.5 downstream VLA에 annotator로 활용

- Real data only: 35.3%
- Real + StableIDM-labeled video: **52.9%** (+17.6pp)

### 실험 설정

- 데이터: **AgiBot 100 episodes**, 10 tasks (articulated, pick-place, cleaning)
- 액션: dual-arm joint angles + gripper opening (continuous)
- `agibot_episodes.csv` 공개 (재현성)

---

## 4. 한계 및 미해결 문제

1. **Truncation 외 failure mode**: Full occlusion, severe motion blur 등은 DFA의 "visible arm direction" 가정이 깨지면 효과가 떨어짐.
2. **Cross-embodiment 일반화**: DFA가 특정 팔 형태에 과적합될 가능성 → 다른 모폴로지로의 전이는 별도 검증 필요.
3. **IDM이 정책의 병목은 아닐 수 있음**: 상류 policy/perception 에러가 크면 IDM 정제의 이익이 제한적.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — Truncation이라는 구체 failure mode를 타겟한 spatial+temporal 정제. 구성 요소는 기존 아이디어의 재조합. |
| **Practical impact** | ★★★★☆ — 데이터 라벨링과 policy 실행 양쪽에서 IDM은 폭넓게 쓰이며, truncation은 실제 시설에서 흔함 → 실용적 가치 높음. |

---

## 6. 🔥 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | DFA는 왜 isotropic feature보다 나은가? | Truncated 영역의 정보는 본질적으로 **방향성** 있음 (팔은 선형 구조). Anisotropic aggregation이 해당 방향으로 정보를 전파하기 적합. |
| 2 | TDR이 진짜 motion continuity를 가정해도 되나? | 대부분 매니퓰레이션은 저가속도 연속 운동이라 합리적. 충격·접촉 순간에는 이 가정이 일시적으로 깨질 수 있음. |
| 3 | 멀티뷰 카메라로 해결하면 되지 않나? | 실제 배포에서 멀티뷰 setup이 항상 가능한 건 아님. StableIDM은 single-view truncation robustness 자체를 목표로 함. |

<!-- VERIFIED: pdf -->
