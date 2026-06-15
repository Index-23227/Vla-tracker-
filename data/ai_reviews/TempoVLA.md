# TempoVLA: Learning Speed-Controllable Vision-Language-Action Policies

> **한 줄 요약**: VLA가 학습 시연 속도 하나에 고정되는 문제를 해결하기 위해, 액션을 **합치거나(merge) 분할(split)** 하여 임의 속도로 재시간화하는 **VSTA** 데이터 증강과 **스칼라 속도 s를 명시적 conditioning**으로 주입하는 정책 설계를 결합. 단일 정책이 양방향 속도 제어를 달성하면서, LIBERO 1× 평균이 단일속도 baseline 96.7 → **96.9**(7-speed)로 향상되고, GPT-4o 스케줄러와 결합 시 실제 로봇에서 **96%** SR(고정 1× 88% 대비 +8p) 도달.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- **VLA는 시연 속도에 silently lock-in**: 학습 데이터의 단일 페이스만 재현
- 가속 연구(모델 압축, KV-cache 재사용, async chunking, RL 롤아웃)는 모두 "**한 고정 속도 → 다른 고정 속도**" 전환에 그침
- **감속(deceleration)** 은 거의 다뤄지지 않음 — 정밀 삽입, 깨지기 쉬운 객체 핸드오버 등 contact-rich 단계에 필수
- 실제 manipulation은 transit (빠르게) ↔ contact (느리게)를 교대로 요구

### 핵심 질문
- **하나의 VLA가 explicit, on-demand, bidirectional 속도 제어를 가질 수 있는가?**
- 기존 base 아키텍처를 처음부터 재학습하지 않고도 가능한가?

### 핵심 관찰
- "**각 예측 액션의 magnitude가 이미 로봇 속도를 지배한다**" → 액션 크기를 스케일하면 속도가 직접 변함

---

## 2. 방법론 심층 분석

### 2.1 Variable-Speed Trajectory Augmentation (VSTA)

데이터 측 핵심 트릭. 시연 trajectory를 target 속도 s로 재시간화.

- **Speed-up (s > 1)**: 연속된 액션 청크를 **선형 합성 공간**(translation, axis-angle rotation)에서 **누적-합쳐** 더 적고 큰 액션으로 변환
- **Slow-down (s < 1)**: 액션을 **분할(split)** 하여 더 많고 작은 액션으로 변환
- **Gripper**: 이산 신호이므로 별도 처리 (segment boundary에서 보존)
- **Segmentation**: 회전 임계값 90°로 segment 경계 정의 (평균 5.96개 segment, 평균 길이 41 steps in LIBERO)
- Motion semantics 보존: end-effector 적분 변위 오차가 **< 5×10⁻⁸** (controller tolerance 대비 무시 가능)

### 2.2 Speed-Integration Scheme

정책에 스칼라 s를 어떻게 주입할지에 대한 세 가지 ablation:

| Scheme | 방식 | LIBERO Avg SR |
|--------|------|--------------|
| Text (prefix) | "speed=1.25x"를 텍스트 프리픽스로 | **96.8** |
| Modulation | Action expert에 FiLM 스타일 변조 | **96.8** |
| Soft Prompt-8 | 8개 anchor token | 96.5 |

→ 세 방법이 0.3% 이내로 거의 동률. **Text prefix가 가장 단순/유연**하므로 default 채택.

### 2.3 Base Model & Training

- **Base**: π0.5 (PaliGemma 기반 flow-matching VLA)
- 30k iterations, batch 512, **32× NVIDIA H20**, 고정 random seed
- 학습 속도 집합: 기본 {0.75, 1, 1.25, 1.5}×, 확장 {0.5, 0.75, 1, 1.25, 1.5, 1.75, 2}×

### 2.4 Dynamic Speed Scheduling (with VLM)

GPT-4o (VLM) 가 매 시점의 RGB + 언어 지시를 보고 다음 phase 속도 라벨 ∈ {slow, normal, fast}를 출력 → TempoVLA가 그 속도로 실행 → transit은 빠르게, contact 직전은 느리게.

---

## 3. 실험 결과 심층 분석

### 3.1 VSTA 재실행 신뢰성 (Table 1)

| Target s | Data Ratio | Replay SR (%) | Motion Err. |
|---------|------------|--------------|-------------|
| 0.5×    | 0.50      | 83.0         | 2.8E-10     |
| 0.75×   | 0.76      | 92.9         | 4.4E-9      |
| 1×      | 1.00      | **97.6**     | –           |
| 1.25×   | 1.20      | 92.4         | 1.1E-8      |
| 1.5×    | 1.43      | 81.6         | 2.2E-8      |
| 2×      | 1.90      | 67.5         | 4.8E-8      |

- 1× 근방은 매우 신뢰 가능, 극단으로 갈수록 monotone degrade
- Motion error는 모두 controller tolerance 대비 무시 가능

### 3.2 LIBERO 메인 결과 (Table 3, 7-speed 정책)

| Speed | Spatial | Object | Goal | Long | **Avg** | Steps |
|------|---------|--------|------|------|---------|-------|
| Baseline 1× (single) | 99.4 | 95.6 | 96.0 | 95.8 | 96.7 | 152 |
| 0.5× | 97.6 | 94.4 | 96.0 | 92.1 | 95.0 | 296 |
| 0.75× | 98.4 | 95.4 | 97.0 | 94.4 | 96.3 | 201 |
| **1.0×** | **99.2** | **98.2** | **98.4** | 91.8 | **96.9** ↑0.2 | 153 |
| **1.25×** | 99.0 | 96.0 | 98.8 | 95.6 | **97.4** ↑0.7 | 129 |
| 1.5× | 98.6 | 98.0 | 96.8 | 95.8 | **97.3** ↑0.6 | 112 |
| 1.75× | 93.6 | 98.0 | 97.0 | 93.6 | 95.6 | 105 |
| 2.0× | 78.6 | 97.0 | 92.4 | 89.6 | 89.4 | 97 |

핵심 발견:
1. **속도-조건부 학습이 1×까지 끌어올림** (96.7 → 96.9): Object/Goal에서 +2.0~+2.6p 큰 향상
2. **Peak는 1×가 아닌 1.25×/1.5×**: 시연 자체의 "리듬 padding" 때문 — VSTA의 merge가 ambiguous transition frame을 압축해 더 결단력 있게 실행
3. **세밀한 stride가 도움**: stride 0.5 → 0.25로 줄이면 같은 속도에서 모두 향상 (0.5× 94.1 → 95.0)
4. **2×에서 Spatial이 78.6으로 급락**: controller bandwidth 한계 (Model Ratio가 1.58로 saturate)

### 3.3 Real-World (Franka, Pick-and-Place)

| 설정 | SR (%) |
|------|--------|
| Single-speed baseline (1×) | 80.0 |
| TempoVLA 고정 1× | **88.0** (+8p) |
| **TempoVLA + GPT-4o dynamic schedule** | **96.0** |

→ 데이터 증강 효과만으로 +8p, 동적 스케줄링으로 추가 +8p.

### 3.4 Soft Prompt 길이 Ablation (Table 6)

P ∈ {4, 8, 16} 모두 평균 SR 평탄 (< 1% 변동) → 적은 anchor token으로 충분.

---

## 4. 강점

- **Architecture-agnostic**: data-side + lightweight conditioning, 어떤 VLA에도 적용 가능
- **Free lunch**: 속도 제어를 얻으면서 1× 성능도 동시 향상 (Object/Goal +2p 이상)
- **양방향 제어**: 감속까지 다루는 첫 VLA 작업 중 하나
- **Plug-and-play VLM 협업**: 별도 학습 없이 GPT-4o를 phase scheduler로 사용
- **명확한 분석**: Data Ratio vs Model Ratio 격차로 controller bandwidth 한계를 정량화

---

## 5. 약점 및 한계

- **극단 속도(2×)에서 Spatial 급락**: controller tracking이 saturate (Model Ratio 1.58 ≪ data 1.90)
- **VSTA의 replay SR도 0.5×/2×에서 83%/67.5%로 떨어짐**: 데이터 자체 noise floor
- **π0.5 base에만 검증**: OpenVLA, RT-2 등 다른 backbone에서의 일반화 미검증
- **VLM scheduler latency**: GPT-4o 호출 cost가 실시간성에 영향 — 더 긴 obs history로 hide 가능하다고만 언급
- **Long suite는 1.5×에서 95.8로 단일속도 baseline과 동률** — long-horizon에서는 속도 이득 제한

---

## 6. 아키텍처 다이어그램

```
[obs (RGB + state)] ─┐
[task text]          ├─► PaliGemma VLM ─► flow-matching action expert ─► action chunk (×s)
[speed s (text prefix "speed=1.25x")] ─┘                                        │
                                                                                ▼
                                                                  low-level controller (untouched)
```

VSTA (offline, data-side):
```
demo trajectory ─► segment by rotation>90°
                ─► for each segment: accumulate then split to N_target chunks
                ─► (s>1: fewer/bigger, s<1: more/smaller)
                ─► re-timed demo @ speed s
```

---

## 7. 베이스라인 비교

| 방법 | 속도 제어 | 양방향 | 1× 성능 영향 | base 재학습 |
|------|----------|-------|-------------|------------|
| KV-cache reuse / 모델 압축 | 고정 가속 | 일방향 | 보통 하락 | 불필요 |
| RL 가속 | 고정 가속 | 일방향 | 변동 | 부분 재학습 |
| Async action chunking | 처리량 | 일방향 | 변동 | 불필요 |
| **TempoVLA (제안)** | **연속 스칼라** | **양방향** | **+0.2~+0.7p** | **불필요** |

---

## 8. 재현성 평가

- 학습 hyperparameter 표 (Appendix Table 4, 5) 제공
- LIBERO 공개 벤치마크 + π0.5 공개 weight 기반
- VSTA 알고리즘은 명세 충분 (segmentation 임계값, accumulate-split 절차)
- 단, 코드 공개 여부 미언급 → **재현 난이도: 중간**

---

## 9. 영향력 및 후속 연구 방향

- **속도를 VLA의 1급 제어 입력으로 격상**시킨 패러다임적 기여
- 후속:
  - VSTA를 force/torque 등 다른 contact-aware 신호로 확장
  - Controller bandwidth와 co-optimization (저자가 future work로 언급)
  - VLM scheduler를 small on-device LM으로 distill
  - Bimanual / humanoid에서 양손 속도 비동기 제어

---

## 10. 실무 적용 가이드

| 사용 사례 | 권장 속도 / 설정 |
|----------|-----------------|
| 일반 pick-and-place 표준 평가 | **1.25×** 또는 1.5× (peak performance) |
| 정밀 삽입 / 깨지기 쉬운 객체 | 0.5× ~ 0.75× |
| 빠른 transit (장애물 없는 경로) | 1.5× ~ 2× (단, controller bandwidth 확인) |
| 자동 phase 전환이 필요한 long-horizon | TempoVLA + VLM scheduler (GPT-4o or local) |

학습 시: **stride 0.25, range [0.5, 2]× 7-speed**가 최고 trade-off.

---

## 11. 핵심 인용 가치

> "**The magnitude of each predicted action already governs how fast the robot moves**, opening a direct route to controllable execution speed."

이 한 문장으로 "속도 제어 = 액션 magnitude 스케일링" 이라는 우아한 reformulation을 제공.

> "**Peak performance shifts away from 1×**" — 시연 데이터의 리듬 padding이라는 그동안 간과된 데이터 품질 측면을 드러냄.

---

## 12. 종합 평가

| 항목 | 점수 (1-5) | 비고 |
|------|-----------|------|
| 새로움 (Novelty) | 4 | 양방향 속도 제어 + VSTA 조합은 신선 |
| 기술적 깊이 | 4 | 선형 합성 공간 분석, Data vs Model Ratio 정량화 |
| 실험 완성도 | 4 | LIBERO + real-world Franka 두 축, ablation 풍부 |
| 재현성 | 3 | 코드 공개 불명, base는 공개 |
| 실용성 | 5 | Plug-and-play, 추가 성능 + 새로운 제어 자유도 |
| 글쓰기 | 4 | 명료, 그림 1의 motion trail 시각화 효과적 |

**최종 평가**: VLA 연구에서 그동안 **암묵적으로 고정**되어 있던 "실행 속도"라는 차원을 **명시적, 양방향, 단일 정책 내**에서 제어 가능하게 만든 **개념적으로 명료하고 실험적으로 견고한** 작업. π0.5에 직접 얹어 LIBERO 96.9%(1×)/97.4%(1.25×)와 real-world 96%(GPT-4o 스케줄)을 달성. 후속 연구는 controller co-tuning과 다른 backbone 일반화로 확장될 가능성이 높다.

<!-- VERIFIED: pdf -->
