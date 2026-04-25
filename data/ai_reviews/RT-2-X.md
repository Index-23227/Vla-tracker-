# Open X-Embodiment: Robotic Learning Datasets and RT-X Models

> **한 줄 요약**: 21개 기관 22 로봇 embodiment에서 527 skill / 160,266 task를 모은 Open X-Embodiment(OXE) dataset을 공개하고, 동일 robotics-mixture로 학습한 RT-1-X(35M)·RT-2-X(55B)가 multi-domain positive transfer를 달성함을 6 robot · 3,600 trial 평가로 실증 (ICRA 2024).

---

## 1. 배경 및 동기 (Sec. I)

- 로봇 학습은 도메인마다 별도 모델을 학습하는 **fragmentation** 문제 → vision/NLP의 "consolidation"을 따라가지 못함.
- Open X-Embodiment 협력으로 21 institution이 데이터셋을 표준화된 RLDS 포맷으로 통합. 9 manipulator-mixture를 구성해 RT-X 학습에 사용.

---

## 2. 데이터셋 & 모델 설계

### Open X-Embodiment dataset (Sec. III)
- 22 robot embodiment, 527 skill, **160,266 task** (Abstract).
- 본 논문의 학습 mixture는 9 embodiment subset (RT-1, QT-Opt, Bridge, TARP, Jaco Play, Cable Routing, RoboTurk, NYU VINN, Austin VIOLA, Berkeley UR5, TOTO, Language Table).

### RT-1-X (Sec. IV)
- FiLM-conditioned EfficientNet + Transformer (Fig. 3).
- 7-DoF end-effector tokenized action.

### RT-2-X (Sec. IV)
- **PaLI-X 55B 기반** (ViT + UL2), web data와 robotics data를 ~1:1 비율로 co-fine-tune.
- Action을 정수 token sequence ("1 128 91 241 5 101 127" 형태)로 출력 → 어떤 VLM도 fine-tune 가능.
- Inference: cloud-hosted, 3-10 Hz.

---

## 3. 핵심 실험 결과 (PDF 수치 인용)

### Small-scale dataset domains (Fig. 4)
- RT-1-X 평균 success rate가 Original Method/RT-1 대비 **+50%** (논문 본문). 5 dataset 중 4개에서 우월 → "데이터가 적은 도메인은 cross-embodiment co-training이 큰 이득".

### Large-scale dataset domains (Table I, in-distribution)
| Method | Bridge IRIS (Stanford) | Bridge RAIL (UCB) | RT-1 paper 6 skills (Google Robot) |
|---|---|---|---|
| Original Method (LCBC) | 13% | 13% | – |
| RT-1 | 40% | 30% | 92% |
| RT-1-X | 27% | 27% | 73% |
| **RT-2-X (55B)** | **50%** | **30%** | **91%** |

→ 작은 모델(RT-1-X)은 large-scale에서 underfit하지만, **고용량 RT-2-X는 모든 setting에서 우월하거나 동등**.

### Generalization & Emergent skills (Table II)
| Row | Model | Size | History | Emergent Skills | RT-2 Generalization |
|---|---|---|---|---|---|
| (1) | RT-2 | 55B | none | 27.3% | 62% |
| (2) | RT-2-X | 55B | none | **75.8%** | 61% |
| (3) | RT-2-X (no Bridge) | 55B | none | 42.8% | 54% |
| (4) | RT-2-X | 5B | 2 | 44.4% | 52% |
| (6) | RT-2-X | 5B | 2 | 0% (from scratch) | 1% |

**핵심 ablation 결과 (Sec. V.C)**:
- (1) vs (2): cross-embodiment 데이터 추가만으로 emergent skills **약 3×** 향상.
- (2) vs (3): Bridge dataset 제거 시 emergent skills 75.8 → 42.8% → **타 robot data가 transfer source임**을 직접 증명.
- (4) vs (5): 짧은 history(2 frame)가 generalization을 크게 향상 (52% vs 30%).
- (4) vs (6): Web pretraining이 large model에게 **결정적** (1% vs 52%).
- (2) vs (4): 55B vs 5B → 더 큰 모델일수록 transfer 능력이 강해짐.

---

## 4. 한계 및 비판적 고찰

1. **22 embodiment 중 9개만 학습 mixture**에 포함 → 데이터 시점 한계.
2. **Action space 정렬 불완전**: 좌표계/단위가 robot마다 다름; 같은 token이 robot에 따라 다른 motion 유발 (Sec. IV.B).
3. **Inference latency**: RT-2 cloud-hosted 3-10 Hz → real-time control에 제약 (Sec. IV.C).
4. **Negative transfer 분석 부족**: 어떤 embodiment 조합이 해로운지 결정 기준 부재 (Sec. VI에서 future work로 언급).
5. **모델 비공개**: RT-2-X weights는 미공개 (YAML `open_source: false`와 일치).

### YAML 검증 결과
- YAML `parameters: 55B`, `simpler_env`의 `google_robot_pick_coke_can: 72.5`, `move_near: 50.3`, `open_drawer: 38.6` 표기. 그러나 본 PDF는 SimplerEnv가 아닌 자체 real-robot evaluation을 보고 → SimplerEnv 점수는 **외부(SimplerEnv 논문)에서 보고된 RT-2-X 수치**여야 함. 출처 라벨이 "RT-2-X paper + SimplerEnv"로 적절히 명시됨 (cross-reference 필요).

---

## 5. 총평

**강점**: 로봇 학습의 "ImageNet moment". RT-2-X가 cross-embodiment data로 emergent skill 3× 향상을 실증한 최초의 **vision-language-action** scale-up. **약점**: closed-source 모델, 9 embodiment subset만 사용, action 정렬 미흡.

---

## 6. 예상 날카로운 질문

| # | 질문 | 답 (논문 근거) |
|---|------|----------------|
| 1 | 모델 크기가 작으면 transfer가 안 되는가? | Table I에서 RT-1-X(35M)는 Bridge IRIS 13→27%로 향상되나, RT-1(40%)을 못 따라잡음. **모델 capacity가 충분해야 cross-embodiment transfer 효과**가 큼 (Sec. V.A). |
| 2 | Bridge 제거 ablation의 의미? | Row (3): emergent skills 75.8→42.8%. WidowX의 Bridge data가 Google Robot tasks에 transfer되었다는 직접 증거 (Sec. V.B). |
| 3 | Web pre-training은 얼마나 중요한가? | Row (4) vs (6): 5B 모델, history=2일 때 web-pretrained 52% vs from-scratch 1%. **scratch는 사실상 작동하지 않음**. |

<!-- VERIFIED: pdf -->
