# HC-VLA: Health-Conditioned Vision-Language-Action Models for Malfunction-Aware Robot Control

> **한 줄 요약**: VLA-Adapter-Pro 위에 7차원 관절 건강 벡터를 받는 900K 파라미터 Health Projector를 zero-init MLP로 얹어, frozen 백본은 그대로 둔 채 단일 관절 약화(w in {0.3, 0.5, 0.7, 0.9}) 조건에서 LIBERO-Spatial 성공률을 베이스라인 대비 최대 +89%p(J1 w=0.3 기준) 끌어올리는 malfunction-aware VLA. 건강 상태에서는 97.5% → 99.0%로 오히려 개선.

---

## 1. 배경 및 동기

### 기존 VLA의 맹점
- 기존 VLA(RT-2, OpenVLA, pi0, VLA-Adapter 등)는 **로봇이 항상 건강하다고 가정**하고 학습된 정책. 실제 배포 환경에서는 모터 토크 저하, 케이블 마찰, 마모로 인한 관절 출력 감소가 점진적으로 발생.
- VLA-Adapter-Pro 같은 강력한 정책도 single-joint degradation 상황에서 J3(엘보) w=0.5에서 0% 성공률로 완전히 실패. 즉, **"본 적 없는 신체"** 에는 일반화하지 못함.
- Hardware-aware control 분야의 고전적 해법(model-based fault-tolerant control)은 dynamics 모델을 요구 — VLA처럼 end-to-end 학습된 정책에 어떻게 끼워 넣을지 미정.

### 핵심 질문
- **"건강 정보"를 VLA에 어떻게 주입해야 backbone을 다시 학습하지 않고도 적응시킬 수 있는가?**
- **소량의 malfunction 데모(128 episodes)만으로 다양한 단일 관절 약화 수준에 일반화 가능한가?**

📌 [Figure 1 삽입] — VLA-Adapter-Pro에 Health Projector 모듈을 삽입한 architecture diagram

---

## 2. 방법론 심층 분석

### 2.1 베이스 아키텍처: VLA-Adapter-Pro

- **Vision**: DINOv2 + SigLIP paired encoder
- **LLM**: Qwen2.5-0.5B (frozen)
- **Action head**: L1 regression on action chunks (chunk size C=8)
- LIBERO-Spatial-Pro pretrained checkpoint를 그대로 사용 — 본 연구의 모든 train시 backbone은 frozen.

### 2.2 Health Vector

7차원 벡터 h ∈ [0, 1]^7로 각 관절의 정상 출력 대비 capability를 표현:
- h_j = 1: 정상 (full torque)
- h_j = 0: 완전 고장 (zero torque)
- 중간값: degradation 정도

> ❓ **예상 질문**: 왜 7D인가? 관절 수와 정확히 일치하나?
> **답변**: Franka Panda 등 7-DoF arm에 맞춰 각 관절당 1차원. Gripper는 별도. 토크/각도 정보가 결합돼 있다고 abstract에서 언급하지만 수식 정의는 단일 scalar h_j로 단순화.

### 2.3 Health Projector (핵심 모듈)

- **구조**: 2-layer MLP: 7D → hidden → 896D (Qwen2.5-0.5B hidden size)
- **파라미터 수**: 810K + 57K (Action Queries) = **900K trainable**
- **Zero initialization**: 마지막 layer를 zero-init → 학습 시작 시 pretrained policy와 동일 출력을 보장 (RT-2 / LoRA의 zero-init trick과 동일 동기)
- Health embedding을 action query 토큰에 더해 LLM 마지막 layer로 주입.

> ❓ **예상 질문**: 왜 LLM을 frozen 둔 채 끝에서 주입하나? Cross-attention으로 vision token에 주입하는 것이 더 강력하지 않을까?
> **답변**: Frozen backbone의 affordance prior를 보존하면서 효율적으로 학습하기 위함. Sample efficiency 우선(128 episodes로 학습) → 자유도가 낮은 late-injection이 안전. 다만 vision-side conditioning은 향후 확장 가능.

### 2.4 학습 손실
- L1 regression on action chunks (VLA-Adapter-Pro와 동일)
- 단, 입력 시퀀스에 health vector embedding이 prepend됨

---

## 3. 데이터 전략

| 데이터 | 규모 | 비고 |
|--------|------|------|
| Malfunction episodes | 128 | 다양한 single-joint w ∈ {0.3, 0.5, 0.7, 0.9} |
| Healthy episodes | 50 | Pretrained model rollout으로 수집 |
| 총합 | **178 episodes** | LIBERO-Spatial pick-and-place tasks only |

- 데이터 효율성이 매우 높음 (수백 episodes로 의미 있는 적응)
- Multi-joint malfunction은 학습 시 보지 못함

> ❓ **예상 질문**: 50개의 healthy episode를 굳이 섞은 이유?
> **답변**: Catastrophic forgetting 방지. Health vector h=[1,...,1]일 때 원래 정책 성능(97.5%)을 유지하면서 추가 robustness를 얻기 위함. 결과적으로 healthy 99.0%로 미세하지만 향상까지 달성.

---

## 4. 실험 결과 심층 분석

### 4.1 Healthy Baseline (정상 조건)

| 모델 | 성공률 |
|------|--------|
| VLA-Adapter-Pro (Libero-Spatial-Pro) | 97.5% |
| **HC-VLA (h=1)** | **99.0%** |

- Health-conditioned 모델이 정상 조건에서도 약간 더 높음 — 추가 학습 데이터(50 healthy ep)의 distillation 효과로 추정.

### 4.2 Single-Joint Degradation (Table II)

7개 관절 × 4개 weakness level(0.3, 0.5, 0.7, 0.9)에서 평가. 대표 결과:

| 관절 | weakness | Baseline | **HC-VLA** | Δ |
|------|---------|----------|-----------|---|
| J1 (Shoulder) | 0.3 | 45% | **89%** | +44%p |
| J3 (Elbow) | 0.5 | 0% | **34%** | +34%p |
| J5 (Wrist) | 0.7 | 65% | **82%** | +17%p |

- **극단적인 degradation (w=0.3)에서 가장 큰 격차** — 정확히 fault-tolerance가 필요한 영역에서 이득.
- J3 w=0.5에서 0% → 34%는 baseline이 완전히 실패하는 시나리오를 부분적으로 구제.
- 그러나 J3 w=0.5에서 34%는 여전히 낮음 — 엘보 관절은 manipulation에 직접적 영향이 커 한계.

### 4.3 Ablation 부재

- **Architectural ablation 없음**: Health Projector 깊이, hidden dim, zero-init 유무, late vs early injection 비교가 빠짐.
- 워크숍 페이퍼 (4-page) 분량 제약 추정.

---

## 5. 관련 연구 비교

| 모델 | 입력 | Robustness | Backbone trainable? |
|------|------|-----------|---------------------|
| VLA-Adapter-Pro | RGB + lang | 없음 | Action head only |
| RoboHorizon (fault-tolerant) | RGB + joint state | Implicit | Full retrain |
| ResiliVLA (제안 가설) | RGB + lang + state | Explicit | LoRA |
| **HC-VLA** | **RGB + lang + health** | **Explicit (degradation conditioning)** | **No (900K only)** |

- "Health"라는 explicit conditioning input은 VLA 문헌에서 거의 처음.
- Parameter-efficient (900K) — 다른 fault-tolerant 접근이 full retrain하는 것과 대조.

---

## 6. 한계 및 미해결 문제

### 방법론적 미비점
1. **단일 관절 약화에만 평가**: 실제 로봇 fault는 multi-joint + 비대칭. Table II는 J1 ~ J7 각각 독립 실험만.
2. **Health vector를 어떻게 얻을 것인가?**: 본 논문은 ground-truth h를 가정. 실제 배포 시 joint encoder current, torque sensor 추정이 필요. 추정 노이즈가 정책 성능에 미치는 영향 미평가.
3. **LIBERO-Spatial 한 도메인**: Object/Goal/Long suite, CALVIN, SimplerEnv 등 추가 평가 없음. Pick-and-place 일반화 한계 불명.
4. **Generalization to unseen weakness**: 학습 시 w ∈ {0.3, 0.5, 0.7, 0.9} 만 봄. w=0.2 또는 w=0.0(완전 고장) 일반화 미평가.
5. **Closed-loop 적응 부재**: 실시간으로 h를 추정해 갱신하는 online adaptation 없음 — 사전 정의된 h로 단발 inference.

### 평가 제약
- 시뮬레이션(LIBERO) only — 실제 로봇 fault 환경 실험 없음.
- Workshop venue 특성상 baseline 다양성 부족 (VLA-Adapter-Pro 1개와만 비교).

---

## 7. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — VLA에 explicit "health" conditioning을 도입한 시각이 신선. Fault-tolerant control과 foundation-model policy를 연결 |
| **Technical depth** | ★★☆☆☆ — Zero-init MLP projector로 간단. Workshop 분량 |
| **Experimental rigor** | ★★☆☆☆ — LIBERO-Spatial 한 도메인, ablation 부재, multi-joint 미평가 |
| **Practical impact** | ★★★☆☆ — 900K parameter overhead로 fault tolerance 추가는 매력적. 실제 sensor에서 h 추정 파이프라인 필요 |
| **Writing quality** | ★★★☆☆ — 명료한 motivation, 짧지만 핵심을 전달 |

**강점**: VLA에 "건강"이라는 새로운 conditioning축을 명시화한 의미 있는 첫걸음. Parameter-efficient (900K) 설계로 frozen backbone에 쉽게 부착 가능. **약점**: 실험 범위가 LIBERO-Spatial 단일 관절로 좁고, real-world 평가 부재, h 추정 파이프라인 미설계 — 이는 실배포로 가는 데 필수 작업.

---

## 8. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | h를 어떻게 얻는가? Sensor noise는? | 본 논문은 GT h 가정. 실배포 시 joint current/torque sensor + Kalman filter로 추정 필요. 이 estimation gap이 정책 성능을 얼마나 떨어뜨릴지는 미평가 |
| 2 | Multi-joint malfunction 일반화? | 학습은 single-joint에만 — multi-joint h vector에 대한 zero-shot transfer는 미평가. Health Projector가 linear MLP라 multi-joint 조합도 표현 가능하나, OOD 위험 큼 |
| 3 | Healthy 성능이 97.5% → 99.0%로 오른 이유? | 50개 healthy episode 추가 학습의 distillation 효과 + Health Projector가 학습 중 healthy/malfunctioning 데이터의 representation을 더 잘 분리 |
| 4 | J3 w=0.5에서 0% → 34%는 의미 있나? | 절대값은 낮으나 baseline이 완전 실패하는 시나리오 → 임의의 부분 성공 확보. Recovery behavior(e.g., 다른 관절로 compensation)가 발현되는지는 미분석 |
| 5 | 왜 LLM 마지막 layer에만 health를 주입하나? | Frozen backbone 보존 + 900K parameter budget. Cross-attention into vision tokens는 표현력은 ↑이나 frozen weight 갱신 위험 ↑ |
| 6 | Zero-init MLP가 정말 도움이 되는지 ablation은? | 없음. ControlNet / LoRA에서 검증된 trick을 채택 — 별도 ablation 없이도 일반적 효과를 인정한 듯 |
| 7 | CALVIN, SimplerEnv 등 다른 벤치마크로 확장 가능한가? | 원리적으로 동일 (frozen backbone + Health Projector). 단, 각 환경의 robot embodiment마다 health vector 차원과 의미가 달라 cross-embodiment health representation 설계가 필요 |
| 8 | 178 episodes로 정말 충분한가? Larger-scale은? | Sample efficiency가 큰 강점. 1000+ episodes로 늘렸을 때 marginal gain이 어떨지는 불명. Real-world에서는 demo 수집 비용이 핵심 제약 |

<!-- VERIFIED: pdf -->
