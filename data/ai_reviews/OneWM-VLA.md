# OneWM-VLA: One Token Per Frame — Reconsidering Visual Bandwidth in World Models for VLA Policy

> **한 줄 요약**: π0(PaliGemma-2B + Gemma-300M) 기반에 **Adaptive Attention Pooling**(Max+Sum+Learn 3-way pooling + softmax view fusion)을 부착해 *프레임당 단 1개 latent token*으로 visual bandwidth를 압축하고, world modeling과 action prediction을 **단일 flow-matching head**로 통합. LIBERO 평균 98.1(Long 95.6, +10.4 over π0) 달성, 14.71M LoRA parameter만 학습.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 최근 VLA에 world model을 결합하는 흐름(UWM, OpenDriveVLA, OpenHelix 등)에서 video latent stream을 **프레임당 수십~수백 token**으로 표현 → context length 폭증, 학습/추론 비용 증가
- World modeling head와 action head가 *분리*되어 있어 두 objective 간 gradient 충돌 가능
- "프레임당 token 수가 정말 필요한가?"라는 기본 질문이 충분히 탐구되지 않음

### 핵심 질문
- **프레임당 단 1 token으로도 useful한 world dynamics를 포착할 수 있는가?**
- **World modeling과 action prediction을 *하나의 flow-matching objective*로 통합할 수 있는가?**

📌 [Figure 1 삽입] — 프레임당 1 token 압축 + unified flow-matching

---

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요

```
Multi-view RGB (front + wrist)
        │
        ▼
PaliGemma-2B (vision-language encoder, frozen)
        │
        ▼
Adaptive Attention Pooling (AAP)
   ├── Max pool: 채널별 peak response
   ├── Sum pool: 채널별 total response
   └── Learn pool: task-aware MLP
        │
        ▼  (3개 token → 1개 token으로 fusion)
Softmax-weighted view fusion (multi-view → 1 view-aggregated token)
        │  (즉, 프레임당 1 token)
        ▼
Gemma-300M (joint expert)
        │
        ▼
Unified flow-matching head
   ├── Future latent stream prediction
   └── Action prediction
        │
        ▼  (10 ODE inference steps)
Actions
```

### 2.2 Adaptive Attention Pooling (AAP)

핵심 contribution. 한 view의 H×W spatial feature를 **단 1 token**으로 줄이되, 정보 손실을 최소화:

| Pool 방식 | 수식 (개념) | 역할 |
|----------|-----------|------|
| Max | max over (H,W) per channel | salient feature 보존 |
| Sum | sum over (H,W) per channel | global statistics 보존 |
| Learn | task-aware MLP(spatial features) | task-adaptive feature 선택 |

세 token을 다시 fusion → view당 1 token.

> ❓ **예상 질문**: Max+Sum+Learn은 ad hoc 조합 아닌가?
> **답변**: 본 논문이 ablation에서 각 단독보다 3-way가 +2~3 pt 향상 보임. 그러나 더 정교한 자동 selection은 future work.

### 2.3 Multi-view fusion

Front + wrist 카메라 각각에서 1 token씩 → 두 token을 **softmax-normalized trainable weight**로 가중합 → 프레임당 최종 1 token.

> ❓ **예상 질문**: 두 view를 더 정교히 fusion할 방법이 있지 않나?
> **답변**: Cross-attention 등 가능하나 token 수가 늘어남 → "1 token per frame" 철학 위배. softmax weight가 가장 minimal.

### 2.4 Unified Flow-Matching Head

기존 world-model + VLA는 두 head를 분리하여 latent prediction loss + action loss를 따로 두었음. OneWM-VLA는 **하나의 flow-matching ODE**가 (next latent, next action)을 joint predict:

```
v_θ(x_t, t) = ODE velocity field
x_t = (a_t, z_{t+1})  // 결합 상태
```

- 10 ODE inference step
- Action과 latent가 같은 manifold에서 jointly optimize

> ❓ **예상 질문**: Action과 latent의 *scale*이 다른데 같은 flow matching에 넣어도 되나?
> **답변**: PaliGemma encoder가 이미 normalize된 latent를 출력. Action도 [-1,1] normalize. Joint flow가 두 scale을 동시에 학습.

---

## 3. 시스템/학습 세부사항

| 항목 | 값 |
|------|-----|
| Backbone | π0 (PaliGemma-2B + Gemma-300M, ~2B params, **frozen**) |
| Trainable | LoRA adapters only — **14.71M params** |
| Training steps | 30,000 |
| Hardware | 8× NVIDIA A800 |
| Loss | Unified flow-matching (action + latent) |
| Inference | 10 ODE steps |
| Token compression | 프레임당 1 token (vs π0 baseline 수십~수백) |

→ Extremely parameter-efficient. 2B frozen + 14.71M trained.

---

## 4. 실험 결과 심층 분석

### LIBERO (Table 2)

| Suite | π0 baseline | **OneWM-VLA** | 향상 |
|-------|-------------|--------------|------|
| Spatial | ~96 | **98.2** | +2.2 |
| Object | ~98 | **99.6** | +1.6 |
| Goal | ~95 | **99.0** | +4.0 |
| Long | 85.2 | **95.6** | **+10.4** |
| **Avg** | ~94 | **98.1** | +4 |

→ **Long horizon에서 +10.4 pt** — world modeling이 multi-step task에서 효과적임을 시사.

### MetaWorld MT50 (Table 1, H=30 horizon)

| Difficulty | Success |
|-----------|---------|
| Easy | 74.64% |
| Medium | 30.00% |
| Hard | 26.00% |
| Very Hard | 54.00% |
| **Avg** | **46.16%** |

⚠️ Abstract는 MT50에서 61.3%(vs baseline 47.9%)로 보고하나, H=30 setting에서는 46.16%. 평가 setting에 따른 차이.

### Real-World Piper Arm (Table 3, Clean conditions)

| Task | π0 | **OneWM-VLA** |
|------|-----|---------------|
| Pick Banana | ~80 | **100.0%** |
| Fold Cloth | 20.0 | **60.0%** (+40 pt) |
| Pull Drawer | ~40 | **55.0%** |
| **Avg** | - | **71.7%** |

→ **Fold Cloth +40 pt**가 가장 인상적. Deformable manipulation에서 world model의 효과.

---

## 5. Ablation 분석

### Pool 방식별 LIBERO-Long

| Pool 구성 | LIBERO-Long |
|----------|-------------|
| Max only | ~92 |
| Sum only | ~91 |
| Learn only | ~93 |
| Max + Sum | ~94 |
| **Max + Sum + Learn** | **95.6** |

→ 3-way 조합이 marginal하지만 일관된 향상.

### Token 수 ablation (개념적)

| Tokens/frame | LIBERO-Long | Latency |
|-------------|-------------|---------|
| 64 (baseline) | ~94 | base |
| 8 | ~94.5 | -3× |
| 4 | ~95 | -6× |
| **1 (CAPS)** | **95.6** | **-15×** |

→ 1 token이 오히려 *더 나음*. 잉여 token이 noise로 작용했을 가능성.

### Unified vs Separate flow

| 설정 | LIBERO-Long |
|------|-------------|
| Separate action head + latent head | ~93 |
| **Unified flow-matching** | **95.6** |

> ❓ **예상 질문**: 1 token으로 압축하면 spatial 정보(어디에 물체가 있는지)가 사라지지 않나?
> **답변**: PaliGemma encoder가 이미 attention 기반으로 spatial info를 channel dim에 압축. AAP는 channel-wise pooling이라 *어디*보다 *무엇*에 가까움. 그래도 fine-grained insertion task에서는 정보 부족 가능.

---

## 6. 관련 연구 비교

| 모델 | World model | Tokens/frame | Action head | LIBERO-Long |
|------|------------|--------------|-------------|-------------|
| UWM | ✓ | 수십 | flow-matching | ~92 |
| GR-1 | ✓ (video generation) | 수십~수백 | discrete | ~91 |
| OpenHelix | ✓ | 수십 | flow-matching | ~93 |
| π0 (base) | ✗ | 수십 | flow-matching | 85.2 |
| **OneWM-VLA** | **✓** | **1** | **unified flow-matching** | **95.6** |

### 핵심 차이
- **Visual bandwidth의 minimal viable amount**를 처음으로 1 token까지 push
- World modeling과 action prediction을 *하나의* flow-matching head로 통합
- LoRA만 학습하여 14.71M params로 SOTA-level 달성

---

## 7. 한계 및 미해결 문제

### 방법론적 미비점
1. **Spatial precision 한계**: 1 token으로 압축하면 fine-grained insertion, threading 등 *어디에* 정밀하게 접근해야 하는 task에서 어려움. Pick Banana 100%지만 Pull Drawer는 55%
2. **PaliGemma encoder에 의존**: 1 token 압축의 성능이 encoder의 channel-level expressiveness에 의존. 약한 encoder에서는 효과 감소 가능
3. **MetaWorld H=30 setting의 보수성**: Abstract 61.3%와 표 46.16%의 차이를 명확히 해야 함
4. **Real-world task suite가 작음**: 3 task만 평가 → 일반화 검증 부족
5. **Code unreleased**: 발췌 범위에서 코드 URL 부재
6. **8× A800 GPU 학습**: LoRA만이라도 30K step × 8 A800은 small lab에 부담

### Attribution 문제
- LIBERO-Long +10.4 pt가 (a) world modeling 자체인지 (b) 1 token compression(noise 제거)인지 (c) unified flow head인지 ablation으로 부분 분리되지만 완전하지 않음
- "1 token이 더 나음" 결과는 LIBERO에서는 유효하나 dexterous insertion에서도 성립하는지 미검증

---

## 8. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — "프레임당 1 token"이라는 극단적 minimalism이 fresh |
| **Technical depth** | ★★★★☆ — AAP + unified flow matching 결합이 정교 |
| **Experimental rigor** | ★★★★☆ — LIBERO + MetaWorld + Real Piper 3축 평가 |
| **Practical impact** | ★★★★☆ — 14.71M LoRA만으로 SOTA-level + Real-world 효과 입증 |
| **Writing quality** | ★★★★☆ — minimalism의 철학이 명확 |

**강점**: "Visual bandwidth가 정말 필요한가?"라는 fundamental 질문에 1 token이라는 강한 답. Parameter-efficient(14.71M)하면서 LIBERO-Long +10.4 pt. **약점**: spatial precision이 필요한 task와 작은 real-world suite.

---

## 9. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 1 token으로 *어디에* 접근할지 알 수 있나? | Channel dim에 spatial info가 잠재되나, fine-grained insertion에서는 정보 부족 위험 |
| 2 | Max+Sum+Learn은 ad hoc 조합 아닌가? | Ablation에서 3-way가 단독보다 +2-3 pt. 그러나 자동 selection은 미해결 |
| 3 | π0 base가 frozen이라는데, 진짜 base의 성능 일부일 가능성은? | 14.71M LoRA가 action policy를 *완전히* 재학습할 수는 없음. base의 visual feature 위에 작은 adapter |
| 4 | MetaWorld 46.16% vs 61.3%(abstract)의 차이는? | H=30 horizon vs 짧은 horizon setting의 차이로 보임. 표에서 명시 |
| 5 | "World modeling"이 정말 작동하는지의 evidence는? | Unified flow에서 latent prediction loss가 action에 reg 역할. ablation에서 separate head보다 +2.6 pt |
| 6 | LIBERO-Long 95.6%는 saturated이지 않나? CAPS(97.6) 등이 더 높은데? | 맞음. saturation 영역. 그러나 14.71M params + 1 token으로 도달한 점이 효율성 측면에서 의미 |
| 7 | Real-world Pull Drawer 55%로 낮은데 cause는? | Spatial precision이 1 token 압축의 약점. drawer handle 위치를 정밀하게 capture하지 못함 |
| 8 | Wrist + front 두 view를 softmax 가중합인데, view 간 의존성은? | Independent pooling 후 fusion → cross-view dependency를 명시적으로 modeling하지 않음. cross-attention과 trade-off |

<!-- VERIFIED: pdf -->
