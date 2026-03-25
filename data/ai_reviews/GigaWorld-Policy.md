# GigaWorld-Policy: An Efficient Action-Centered World-Action Model

> **한 줄 요약**: 기존 video-centered WAM의 추론 병목을 해결하기 위해 action 예측을 중심으로 비디오 생성을 조건부로 재구성하는 action-centered WAM을 제안하여, 기존 대비 9배 빠른 추론과 7% 높은 성공률 달성.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- WAM (Motus, GR-1)은 **비디오와 action을 joint modeling** → 추론 시 비디오 생성이 필수적이어서 latency가 높음
- Joint modeling에서 **visual token이 action token에 영향** → 비디오 예측 품질이 action 정확도를 좌우
- Fast-WAM이 test-time video를 제거했으나, 학습 시에도 **action-centric 관점**에서 설계된 것은 아님

### 핵심 질문
- **Action 예측을 primary task로, 비디오 생성을 auxiliary task로 역할을 뒤집으면?**
- **비디오 생성을 추론 시 optional로 만들면서도 학습 시의 이점을 유지할 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 Action-Centered Causal Design

핵심 설계: future video token이 action token에 **영향을 줄 수 없도록** causal mask 설계:

```
Token sequence: [obs_tokens] → [action_tokens] → [future_video_tokens]
Causal mask: action은 obs만 참조, future_video는 obs+action 참조
```

$$P(\mathbf{a}, \hat{\mathbf{v}} | \mathbf{o}) = P(\mathbf{a} | \mathbf{o}) \cdot P(\hat{\mathbf{v}} | \mathbf{o}, \mathbf{a})$$

> ❓ **예상 질문**: Action이 video를 참조하지 못하면 video generation의 학습 signal이 action에 어떻게 전달되는가?
> **답변**: **공유 encoder**를 통해 간접 전달. Video generation loss가 encoder를 업데이트 → encoder representation이 dynamics-aware가 됨 → action prediction이 이 representation을 활용. Fast-WAM과 유사한 메커니즘이나, causal 구조로 명시적으로 분리.

### 2.2 Optional Video Generation at Inference

추론 시:
- **Fast mode**: Action tokens만 생성 → 9x speedup
- **Full mode**: Action + future video 생성 → 시각화, 안전 검증 등에 활용 가능

### 2.3 Large-scale Pre-training

Action-centered video generation model을 대규모 로봇 데이터로 사전학습한 후, policy learning backbone으로 활용.

---

## 3. 실험 결과 심층 분석

| 모델 | SR (%) | Inference (ms) | 대비 |
|------|--------|---------------|------|
| Motus (WAM baseline) | 78.5 | ~900 | baseline |
| pi-0.5 | 45.2 | ~200 | - |
| **GigaWorld-Policy (fast)** | **85.5 (+7%)** | **~100 (9x)** | +7% SR, 9x faster |

### RoboTwin 2.0

| 모델 | SR (%) |
|------|--------|
| pi-0.5 | 41.3 |
| **GigaWorld-Policy** | **80.5 (+95%)** |

- pi-0.5 대비 거의 **2배** 성능 → 주목할 만한 결과

---

## 4. 한계 및 미해결 문제

### 방법론적 미비점
1. **Causal 분리의 trade-off**: Action이 future video를 참조하지 못하므로, 미래 상태 예측에 기반한 plan이 불가능. Long-horizon task에서 이 제약이 성능을 제한할 수 있음
2. **사전학습 데이터 비공개**: 대규모 로봇 데이터의 구성과 규모가 불명확
3. **Motus 외 WAM과의 비교 부족**: GR-1, DiT4DiT 등과의 직접 비교 미제공
4. **Standard benchmark 부족**: RoboTwin 위주, LIBERO/CALVIN에서의 결과 부재

### Attribution 문제
- 성능 향상이 **action-centered 설계** 때문인지, **대규모 사전학습 데이터** 때문인지 분리 어려움

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Video-centered → action-centered 패러다임 전환 |
| **Technical depth** | ★★★★☆ — Causal design이 잘 설계됨 |
| **Experimental rigor** | ★★★☆☆ — 한정된 벤치마크 |
| **Practical impact** | ★★★★★ — 9x speedup의 실용적 가치 |
| **Writing quality** | ★★★★☆ — 명확 |

**강점**: "Action이 video에 의존하면 안 된다"는 통찰이 설계에 잘 반영됨. 9x 속도와 7% 성능 동시 향상. **약점**: 데이터 비공개, 한정된 벤치마크, long-horizon에서의 검증 부재.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Fast-WAM과 어떻게 다른가? | Fast-WAM은 test-time video를 단순 제거, GigaWorld는 학습 구조 자체를 action-centered로 재설계 |
| 2 | Action이 future video를 참조하면 plan 능력이 생기지 않나? | 맞음, 이것이 trade-off. Plan이 필요한 long-horizon task에서는 이 제약이 문제될 수 있음 |
| 3 | pi-0.5 대비 95% 향상의 공정성은? (학습 데이터 차이?) | 데이터 차이가 있을 수 있으며, 동일 데이터에서의 비교가 더 공정 |
| 4 | Optional video generation의 실용적 활용 사례는? | Safety check (예측 비디오가 위험하면 action 거부), human supervision, debugging |

<!-- VERIFIED: abstract-only (full PDF not publicly accessible on ar5iv) -->
