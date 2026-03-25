# GR-1: Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation

> **한 줄 요약**: 195M GPT-style transformer (46M trainable)를 Ego4D 800K 비디오 클립으로 video generation 사전학습한 후, CALVIN에서 **ABCD→D avg len 4.21** (HULC 3.06 대비 +37.6%), unseen scene zero-shot **85.4%** (기존 53.3% 대비 +60%) 달성.

---

## 1. 배경 및 동기

- 로봇 데이터만으로는 diversity와 규모 부족 → 일반화 한계
- 인터넷 비디오에 풍부한 physical dynamics knowledge가 있으나 action label 없음
- **Video generative pre-training**이 manipulation의 일반화에 직접 기여하는가?

---

## 2. 방법론 심층 분석

### 2.1 Architecture

| Component | Spec |
|-----------|------|
| Transformer | **12 layers, 12 heads, hidden 384** |
| Total params | **195M** (46M trainable) |
| Language | CLIP text encoder (frozen) |
| Vision | MAE pre-trained ViT + Perceiver Resampler (frozen) |
| Robot state | Linear layers (6D EE pose + gripper) |

### 2.2 Token Sequencing

```
(l, s_{t-h}, o_{t-h}, [OBS], [ACT], l, s_{t-h+1}, ..., l, s_t, o_t, [OBS], [ACT])
```

Causal attention, [ACT]과 [OBS]은 미래 position에 attend하지 않음.

### 2.3 Output

- **Video decoder**: Transformer + mask tokens → image patch reconstruction (MSE loss)
- **Action decoder**: Linear layers → arm (Smooth-L1) + gripper (BCE)

### 2.4 Training

| Stage | Data | Batch | LR | Epochs |
|-------|------|-------|-----|--------|
| Pre-training | **Ego4D** (800K clips, 8M frames, 3500 hours) | 1024 | 3.6e-4 | 50 |
| Fine-tuning | CALVIN (22,966 trajectories) | 512 | 1e-3 | 20 |

---

## 3. 실험 결과 심층 분석

### CALVIN ABCD→D

| Model | Avg Len | 5-task SR (%) |
|-------|---------|-------------|
| HULC | 3.06 | 38.3 |
| MT-R3M | - | - |
| **GR-1** | **4.21** | **73.1** |

### CALVIN ABC→D (Unseen Scene Zero-shot)

| Model | 1-task SR (%) |
|-------|-------------|
| MT-R3M | 53.3 |
| **GR-1** | **85.4 (+60%)** |

### Data Efficiency

| Data | Avg Len |
|------|---------|
| 10% of CALVIN | GR-1: **2.00** vs HULC: 1.11 |
| Full | GR-1: **4.21** vs HULC: 3.06 |

### Unseen Language

| Model | Avg Len |
|-------|---------|
| HULC | 1.82 |
| **GR-1** | **2.17** |

### Real Robot

| Setting | SR (%) |
|---------|--------|
| Seen objects | 79 |
| Unseen instances | 73 |
| Unseen categories | 30 |
| Drawer manipulation | 75 (vs RT-1: 35) |

---

## 4. Ablation 분석

### Video Pre-training 기여 (ABCD→D)

| Setting | Avg Len |
|---------|---------|
| No video pred + No pre-train | 3.33 |
| Video prediction only | 3.82 |
| **Full (pre-train + video pred)** | **4.21** |

### Future Prediction Step (Δt)

| Δt | Avg Len |
|----|---------|
| 1 | 3.61 |
| **3** | **3.82 (optimal)** |
| 5 | 3.67 |

---

## 5. 한계 및 미해결 문제

1. **LIBERO 결과 없음**: CALVIN only. LIBERO에서의 성능 미검증
2. **Image generation overhead**: 추론 시 future image token 생성 → latency 증가 (수치 미보고)
3. **195M은 소형**: 이후 GR-2 (수십B)로 scaling. 195M의 capacity 한계
4. **Unseen categories 30%**: 완전히 새로운 물체에서 약함
5. **Pre-training 비용**: Ego4D 800K clips, 50 epochs → 상당한 compute

---

## 6. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — Video generative pre-training for robots의 선구적 연구 |
| **Technical depth** | ★★★★☆ — Δt ablation, data efficiency 분석 |
| **Experimental rigor** | ★★★★☆ — CALVIN + real robot. LIBERO 부재 |
| **Practical impact** | ★★★★★ — WAM 패러다임의 기초 확립 |
| **Writing quality** | ★★★★☆ |

**강점**: Unseen scene 53.3%→85.4%는 video pre-training의 결정적 기여. 195M의 효율적 설계. **약점**: CALVIN only, latency 미보고, unseen category 30%.

---

## 7. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Fast-WAM처럼 test-time video를 skip하면? | 이 ablation 미수행. 이후 Fast-WAM이 이를 검증 |
| 2 | Ego4D 대신 다른 video dataset이면? | Domain match가 중요. Kitchen video → kitchen task에 유리 |
| 3 | 195M을 1B+로 scaling하면? | GR-2가 이를 수행. 97.7% on 100+ tasks |

<!-- VERIFIED: pdf -->
