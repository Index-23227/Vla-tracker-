# Octo: An Open-Source Generalist Robot Policy

> **한 줄 요약**: 93M transformer를 OXE 800K trajectory (25 datasets)에서 학습한 오픈소스 generalist policy로, 9개 로봇 플랫폼에서 검증. Fine-tuning 시 baseline 대비 **+52%p**, 100 target demos + 5시간 A5000으로 새 로봇·센서·action space에 적응.

---

## 1. 배경 및 동기

- 로봇 학습의 데이터 파편화 해소 → Open X-Embodiment 기반 범용 정책
- **오픈소스**: 누구나 접근·fine-tuning 가능한 generalist policy

---

## 2. Architecture (93M)

| Component | Spec |
|-----------|------|
| Transformer | **12 layers, 12 heads, hidden 768, MLP 3072** |
| Total params | **93M** |
| Language encoder | T5-base (111M, frozen) → 16 tokens |
| Image encoder | Shallow CNN → 16×16 patches (256 tokens/view) |
| Action head | **Diffusion** (3-layer MLP, hidden 256, 20 denoising steps, cosine schedule) |

Block-wise masked attention으로 flexible input/output adaptation.

### Training

| 항목 | 값 |
|------|-----|
| Data | **800K trajectories** (25 OXE datasets) |
| Hardware | **TPU v4-128** |
| Duration | **14 hours** (300K steps) |
| Batch | 2048 |
| LR | 3e-4 (inverse sqrt decay) |
| Fine-tuning | **~5 hours on A5000** (50K steps) |

---

## 3. 실험 결과 심층 분석

### Zero-Shot (3 platforms)

| Platform | Octo | RT-1-X | RT-2-X (55B) |
|---------|------|--------|-------------|
| WidowX (language) | **70%** | - | ~70% |
| UR5 | **75%** | - | - |
| RT-1 Robot | **50%** | - | ~50% |

"Octo performed similarly to RT-2-X (55 billion parameters)"

### Fine-Tuning (6 new domains, Table I)

| Domain | Octo | Scratch | VC-1 |
|--------|------|---------|------|
| Berkeley Insertion (force-torque) | **70%** | 10% | 5% |
| Stanford Coffee | **75%** | 45% | 0% |
| CMU Baking | **50%** | 25% | 30% |
| Berkeley Pick-Up (joint pos) | **60%** | 0% | 0% |
| Berkeley Coke (new embodiment) | **100%** | 20% | 10% |
| Berkeley Bimanual (ALOHA) | **80%** | 20% | 50% |
| **Average** | **72%** | **20%** | **15%** |

- **+52%p vs scratch**, +57%p vs VC-1
- 100 target demos로 new sensor (force-torque), new action space (joint), new embodiment 적응

### Model Ablation (WidowX)

| Setting | SR (%) |
|---------|--------|
| Octo-Small (full) | 83% |
| RT-X dataset mix | 60% |
| Single robot (Bridge only) | 43% |
| Discretized actions | 18% |
| MSE continuous | 35% |

- **Diffusion >> discrete (18%)**: Diffusion head의 결정적 중요성
- **More datasets = better** (43% → 83%)

---

## 4. 한계 (저자 명시)

1. **Wrist camera 처리 약함**: 27% 데이터만 wrist camera 포함
2. **Language vs goal gap**: Language-conditioned가 goal-conditioned보다 일관적으로 우수 → goal conditioning 구조 개선 필요
3. **93M capacity 제한**: 최신 VLA (7B+) 대비 capacity 부족 → 복잡한 reasoning 불가
4. **이후 OpenVLA에 의해 추월**: VLM backbone의 중요성을 역설적으로 보여줌

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★★ — 최초의 실용적 open-source generalist policy |
| **Technical depth** | ★★★★☆ — 유연한 아키텍처, 9 platforms |
| **Experimental rigor** | ★★★★★ — 9 platforms, fine-tuning 6 domains |
| **Practical impact** | ★★★★★ — VLA 연구의 baseline |
| **Writing quality** | ★★★★★ |

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | VLM 추가하면? (→ OpenVLA) | OpenVLA가 정확히 이를 수행. Language understanding 향상 + 성능 향상 |
| 2 | Diffusion head 대신 token prediction이면? | Ablation에서 18% (discrete) vs 83% (diffusion). Diffusion이 압도적 |
| 3 | 93M이 충분한가? | No. HPT, CrossFormer 등이 더 큰 trunk으로 더 나은 결과. Scaling이 필요함을 시사 |

<!-- VERIFIED: pdf -->
