# GR00T N1: An Open Foundation Model for Generalist Humanoid Robots

> **한 줄 요약**: NVIDIA GEAR이 공개한 humanoid 전용 2.2B VLA로, Eagle-2 VLM(SigLIP-2 vision + SmolLM2 LLM, 1.34B)을 System 2(10Hz)로, flow-matching diffusion transformer를 System 1(120Hz)로 결합한 dual-system 구조이며, 88h 실제 GR-1 + 827h neural trajectory + 6,500h sim의 heterogeneous data로 학습해 RoboCasa(32.1%)·DexMimicGen(66.5%)·GR-1 tabletop(50.0%)에서 BC-Transformer/Diffusion Policy 베이스라인을 능가했다.

---

## 1. 배경 및 동기

기존 VLA(OpenVLA 등)는 single-arm tabletop 편향이며, humanoid의 양팔+상체+양손(40+ DoF) 제어에는 부적합. 또한 robot demonstration 수집 비용이 높아 data efficiency가 핵심. 저자들은 (1) humanoid에 특화된 dual-system foundation model, (2) real + neural-generated + synthetic + human-video를 혼합한 "data pyramid", (3) open weights 공개로 커뮤니티 가속을 목표로 한다.

---

## 2. 방법론 심층 분석

### 2.1 Dual-System Architecture

- **System 2 — Eagle-2 VLM (1.34B/2.2B)**: SigLIP-2 vision encoder(224×224, frame당 64 image token) + SmolLM2 LLM. 10Hz on L40 GPU.
- **System 1 — Diffusion Transformer (DiT)**: flow matching 기반 action head, action chunk H=16, denoising K=4 steps. Sampling latency 63.9ms on L40 (bf16) → 120Hz motor control.
- 두 모듈은 end-to-end joint training, latent token으로 결합.

### 2.2 Heterogeneous Training Data ("Data Pyramid")

| Source | Scale | 비고 |
|---|---|---|
| Real GR-1 teleop | 88 hours | Bimanual humanoid |
| Neural trajectories | 827 hours | Video-generated synthetic |
| Simulation (DexMimicGen, Isaac Sim) | 6,500 hours | 11h만에 780K trajectory 생성 |
| Human videos (Ego4D, Assembly-101) | 대규모 | Web-scale |

DexMimicGen 기반 synthetic generation으로 real data 대비 ~74× 규모의 sim data 확보.

### 2.3 학습 인프라

H100 1024 GPUs(NVIDIA Quantum-2 InfiniBand), GR00T-N1-2B 사전학습에 약 50,000 H100 GPU-hours.

---

## 3. 실험 결과

### 3.1 Simulation Benchmarks (100 demos/task)

| Benchmark | BC-Transformer | Diffusion Policy | **GR00T-N1-2B** |
|---|---|---|---|
| RoboCasa Kitchen (24 atomic) | 26.3% | 25.6% | **32.1%** |
| DexMimicGen | 53.9% | 56.1% | **66.5%** |
| GR-1 Tabletop | 16.1% | 32.7% | **50.0%** |

### 3.2 Real-World GR-1 Humanoid

| Setting | Diffusion Policy | GR00T-N1-2B |
|---|---|---|
| 10% data | 10.2% | **42.6%** |
| Full data | 46.4% | **76.8%** |

10% data regime에서 +32.4%p 차이 — 강력한 data efficiency.

### 3.3 Ablation: Neural Trajectory Co-training

RoboCasa 30/100/300 demo regime에서 +4.2 / +8.8 / +6.8%p 향상; real-world 8 task 평균 +5.8%p. Synthetic data가 실제 성능 기여 입증.

### 3.4 Fine-tuning 비용

Single A6000 GPU + adapter batch=200 또는 vision encoder unfreeze batch=16으로 적응 가능 — deployment 친화적.

---

## 4. 한계

1. **Locomotion 미포함**: 본 논문은 manipulation 한정, walking/balance는 별도 controller 필요.
2. **LIBERO 평가 부재**: 본 paper에는 LIBERO score가 보고되지 않음. **YAML의 `libero_spatial=93.2 / object=94.8 / goal=88.6 / long=66.4`는 본 paper의 Table에서 확인 불가** — 외부 출처(예: 후속 GR00T-N1.5 또는 community 평가) 추정. RoboCasa 32.1%만 paper-grounded.
3. **하드웨어 의존**: Fourier GR-1에 특화된 데이터 — 다른 humanoid로의 zero-shot transfer 미검증.
4. **연산 비용**: 50K H100-hours는 학계 재현 불가 수준.
5. **Sim-to-real gap 정량화 부족**: 6,500h sim의 실제 성능 기여를 RoboCasa 외에서 자세히 분리하지 않음.

---

## 5. 총평

| 항목 | 평가 |
|---|---|
| Novelty | ★★★★☆ — Dual-system + heterogeneous data pyramid |
| Technical depth | ★★★★☆ — Eagle-2 + DiT의 명확한 통합 |
| Experimental rigor | ★★★★☆ — 다중 sim + real humanoid |
| Practical impact | ★★★★★ — Open weights, humanoid 생태계 마중물 |

**강점**: Humanoid VLA 표준 정립, 88h real data만으로 76.8% real success. **약점**: LIBERO 직접 보고 없음, locomotion 미포함, 학계 재현 불가.

---

## 6. 예상 질문

| # | 질문 | 답변 요점 |
|---|---|---|
| 1 | π0의 single-system flow matching 대비 dual-system이 정말 필요한가? | 120Hz motor control + 10Hz reasoning 분리는 humanoid의 high-DoF 제어에 합리적. 단, π0도 50Hz까지 도달 가능하므로 ablation 부족. |
| 2 | Neural trajectory 827h가 실제로 필요한가? | Ablation에서 RoboCasa +4.2~8.8%p 기여(§3.3) — 유의미하나 critical은 아님. |
| 3 | Eagle-2 VLM이 Qwen2.5-VL이나 PaliGemma 대비 우월한가? | 비교 ablation 부재. SmolLM2의 작은 크기(SigLIP-2 + 1.34B)가 inference latency에는 유리. |

<!-- VERIFIED: pdf -->
