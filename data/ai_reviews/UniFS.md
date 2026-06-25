# UniFS: 통합 Fast-to-Slow 계층 구조 VLA 모델 리뷰

> arXiv:2606.22794 | JD.com, Tianjin University, Fudan University | 2026-06-22

## 1. 한 줄 요약
UniFS는 단일 VLM 백본 안에서 층(layer)별로 서로 다른 업데이트 주파수를 부여하는 "Fast-to-Slow" 통합 계층 구조를 도입하여, 기존 이중 시스템(Fast-Slow Dual System) VLA의 주파수 딜레마를 해소하고 LIBERO 평균 98.3% 성공률과 17.8ms 평균 추론 지연을 동시에 달성한 새로운 VLA 아키텍처이다.

## 2. 문제 정의 및 동기
주류 VLA 모델은 7B+ 규모 VLM 백본 때문에 단일 forward pass 지연이 커서 1–10 Hz의 제어 주파수에 머문다. 이를 해결하려 GR00T-N1, Fast-in-Slow 같은 이중 시스템이 등장했으나, (1) 큰 업데이트 간격은 stale context로 인한 semantic drift를 일으키고 (2) 작은 간격은 가속 효과를 상쇄하는 근본적인 "주파수 딜레마"가 있다. 또한 action expert는 VLM 최종 layer 표현만 받기 때문에 중간 layer의 풍부한 다중 스케일 의미 정보를 버린다.

## 3. 핵심 아이디어 (생물학적 영감)
인간 뇌의 대뇌피질–소뇌 통신이 30–150Hz 감마파(감각)와 13–30Hz 베타파(고차원 인지)처럼 다중 주파수 대역으로 동시에 정보를 전달하는 multi-timescale 신경 처리 원리에서 영감을 얻었다. 저자들은 π0와 VLA-Adapter의 layer별 hidden state cosine distance를 시각화하여, 실제로 layer 깊이에 따라 시간적 변화 크기가 크게 다르다는 것을 경험적으로 확인했다(Figure 2).

## 4. 아키텍처 개요
- **Vision Encoder**: DINOv2 + SigLIP (OpenVLA, VLA-Adapter 계승)
- **LLM Backbone**: Qwen2.5-0.5B
- **Action Expert**: noisy action prior를 입력받아 cross-attention으로 multi-scale VLM latent와 융합 후 MLP head로 action chunk를 디코딩하는 transformer block 스택
- **VLM 분할**: 5개 그룹, time scale 1, 2, 4, 8, 16
- **출력**: 7-DoF action (Δx, Δy, Δz, Δϕ, Δθ, Δψ, gripper), chunk length 8

## 5. 세 가지 핵심 설계
### (1) Fast-to-Slow Architecture (FSA)
VLM의 layer를 K개 그룹으로 계층화하고 각 그룹 k에 업데이트 주기 n_k를 부여 (n_1 < n_2 < ... < n_K). 얕은 layer는 매 step마다(n_1=1) 갱신하여 fast 동역학을 포착하고, 깊은 layer는 N step마다(n_K=N) 갱신하여 안정된 의미 컨텍스트를 캐싱한다. t step에서 t mod n_k = 0일 때만 재계산하고 그 외에는 이전 출력을 그대로 사용.

### (2) Latent Vector Inversion (LVI)
경험적으로 deeper layer가 오히려 더 높은 시간적 주파수를 보이는 현상(action 출력에 가까워 fast-frequency supervision의 영향을 받기 때문)을 관찰. 이에 LLM과 action expert 간 latent 상호작용 순서를 뒤집어, 초기 noisy action proposal은 깊은 (semantic) feature와 상호작용하고, 정제된 action 출력은 얕은 (high-frequency) feature와 상호작용하도록 재배선. 결과적으로 fast 동역학이 얕은 layer로 이동하여 깊은 layer가 캐시 가능해진다.

### (3) Multi-Level Supervision (MLS)
LVI로 인해 모델이 fast-frequency component를 통한 shortcut을 학습할 위험을 막기 위해, 최종 layer뿐 아니라 action expert 내 각 frequency group에 보조 L1 loss를 부여. 낮은 frequency group은 동일 feature에 다른 GT action을 매핑하도록 강제되어 coarse한 장기 계획을, 높은 frequency group은 fine-grained 즉각 보정을 학습.

## 6. 학습 전략
- **Temporal Batch Sampling (Alg. 1)**: 한 trajectory에서 연속 window를 무작위로 선택 후 K step을 시간 순으로 정렬해 배치 구성. 인과적 의존성을 보존하면서도 stochasticity를 부여.
- **Frequency Feature Replacement (FFR, Alg. 2)**: 비동기 layer 실행을 GPU 병렬화와 양립시키기 위한 학습 시 트릭. 모든 layer/모든 timestep을 표준 forward로 계산한 뒤, 인덱싱 연산으로 H'[t, i] ← H[⌊t/f⌋·f, i]를 적용해 사후에 frequency-aware feature를 정렬. 정적 계산 그래프를 유지하면서 비동기 효과를 모사.
- **Optimization**: AdamW, LR 1e-5 constant, 4×A100, per-GPU batch 16 (global 64), effective window 128, TFDS shuffle buffer 10000.

## 7. 추론 가속
- VLM 백본의 lower-frequency group은 f_k step마다 한 번만 계산하고 hidden state를 캐싱하여 재사용.
- Action expert도 LVI 덕분에 초기 layer는 느린 semantic 입력을 받아 낮은 주기로 동작하고, 마지막 layer만 매 step 실행.
- 이론적 가속 ≈ 2.6×.

## 8. 실험 결과 — LIBERO
| Suite | UniFS | VLA-Adapter (baseline) | Δ |
|---|---|---|---|
| Spatial | 99.6 | 97.2 | +2.4 |
| Object | 99.6 | 98.8 | +0.8 |
| Goal | 98.2 | 93.4 | +4.8 |
| Long | 95.6 | 93.6 | +2.0 |
| **Avg** | **98.3** | 95.8 | **+2.5** |

비교 SOTA: EO-1 98.2, X-VLA 98.1, MemoryVLA 96.5, CogVLA 97.4, π0.5 94.3, OpenVLA-OFT 95.3. UniFS는 평균 98.3%로 신규 SOTA를 기록.

## 9. 효율성 (Table 2)
| 모델 | Throughput (Hz) ↑ | Latency (ms) ↓ |
|---|---|---|
| OpenVLA | 4.2 | 239.6 |
| OpenVLA-OFT | 109.7 | 72.9 |
| VLA-Adapter | 219.2 | 36.5 |
| **UniFS (mean)** | **449.4** | **17.8** |
| UniFS (fastest) | 650.4 | 12.3 |
| UniFS (slowest) | 245.4 | 32.6 |

VLA-Adapter 대비 평균 2.1× 가속.

## 10. Ablation (Table 3)
| FSA | LVI | MLS | Avg SR (%) | Latency (ms) |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | 95.8 | 36.5 |
| ✓ | ✗ | ✗ | 70.3 | 22.6 |
| ✓ | ✓ | ✗ | 94.3 | 17.8 |
| ✓ | ✓ | ✓ | **98.3** | **17.8** |

FSA만 켜면 LIBERO-10이 93.6→28.4로 붕괴 (주파수–action expert 불일치). LVI를 더하면 94.3까지 회복, MLS까지 더하면 98.3 SOTA. 세 구성요소 모두 필수임이 증명됨.

## 11. 강점 및 한계
**강점**: (a) 이중 모듈 분리 없이 단일 VLM 내부에서 fast/slow 통합 → end-to-end gradient flow가 보장됨; (b) feature caching이 train-test discrepancy 없이 자연스럽게 학습됨; (c) implicit memory mechanism (slow 경로가 historical dependency를 보존); (d) Franka 실로봇 검증.
**한계**: (a) 5개 frequency group, time scale {1,2,4,8,16}은 휴리스틱 — 자동 탐색은 미진; (b) LLM 백본이 Qwen2.5-0.5B로 매우 작아 대규모 backbone에서의 확장성은 미검증; (c) 비교가 LIBERO 시뮬레이션 중심 (CALVIN, SimplerEnv 등 다른 벤치마크 부재); (d) Latent inversion이 backbone과 action expert layer 수의 정합성을 요구.

## 12. 의의 및 향후 방향
UniFS는 "분리된 dual-system이 아닌 통합 backbone 안에서 주파수 계층을 부여한다"는 새로운 설계 철학을 제시한다. Brain-inspired multi-timescale 처리를 VLA에 본격 도입한 첫 사례 중 하나로, π0/VLA-Adapter 계열의 효율-성능 트레이드오프 곡선을 명확히 외측으로 이동시켰다. 향후 (a) 더 큰 VLM 백본 (Qwen2.5-7B, LLaMA3 등)과의 결합, (b) frequency group/time scale의 자동 탐색, (c) 실세계 dynamic 환경에서 closed-loop 평가, (d) 다른 벤치마크 (CALVIN, RoboTwin, SimplerEnv)로의 일반화 검증이 후속 연구의 핵심 과제가 될 것이다.

<!-- VERIFIED: pdf -->
