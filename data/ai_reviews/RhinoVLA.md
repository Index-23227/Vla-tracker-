## RhinoVLA: Edge-deployable VLA via Token-Efficient Qwen3-VL Backbone and 72D Unified Cross-Robot Interface

### 1. 한 줄 요약

Huixi Intelligence가 발표한 RhinoVLA는 token-efficient **Qwen3-VL-2B** backbone과 0.40B continuous flow-matching Action Expert를 자체 개발 edge SoC **Huixi R1**(500 TOPS INT8, 7nm)과 알고리즘-시스템 co-design한 deployment-oriented VLA로, **View Registry / 72D 통일 슬롯 공간 / robot-instance LoRA**라는 cross-robot interface를 통해 이종 로봇 데이터를 통합 학습하여, π0.5와 비슷한 파라미터 규모에서 동등한 LIBERO 90.0% 성능을 달성하면서 edge에서 **11.69 Hz end-to-end inference**를 실현했다(arXiv 2606.07383, 2026-06-05).

---

### 2. 배경 및 동기

- 현대 VLA(π0/π0.5, GR00T N1, RDT 등)는 강력한 VLM backbone + iterative action generation으로 성능은 좋아졌지만 **edge 배포에서 10 Hz 실시간 closed-loop 요건을 충족하기 어려움**.
- 저자들의 Orin roofline 분석(Fig. 2): π0.5, GR00T, RDT는 5 Hz에서도 effective 17.2 TFLOPS roofline에 근접/초과 → 10 Hz는 사실상 불가능.
- π0.5 PyTorch-SDPA 분석(858.3 ms 총 latency 중 vision encoder 69.3 / **VLM backbone 528.0** / action expert 257.0 ms): VLM과 action expert가 90% 이상 차지.
- VLM 내부 operator breakdown: gate/up/down_proj(MLP)이 **74.7%**, attention proj는 단지 7.2%. MLP GEMM의 FLOPs = 2BSd_in d_out → token 수 S에 선형 → **visual/context token 수 감소가 핵심 lever**.

---

### 3. 방법론

#### 3.1 Architecture (Fig. 3, Table 8)

- **VLM backbone**: Qwen3-VL-2B (2.13B, 28 layers, hidden 2048, MLP 6144, 16 attn heads / 8 KV heads, head dim 128).
- **Visual token 압축**: 256×256 입력 → 16×16 raw grid → spatial merging으로 **이미지당 64 token** (PaliGemma-224의 256 대비 4× 감소).
- **Action Expert**: 0.40B, 18 layers, hidden 1024, MLP 3072, 16/8 attention heads (GQA grouping 2). Qwen-compatible attention과 cache 사용. Qwen3-VL의 마지막 18 layer **KV cache를 직접 conditioning**.
- **Flow matching**: clean target $z\in\mathbb{R}^{H\times 72}$, noise $a\sim\mathcal{N}(0,I)$, $x_t = (1-t)a + tz$, predict $\hat{v}_\theta = f_\theta(x_t, t, s, m_s, m_a, c_{vlm}, r)$, target $z-a$.
- Suffix token = 1 state + 30 noisy action tokens.

#### 3.2 통일 Cross-Robot Interface (3가지 핵심)

**A. View Registry**: 각 이미지에 `[head|rgb]`, `[left_wrist|rgb]`, `[head|depth]` 같이 **role-modality 태그**를 명시적으로 prefix 삽입 → 이미지 순서 dependence 제거 (Table 1).

**B. Unified 72D physical slot space** (Table 2):
- D0–D6 Arm 0 (rad), D7–D13 Arm 1 (rad), D14–D15 parallel grippers (closed ratio), D16–D31 Hand 0 active DoF (4-3-3-3-3 thumb-to-little 할당), D32–D47 Hand 1, D48–D50 head RPY, D51–D52 torso pitch/lift, D53–D54 folded-leg, D55–D57 waist RPY, D58–D60 base velocity (m/s, rad/s), D61–D71 reserved.
- Binary state/action mask로 유효 dimension만 supervision → 결손 slot이 zero target으로 오해되지 않음.

**C. Robot-instance LoRA**: shared Action Expert FFN에 LoRA(r=64) 삽입, `instance_id`로 **hard-routed**. 추론시 선택된 LoRA를 base 가중치에 merge → 모든 로봇이 동일 18-layer graph 공유 → kernel/최적화 재사용. 새 로봇 추가 시 LoRA + normalization stat만 추가하면 됨.

#### 3.3 Training Strategy

- Qwen3-VL **frozen**, VLM-LoRA + shared Action Expert + instance-LoRA 공동 최적화.
- 데이터셋 sampling: $p_i \propto N_i^{0.43}$ (π-style power-law).
- Masked flow-matching loss: $L_{FM} = \frac{\sum_{h,d} m_a(d) w(h,d) \lVert \hat{v}_\theta - (z-a) \rVert^2}{\sum_{h,d} m_a(d) w(h,d) + \epsilon}$.
- Base prediction + adapter residual regularization → instance LoRA가 전체 정책을 대신하지 않게 보장.
- **Post-training**: VLM/VLM-LoRA + Action Expert 대부분 frozen, target 로봇의 instance LoRA만 업데이트.

#### 3.4 Huixi R1 배포 최적화

- **Compilation**: (i) FlashAttention-style tiling을 R1 software-managed SPM에 적응 → attention kernel이 peak compute의 80%+ 도달. (ii) Aggressive operator fusion(RMSNorm/proj/bias/activation/residual). (iii) Fine-grained operator task scheduling.
- **Mixed-precision (W8A16)**: INT8 weight + FP16 activation. W8A8는 task SR 저하 확인. 커스텀 W8A16 GEMM kernel은 weight load / dequant / MAC을 **단일 pipeline에 fuse**, per-channel scaling, memory-channel-aware weight layout. 측정: π0.5 up_proj W16A16 191µs → **W8A16 113µs (1.69× speedup, 50.6% compute utilization)**.
- **Parallel encoding**: 3-view ViT를 **batched로 동시 처리** → 34.52 ms → **24.31 ms**.

---

### 4. 실험 결과

#### 4.1 LIBERO (Table 4, 단일 jointly-trained checkpoint)

| Suite | Spatial | Object | Goal | Long | **Avg** |
|---|---|---|---|---|---|
| RhinoVLA | 93.0 | 91.0 | 93.4 | **82.4** | **90.0** |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| π0 | 90.0 | 86.0 | 95.0 | 73.0 | 86.0 |
| π0-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| SmolVLA | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |

- π0, π0-FAST 대비 **Long suite에서 +9.4, +22.2%p**로 큰 우위.
- π0.5와 갭(6.9%p)이 있지만 π0.5는 multi-source co-training; RhinoVLA는 Qwen3-VL-2B-Instruct에서 시작해 action expert를 scratch로 학습.

#### 4.2 Real-robot (Table 5)

| Robot | Task | Setting | π0.5 SR | RhinoVLA SR |
|---|---|---|---|---|
| Galbot G1 | Red bag → far bin | Unseen | 100% | 100% |
| Galbot G1 | Black fan → middle | Unseen | – | 40% |
| Galbot G1 | White foam → near | Unseen | – | 20% |
| AgiBot G2 | 3-step sequence | Seen | – | 58% |
| AgiBot G2 | 3-step sequence | Unseen | 18% | **24%** |
| AgiBot G1 | Towel folding | Seen | – | 67% |
| AgiBot G1 | Towel folding | Unseen | – | 43% |

- Galbot G1은 pretraining에 포함되지 않은 embodiment → 일정 수준 일반화 확인.
- AgiBot G2 unseen에서 π0.5 18% 대비 **24% (+6%p)**.
- AgiBot G1 양손 towel folding으로 **deformable manipulation** 능력 입증.

#### 4.3 Instance LoRA ablation (Table 3)

| Method | Masked FM Loss | Arm MAE | Base vel MAE | Yaw MAE | Gripper MAE |
|---|---|---|---|---|---|
| Base only | 0.0192 | 0.0446 | 0.0187 | 0.0195 | 0.1064 |
| Instance LoRA | 0.0191 | **0.0440** | 0.0188 | **0.0194** | **0.1056** |

- arm/gripper에서 일관된 소폭 개선 → embodiment-specific residual은 주로 manipulator 쪽에서 발생함을 확인.
- LoRA residual similarity ≈ action-mask Hamming distance와 상관 (Fig. 6) → 어댑터가 dataset ID가 아닌 **embodiment 구조 자체를 학습**.

#### 4.4 Inference efficiency (Table 6, Fig. 4)

- **Baseline → Compilation → Mixed-precision → Parallel encoding**: 5.84 → ... → **11.69 Hz**.
- 최종 per-stage latency: Vision encoder (3 views) 24.31 ms (28.4%) / VLM backbone 20.78 ms (24.3%) / Action Expert **36.71 ms (42.9%)** / Others 3.74 ms / **Total 85.54 ms**.
- 10 Hz closed-loop 요건 충족.

---

### 5. RhinoVLA vs π0.5 핵심 비교 (Table 8)

| Item | RhinoVLA | π0.5 |
|---|---|---|
| VLM backbone | Qwen3-VL-2B | PaliGemma |
| VLM inference-path params | 2.13B | 2.92B |
| Visual tokens / image (256² vs 224²) | **64 (merged from 16×16)** | 256 |
| Action Expert params | 0.40B | 0.43B |
| Action Expert depth/width | 18 layers / 1024 | 18 layers / 1024 |
| Action Expert MLP hidden | 3072 | 4096 |
| Attention heads / KV heads | 16 / 8 (GQA 2) | 8 / 1 (GQA 8) |
| Head dim | 128 | 256 |
| Action interface | **Masked 72D physical slots** | Conventional |

---

### 6. 강점

1. **명확한 첫 원리적 분석**: Roofline + operator breakdown → "VLM MLP의 token-linear GEMM이 병목"이라는 정량적 진단 후 backbone 선택을 정당화.
2. **Cross-embodiment interface가 깔끔**: 72D slot space + binary mask + view registry는 robot-specific output head 없이도 통합 학습/배포 graph 유지.
3. **Algorithm-system co-design 일관성**: token 효율 backbone + custom W8A16 fused kernel + batched ViT가 모두 동일 병목을 공략.
4. **Edge 배포 실증**: Orin이 아닌 자체 R1에서 진짜 11.69 Hz 달성 → 산업 응용 가능성.
5. **다양한 embodiment 실제 평가**: Galbot G1 / AgiBot G1 / G2(단일/양손, rigid/deformable, 단/장기 horizon) 모두 커버.

---

### 7. 약점 및 의문점

1. **π0.5와 LIBERO 6.9%p 갭**: Action expert를 scratch로 학습한 setting이 원인이라고 명시했으나, π0.5와 동일 co-training recipe 사용시의 ablation은 없음.
2. **Real-robot 베이스라인 부족**: π0.5 외 비교 미흡. SR 절대치(20–67%)가 production 수준은 아님 → adaptation data 양과 task 난이도 trade-off 불투명.
3. **Galbot G1 unseen SR 편차(100% vs 40% vs 20%)**: object/bin 페어에 따라 큰 차이 → 일반화 한계.
4. **Open X-Embodiment subset 명세 부족**: 어떤 subset을 어떻게 72D에 매핑했는지(특히 dexterous hand 매핑) appendix B에 일부 있으나 재현 어려움.
5. **R1 외 hardware 비교 부재**: Orin에서 동일 RhinoVLA가 몇 Hz인지 정확 보고 없음 (π0.5 @Orin 1.17 Hz vs RhinoVLA @R1 11.69 Hz 비교는 platform이 다름).
6. **Reserved D61–D71의 미래 확장성**: legged/foot joint는 현재 supervision에서 제외 → humanoid full-body manipulation에 미적용.

---

### 8. 핵심 기여 (저자 contribution)

1. VLM visual/context token이 VLA 배포 비용의 핵심 원천임을 식별 → token-efficient Qwen3-VL backbone 채택.
2. View Registry + 72D physical slot space + robot-instance LoRA로 구성된 통일 cross-robot training framework.
3. R1 hardware-aware compilation + W8A16 fused GEMM + parallel encoding으로 **11.69 Hz edge inference + π0.5 비슷한 task accuracy** 달성.

---

### 9. 후속 연구 방향 (저자 미래 작업 포함)

- R1 위에서 **training pipeline까지 통합** → 데이터 수집/학습/배포 same edge SoC.
- RL/online policy improvement와 연결.
- Future Qwen-VL backbone(차세대) 채택으로 VLM 모듈 지속 업그레이드.
- Leg/foot joint, humanoid whole-body로 D61–D71 활용 확장.
- W8A16 kernel을 더 다양한 transformer block에 일반화.

---

### 10. 실험 재현·검증 체크리스트

- [ ] Open X-Embodiment subset 정확 list와 72D 매핑 코드 공개 여부.
- [ ] LoRA rank=64, instance 수, 학습 step 수, lr schedule 공개.
- [ ] Qwen3-VL-2B-Instruct vs base 차이가 결과에 미치는 영향.
- [ ] R1 deploy SDK / compiler 공개 여부 (재현 핵심).
- [ ] LIBERO 90.0이 multi-seed 평균인지 단일 seed인지.

---

### 11. VLA-Tracker 관점에서의 의의

- 본 연구는 **edge-first VLA**의 종합 reference design을 제공 → Tracker에서 "real-time deployment" 카테고리의 대표 모델.
- π0/π0.5 family와 비교 가능한 LIBERO 점수를 가지면서 **시스템 efficiency 축**에서 새로운 frontier.
- Qwen3-VL backbone 채택은 PaliGemma 중심 흐름에서 Qwen 계열로의 이동을 가속 → VLA Foundry, InternVLA-A1 흐름과 합류.
- 72D unified slot space 디자인은 cross-embodiment dataset 표준화 측면에서 RDT의 motivation을 확장한 유의미한 contribution.

---

### 12. 결론

RhinoVLA는 **"VLA 배포 병목 = VLM token 수 × MLP GEMM"** 이라는 명확한 진단에서 출발해, token-efficient Qwen3-VL backbone, masked 72D physical slot interface, instance-LoRA, 그리고 자체 edge SoC R1을 위한 W8A16 fused kernel·parallel encoding까지 알고리즘부터 chip까지 일관된 co-design을 제시한다. 결과적으로 LIBERO 90.0% (Long 82.4%)로 π0.5 다음 수준의 정책 성능을 유지하면서, π0.5 @Orin 1.17 Hz 대비 **10배 빠른 11.69 Hz edge inference**를 달성했고, AgiBot G1/G2 및 Galbot G1까지 real-robot에서 검증되었다. π0.5와의 task SR 갭은 남아있고 real-robot 베이스라인은 제한적이지만, **edge VLA의 deployment-oriented 표준**을 제시한 의미 있는 technical report로 평가된다.

<!-- VERIFIED: pdf -->
