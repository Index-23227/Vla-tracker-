# X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment VLA Model

> **한 줄 요약**: 임바디먼트별 soft prompt(학습 가능한 임베딩)와 flow matching 디코더를 결합하여 0.9B 파라미터로 LIBERO 98.1 / Simpler-WidowX 95.8 등 6개 시뮬레이션 + 3개 실제 로봇 플랫폼에서 SOTA를 달성한 cross-embodiment VLA. ICLR 2026.

## 1. 배경 및 동기

- 일반화된 VLA는 multiple robotic platforms × 다양한 task의 heterogeneous 데이터셋에서 학습되어야 하지만, hardware/camera/task 분포가 매우 이질적이어서 분포 이동 및 의미 정합 문제가 발생한다 (§1).
- 기존 방법은 주로 embodiment별 action decoder head만 분리(π₀, GR00T-N1)하여 후단에만 heterogeneity를 반영, 초기 feature fusion 단계의 정렬은 놓친다.
- HPT-style projection은 pretrained VLM 표현을 손상시키고, language prompt는 hand-crafted 텍스트 설명에 의존해 확장성이 떨어진다 (Fig. 2 비교).
- 저자들은 prompt-learning(Lester et al., 2021)을 차용해 데이터 소스마다 별개의 학습 가능한 embedding을 부여하는 **Soft Prompt** 방식을 제안.
- AgiBot World Challenge (IROS 2025) 1위 수상.

## 2. 방법론

### 2.1 Soft Prompt 메커니즘 (§3, Fig. 2)
- 데이터 소스마다 학습 가능한 prompt p_i ∈ ℝ^k를 부여하여 hardware configuration h_i를 implicit하게 인코딩: p_i ≈ Φ(h_i).
- Random init 후 end-to-end로 최적화되며, action 생성 초기 단계에 inject되어 backbone이 embodiment-aware 학습을 하도록 유도. Fig. 4에서 다른 방식 대비 가장 안정적인 학습 곡선을 보인다.

### 2.2 X-VLA 아키텍처 (§4.1)
1) **High-dimensional 스트림**: 메인 view + 언어는 Florence-Large VLM 인코더, 보조 wrist view는 별도 shared ViT로 처리. Wrist view의 fast-changing noisy 특성을 분리.
2) **Low-dimensional proprio-action 스트림**: proprioceptive state R_t와 noisy action A_t를 time embedding T와 함께 concat → linear projection → flow-matching 파이프라인에 투입.
- Backbone은 24-block Transformer encoder(hidden=1024, 0.9B), DiT 대신 vanilla self-attention block을 stack.

### 2.3 Flow Matching 정책 (§2)
- Gaussian noise A_0에서 OT(optimal transport) path A_t = (1-t)A_0 + t·A로 chunk size 32의 액션을 생성.
- Loss: L_BC^FM = E[‖v_θ(A_t,o,t) − (A − A_0)‖²] (§2 Eq.).

### 2.4 학습 레시피 (§4.2)
- **Phase I 사전학습**: AGIBOT(48.8%) + Droid + RoboMind 등 290K 에피소드, 5종 로봇 7개 hardware setup. 액션 표현은 EEF xyz + Rotate6D + binary gripper로 표준화.
- **Intention abstraction**: 30 anchor points로 4초 trajectory를 다운샘플하여 high-level intention을 학습.
- **Custom LR + balanced sampling**: soft prompt와 VLM 모듈에 낮은 lr 적용, cross-domain & within-domain shuffling.
- **Phase II 적응 (two-step)**: (1) Prompt warm-up — backbone freeze 후 새 p_new만 학습, (2) Joint adaptation — backbone과 prompt 공동 최적화. LoRA로 9M (≈1%) 파라미터만 튜닝해도 강력함.

## 3. 실험 결과

X-VLA-0.9B vs 기존 SOTA (Table 2):
- **LIBERO**: Spatial 98.2 / Object 98.6 / Goal 97.8 / Long 97.6 → **Avg 98.1** (기존 max 97.1, OpenVLA-OFT). 본 데이터셋 키지의 libero_avg는 97.8(긴 평가 평균치).
- **Simpler-VM 80.4 / Simpler-VA 75.7**: 기존 max(MemoryVLA 77.7/72.7) 대비 우위. **Simpler-WidowX 95.8** (기존 SOTA MemoryVLA 71.9 대비 +23.9).
- **Calvin ABC→D 4.43 avg len**: FLOWER 4.53과 근접.
- **RoboTwin-2.0 Easy 70.0 / Hard 39.0** (기존 max π₀ 46.4/16.4 대비 압도적, Hard +22.6).
- **VLABench PS 51.1** (기존 max GR00T-N1 39.7 대비 +11.4).
- **NAVSIM PDMS 87.3** (UniVLA 81.7 대비 +5.6) — 자율주행 벤치까지 SOTA.
- **Ablation (Table 1)**: heterogeneous PT + custom LR + soft prompt + scaling → val error 0.11 → 0.032, AD acc 25 → 89.6 (Simpler-WidowX).
- **PEFT (Table 3)**: 9M LoRA 파라미터로 LIBERO Spatial 95.4 / Object 96.6 / Goal 96.0 / Long 84.2 / Simpler-WidowX 54.2 — full fine-tune π₀ 3B와 비교가능.
- **Real-world cloth folding**: 1,200 trajectories로 finetune, ~100% 성공률 + 33 folds/hour로 closed-source π₀-folding과 동급. ACT, π₀-base는 모두 미달.
- **Soft prompt 해석성 (Fig. 8)**: t-SNE에서 hardware별 클러스터 형성, Droid의 두 Franka view는 자연스럽게 섞여 cross-embodiment 유사성을 자동 포착.

## 4. 한계 및 미해결 문제

1. **새 임바디먼트마다 prompt 학습 필요**: 완전 zero-shot transfer는 어렵고, Phase II에서 새 hardware에 대한 prompt warm-up 과정이 필수. 저자도 §5.3에서 frozen UR5 prompt → WidowX 전이는 초반엔 좋지만 final performance에 한계가 있다고 인정.
2. **0.9B는 여전히 작은 규모**: Fig. 5의 scaling 곡선은 saturation을 보이지 않으나, π₀(3B), OpenVLA(7B)와 같은 규모에서 soft prompt 효과가 어떻게 변하는지는 미검증.
3. **Hardware setup이 명시적으로 알려져야 함**: dataset ID로 prompt를 querying하므로 새 unlabeled mixed 데이터에 자동 적용은 불가. 실세계 배포 시 hardware ID metadata 관리가 필수.

## 5. 총평

- **Novelty: ★★★★☆** — Soft prompt 자체는 기존 NLP/Vision 영역 기법이지만, cross-embodiment VLA에 명료하게 도입하고 다른 방식(HPT, language prompt) 대비 체계적으로 우위를 입증.
- **Practical impact: ★★★★★** — 0.9B로 6 시뮬레이션 + 3 실제 로봇 SOTA. 1% 파라미터 LoRA만으로 π₀ 3B와 동급 성능을 내는 것은 배포 측면에서 매우 실용적. 코드/데이터셋 공개(2toinf/X-VLA)와 IROS 2025 챔피언 실적이 신뢰도를 더해줌.

"작지만 강한" generalist VLA의 모범 사례. Soft prompt가 단순한 dataset ID embedding을 넘어 cross-embodiment 유사성까지 학습한다는 t-SNE 결과(Fig. 8)는 향후 prompt-based zero-shot embodiment transfer의 가능성을 시사한다.

## 6. 예상 질문

| Q | A |
|---|---|
| Soft prompt가 단순한 dataset ID one-hot embedding과 본질적으로 다른가? | Fig. 8 t-SNE에서 같은 robot의 다른 view는 섞이고 다른 robot은 분리됨. 이는 dataset ID 분리가 아닌 hardware feature space 학습을 의미. 또한 §5.3에서 UR5 prompt가 WidowX 적응을 가속하는 transfer 효과 확인. |
| 왜 DiT 대신 vanilla Transformer encoder를 쓰는가? | Table 1 ablation에서 DiT → Transformer encoder로 교체 시 val error 0.077→0.071, 그리고 encoding pipeline 추가로 0.053까지 감소. Streamlined encoding으로 충분하다는 실증. |
| RoboTwin Hard에서 70 → 39%로 큰 격차가 나는 이유? | Hard split은 더 많은 randomization과 long-horizon을 요구. X-VLA의 cross-embodiment pretraining이 spatial reasoning에 강점을 주지만 정밀 양팔 협응에는 한계가 있을 수 있음. |
| LoRA 9M로 LIBERO 93%인데 풀 fine-tune은 98.1%. 왜 차이? | Phase II joint adaptation은 backbone 전체를 적응시켜 도메인 specialization이 더 강함. PEFT는 cost-efficiency 시연용이며, 실제 SOTA 수치는 full fine-tune 결과 (Table 2). |

<!-- VERIFIED: pdf -->
