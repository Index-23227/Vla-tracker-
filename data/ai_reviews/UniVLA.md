# UniVLA: Learning to Act Anywhere with Task-centric Latent Actions

**Bu, Yang, Cai, Gao, Ren, Yao, Luo, Li, 2025 (arXiv:2505.06111v3, HKU / OpenDriveLab / AgiBot, RSS 2025)**

## 한 줄 요약
DINOv2 feature space 위에서 language-conditioned VQ-VAE inverse/forward dynamics model로 task-centric latent action을 비지도 학습하고, Prismatic-7B VLM에 |C|=16개 ACT 토큰을 추가해 cross-embodiment generalist policy를 학습 — LIBERO 95.2%, CALVIN ABC→D avg.len 3.80, R2R navigation 47.1% oracle 성공률을 OpenVLA 대비 1/20 GPU-hour로 달성.

## 배경
OpenVLA·RT-2 등 기존 VLA는 action label 보유 데이터에만 의존해 internet-scale 비디오를 활용하지 못했고, embodiment 간 action space 이질성이 transfer를 막았다 (§1). LAPA[87], IGOR[16], Genie[12]는 raw pixel 재구성 기반 latent action을 도입했으나, 카메라 흔들림·외부 agent 움직임 등 task-irrelevant dynamics까지 포착해 policy 학습에 노이즈로 작용한다는 한계가 있다 (§II.C). UniVLA의 핵심 통찰은 "DINOv2 patch feature 공간에서 inverse dynamics를 학습하고, 언어 instruction을 conditioning으로 줘서 task-irrelevant 변화를 별도 codebook으로 분리(decouple)"하는 것이다 (§III.A, Fig. 2).

## 방법론
- **Stage 1: Task-irrelevant latent action 분리 (Fig. 2 Left)**: 약 1초 간격의 frame pair {o_t, o_{t+k}}를 DINOv2[62]로 임베딩 → spatial-temporal transformer encoder I가 inverse dynamics 추출, learnable action token a^{TI}와 T5 instruction embedding ℓ을 입력. VQ-VAE로 양자화해 ã^{TI} 생성, decoder F는 ã^{TI} + ℓ로 미래 DINO feature 예측. 언어가 디코더에 직접 주어지므로 ã^{TI}는 task-irrelevant 잔여 변화만 학습.
- **Stage 2: Task-centric latent (Fig. 2 Right)**: Stage 1 codebook을 freeze, 새 codebook VQ_{TC}와 token a^{TC}를 추가해 {ã^{TI}, ã^{TC}}로 미래 frame 예측. ℓ을 제거해 a^{TC}가 task-relevant dynamics를 담도록 강제. 이중 codebook으로 explicit decoupling.
- **Generalist policy (Fig. 3)**: Prismatic-7B VLM(SigLIP+DINOv2 + LLaMA-2 7B). LLaMA tokenizer에 |C|=16 special token {ACT_1,...,ACT_C} 추가, latent action을 next-token으로 autoregressive 예측 (§III.B). 압축된 16⁴=65,536 vs OpenVLA의 256⁷ 공간으로 수렴이 훨씬 빠르고, 사전학습은 960 A100-hour로 OpenVLA(21,500)의 ~1/20.
- **Action decoder (Fig. 3, §III.C)**: 12.6M param의 lightweight transformer head가 latent action 토큰을 query로 받고 visual feature를 KV로 cross-attention하여 heterogeneous embodiment의 연속 action chunk로 디코딩. LoRA fine-tuning, total trainable ~123M, L1 loss + next-latent CE 공동 최적화.
- **History-as-CoT**: 직전 latent action 토큰들을 prompt에 포함시켜 CoT 식 sequential context 제공 — multi-frame visual history 없이도 +16.5% R2R, +3.9% LIBERO-Long.

## 실험 결과
- **LIBERO (Table I)**: Full pretraining 모델이 Spatial 96.5 / Object 96.8 / Goal 95.6 / Long 92.0, 평균 95.2%로 OpenVLA(76.5%)·LAPA(65.7%)·MaIL(83.5%)·MDT(76.1%)를 모두 능가. Bridge-only 사전학습만으로도 평균 92.5%, Human(Ego4D) 단독은 88.7%로, 데이터 scaling이 monotonic하게 도움.
- **CALVIN ABC→D (Table A-II)**: avg.len 3.80, 1→5 task 95.5/85.8/75.4/66.9/56.5. RoboDual(3.66), CLOVER(3.53), GR-1(3.06), OpenVLA(3.27) 대비 SOTA. UniVLA는 단일 third-view RGB만 사용한다는 제약 하의 결과.
- **VLN-CE R2R (Fig. 6)**: Oracle success 47.1%, OpenVLA(17.5%) 대비 +29.6%. NaVid(49.1%) 수준에 단일 RGB만으로 도달. Seq2Seq 8.1, CMA 14.0 대비 압도적.
- **SimplerEnv-WidowX (Table A-III)**: Full fine-tune 47.9%, Decoder-only 35.4%. OpenVLA 1.0%, Octo-Small 30.0%, RoboVLM 31.3% 대비 SOTA. Bridge가 사전학습에 포함된 점을 고려해 decoder-only 평가도 함께 보고하는 점이 인상적.
- **Real-world (Fig. 5)**: Store screwdriver, Clean cutting board, Fold towel twice, Stack tower of Hanoi 4 task에서 평균 81.7% 성공률, OpenVLA 45.0%, LAPA 38.3%, Diffusion Policy 33.3% 대비 +36.7%. 변형 객체(towel) 시나리오 86.7%.
- **Ablation (Table III, IV, Fig. 10)**: Task-irrelevant latent만 사용 시 LIBERO-Long ≈0%, task-centric 사용 시 92.0%로 decoupling 효과 입증. 10% LIBERO-Goal 데이터로 86.3% (OpenVLA 100% 학습본의 79.2% 능가). Action decoder의 cross-attention 설계가 autoregressive decoder 대비 LIBERO-Long +42.1%pp 이득.

## 한계
- 모든 사전학습이 BridgeData·OXE·GNM·Ego4D에 한정 — 비-egocentric 산업 영상이나 다양한 카메라 위치로의 일반화 검증 부족.
- VQ codebook |C|=16은 fine-grained motion(예: 정밀 회전)에서는 표현력 한계 가능성. 저자들이 codebook 크기 ablation은 일부만 제공.
- Action decoder는 embodiment별 별도 학습이 필요해 "한 번 학습 후 zero-shot 어디든"은 아니며, decoder fine-tune이 항상 필요하다.
- 역동적 외부 agent가 많은 비디오에서 stage-1 disentanglement가 충분히 깨끗하지 않을 가능성(저자도 ablation에서 task-irrelevant latent의 일부 leakage 언급).
- **YAML 점수 불일치 주의**: 본 리포지토리 `data/models/univla.yaml`은 LIBERO Avg 90.1, CALVIN avg.len 4.1로 표기되어 있으나, 논문 Table I의 Full 모델은 95.2%, Table A-II는 3.80이다. YAML 검토 필요.

## 총평
"action label 없이 비디오에서 학습한 task-centric latent space + 768→16 vocab compaction + Prismatic-7B 위 next-token prediction"이라는 아이디어를 일관성 있게 밀어붙여, OpenVLA 대비 1/20 compute·1/10 downstream data로 다중 벤치마크에서 SOTA를 달성한 모범적 generalist VLA 레시피다. DINOv2 + 언어 conditioning으로 task-relevant dynamics를 분리하는 설계는 이후 IGOR-2, LAPA 후속 연구에 직접 영향을 끼쳤고, navigation·manipulation·real-world 모두에서 통하는 unified action space의 가능성을 확실히 보였다. RSS 2025 채택.

## 예상 질문
1. Latent action |C|=16이 너무 작지 않나? — Table III에서 task-centric vs task-irrelevant 분리만으로도 충분히 표현력 확보, 압축 자체가 학습 수렴을 4× 가속한다고 §III.B에서 주장. Action chunk N=4와 결합되면 16^4=65k 공간이라 충분하다는 입장.
2. OpenVLA보다 적은 compute로 어떻게 SOTA를 달성하나? — Action vocabulary 압축 + DINOv2 feature 공간의 prediction(픽셀 vs DINO)이 핵심. Bridge-only로 사전학습해도 OpenVLA OXE-full보다 LIBERO 평균 92.5% > 76.5% (Table I).
3. "History-as-CoT"는 정확히 무엇인가? — 직전 latent action 토큰들을 prompt prefix로 주입하는 방식. R2R에서 +16.5%, LIBERO-Long에서 +3.9% 이득 (§III.B 후반 + Sec. C ablation).

<!-- VERIFIED: pdf -->
