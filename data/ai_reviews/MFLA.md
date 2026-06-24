# MFLA: Motion-Focused Latent Action Enables Cross-Embodiment VLA Training from Human EgoVideos

**Xu, Zhang, Wang, Wang, Yu, 2026 (arXiv:2606.18955v1, Tsinghua University / Tianfu Jiangxi Laboratory)**

## 한 줄 요약
물리적 hand/arm mask로 motion과 background를 분리하는 Hybrid Disentangled VQ-VAE를 도입해 unlabeled 인간 egocentric 비디오에서 cross-embodiment latent action codebook을 학습하고, Prismatic-7B VLM으로 인간→로봇 action intent를 옮긴 뒤 flow matching action expert로 실행 — LIBERO 평균 91.8% (third-person만), RoboTwin 2.0 67.7%로 인간 비디오 사전학습만으로 robot-label 기반 SOTA(pi0/villa-x/UniVLA)와 동급 이상.

## 핵심 기여
- **Hybrid Disentangled VQ-VAE**: SAM2/RoboEngine로 얻은 hand/arm physical mask로 foreground(motion)와 background를 dual-path VQ로 강제 분리 — language signal에만 의존하던 UniVLA의 한계 극복.
- **Intention-Perception Decoupling**: VLM이 latent action intent만 만들고, 실시간 perception은 별도 frozen DINOv2가 담당해 action hallucination(닫힌 컨테이너에 물체 넣기 등) 억제.
- **Human-only pre-training으로 cross-embodiment 일반화**: EgoDex 비디오만으로 사전학습한 모델이 dual-arm Aloha-Agilex(RoboTwin 2.0)·실세계 ARX R5에서 RDT/pi0와 비교 가능한 성능 달성, 단 ~50 trajectory/task로 downstream 적응.

## 배경
기존 generalist VLA(OpenVLA, pi0, RDT)는 OXE·AgiBot 등 action-labeled 로봇 데이터셋에 의존하고, 인간 비디오를 활용하는 EgoMimic/MotionTrans/H-RDT는 AR/VR 하드웨어로 수집한 hand-pose label에 묶여 internet-scale로 확장 불가능하다(§I, §II.C). LAPA·IGOR·UniVLA·villa-x는 VQ-VAE로 frame 전이를 latent action으로 압축했지만, 카메라 흔들림·배경 변화 같은 task-irrelevant dynamics를 함께 학습해 정책 품질을 떨어뜨린다(§II.B). UniVLA는 언어 conditioning으로 task-relevant signal만 분리하려 했으나, 환경 다양성이 큰 인간 영상에서는 language signal만으로 분리가 부족하다는 게 본 논문의 핵심 진단(§I).

## 방법론
- **Stage 1 — Hybrid Disentangled VQ-VAE (§III.B, Fig. 2)**: 1초 간격 인접 frame V ∈ R^{T×C×H×W}를 frozen DINOv2로 인코딩, learnable query Q_act / Q_bg를 visual patch와 concat해 spatial-temporal transformer encoder에 통과. dual-path VQ로 action codebook(size 16)과 background codebook(size 16)을 분리 학습, 각 frame pair는 4개의 discrete latent action token으로 표현. 공유 decoder가 (1) 전체 재구성, (2) action ablation(action token + 초기 frame만, foreground mask 영역만 loss), (3) background ablation(bg token만, background 영역만 loss)을 동시에 수행해 semantic isolation 강제. Mask는 BridgeV2에서는 RoboEngine, EgoDex에서는 SAM2로 생성. Loss = λ_recon L_recon + λ_vq L_vq + λ_commit L_commit (Eq. 1).
- **Stage 2 — VLM Pre-training (§III.C)**: VQ-VAE의 action codebook을 VLM vocabulary에 추가(UniVLA 방식). Prismatic-7B 백본이 (I_t, I_{t+T}, language L)로부터 4-token latent action sequence z^{act}를 autoregressive NLL로 예측 (Eq. 2). 인간 비디오만으로 action intent prior를 학습.
- **Stage 3 — Downstream Adaptation (§III.D)**: VLM은 LoRA fine-tune로 task-specific intent를 만들고, 마지막 transformer hidden state를 latent action embedding f_act로 집계. 별도 frozen DINOv2가 f_obs = DINO(I_main) 제공, robot proprioception f_proprio와 concat해 F_full = Concat(f_act, f_obs, f_proprio) (Eq. 3). Transformer 기반 flow matching action expert가 F_full로 cross-attention하여 vector field v_θ를 예측 — L_flow = ‖v_θ(x_t, t, F_full) − (a − ε)‖² (Eq. 4). 최종 손실 L_total = L_flow + λ_intent L_intent (Eq. 5)로 intent CE와 flow를 공동 최적화.
- **RoboTwin 2.0 변형**: 주 카메라 self-occlusion 때문에 추가 wrist view I_wrist를 도입, f_obs = DINO([I_main, I_wrist]).

## 실험 결과
### LIBERO (Single-arm, robot-to-robot transfer; Table I)
- Bridge-only VQ-VAE/VLM 사전학습 → LIBERO에서 post-training. **Spatial 95.5 / Object 94.0 / Goal 93.5 / Long 84.0 / Avg 91.8** (third-person view only, no wrist).
- 비교: LAPA 65.7, OpenVLA 76.5, SpatialVLA 78.1, Diffusion Policy 72.4, pi0* 81.1, pi0-fast* 85.5, villa-x* 90.1, UniVLA-Bridge 88.1. * = wrist camera 사용. wrist 카메라 없이도 wrist 사용 villa-x 대비 Goal +2.0%, Long +9.5%로 long-horizon에서 두드러진 우위.
- **Ours w/o DINO (VLM 임베딩만)**: 85.4 → intent-perception decoupling이 +6.4% 기여.

### RoboTwin 2.0 (Human-to-dual-arm transfer; Table II, 10 tasks, Aloha-Agilex)
- EgoDex 비디오만으로 사전학습 → RoboTwin 2.0 post-training. **Average 67.7%** > RDT 52.5, pi0 65.2, ACT 51.2, DP 49.7, UniVLA 63.6.
- Per-task highlight: Adjust bottle 97, Grab roller 90, Open laptop 87, Move can pot 65, Handover mic 92. 어려운 Place phone stand는 28%(저자 SOTA보다 낮지만 평균은 최고).
- Ablations: Ours w/o DINO 62.8 (visual decoupling 영향 -4.9%); Ours (Freeze VLM) 52.4 (VLM 사전학습 critical, 하지만 frozen이어도 RDT 수준 유지 — pre-training 자체가 매우 효과적).

### 실세계 (ARX R5 dual-arm, Fig. 3)
- EgoDex 사전학습 + 50 real trajectory post-training, 3 task. Place bottle on plate / Unplug power cord / Fold towel에서 **Ours 0.30 / 0.75 / 0.75** vs UniVLA 0.30 / 0.35 / 0.50 — 두 dual-arm task에서 +0.40 ~ +0.25. Place Bottle은 양쪽 모두 0.30인데, 병의 무게중심이 높아 한 번 쓰러지면 회복 불가하기 때문이라고 분석.

### Latent Action 표현 분석 (§IV.C, Fig. 4-5)
- Bridge+FurnitureBench 혼합 학습 후, domain subspace를 PCA로 iterative하게 제거한 뒤 CKA 측정. **MFLA 0.9139 ± 0.0099 vs UniVLA 0.8659 ± 0.0064** — domain bias 제거 후에도 cross-embodiment 일관성이 유의미하게 높음.
- Fig. 5 정성: EgoDex(인간 손)와 BridgeV2(WindowX) 양쪽에서 "Move Forward / Move Right / Lift Object"에 동일 token sequence((1,12,1,3), (6,9,9,4), (2,9,6,7))가 할당 — embodiment-agnostic 의미 구조.

## 한계
- 사전학습 데이터가 BridgeV2 또는 EgoDex로 한정 — 산업 환경·다양한 카메라 위치·non-manipulation 도메인 일반화는 미검증.
- Mask 품질 의존도: SAM2/RoboEngine이 hand·arm을 잘 못 잡는 occlusion-heavy 장면에서 disentanglement가 깨질 가능성. Mask 품질 ablation 없음.
- Codebook size 16, K=4 token이 fine-grained manipulation(작은 회전·grasp 강도)에서 충분한지 codebook size sweep 없음.
- RoboTwin 2.0에서 wrist view를 추가로 사용 — "pure third-person from human videos"의 cleanness가 약간 희석됨. Wrist 도움 정도의 ablation 부재.
- 실세계 task가 3개로 적고, Place Bottle 30%는 절대 성능이 낮음. UniVLA와 격차는 보였지만 task 다양성·반복 횟수 정보가 제한적.
- Open-source 여부, 학습 GPU·시간 등 비용 정보 미공개. Prismatic-7B 기반이라 OpenVLA 계열 reproducibility는 가능할 전망.

## 총평
"인간 비디오의 motion vs background 엉킴"이라는 latent action 학습의 잔존 문제를 SAM2 기반 physical mask로 정면 돌파한, latent-action VLA 계보(LAPA → IGOR → UniVLA → villa-x)의 자연스러운 진화형이다. 언어 conditioning만으로 task-irrelevant signal을 분리하던 UniVLA를 한 단계 더 밀고 가, 동일 백본(Prismatic-7B)에서 LIBERO 평균 +3.7%pp, RoboTwin 2.0 +4.1%pp, CKA +0.048을 달성한 점이 깔끔하다. 특히 "VLM은 intent, DINOv2는 perception"이라는 decoupling 설계가 단순한 ablation 차원을 넘어 action hallucination이라는 구체적 실패 모드를 해결한 점에서 설득력이 크다. ~50 trajectory/task로 dual-arm까지 적응한다는 데이터 효율은 인간-스케일 사전학습의 실용성을 잘 보여준다. 다만 codebook 설계 hyperparameter(K=4, |C|=16)와 mask 품질 의존도에 대한 추가 분석이 차기 버전에서 필요하다.

## 예상 질문
1. **UniVLA와의 본질적 차이는 무엇인가?**
   - UniVLA: 언어 conditioning + 이중 codebook(task-irrelevant / task-centric)으로 의미 단위 분리. 환경 다양성이 큰 인간 영상에서는 language signal만으로 motion/background 분리가 약함.
   - MFLA: physical mask(SAM2/RoboEngine)로 명시적 공간 분리 + 별도 background codebook + mask-guided reconstruction loss. CKA 0.866 → 0.914로 cross-embodiment alignment 정량 개선 입증.
2. **wrist 카메라 없이 LIBERO에서 villa-x를 어떻게 이기나?**
   - villa-x는 wrist + action label, MFLA는 third-person + no action label. Long-horizon에서 MFLA가 +9.5%pp 우위 — 저자는 high-level intent planning이 우월하고 villa-x는 immediate motor precision에 집중되어 long sequence에서 drift가 발생한다고 해석(§IV.A).
   - intent(VLM) / perception(DINOv2) 분리가 단순 wrist 보강보다 더 큰 일반화 신호를 제공.
3. **Mask 품질에 얼마나 의존하나?**
   - Bridge에서는 RoboEngine(SOTA segmentation), EgoDex에서는 SAM2를 사용 — 둘 다 강력한 사전학습된 segmenter. Mask quality ablation은 논문에 없음. Heavy occlusion이나 정밀 도구 사용 task에서 잠재적 실패 요인.
4. **K=4 / |C|=16 latent action token이 정밀 작업에 충분한가?**
   - 명시적 codebook size ablation 없음. 다만 RoboTwin Place phone stand 28%, Place object basket 25% 등 정밀 task에서 절대 성능이 낮은 점은 representation capacity 한계 가능성 시사.
5. **Open-source인가?**
   - 본 v1 preprint에 code release 명시 없음. Prismatic-7B + SAM2 + DINOv2 모두 공개되어 있으므로 재현 가능하지만 EgoDex/BridgeV2 mask 전처리 파이프라인이 관건.
6. **flow matching head를 쓴 이유?**
   - 본 논문이 latent intent와 perception을 분리하는 데 집중하므로, action expert는 villa-x·pi0와 동일한 modern 선택(flow matching)으로 multimodal action distribution을 표현. UniVLA fairness 비교에서는 head를 동일 flow matching으로 맞춰 평가.

<!-- VERIFIED: pdf -->
