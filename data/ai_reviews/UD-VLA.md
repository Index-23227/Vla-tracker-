# UD-VLA: Unified Diffusion VLA via Joint Discrete Denoising Diffusion Process (JD3P)

**Chen, Song, Ding, Zhou, Zhao, Tang, Wang, Li, ICLR 2026 (arXiv:2511.01718v2, HKUST-GZ / Westlake / ZJU / Monash)**

## 한 줄 요약
이미지 토큰(VQ visual tokenizer)과 action 토큰(FAST tokenizer)을 단일 시퀀스에 놓고, hybrid attention(intra-modal bidirectional + inter-modal causal)과 동기적 이산 확산 과정 JD3P로 future image와 action을 함께 denoise함으로써, CALVIN ABCD→D avg.len 4.64, LIBERO 92.7%, SimplerEnv-WidowX 62.5%를 autoregressive 대비 4.3× 빠른 속도로 달성한 통합 VLA.

## 배경
"Unified VLA"는 이미지 생성과 action 예측을 함께 다루어 시각적 chain-of-thought를 활용하는 흐름이지만, 기존 방법은 두 패러다임 모두 단점이 있다 (Table 1):
1) **Extrinsic experts** (GR-1, SEER, DreamVLA, F1, UP-VLA): 외부 vision encoder/decoder를 두어 modality 정렬에서 misalignment, 복잡성, weak coupling이 발생.
2) **Unified token space, separate decoding** (CoT-VLA, WorldVLA, UniVLA, FlowVLA): vocabulary는 하나로 합치지만 image 생성과 action 생성이 별도 과정으로 일어나 future image의 정보를 inference 시점에 충분히 활용하지 못한다.
저자들은 두 modality가 "synchronous denoising trajectory" 안에서 매 step 서로 attention함으로써 image 생성이 action 생성을 지속적으로 가이드해야 한다고 주장한다 (§1, Fig. 1).

## 방법론
- **Unified tokenization (§3.1, Eq. 1)**: 시퀀스 [text ; current image ; future image ; action]. text는 Emu3 방식, image는 VQ tokenizer(MoVQ[Zheng+ 2022], V_v vocab), action은 FAST tokenizer(Pertsch+ 2025, V_a vocab). <BOI>/<EOI>, <BOA>/<EOA> special token으로 modality 경계 표시.
- **Hybrid attention (Fig. 2, §3.1)**: text는 causal, current image는 bidirectional. 출력 future image block과 action block 각각은 internal bidirectional이지만 modality 간에는 causal(future image → action). 한 image 내 token들은 global consistency를 위해 서로 attention, action의 position·rotation 같은 dimension들도 strict causal이 아니므로 bidirectional이 효과적이라는 논거. Table 5에서 hybrid 4.64 vs bidirectional 4.32 vs causal 4.04로 검증.
- **JD3P (§3.2)**: 이산 확산 과정에서 image와 action 토큰이 하나의 동기적 trajectory를 따라 noise → clean으로 진화. 매 denoising step마다 action 토큰이 그 시점의 future image 토큰에 attend → 정보가 반복적으로 흐른다. Confidence-based unmasking + remapping으로 candidate 좁힘.
- **Two-stage training (Fig. 1, §3)**: Stage (i) 사전학습 VLM(Emu3 기반)에 image-only future prediction 학습으로 world dynamics 주입; Stage (ii) robot dataset에서 image+action joint diffusion training (LIBERO 4 suite는 8× H100, ~24h).
- **Inference 가속 (§3.3)**: KV-cache, prefilled special token, vocabulary remapping(작은 후보 집합), confidence-based decoding. Table 7에서 AR 50.2 tok/s → JD3P 219.3 tok/s (×4.3), 동시에 avg.len 4.18 → 4.64로 정확도까지 개선.

## 실험 결과
- **CALVIN ABCD→D (Table 2)**: avg.len **4.64** (1→5: 99.2/96.8/93.6/90.4/84.0). 기존 SOTA MDT 4.52, UP-VLA 4.42, UniVLA* 4.26 대비 큰 폭 우위. 5-task 연속 성공률 84.0%로 가장 어려운 long-horizon에서 두드러짐.
- **LIBERO (Table 3)**: 평균 **92.7%** (Spatial 94.1 / Object 95.7 / Goal 91.2 / Long 89.6). DreamVLA(92.6)와 거의 동률이나 Long 89.6%로 모든 비교 모델 중 최고. FlowVLA 88.1, MolmoAct 86.6, π0-FAST 85.5 대비 우위.
- **SimplerEnv-WidowX (Table 4)**: 평균 **62.5%** (Put Spoon 58.3 / Put Carrot 62.5 / Stack Block 54.1 / Put Eggplant 75.0). F1(59.4), π0-FAST(48.3), SpatialVLA(42.7) 대비 SOTA. Stack Block(고정밀 grasp+place)에서 SpatialVLA(3D 입력) 대비 +24.9%.
- **Hybrid attention ablation (Table 5)**: Causal 4.04 / Bidirectional 4.32 / Hybrid 4.64. Bidirectional은 modality 간 정보 누설로 -0.3 손실.
- **Visual generation target ablation (Table 6)**: Null(미생성) 4.21 / Current image 재구성 4.39 / **Future image 4.64**. 미래 이미지 예측이 명시적 visual CoT 역할을 한다는 증거.
- **Decoding mechanism (Table 7)**: AR 4.18@50tok/s, Jacobi 4.16@102, Independent diffusion 4.35@144, **JD3P 4.64@219**. 정확도와 속도 동시 개선이 핵심 기여.

## 한계
- 사전학습 비용이 명시적으로 공개되지 않았고, 학습 코드/체크포인트가 공개되지 않은 상태(YAML `code_url: null`, `open_source: false`).
- VQ visual tokenizer(MoVQ)와 FAST action tokenizer 두 외부 도구에 의존 — 이들의 reconstruction loss가 상한이 됨.
- 실세계 평가는 Fig. 3 등 정성 위주로 보고되어 있으며, 다양한 embodiment에서의 정량 robustness 측정은 제한적.
- Long-horizon temporal CoT(여러 sub-goal image 시퀀스 예측)나 sparse-reward RL과의 결합은 future work.
- **YAML 점수 불일치**: 본 리포 `data/models/ud_vla.yaml`은 CALVIN 4.5, LIBERO 평균 91.75 (97.2/97.8/93.6/78.4)로 표기되어 있으나, 논문 Table 2/3의 실제 값은 4.64, 92.7% (94.1/95.7/91.2/89.6). 특히 Long suite 78.4 vs 89.6은 큰 차이라 데이터 갱신 필요.

## 총평
"image와 action을 같은 vocabulary와 같은 denoising step 안에 묶어 매 iteration마다 cross-modal interaction을 강제"한다는 단순하지만 강력한 통일 원리를 제시한 ICLR 2026 논문. CoT-VLA(autoregressive image + diffusion action)와 비교했을 때, 동일한 unified token space를 공유하면서도 디코딩 단계까지 합쳐 4× 가속 + 정확도 개선을 동시에 달성한 점이 핵심 기여. Hybrid attention의 modality-별 분리 설계와 future image generation을 visual CoT로 활용한다는 ablation 결과는 후속 unified VLA 연구에 강한 베이스라인을 제공한다.

## 예상 질문
1. Future image 예측이 정말 action에 도움이 되는가, 아니면 단순 정규화 효과인가? — Table 6에서 Null vs Current vs Future가 4.21 → 4.39 → 4.64로 monotonic 증가, "단지 fine-grained perception" 가설(Current)을 넘어서는 추가 이득 0.25 → temporal anticipation의 직접 기여 (§4.3).
2. CoT-VLA와 무엇이 본질적으로 다른가? — CoT-VLA는 이미지를 AR로 먼저 다 그리고 그 후 action을 diffusion으로 디코드(AR-Diff). UD-VLA는 두 modality를 동일 trajectory에서 동시 denoise(JD3P). 매 step cross-modal attention이 가능하다는 점이 결정적 (Table 1, §1).
3. 4× 속도 개선의 출처는? — KV-cache + prefilled special token + remapping + confidence-based parallel unmasking. AR의 token-by-token 50 tok/s → JD3P 219 tok/s (Table 7).

<!-- VERIFIED: pdf -->
