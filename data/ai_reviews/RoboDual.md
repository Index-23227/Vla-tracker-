# RoboDual: Synergistic Dual-System for Robotic Manipulation

**Bu et al., 2024 (arXiv:2410.08001v3, Shanghai Jiao Tong / HKU / AgiBot / Shanghai AI Lab)**

## 한 줄 요약
OpenVLA(7B) generalist를 느리지만 일반화 능한 "System-2"로, 20M 파라미터 Diffusion Transformer specialist를 빠른 "System-1"로 결합하여 cross-attention 기반의 dual-system VLA를 제안한 연구로, CALVIN ABC→D에서 avg.len 3.66, 실세계에서 OpenVLA 대비 +20% 성능 향상을 달성했다.

## 핵심 기여
- "느린 generalist + 빠른 specialist"의 인지과학적 dual-process 모델을 robotic manipulation에 구현.
- Discretized action token + task/action latent의 cross-attention conditioning으로 두 시스템을 느슨하게 결합.
- 단 20M trainable parameter, 8 GPU-hour로 OpenVLA 대비 CALVIN +12%, 실세계 +26.7%, 제어 주파수 3.8× 향상.

## 배경
저자들은 generalist VLA(예: OpenVLA, RT-2)와 specialist policy(예: ACT, Diffusion Policy)의 보완적 trade-off를 출발점으로 삼는다. Generalist는 OXE/Bridge V2/DROID 등의 web-scale 데이터로 학습되어 instruction-following과 cross-domain 일반화에 강하지만 (i) 새 embodiment에 직접 배포 불가, (ii) 추론 지연이 크고 (논문 본문 §1, 4 Hz vs 20 Hz target), (iii) 단일 RGB 외 modality 추가가 비싸다. Specialist는 task-specific data로 정밀하지만 일반화·언어 입력이 부족하다. RoboDual은 두 정책을 단일 framework로 묶되, 둘을 end-to-end로 공동 학습하지 않고 "interface"(discretized action + latent)로 느슨하게 연결한다는 점이 특징이다 (Fig. 1, Fig. 2).

## 방법론
- **Generalist (Fig. 2a)**: OpenVLA(Prismatic-7B + LLaMA-2 7B) 위에 LoRA fine-tuning. RT-2/OpenVLA 식대로 LLaMA tokenizer의 비활성 256개 단어를 [-1,1] action bin에 매핑하고, action chunk(길이 k_g)를 [space] 토큰으로 분리하여 autoregressive하게 출력 (§3.1).
- **Specialist (Fig. 2b)**: stacked DiT(Peebles & Xie, 2023) block들로 구성. 각 block은 causal self-attention(temporal action) → cross-attention(conditioning fusion) → FFN. 7-DoF action을 1 토큰으로 linear projection하여 chunk 길이 k_s 예측. ViT(DINO 사전학습, frozen)로 RGB encoding하고, depth/tactile은 6-layer/256-dim 작은 ViT로 인코딩 (§3.2).
- **Conditioning 4종**: (1) proprioceptive state (MLP + timestep embedding으로 adaLN의 γ, β, α 회귀), (2) 다중 sensory ViT 임베딩(8 learnable query Perceiver Resampler로 압축), (3) generalist의 discretized action(noised action과 채널 concat 후 latent space에 투영, video diffusion에서 영감), (4) generalist의 task/action latent(linear projection으로 hidden space 정렬). (2)+(4)는 cross-attention의 K,V로 사용된다.
- **Asynchronous inference**: 1회 generalist 추론이 여러 specialist rollout을 지원. Shifted-window conditioning으로 specialist가 τ_s step 진행 후 가장 최근 k_g - τ_s개 generalist action만 유지. Latency-aware augmentation(τ ∈ [0,k_g] offset)으로 deployment 비동기성 보정 (§3.2 후반).
- **Loss**: generalist는 next-token NLL (Eq. 1), specialist는 DDPM noise prediction L2 (Eq. 2, T=100). Specialist는 단 20M trainable parameter, 8 GPU-hour로 학습 가능하다고 명시 (§3.3).

## 실험 결과
### 시뮬레이션
- **CALVIN ABC→D (Table 1)**: avg.len 3.66, 1→5 task 성공률 94.4/82.7/72.1/62.4/54.4. 직전 SOTA인 3D Diffuser Actor(3.27) 대비 +0.39, 5-task 연속 성공률은 +13.2%p.
- **Free-form instruction robustness (Table 2)**: GPT-4로 enrich한 명령에 대해 avg.len 3.47로, LCB(1.78), 3D Diffuser Actor(1.42) 대비 약 2배. Generalist의 의미 이해와 specialist의 conditioning robustness 결합 효과로 해석.

### 실세계
- **Real-world ALOHA (Fig. 3)**: 단일·다중 instruction 5개 task 평균에서 RoboDual single-task 62.7%, multi-task 56.0%로 OpenVLA(40%대), Diffusion Policy(40%대)를 ~+20% 상회. 특히 "Put <obj> into basket" 다중 instruction에서 Diffusion Policy 20% vs RoboDual 73-87%.
- **Generalizability (Table 3, Fig. 4)**: position variation/visual distractor/unseen background/novel object 4축 평균에서 RoboDual 70.0%(single)·68.3%(multi). DP 40.0%, OpenVLA 41.7% 대비 큰 폭 우위. Position 단독에서는 93.3%로 ACT(46.7), DP(53.3), OpenVLA(26.7)를 압도.

### 효율성
- **Training efficiency (Fig. 5)**: Generalist만으로 ~1400 GPU-hour 후 3.27에 수렴, RoboDual은 추가 1 GPU-hour(8×A100)만으로 3.52, full 학습 후 3.66. Generalist 학습 자체가 부족할 때(Ours*)도 1 hour 추가만으로 3.44 도달.
- **Data efficiency (Table 4)**: 5% CALVIN demo로 avg.len 3.59 (RoboFlamingo 1.35). 실세계 5 demo로 "put block into bowl" 73.3% (DP 20%, ACT 0%).
- **Latency**: A5000 Ada에서 specialist 0.035s → 15 Hz 제어 주파수, OpenVLA(3.9 Hz) 대비 3.8× 향상 (§4.4).

### Ablation
- **Conditioning sources (Fig. 6a)**: action latent + task latent + discretized action 모두 기여 (각각 제거 시 -0.14~-0.8 avg.len). Discretized trajectory가 가장 큰 단일 conditioning 기여.
- **Sensory inputs (Fig. 6b)**: depth, tactile, wrist 추가 시 점진적 개선.
- **Conditioning method (Fig. 6c)**: cross-attention > in-context > FiLM. Cross-attention이 약간 더 우수하면서 계산 효율도 우수.

## 한계
- 두 시스템의 inference time을 상수로 가정해 generalist 1회당 specialist 호출 횟수를 고정 (§5).
- Generalist 출력이 discretized action 한정 — task decomposition, affordance grounding, image goal generation 등으로 확장 시 더 풍부한 "bridge"가 가능하다고 future work에 언급.
- 실세계 평가는 ALOHA + 단일 third-view RGB 한정, 더 다양한 embodiment·해상도에서의 검증 부재.
- Code "would be made publicly available"이라 명시되어 있으나 본 리포지토리 YAML에는 `open_source: false`로 등록 — 시점에 따라 상태 변동 가능.

## 총평
"느린 generalist + 빠른 specialist"라는 dual-process 인지과학 비유를 cross-attention 기반의 lightweight DiT specialist로 깔끔하게 구현한 작품이다. 20M trainable param·8 GPU-hour라는 매우 가벼운 specialist만으로 OpenVLA 대비 CALVIN +12%, 실세계 +26.7%(논문 abstract)까지 끌어올린 점이 핵심 기여이며, multi-modality 통합과 latency 완화를 동시에 해결하는 실용적 레시피로 후속 dual-system 연구(예: HiRT, π0 hierarchy)에 영향이 컸다.

## 예상 질문
1. **Specialist를 generalist와 함께 end-to-end로 학습하지 않은 이유는?**
   - Decoupled training이 modality·embodiment 별 독립 확장을 가능케 하고, OpenVLA의 large-scale 사전학습 지식 망각(catastrophic forgetting)을 막기 위함 (§3, §3.3).
   - 또한 specialist만 task-specific data로 빠르게 fine-tune 가능 → 적은 데이터로도 73.3% 달성 (Table 4b).
2. **Generalist 비동기 호출 시 stale conditioning 문제는?**
   - Shifted-window mechanism: specialist가 τ_s step 진행 후 generalist의 가장 최근 k_g - τ_s개 action만 유지.
   - Latency-aware augmentation: 학습 시 τ ∈ [0,k_g] offset을 무작위로 적용해 deployment의 비동기성을 명시적으로 시뮬레이션 (§3.2 후반).
3. **CALVIN avg.len 3.66은 더 큰 generalist 단독 학습으로 도달 가능한가?**
   - Fig. 5에서 generalist 단독은 ~1400 GPU-hour 학습 후 3.27에 정체. RoboDual은 1 GPU-hour 추가만으로 3.52에 도달, 동일 specialist를 inadequately-trained generalist에 붙이면 3.44 (Ours*) — 단순 scale-up 보다 효율적.
4. **다른 modality(depth/tactile) 추가는 어떻게 처리하나?**
   - Specialist의 ViT encoder 패치 레이어만 channel 수에 맞게 수정 (6-layer/256-dim). Generalist는 그대로. Fig. 6(b)에서 depth/tactile/wrist 추가가 평균 길이를 증분적으로 개선.

<!-- VERIFIED: pdf -->
