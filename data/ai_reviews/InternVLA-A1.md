# InternVLA-A1: Understanding-Generation-Action 통합 프레임워크 세미나 리뷰

## 1. 연구 배경 및 동기
InternVLA-A1(Shanghai AI Lab InternRobotics, arXiv:2601.02456)은 기존 VLA의 두 가지 한계 — (1) MLLM 기반 의미 이해는 강하지만 물리 동역학 추론이 부족하고, (2) world model 기반 video prediction은 의미 grounding이 약하고 prediction error에 취약 — 를 동시에 해결한다. 핵심 아이디어는 understanding, generation, action 세 expert를 단일 Mixture-of-Transformers(MoT) 아키텍처로 통합하고 unified masked self-attention으로 조율하는 것이다. Figure 1의 요약 차트에 따르면 π0.5 대비 정적 manipulation +4.4%(70.7%→75.1%), 동적 manipulation +26.7%(60.0%→86.7%), RoboTwin 2.0 simulation +2.6%(86.9%→89.5%) 향상을 달성한다.

## 2. 아키텍처: Mixture-of-Transformers (MoT)
Figure 2는 세 expert의 정보 흐름 understanding → generation → action을 보여준다. 각 expert는 decoder-only transformer로, blockwise attention mask가 cumulative segment mask로 한 방향(앞→뒤) flow를 강제한다.

- **Understanding Expert**: InternVL3 (2B 변형) 또는 Qwen3-VL (3B 변형). 다중 시점 관측 o_t와 instruction l을 contextual embedding h_und로 인코딩.
- **Generation Expert (Figure 3)**: COSMOS CI8×8 continuous VAE encoder로 256×256 입력 6장(3 view × 2 timestamp t-15, t)을 32×32 latent grid (1024 tokens/image)로 인코딩 후, 8×8 conv로 4×4 (16 tokens)로 token compression → 96 tokens. Parallel decoding(temporal average pooling)으로 48 tokens(16/view)를 deconv 후 COSMOS VAE decoder로 t+15 frame 복원. Auto-regressive next-token이 아닌 single-forward parallel decoding 채택.
- **Action Expert**: Flow matching 목적함수(식 (2))로 학습. v_θ(q_t, a^τ_{t:t+k}; h_und, h_gen)이 (a-ε) velocity field를 예측. 추론 시 Euler update(식 (3))로 K step ODE 적분.

Table 1에 따르면 InternVLA-A1(2B): InternVL3 0.94B + Qwen2.5 0.36B + Qwen2.5 0.36B = 1.8B; InternVLA-A1(3B): Qwen3-VL 2.13B + Qwen3 0.44B + Qwen3 0.44B = 3.2B. 두 변형 모두 RTX 4090에서 torch.compile 사용 시 ~13Hz 추론.

## 3. 학습 데이터 및 최적화
**Pre-training 데이터 (Table 3)**: 총 692M frames의 이질적 데이터 — InternData-A1 (Sim, 396M frames, weight 0.64), AgiBot-World (Real, 206M, 0.18), RoboTwin (Sim, 17M, 0.08), EgoDex (Human video, 68M, 0.08), RoboMind (Real, 5M, 0.02). InternData-A1은 630K trajectories/7,433h 규모의 합성 dataset(4 embodiments, 18 skills, 70 tasks, 227 scenes). EgoDex(Hoque et al., 2025) 829h egocentric dexterous manipulation video 활용. Load-balanced Parallel Training(LPT)으로 worker간 dataset 크기 불균형 완화.

**최적화 (Table 2)**: AdamW, lr 5e-5(constant), batch size 512, 700K steps pre-training; post-training은 lr 5e-5→5e-6, warmup 2000, decay 60K, batch 128, 60K steps. λ=0.01로 L_gen과 L_act 가중합(식 (4)), m=15(history/future stride). 정밀도 bfloat16, grad clipping 1.0.

## 4. 핵심 실험 결과
**Static Manipulation (Table 4, 10 tasks)**: InternVLA-A1(3B) 평균 75.1% vs. π0.5 70.7%, π0 60.6%, GR00T N1.5 33.0%. 주요 task별: Make Sandwich 93.3% (π0.5 73.3%), Operate Oven 86.7%(80.0%), Sort Rubbish 97.3%(97.3%, tie), Wipe Stain 86.7%(93.3%), Zip Bag 73.3%(60.0%), Place Flower 60.0%(66.7%, π0.5 우세).

**Dynamic Manipulation (Figure 6, 2 tasks)**: Express Sorting (3B) 80.0% vs. π0.5 53.3%(+26.7%), GR00T 40.0%, π0 36.7%. In-motion Ingredient Picking (3B) 93.3% vs. π0.5 66.7%(+26.6%), π0/GR00T 모두 20.0%. 평균 86.7%로 동적 환경 적응성 입증.

**RoboTwin 2.0 Simulation (Figure 7)**: 50 bimanual tasks, 27,500 demos(2,500 clean + 25,000 randomized) fine-tuning. Easy 89.4% (π0.5 86.8%, π0 80.0%), Hard 89.6% (π0.5 87.0%, π0 79.5%). 두 setting 모두 +2.6% 향상.

## 5. Ablation 분석
**Pre-training의 영향 (Figure 8)**: pre-training 제거 시 평균 SR 77.0% → 25.4%로 51.6%p 폭락, Express Sorting/Sweep Trash/Place Markpen 등은 거의 0~13.3% 수준으로 붕괴. Pre-training이 inductive prior로 작용한다는 강력한 증거.

**Pre-training Dataset 구성 (Table 5)**: Sim only는 RoboTwin 2.0 Easy/Hard 88.3/88.5 달성하나 real-world Place flower 53.3%, Sort parts 33.3%로 일반화 부족. Sim+Human 추가로 89.4/89.3 (Sort parts 40.0%로 향상), Sim+Real+Human 결합 시 89.4/89.6 + Place flower 60.0% + Sort parts 53.3%로 최고 성능. 세 source 결합이 sim-to-real gap 완화에 결정적.

**Generation Expert 영향 (Figure 9)**: 제거 시 평균 SR 77.0% → 57.6%, Express Sorting 80.0%→46.7%, In-motion Ingredient Picking 93.3%→40.0% 등 동적 task에서 가장 큰 하락. Generation expert의 visual foresight가 동적 환경 closed-loop control의 핵심임을 입증. Figure 10의 future frame 시각화는 high-frequency detail을 일부 희생하지만 motion trend는 정확히 포착함을 보여준다.

## 6. 평가 및 한계
InternVLA-A1은 understanding-generation-action을 MoT로 통합한 명확한 설계와 광범위한 ablation으로 동적 환경에서의 우수성을 설득력 있게 입증한다. 단, 논문 자체가 인정하는 한계 두 가지: (1) understanding expert가 대규모 multimodal VQA 데이터로 joint training되지 않아 일반 의미 추론/복잡 instruction-following이 약하고, (2) 추론 효율을 위해 image prediction 충실도(high-frequency detail)를 희생했다.

**YAML 점검**: organization, paper_url, code_url, open_source=true, parameters "2B/3B"는 Table 1의 실제 1.8B/3.2B와 부합한다. action_head_category=hybrid는 MoT(understanding+generation+action) + flow matching 액션이라는 복합 구조를 잘 반영한다. backbone "InternVL3 (2B) / Qwen3-VL (3B)" 표기는 정확하나 llm 필드의 "Qwen2.5 (2B variant)"는 InternVL3-1B 위에 Qwen2.5 0.36B gen/act expert를 사용한다는 점에서 다소 단순화된 표현이다 — 보다 명확히 "InternVL3-1B (und) + Qwen2.5 (gen/act)" 형식이 더 정확할 수 있다 (**경미한 YAML 불일치 가능성**).

<!-- VERIFIED: pdf -->
