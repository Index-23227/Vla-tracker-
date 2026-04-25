# UP-VLA: Unified Understanding and Prediction Model 세미나 리뷰

## 1. 연구 배경 및 동기
UP-VLA(Zhang et al., ICML 2025, arXiv:2501.18867, Tsinghua/Shanghai Qi Zhi)는 기존 VLA 모델이 사전학습 VLM의 high-level semantic 지식을 잘 활용하는 반면 low-level 시각/공간 정보 캡처가 부족하다는 문제를 지적한다. 선행 연구(Zheng et al. 2024a; Chen et al. 2024a; Wen et al. 2024)는 VLM이 distance/size 같은 low-level detail과 physical dynamics 이해에 약하다는 점을 지적해 왔다. UP-VLA는 multi-modal understanding(MMU)과 future prediction(PRE) 두 objective를 하나의 autoregressive 모델로 통합해 high-level semantic + low-level spatial 표현을 동시에 학습한다.

Figure 1과 Figure 2는 UP-VLA의 위치를 보여준다 — VLM-based VLA(RT-2/RoboFlamingo)는 ABCD→D, real-unseen 같은 in-distribution multitask와 real-world 일반화에 강하고, prediction-based(GR-1/SuSIE)는 ABC-D, real-precise 같은 적응/정밀 제어에 강한데, UP-VLA는 두 강점을 모두 흡수한다.

## 2. 아키텍처 및 통합 토큰 형식
**Backbone (Section 4.1)**: Phi-1.5(Li et al. 2023c) LLM(1.3B)을 backbone으로 사용. MMU 입력은 CLIP-ViT(Radford et al. 2021) → projector로 continuous embedding u={u_i}, future prediction 입력은 VQ-GAN(Esser et al. 2021)으로 discrete tokens v={v_i}로 인코딩. Show-o(Xie et al. 2024)에서 backbone 가중치 초기화.

**Unified Prompting (Figure 4, Section 3 식 2개)**:
- MMU: `{|MMU|, (u_1, ..., u_n), (l_1, ..., l_m)}` — image tokens가 서로 attend, language token은 다음 토큰 예측.
- PRE: `{|T2I|, (l_1, ..., l_m), (v_1, ..., v_n)}` — language → image 순서로 image가 모든 prior에 attend, 단 next-token 대신 동일 위치(같은 index)에서 future image token v'_t+Δt를 직접 예측.
- Action token은 PRE 시퀀스 끝에 SOA/EOA로 segmentation, MAP(single-layer attention) + linear policy head로 action a_{t:t+Δt} 출력.

**Joint Prediction-Understanding (Section 4.3)**: 식 (Ô_{t+Δt}, Â_{t:t+Δt}) = π_θ^PRE(O_t, L'). 여기서 L' = [E1(O_t'), π_θ^MMU(O_t, L_prompt), L]로, MMU output을 self-prompting 형태로 재투입해 scene description을 embedding 공간에 추가한다.

## 3. 학습 절차 및 손실 함수
**Pretraining Stage**: Bridge dataset(Walke et al. 2023, 25K robotic arm demos)에서 future prediction, LLaVA-tuning-665k(Liu et al. 2024)에서 MMU 학습. Encoder는 freeze, LLM은 full fine-tune.

**Fine-tuning Stage**: 다운스트림 robot data로 joint training, MMU/PRE/ACT 3 loss를 가중합:
- L_MMU = Σ log p_θ(l_i | u, l_<i) (cross-entropy, next-token)
- L_PRE = Σ log p_θ(v_j' | l, v_1, ..., v_M) (동일 position에서 future image 예측)
- L_ACT = Σ ||â_pos - a_pos||² + BCE(â_end, a_end)
- 전체: L = λ_1 L_MMU + λ_2 L_PRE + λ_3 L_ACT

## 4. CALVIN 시뮬레이션 결과
**ABC→D (Table 1)** — zero-shot generalization (4 chained tasks 평균 길이):
- UP-VLA **4.08** (best)
- UP-VLA-phi-w/o-mmu 3.13, GR-1 3.06, 3D Diffuser Actor 3.35, SuSIE 2.69, RoboFlamingo 2.47, UP-VLA-RT-2 1.44, RT-1 0.90
- UP-VLA의 5-step 성공률: 0.928/0.865/0.815/0.769/0.699 — GR-1(0.854/0.712/0.596/0.497/0.401) 대비 long-horizon에서 큰 격차. 선행 SOTA(GR-1 3.06) 대비 +33.3% 향상이 논문 abstract에 인용된 수치다.

**ABCD→D (Table 2)**: UP-VLA 4.42 (5-step 0.812), GR-1 4.21, RoboFlamingo 4.09, RT-1 2.45. In-distribution multitask에서도 SOTA.

**효과 분석**: 
- Visual prediction의 효과 — UP-VLA-RT-2(action only) 1.44 → UP-VLA(prediction 추가) 4.08으로 +2.64 향상.
- MMU의 효과 — UP-VLA-phi-w/o-mmu(prediction only) 3.13 → UP-VLA(MMU 추가) 4.08으로 +0.95 향상.

## 5. 실세계 Franka-Emika Panda 평가
**Setup (Section 5.3)**: 6 skill, 2K+ demos 수집(picking, placing, cable routing, button pressing, drawer opening 등). Simple scene으로 학습, 복잡 scene(distractor 있음)으로 평가, task 당 20회 시도.

**Figure 6 결과 (Real-World)**:
- **Seen tasks**: UP-VLA, UP-VLA-RT-2, UP-VLA-phi-w/o-mmu 모두 LLM backbone 효과로 RT-1/Diffusion Policy 대비 강함.
- **Unseen objects (semantic generalization)**: UP-VLA > UP-VLA-RT-2 > UP-VLA-phi-w/o-mmu — MMU 통합이 semantic grounding에 결정적. 
- **Precise operations** (cable routing, small block grasping, pen pickup): UP-VLA > UP-VLA-phi-w/o-mmu > UP-VLA-RT-2 — visual prediction이 spatial detail 캡처에 결정적.

종합적으로 두 objective의 통합이 두 측면 모두에서 시너지 효과를 만든다.

## 6. Ablation 및 평가
**Table 3 Ablation**:
- w/o MMU: ABC→D 3.89 (vs 4.08), Real Seen 0.85, Unseen 0.20 — unseen 일반화에 큰 타격.
- w/o Bridge-Pretrain: ABC→D 2.74, Seen 0.65, Unseen 0.30 — pre-training이 dynamics 학습에 필수.
- w/o Prediction: ABC→D 1.44(가장 큰 하락), Seen 0.65, Unseen 0.35.
- w/o MMU-Condition: ABC→D 3.99, Seen 0.80, Unseen 0.50.
- Full: ABC→D 4.08, Seen 0.80, Unseen 0.58.

각 component가 명확한 기여를 보이며, 특히 visual prediction이 ABC→D 일반화에 가장 큰 영향(+2.64).

**Figure 7 Visualization**: VQA에서 UP-VLA가 scene 내 객체 식별 및 상대 위치 추정 가능하나, 특정 객체 식별이 부정확한 경우 존재(데이터/backbone 규모 한계). Future prediction은 robot arm/object 위치를 instruction에 따라 정확히 예측, 단 CALVIN D 환경에서 background color가 학습 분포(ABC)로 회귀하는 경향 — generation 일반화 한계.

**평가 종합**: UP-VLA는 비교적 작은 1.3B 규모로 33% CALVIN ABC→D 성능 향상을 이룬 efficient한 unified VLA의 명확한 사례다. Show-o 기반 unified MLLM을 robotics에 성공적으로 적용했고, ablation이 각 component의 기여를 명확히 분리한다. 한계는 (a) Phi-1.5 1.3B로 인한 객체 식별 정밀도 한계, (b) generation 일반화 부족(CALVIN D background bias).

**YAML 점검**: open_source=true, code_url(github.com/CladernyJorn/UP-VLA), venue="ICML 2025"는 논문 footnote("Proceedings of the 42nd ICML, PMLR 267")와 일치. parameters="1.3B"는 Phi-1.5 backbone(1.3B)을 정확히 반영. backbone="Show-o (CLIP ViT)"는 다소 단순화 — 정확히는 "CLIP-ViT (MMU encoder) + VQ-GAN (PRE encoder) + Phi-1.5 LLM, Show-o weight init". action_head_category="other"는 unified understanding + future prediction 기반 token autoregression이라는 점에서 적절. **YAML 불일치 가능성**: benchmarks.calvin에 calvin_note="+33% over prior SOTA on ABC→D"만 기재되어 있는데, paper의 정확한 수치(ABC→D Avg.Len 4.08, ABCD→D 4.42)를 추가 기재하면 leaderboard.json의 calvin_avg 활용도가 향상된다. eval_condition: fine-tuned 표기는 paper의 fine-tuning evaluation과 일치.

<!-- VERIFIED: pdf -->
