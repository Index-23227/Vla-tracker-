# InternVLA-M1: Spatially Guided VLA Framework 세미나 리뷰

## 1. 연구 배경 및 핵심 아이디어
InternVLA-M1(Shanghai AI Lab InternRobotics, arXiv:2510.13778)은 instruction-following 로봇이 "where to act"(공간적 위치)를 결정하기 위한 embodiment-agnostic spatial prior를 학습하고, 이를 기반으로 "how to act"(embodiment-aware action)를 생성하는 dual-system VLA를 제시한다. 핵심 모토는 spatially guided VLA training — spatial grounding이 instruction과 robot action을 연결하는 핵심 고리. Figure 1은 box, point, trace, sub-task, latent planning 등 다양한 spatial 표현이 VLM Planner를 통해 DiT Actor의 action 생성을 guide하는 구조를 보여준다.

대표 성과: SimplerEnv Google Robot에서 vanilla VLA 대비 +14.6%, WidowX +17.0%, LIBERO Franka +4.3% 향상. 200 task / 3K+ object generalizable pick-and-place에서 +6.2%, real-world cluttered pick-and-place에서 +20.6% (unseen objects), long-horizon reasoning에서 baseline 대비 +10% 이상 우위.

## 2. 아키텍처: Dual-System (System 1 + System 2)
Figure 2는 2-stage pipeline을 보여준다. System 2(slow planner)는 Qwen2.5-VL-3B-Instruct 기반 multimodal encoder로 spatial prior를 캡처하고, System 1(fast executor)은 86M Diffusion Policy(Chi et al., 2023) action expert + DINOv2 encoder(21M) + state encoder(0.4M)로 구성된 DiT Actor다. 총 4.1B params, 단일 RTX 4090 12GB 메모리에서 추론 가능. FlashAttention으로 VLM은 ~10 FPS, action 실행은 chunking + KV caching으로 가속.

**Latent planning via spatial prompting**: lightweight querying transformer(8.7MB)가 VLM의 latent planning embedding을 받아 fixed query token으로 매핑(k-layer cross-attention; k=1이면 final layer만). Action expert에서 VLM으로의 gradient flow는 0.5 factor로 감쇠시켜 multimodal knowledge 보존(Bjorck et al. 2025; Driess et al. 2025 motivated).

## 3. 학습 전략 및 데이터
**Stage 1: Spatial Grounding Pre-training** (VLM only). 3.0M+ multimodal samples = 2.3M spatial grounding (box/point/trace) + 0.7M general MM understanding. 데이터 구성: General QA 637K (LLaVA-OneVision, InternVL3), Box QA 879K (RefCOCO, ASv2, COCO-ReM, RoboRefIt, InternData-M1), Trajectory QA 684K (A0 ManiSkill, MolmoAct, InternData-M1 waypoint), Point QA 832K (Pixmo-Points, RoboPoint, RefSpatial). 모든 좌표는 absolute, JSON/XML 출력.

**Stage 2: Spatially Guided Action Post-training** (VLM + Action Expert 동시). 두 전략: (a) Spatial Prompting — instruction에 "Identify all relevant toys and their spatial relationships..." 같은 auxiliary cue를 prepend. VLM은 직접 응답하지 않더라도 spatial awareness가 향상. (b) Co-training — robot trajectory data(L2 noise loss for both VLM+expert)와 spatial grounding data(next-token prediction for VLM only)를 alternating training.

Stage 2는 16×A100 GPU, 50K steps(~2.5 epoch), batch 256(robot)/64(MM)로 학습.

## 4. 핵심 실험 결과: SimplerEnv 및 LIBERO
**SimplerEnv Google Robot (Table 1)**: InternVLA-M1 Visual Matching 평균 80.7% (Vanilla VLA 66.1%, +14.6%), Variant Aggregation 76.0% (63.5%, +12.5%). 세부 task: Pick Coke Can 95.3, Move Near 90.0, Open/Close Drawer 75.5, Open Top Drawer & Place Apple 62.0 (VM). CogACT(74.8), SpatialVLA(75.1), π0(58.8), π0-FAST(61.9) 모두 능가.

**SimplerEnv WidowX (Table 2)**: 평균 71.7% (Vanilla VLA 54.7%, +17.0%). 세부: Put Spoon on Towel 87.5, Put Carrot on Plate 67.9, Stack Green Block 31.3, Put Eggplant in Yellow Basket 100.0. GR00T N1.5 61.9, CogACT 51.3 등 능가.

**LIBERO (Table 4)**: Spatial 98.0, Object 99.0, Goal 93.8, Long 92.6, **평균 95.9%**. π0.5-KI 94.3, π0 94.2, GR00T N1 93.9 모두 상회. 8×A100 GPU, batch 128, action chunk 8, 30K step(~20h), 500 trial/suite 평가.

**Ablation (Table 3)**: Vanilla co-train(MME 1106, RefCOCO-g IoU 47.1, Google VM 70.2, WidowX 61.1) 대비 InternVLA-M1(MME 1411, RefCOCO-g 71.2, VM 80.7, WidowX 71.7) — multimodal understanding/spatial grounding/manipulation 모두 동시 향상. Figure 5(c): vanilla co-train의 Projection-space Similarity(SVD 기반)는 0.25, spatially guided는 0.42로 gradient subspace alignment 개선.

## 5. 200 Tasks 및 Real-world Cluttered Pick-and-Place
**200 tabletop simulation (Figure 7)**: 4 generalization setting(In-distribution, Unseen Object, New Background, Unseen Instruction). InternVLA-M1 w/ mid-train(InternData-M1으로 mid-train 후 fine-tune) 평균 SR가 In-distribution 80, Unseen objects 68, New background 72, Unseen instruction 78로 GR00T N1.5 대비 평균 +6.2%, π0 baseline은 51/42/33/47.

**Real-world cluttered pick-and-place (Figure 10)**: Franka Research 3 + Robotiq 2F-85 gripper, 23 seen + 27 unseen objects, 5 seen + 6 unseen containers. 6h teleoperated demo + InternData-M1 co-training. In-distribution 92(M1 w/co-train)/88/78/45(π0); Unseen objects 58(M1)/29/49/44; Unseen orientation 56/34/40/32; Unseen instruction 67/37/63/25; By spatial 61/58/53/18 등 모든 setting에서 일관된 우위. Co-training이 unseen 일반화에 +20.6% 평균 기여.

## 6. Long-horizon 추론 및 평가
**Long-horizon tasks (Figure 12)**: Sort items into Drawers, Make Sandwiches, Desktop Sorting을 In-distribution / Physical interference / Task replanning 3 setting으로 평가. 예: Desktop Sorting In-distribution InternVLA-M1 67%(π0 30, GR00T 38), Physical interference 67/55/57, Task replanning 63/62. Make Sandwiches In-distribution 70/51/45, Physical 67/52/45 등 모든 setting에서 baseline 능가.

**VLM planner scheduling (Table 5)**: Sort into Drawers 90 / Make Sandwiches 91 / Desktop Sorting 91 / Math calc 93 / Goods Purchase 92로 GPT-5(75/67/62/79/82), Gemini-2.5 Pro(57/62/83/53/61), Qwen2.5-VL-72B(31/71/34/33/29) 모두 능가. Post-training의 task-planning 효과 입증.

**평가 종합**: spatial grounding을 명시적으로 학습하고 latent planning + spatial prompting으로 action expert에 transfer하는 설계가 generalization 일관성에 크게 기여. 단점은 (a) 4.1B parameter / DINOv2 + Qwen2.5-VL 의존도가 높아 edge deployment 제약 가능, (b) spatial grounding data(2.3M)가 web/real/sim 혼합으로 노이즈 가능성. 

**YAML 점검**: open_source=true, code_url 일치. backbone="Qwen2.5-VL"(=Qwen2.5-VL-3B-Instruct), action_head="dual-system: Qwen2.5-VL spatial planner + DiT action head" 정확. parameters "based on Qwen2.5-VL"는 모호 — **YAML 보완 필요**: 총 ~4.1B(VLM 3B + DiT 86M + DINOv2 21M + state 0.4M + querying transformer 8.7MB)로 명시 권장. benchmarks.simpler_env.simpler_note "SOTA on SimplerEnv"는 정성적 — **YAML 불일치 가능**: 실제 SimplerEnv Google VM 80.7 / VA 76.0 / WidowX 71.7, LIBERO Avg 95.9 등 정량값을 추가하면 leaderboard.json 활용도가 향상된다 (eval_condition은 fine-tuned로 일치).

<!-- VERIFIED: pdf -->
