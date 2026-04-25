# InternVLA-M1: Spatially Guided VLA Framework 세미나 리뷰

## 1. 연구 배경 및 핵심 아이디어

InternVLA-M1(Shanghai AI Lab InternRobotics, arXiv:2510.13778)은 instruction-following 로봇이 "where to act"(공간적 위치)를 결정하기 위한 embodiment-agnostic spatial prior를 학습하고, 이를 기반으로 "how to act"(embodiment-aware action)를 생성하는 dual-system VLA를 제시한다.

핵심 모토는 **spatially guided VLA training** — spatial grounding이 instruction과 robot action을 연결하는 핵심 고리 역할을 한다. Figure 1은 box, point, trace, sub-task, latent planning 등 다양한 spatial 표현이 VLM Planner를 통해 DiT Actor의 action 생성을 guide하는 구조를 보여준다.

**대표 성과** (vanilla VLA 대비): SimplerEnv Google Robot +14.6%, WidowX +17.0%, LIBERO Franka +4.3%, 200 task / 3K+ object pick-and-place +6.2%, Real-world cluttered pick-and-place(unseen) +20.6%, Long-horizon reasoning baseline 대비 +10% 이상 우위.

## 2. 아키텍처: Dual-System (System 1 + System 2)

Figure 2는 2-stage pipeline을 보여준다.

**System 2 (slow planner)**: Qwen2.5-VL-3B-Instruct(Bai et al., 2025a) 기반 multimodal encoder가 spatial prior를 캡처하는 reasoner 역할.

**System 1 (fast executor)**: Diffusion Policy(Chi et al., 2023) 기반 86M Action Expert + DINOv2(Oquab et al., 2023) 21M visual encoder + 0.4M state encoder. 총 4.1B params, 단일 RTX 4090 12GB 메모리에서 추론. FlashAttention으로 VLM ~10 FPS, action은 chunking + KV caching으로 가속.

**Latent planning via spatial prompting**: Lightweight querying transformer(8.7MB)가 VLM의 latent planning embedding을 받아 fixed query token으로 매핑. k-layer cross-attention(k=1이면 final layer만), action expert에서 VLM으로의 gradient는 0.5 factor로 감쇠해 multimodal knowledge 보존(Bjorck et al. 2025; Driess et al. 2025 motivated).

## 3. 학습 전략 및 데이터

**Stage 1: Spatial Grounding Pre-training** (VLM only). 3.0M+ samples = 2.3M spatial grounding(box/point/trace) + 0.7M general MM understanding. 데이터 구성(Figure 3):
- General QA 637K (LLaVA-OneVision, InternVL3)
- Box QA 879K (RefCOCO, ASv2, COCO-ReM, RoboRefIt, InternData-M1)
- Trajectory QA 684K (A0 ManiSkill, MolmoAct, InternData-M1 waypoint)
- Point QA 832K (Pixmo-Points, RoboPoint, RefSpatial)

모든 좌표는 absolute(Bai et al., 2025a)로 변환되며 JSON/XML 출력 동시 지원.

**Stage 2: Spatially Guided Action Post-training** (VLM + Action Expert 동시):
- **(a) Spatial Prompting**: instruction에 "Identify all relevant toys and their spatial relationships..." 같은 cue를 prepend, VLM 직접 응답 없어도 spatial awareness 향상
- **(b) Co-training**: robot trajectory data(L2 noise loss for both)와 spatial grounding data(next-token for VLM only) alternating training

16×A100 GPU, 50K steps(~2.5 epoch), batch 256(robot)/64(MM). **Synthetic Data Engine (Figure 4)**: GenManip+Isaac Sim 기반, 14K objects, 211 tables, 1.6K textures, 87 dome lights, 244K closed-loop episodes. ArUco marker로 sim-real intrinsic/extrinsic 일치.

## 4. 핵심 실험 결과: SimplerEnv 및 LIBERO

**SimplerEnv Google Robot Visual Matching (Table 1)**: InternVLA-M1 평균 80.7%, Vanilla VLA 66.1% (+14.6%). Pick Coke Can 95.3, Move Near 90.0, Open/Close Drawer 75.5, Open Top Drawer & Place Apple 62.0.

**Variant Aggregation**: 평균 76.0% (Vanilla 63.5%, +12.5%). CogACT 74.8, SpatialVLA 75.1, π0 58.8 모두 능가.

**SimplerEnv WidowX (Table 2)**: 평균 71.7%, Vanilla 54.7% (+17.0%). Put Spoon on Towel 87.5, Put Carrot on Plate 67.9, Stack Green Block 31.3, Put Eggplant in Yellow Basket 100.0.

**LIBERO (Table 4)**: Spatial 98.0, Object 99.0, Goal 93.8, Long 92.6, **평균 95.9%**. π0.5-KI 94.3, π0 94.2, GR00T N1 93.9 모두 상회. 8×A100, batch 128, action chunk 8, 30K step, 500 trial/suite.

**Table 3 Ablation** (Vanilla co-train → InternVLA-M1): MME 1106→1411, RefCOCO-g IoU 47.1→71.2, Where2place 21.4→25.5, A0-maniskill MAE 6.4→5.1, Google VM/VA 70.2/66.5→80.7/76.0, WidowX 61.1→71.7. Multimodal understanding/spatial grounding/manipulation이 **동시에** 향상됨이 핵심 발견.

Figure 5(c) PSS(Projection-space Similarity, SVD 기반): Vanilla co-train 0.25 → Spatially guided 0.42, gradient subspace alignment 명확 개선.

## 5. 200 Tasks 및 Real-world Cluttered Pick-and-Place

**200 tabletop simulation (Figure 7)** 4 generalization setting 평균:
- π0: 51/42/33/47
- GR00T N1.5: 76/65/62/75
- M1 w/o mid-train: 75/62/59/69
- M1 w/ mid-train: 80/68/72/78 (+6.2% 평균)

**Real-world cluttered pick-and-place (Figure 10)**: Franka Research 3 + Robotiq 2F-85, 23 seen + 27 unseen objects, 5 seen + 6 unseen containers, 6h teleoperated demo, 300 rollout/model.

세부 (M1 w/co-train / w/o / GR00T / π0):
- In-distribution: 92/88/78/45
- Unseen objects: 58/29/49/44
- Unseen orientation: 56/34/40/32
- Unseen instruction: 67/37/63/25
- By spatial: 61/58/53/18

Co-training이 unseen 일반화에 평균 +20.6% 기여, 모든 setting에서 일관된 우위.

## 6. Long-horizon 추론 및 평가

**Long-horizon tasks (Figure 12)** Desktop Sorting: M1 67% / π0 30 / GR00T 38 (In-distribution); Physical interference 67/55/57; Task replanning 63/62. Make Sandwiches: 70/51/45 (In-dist), 67/52/45 (Physical).

**VLM planner scheduling (Table 5)** (M1 / GPT-5 / Gemini-2.5 Pro / Qwen2.5-VL-72B / Qwen2.5-VL-3B):
- Sort into Drawers: 90/75/57/31/30
- Make Sandwiches: 91/67/62/71/49
- Desktop Sorting: 91/62/83/34/52
- Math calc: 93/79/53/33/41
- Goods Purchase: 92/82/61/29/38

Post-training이 commercial LLM(GPT-5, Gemini-2.5)을 일관되게 능가. 22h long-horizon teleoperated demos(약 500 demo/task)로 fine-tuning, subtask-level transition을 위한 zero-action vector padding 도입.

**강점**: spatial grounding 명시적 학습 + latent planning + spatial prompting을 action expert에 transfer하는 설계가 generalization 일관성에 크게 기여; 200 task/3K+ object scale generalizable benchmark 자체 구축; LIBERO 평균 95.9%, SimplerEnv 다수 SOTA.

**약점**: 4.1B params / DINOv2 + Qwen2.5-VL 의존도가 높아 edge deployment 제약; spatial grounding data(2.3M)가 web/real/sim 혼합으로 노이즈 가능성.

**YAML 점검**:
- `open_source=true`, `code_url` 일치
- `backbone="Qwen2.5-VL"` (=Qwen2.5-VL-3B-Instruct), `action_head` "dual-system" 정확
- `parameters="based on Qwen2.5-VL"`는 모호 — **YAML 보완 필요**: 총 ~4.1B(VLM 3B + DiT 86M + DINOv2 21M + state 0.4M + querying transformer 8.7MB) 명시 권장
- `benchmarks.simpler_env.simpler_note="SOTA on SimplerEnv"`는 정성적 — **YAML 불일치 가능**: 실제 SimplerEnv Google VM 80.7 / VA 76.0 / WidowX 71.7, LIBERO Avg 95.9 등 정량값 추가 시 leaderboard.json 활용도 향상
- `eval_condition: fine-tuned`는 paper 평가 조건과 일치

<!-- VERIFIED: pdf -->
