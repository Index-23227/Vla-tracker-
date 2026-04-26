# VLA-Thinker: Boosting VLAs through Thinking-with-Image Reasoning

**Wang, Bao, Gao, Xu, Tian, Rawat, Ge, Shang, 2026 (arXiv:2603.14523v1, UCF / U. Würzburg / USC / NVIDIA Research)**

## 한 줄 요약
OpenVLA-OFT 백본 위에 "perception을 동적으로 호출 가능한 reasoning action"으로 본 thinking-with-image 프레임워크를 도입, ZOOM-IN tool 호출과 (think) 토큰을 interleave하는 multimodal CoT를 SFT cold-start + GRPO RL 2단계로 학습 — LIBERO 평균 97.5% (OpenVLA-OFT 91.0 대비 +6.5%pp), RoboTwin 2.0의 Short/Medium/Long horizon에서 각 62.3 / 70.7 / 64.6%를 기록.

## 핵심 기여
- **Thinking-with-image paradigm**: 시각을 정적 컨텍스트가 아닌 동적 호출 가능한 reasoning action으로 격상시킨 첫 VLA. 매 추론 step에서 ZOOM-IN tool을 호출해 새로운 visual evidence 획득 가능.
- **2-stage 학습 레시피**: (i) Qwen3-VL-30B-A3B-Instruct로 합성한 visual CoT 데이터 SFT cold-start, (ii) sparse task success 보상의 GRPO로 trajectory-level alignment.
- **Long-horizon 우위**: LIBERO Long +10.4%pp, RoboTwin 2.0 Extra-Long(blocks_rank_rgb 79.3, put_bottles_dustbin 55.4)에서 OpenVLA-OFT 압도.

## 배경
기존 CoT-VLA류는 시각 입력을 한 번 임베딩한 뒤 모든 추론을 언어 공간에서 수행한다 (§1, Fig. 1 left). 이는 사람의 능동적 시지각(uncertainty 발생 시 줌인·재관찰)과 어긋나며, 특히 long-horizon에서 ambiguity·중간 오류 회복이 어렵다. VLA-Thinker는 perception을 "tool call로서의 reasoning action"으로 격상시켜 reasoning–perception–action을 interleave한다 (Fig. 1 right). 핵심 도전은 (i) 무엇을 추론할지뿐 아니라 언제·어떻게 시각 도구를 호출할지 학습, (ii) 전체 reasoning–action trajectory를 sparse task-success 보상으로 align하는 것이다.

## 방법론
- **문제 정식화 (§2.1)**: τ = {T_1, C_1, V_1, ..., T_k, A_k} (Eq. 2). T_k는 think text, C_k는 perception 호출(tool name + args), V_k는 도구 반환 시각 증거, A_k는 environment action. Controller가 매 step "다음 reasoning + perception 호출" vs "action 출력"을 결정.
- **Visual tool**: ZOOM-IN(crop_image, bbox_2d) 한 종류로 시작 — fine-grained 영역 inspection. Appendix에서 protocol 명시. 향후 detection, segmentation 등 확장 가능성을 future work로 둠 (§2.1 후반).
- **백본 (§3.1)**: OpenVLA-OFT[26] 채택 (LLaMA2-7B + SigLIP+DINOv2 vision encoder, action chunking + parallel decoding). 효율성을 위해 wrist camera 미사용 — 단일 third-view RGB + 언어 + proprio.
- **Stage 1 SFT cold-start (§2.2)**: Qwen3-VL-30B-A3B-Instruct로 embodied CoT 데이터 합성. Gripper state 변화로 keyframe 식별 → keyframe에는 "tool 호출 + think" 모두, intermediate frame에는 pure text think 생성. Schema validation + temporal consistency 강제.
- **Stage 2 GRPO RL (§2.2, Eq. 3-7)**: π_θ가 trajectory τ 샘플링, 보상 R(τ) = α_s·I_success + α_f·I_format (Eq. 4). 그룹 M개 trajectory의 평균/표준편차로 advantage 정규화 (Eq. 6, DeepSeek-R1식). Eq. 7의 PPO-style clipped objective + KL 정규화. Sparse 성공 보상만으로 reasoning과 action을 jointly 최적화. SFT batch 64, RL batch 128.

## 실험 결과
### LIBERO
- **LIBERO (Table 2)**: 평균 97.5% — Spatial 98.7 / Object 99.0 / Goal 95.2 / Long 96.9. OpenVLA-OFT(91.0) 대비 +7.1/+3.7/+4.6/+10.4pp, 평균 +6.5pp. Long suite에서 +10.4pp 개선이 가장 두드러져 thinking-with-image의 long-horizon 효과를 입증. 비교군: PD-VLA 94.7, π0-FAST 85.5, NORA 87.9, FlowVLA 88.1, UnifiedVLA 95.5, MolmoAct 86.6 등.
### RoboTwin 2.0 (bimanual, 50 task suite)
- **Short horizon (Table 3, 112-130 steps)**: 4 task 평균 62.3% (lift_pot 64.8 / beat_block_hammer 82.5 / pick_dual_bottles 65.4 / place_phone_stand 36.6). OpenVLA-OFT 21.3, π0 45.5, π0-FAST 27.3 대비 압도적.
- **Medium horizon (151-223 steps)**: 평균 70.7% (move_can_pot 61.0 / place_a2b_left 39.1 / place_empty_cup 92.7 / handover_mic 89.9). OpenVLA-OFT 47.1 대비 +23.6pp.
- **Long & Extra-Long (283-637 steps)**: 평균 64.6% (handover_block 52.8 / stack_bowls_two 71.1 / blocks_rank_rgb 79.3 / put_bottles_dustbin 55.4). OpenVLA-OFT 46.5, RDT 27.8 대비 큰 폭 우위.

### Ablation
- **Training stages (Table 4)**: SFT 단독 시 LIBERO 평균 88.2%로 OpenVLA-OFT(91.0)보다 오히려 떨어짐 — RL 정렬 없이는 reasoning이 노이즈로 작용. SFT + GRPO 모두 적용 시 97.5%로 최고. 이는 cold-start만으로는 부족하고 trajectory-level 보상 정렬이 필수임을 시사.
- **GRPO 단독**: 본문 §3.2 후반에 "RL 단독 초기화는 형식·도구 사용 패턴 미정착으로 학습 불안정"이라고 보고 — 두 stage가 보완적.

### 정성 분석
- **Fig. 1 (right)**: 스토브 점등 + moka pot 옮기기 task에서 ZOOM-IN을 호출해 가스밸브 위치를 재확인 후 정확히 grasp — text-only CoT 모델은 동일 시나리오에서 grasp 실패로 task 미완료.
- 모델은 `<think>...zoom in...</think><tool_call>{"name": "crop_image", "arguments": {"bbox_2d": [...]}}</tool_call>` 형식으로 도구 호출, 이후 반환 sub-image를 컨텍스트에 추가해 다음 reasoning step 진행.

## 한계
- ZOOM-IN(crop_image)이라는 단일 시각 도구만 검증되어 있어, segmentation·depth·tracking 등 도구 다양화의 효과는 미검증 (§2.1 후반에서 future work로 명시).
- SFT 데이터를 Qwen3-VL-30B로 합성 — 합성 라벨의 잠재 편향이 RL 단계까지 전이될 가능성. Filtering(schema/temporal) 외 검증 절차가 정량적으로 보고되지 않음.
- LIBERO의 wrist camera 미사용 등 평가 조건이 OpenVLA-OFT 원래 셋업과 다름 — 동일 입력 조건의 직접 비교 시 격차 해석에 주의 필요.
- RoboTwin 2.0 일부 task(place_phone_stand 36.6, place_a2b_left 39.1)는 여전히 낮아 dual-arm 조정이 어려운 영역에서는 thinking-with-image도 한계.
- 추론 latency(매 step 도구 호출 + reasoning) 측정이 본문에 명시적이지 않아 실시간 제어 적용성 평가가 어렵다.
- **YAML 점수 일치 확인**: `data/models/vla_thinker.yaml`의 LIBERO(98.7/99.0/95.2/96.9) 및 RoboTwin v2 sub-task 점수, short/medium/long 평균은 논문 Table 2/3과 일치. 점수 신뢰성 양호.

## 총평
"text-only CoT의 한계를 perception tool call로 명시적으로 깬다"는 단순하지만 강한 메시지를 (i) 합성 visual CoT 데이터로 SFT cold-start, (ii) sparse success 보상의 GRPO로 trajectory-level alignment 하는 깔끔한 2단 학습으로 구현했다. LIBERO 97.5%(특히 Long +10.4pp)와 RoboTwin 2.0의 모든 horizon group에서 OpenVLA-OFT를 큰 폭으로 능가한 결과는, 시각을 능동적 도구로 다루는 패러다임이 long-horizon embodied reasoning에 실질적 이득을 가져옴을 보여준다. Ablation에서 SFT 단독이 오히려 성능을 떨어뜨리는 점은 RL alignment의 필수성을 명확히 한 정직한 발견이다.

## 예상 질문
1. **ZOOM-IN 외의 도구 없이도 정말 thinking-with-image라 부를 수 있는가?**
   - 저자들도 이를 인정하고 "fundamental effectiveness 검증을 위한 대표 instance"로 사용. 다양 도구 확장(detection, segmentation, tracking)은 future work (§2.1).
   - 단일 도구만으로도 LIBERO +6.5%pp, RoboTwin 2.0 +20pp 이상 — paradigm 자체의 효과 입증.
2. **SFT만으로는 왜 떨어지는가(88.2%)?**
   - 저자 분석(§3.2): SFT는 reasoning/tool format은 학습하지만 어느 시점에 호출이 task 성공에 기여하는지를 학습 못 함.
   - 합성 데이터의 도구 호출 패턴이 실제 manipulation task의 success criterion과 어긋날 수 있어, 오히려 추론 비용만 증가시킴.
   - RL의 sparse reward(I_success)만이 "언제 호출할지"를 align한다.
3. **RoboTwin 2.0에서 horizon이 길수록 격차가 줄어드는 이유?**
   - Long(64.6) < Medium(70.7) > Short(62.3) 패턴. 매우 긴 horizon에서는 제한된 도구(ZOOM-IN)와 sparse reward만으로는 충분한 신호 확보가 어렵다고 §3.2에서 언급.
   - 더 풍부한 도구·dense reward·hierarchical RL 도입을 후속 과제로 제시.
4. **GRPO를 선택한 이유는? PPO/SAC와 비교는?**
   - DeepSeek-R1 식 group-relative advantage(Eq. 6)로 explicit value function 불필요 → long-horizon trajectory의 variance 감소.
   - Sparse reward (Eq. 4)에서 PPO보다 안정적 학습이 가능하다는 게 저자 입장. 다만 다른 RL 알고리즘과의 직접 비교 ablation은 본문에 없음.

<!-- VERIFIED: pdf -->
