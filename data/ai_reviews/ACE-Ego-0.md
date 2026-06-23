# ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining

**arXiv:** 2606.17200 · **Date:** 2026-06-15 · **Org:** ACE Robotics / CUHK MMLab / CUHK-Shenzhen / SJTU / THU
**Project:** https://acerobotics-vla.github.io/ACE-Ego/ · **Code:** https://github.com/ACERobotics-VLA/ACE-Ego

## 1. 한 줄 요약
ACE-Ego-0는 4.53K시간의 로봇/시뮬레이션 데이터와 1.48K시간의 pseudo-action 레이블된 에고센트릭 사람 영상을 **단일 표현 공간**으로 통합하여 사전학습하는 VLA 프레임워크로, RoboCasa GR1 TableTop과 RoboTwin 2.0에서 SOTA를 달성한다.

## 2. 문제 정의
대규모·다양한 embodied 데이터는 VLA 일반화의 핵심이지만, 로봇 텔레오퍼레이션은 비용·노동 집약적이다. 사람 에고센트릭 영상은 보완적 supervision을 제공하지만 (i) 좌표계(MANO local vs. global world), (ii) 키네매틱 구조, (iii) 제어 주파수, (iv) supervision 품질(센서 vs. 비전 추정 노이즈)에서 로봇 데이터와 충돌한다. 기존 방법은 이 네 축을 동시에 다루지 못한다.

## 3. 핵심 기여
1. **Unified action representation**: spatial(canonical camera-space action) + structural(cross-embodiment morphology conditioning) + temporal(time-aligned action chunking) 정렬.
2. **Reliability-aware training objective**: 로봇 데이터는 primary flow-matching loss로, 사람 pseudo-action은 noise-robust 채널(position)만에 가중된 **human auxiliary loss**로 학습.
3. **5단계 에고센트릭 비디오→pseudo-action 파이프라인**: 6개 데이터셋에서 1.48K시간 변환.

## 4. 아키텍처
- **VLM backbone**: Qwen3-VL-4B-Instruct (~4B), 24-layer vision encoder, patch 16×16, 256×256 multi-view 입력 (head + wrist).
- **Action expert**: Flow-matching Diffusion Transformer (DiT), 36 layers, hidden 1024, 16 heads, **~600M params**, 4 flow-matching steps at inference.
- **Morphology conditioning**: 로봇은 URDF→kinematic graph→message-passing 인코더 `E_urdf`로 morphology token 생성; 사람 비디오 소스는 학습 surrogate embedding. 토큰은 VLM과 분리되어 action decoder에만 주입.
- **Action chunking**: 프레임 인덱스가 아닌 **물리 timestamp** 기준으로 인덱싱하여 제어 주파수 이질성 흡수.

## 5. Action Head Category
**Flow matching** (DiT 기반 conditional flow-matching, position 채널 보조 손실 추가). VLA-Tracker enum 상 `flow_matching`.

## 6. 학습 데이터 & 레시피
- **총 6.0K+ 시간**: 71.6% 시뮬레이션, 24.9% 사람, 3.4% 로봇 (749 그룹).
- 로봇/시뮬: Open X-Embodiment 일부, AgiBot R1Lite, AgiBot DigitalWorld, RoboCasa GR1 Tabletop (24 tasks × 1,000 ep), Galbot 자체 수집 1,800+시간.
- 사람: Ego4D, EPIC-KITCHENS, EgoExo4D, EgoDex, EgoScale 등 6개 소스 → 5단계 파이프라인 (clip 분할, hand detection, MANO 재구성, 좌표 정렬, 신뢰도 추정).
- VLM lr 2e-5 / action expert lr 1e-4. SFT는 16×A800 GPU.

## 7. RoboCasa GR1 TableTop 결과 (Table 3)
24 tasks, 50 rollouts/task. ACE-Ego-0 **72.8%** (Avg).
- 베이스라인: GR00T-N1.6 47.6 · Qwen3π 43.9 · FLARE 55.0 · ABot-M0 58.3 · JoyAI-RA 63.2 · DIAL 70.2.
- 대표 태스크: CuttingboardToCardboardbox 84.0, PlateToPlate 98.0, TrayToPlate 90.0, PlacematToTieredshelf 44.0.

## 8. RoboTwin 2.0 결과 (Table 4)
50 tasks × 100 trials, Easy/Clean & Hard/Randomized. ACE-Ego-0 **Easy 91.12% / Hard 90.62%** (Avg 90.87%).
- 베이스라인: π0.5 82.74/76.76 · Motus 88.66/87.02 · LingBot-VLA 88.56/86.68 · ABot-M0 86.06/85.08 · JoyAI-RA 90.48/89.28 · Hy-VLA 90.9/90.1.
- JoyAI-RA 대비 +0.64 / +1.34. 학습 데이터: 2,500 clean + 25,000 randomized demos.

## 9. Real-World ARX 이중팔 결과
head-mount RGB-D, camera-space delta EE 명령. Scooping/Pouring, Packing Shoes, Stacking Bowls, Dustpan Sweeping 등 장기·접촉-풍부 태스크에서 π0.5 대비 일관된 우위(논문 Fig. 5a; 대부분 태스크 ≥83%).

## 10. Ablation 결과 (RoboCasa)
- **w/o morphology conditioning**: 72.8 → 70.9 (−1.9). 같은 카메라-공간 action 포맷이라도 키네매틱 구조 차이는 잔존.
- **w/o time-aligned chunking**: 추가 하락 (논문 −1.1).
- **w/o reliability-aware weighting** (사람 영상에 robot과 동일 가중치): action expert 학습이 혼란.
- **데이터 ablation**: Robot-only → Robot+Sim → Robot+Human(full) 진행에서 +4.5pp 최대 단일 이득이 human 영상 추가에서 발생, 사람 비디오가 행동 다양성에 기여함을 입증.

## 11. 한계 & 토의
- VLM/action expert 모두 inference 시 ~5B params, 4-step flow matching이지만 실시간성(Hz) 보고는 제한적.
- Pseudo-action 품질은 hand reconstruction(MANO) 정확도에 의존; occlusion·jitter는 reliability weight으로 완화하나 완전 제거 안 됨.
- 평가 시뮬은 RoboCasa GR1 + RoboTwin 2.0 2종에 한정 (LIBERO/CALVIN/SimplerEnv 미보고).
- 실 로봇은 ARX 단일 플랫폼; 다중 embodiment 일반화 증명은 시뮬 위주.

## 12. VLA-Tracker 관점에서의 의의
2026년 6월 시점에서 (i) **사람 에고센트릭 영상을 행동-레벨 supervision으로** 단순 BC가 아닌 reliability-aware auxiliary loss로 결합한 첫 대규모 사례 중 하나이며, (ii) RoboTwin 2.0 Hard에서 90%대 진입, RoboCasa GR1 TableTop에서 DIAL 대비 +2.6pp 달성. flow-matching DiT + Qwen3-VL-4B 조합은 π0/π0.5 계열의 **사람 데이터 통합 후속**으로 자리매김.

<!-- VERIFIED: pdf -->
