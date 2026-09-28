# Fine-Tuning VLAs with Self-Demonstrated Generative Control for Multi-Task Manipulation (SelfDemo-VLA)

> **한 줄 요약**: 새 로봇에 π0.5를 파인튜닝하면 파지는 좋아지지만 지시 따르기와 사전학습 행동(place, push 등)을 잊어버린다. 저자들은 **동결된 zero-shot π0.5를 타깃 로봇에서 직접 굴려 얻은 (대부분 실패한) 자기 롤아웃**을 "self-demo"로 기록해 소량의 전문가 데이터와 1:1로 섞어 전체 파인튜닝한다. 실물 ALOHA에서 pick-and-place SR 0% → 55%(place 전문가 데모 없이), 큐브 집기 SR 65% → 90%, 기어 삽입 SR 30% → 90%; RoboTwin에서 이전 과제 유지율 16.6% → 70.6%.

- **arXiv**: 2608.19490v1 (2026-08-19, cs.RO)
- **소속**: University of Illinois Urbana-Champaign (Prachi Garg, Saurabh Gupta, Derek Hoiem 외)
- **프로젝트**: https://self-supervised-control.pages.dev/
- **참고**: 논문은 방법에 이름을 붙이지 않았다. 트래커에서는 "π0.5 Multi-Task ES+SS" 정책을 `SelfDemo-VLA`로 표기한다.

---

## 1. 배경 및 동기

π0.5 같은 VLA는 사전학습 로봇과 하드웨어(그리퍼, 카메라 자세)가 조금만 달라도 성능이 급락한다. 저자들의 ALOHA-1에서 zero-shot π0.5는 지시된 물체로 80%까지 정확히 접근하지만 **파지는 0%**다. 전문가 텔레옵 데이터로 파인튜닝하면 파지는 회복되지만, (i) 텍스트 지시를 무시하고 시각만 보고 행동하는 "visual-action shortcut", (ii) 파인튜닝 과제에 없는 행동(pick 이후의 place)을 완전히 잊는 파국적 망각이 생긴다. 사용자는 대개 사전학습 데이터에 접근할 수 없으므로 일반적인 rehearsal도 불가능하다.

## 2. 핵심 관찰

기본 정책은 과제를 끝내지 못해도 **롤아웃 궤적이 지시와 의미적으로 상관**되어 있다(올바른 물체로 향하고, 컨테이너 쪽으로 이동하려 함). 따라서 이 롤아웃 자체를 증류 타깃으로 쓰면, 타깃 로봇·타깃 장면·타깃 프롬프트에서 수집되므로 도메인 갭이 없는 rehearsal 데이터가 된다.

## 3. 방법

- **Expert-supervised (ES)**: 텔레옵/모션플래너 데이터 D_ES. 한 장면에 여러 물체를 두고 서로 다른 지시로 수집해 텍스트가 과제를 결정하도록 설계(Multi-Task ES).
- **Self-supervised (SS)**: 동결 base π0.5를 사전학습 과제군의 프롬프트 P_SS로 타깃 로봇에서 실행, 매 결정 스텝 샘플한 action chunk â_t ~ π_base(·|I_t, p_SS, q_t)를 실행·기록해 D_SS = {(o_t, â_t)} 구성.
- **손실**: ℓ = CE(FAST 토큰) + α·‖(ω − a) − f_θ(a^{τ,ω}, o)‖² (FAST+Flow 이중 목표, openpi에 없는 FAST 부분 재구현). L = L_ES + λ·L_SS.
- **데이터 혼합**: ES:SS = 1:1, 각 버퍼 내부는 균일 샘플링(task-weighted는 성능 저하). 위험하거나 원치 않는 self-demo는 수동 필터링.
- 기본 설정은 VLM + action expert **전체 파인튜닝**.

DreamBooth식 prior preservation과 유사하지만, (a) 오프라인 샘플링이 아니라 타깃 로봇에서의 **온라인 물리 상호작용**으로 생성하고, (b) 같은 클래스가 아니라 전문가 과제와 다른 사전학습 과제를 재생한다는 점이 다르다.

## 4. 벤치마크 구성

- **Benchmark 1 (실물 ALOHA, 오른팔)**: 전문가 30 demo("Pick up the red/green cube", 약 14분 텔레옵) + self-demo 15개(pick-and-place, laundry). 테스트: T_ES, T_NO(새 물체), T_SS¹(pick & place), T_SS²(laundry).
- **Benchmark 2 (실물 ALOHA, 양팔 caterpillar 기어 삽입)**: 전문가 60 demo(3색) + self-demo 29개. 테스트: 학습 색 T_ES, 미학습 3색 T_NO, 각 10 trial. 판이 고정되지 않아 다른 팔이 판을 안정화해야 함.
- **Benchmark 3 (RoboTwin 2.0, ALOHA-AgileX)**: π0.5 zero-shot이 0%여서 10개 과제로 stage-1 mid-training(90.8%)한 모델을 base로 삼음. stage-2에서 두 개의 stacking 과제를 전문가 데이터로 학습하며, 10개 stage-1 과제는 과제당 10개 self-demo로만 유지. 테스트: T_SS, T_ES, T_NO, T_NC, 과제당 50 seed.

## 5. 구현 세부

- H=50, D=32, 쿼리당 25 action open-loop 실행, 30–50 Hz.
- 실물: 10k iter, batch 64, α=8.0 (기어 과제 16–18k iter, α=10.0).
- 시뮬 stage-2: 6k iter, batch 64, LR 2.5e-5. stage-1은 10k iter, batch 128, 과제 frame 수 역수 가중.
- 학습: H200 1장 또는 H100 2장(전체 파인튜닝에 120GB+ VRAM 필요), 추론은 RTX 4090.

## 6. 실험 설정 및 지표

- **SR**: 지정 물체와 상호작용하고 올바른 행동으로 과제를 완료.
- **IF (pre-grasp)**: 지정 물체로 접근해 파지 시도하는 비율. **IF (post-grasp)**: 파지 후 올바른 컨테이너/샤프트로 이동하는 비율.
- laundry는 옷 일부라도 바구니에 들어가면 partial success.
- 기준선: zero-shot, Single-Task ES, Multi-Task ES(단일 물체/장면 변형 포함), 상한 ES+ES(모든 과제 전문가 데이터), RoboTwin에서는 rehearsal oracle, LoRA/부분 동결 PEFT, flow-only 목표.

## 7. 주요 결과

**Table 2 – ALOHA Benchmark 1 (%, 20 trial)**

| Method | T_ES IF / SR | T_NO IF / SR | Pick&Place IF-pre / IF-post / SR | Laundry IF-pre / IF-post / Partial SR |
|---|---|---|---|---|
| π0.5 Zero-Shot | 80 / 0 | 70 / 0 | 70 / 25 / 0 | 65 / 30 / 5 |
| Multi-Task ES | 100 / 65 | 50 / 50 | 75 / 0 / 0 | 100 / 50 / 10 |
| **Multi-Task ES+SS** | **100 / 90** | **55 / 55** | **90 / 90 / 55** | **100 / 95 / 40** |

**Table 3 – 기어 삽입 (%, 10 trial)**: ES+SS T_ES SR 90 (ES 30), T_NO SR 30 (ES 0), post-grasp IF 90 vs 40.

**Table 6/7 – 상한 비교 및 held-out push**: 모든 과제 전문가 데이터(ES+ES)는 T_ES SR 95, pick&place 65, laundry 90으로 약간 우위지만, 학습에 없던 "push" 과제에서 ES+SS가 push SR 60% vs ES+ES 5%(ES+ES는 35% 경우 물체를 집어버림).

**Table 4/5 – RoboTwin**

| Method | T_SS | T_ES | T_NO | T_NC | Overall |
|---|---|---|---|---|---|
| RoboTwin base (stage 1) | 90.8 | – | – | – | – |
| Rehearsal oracle (frame) | 85.6 | 97.0 | 30.0 | 15.3 | 57.0 |
| LoRA (ES) | 28.0 | 87.0 | 37.5 | 2.0 | 38.6 |
| Multi-Task ES | 16.6 | 93.0 | 57.0 | 6.7 | 43.3 |
| **Multi-Task ES+SS** | **70.6** | **98.0** | 44.5 | 14.0 | 56.8 |

self-demo 과제당 10개만으로 망각된 T_SS를 16.6 → 70.6으로 회복하고, 새 과제 성능도 93 → 98로 상승. 사전학습 데이터에 접근하는 oracle과 overall에서 0.2p 차이.

**부가 결과**: FAST+Flow 이중 목표가 flow-only보다 망각이 적음(T_SS 16.6 vs 11.2); LoRA·부분 동결은 새 과제 학습이 약함. 복제 비율 1:1이 비례 샘플링(69.2 vs 56.4, 검증셋)보다 좋고, self-demo를 과제당 5개로 줄이면 56.8로 하락.

## 8. Related Work 상의 위치

- VLA 연속학습(표현 정렬, adapter routing·확장, weight merging, RL 미세조정)과 경험 재생(ExpReS-VLA 등) 계열 중 **사전학습 데이터 없이** 동작하는 방법.
- 연속학습의 generative replay(Shin et al.)와 DreamBooth prior preservation의 로봇판이지만, 별도 생성기 없이 **VLA 자신을 궤적 생성기**로 쓰는 self-distillation.
- Pinto & Gupta류 로봇 self-supervision(보상/동역학 학습)과는 다른 의미의 "self-supervision".

## 9. 강점

1. **실용적 설정**: 사전학습 데이터·과제 목록이 없는 다운스트림 사용자 관점. 14분 텔레옵 + 자기 롤아웃만으로 다과제 정책 확보.
2. **실패 롤아웃도 유용**하다는 반직관적 결과를 실물과 시뮬 모두에서 보임(place 0 → 55%).
3. held-out push 실험으로 "전문가 데이터로 모든 과제를 덮으면 오히려 사전학습 prior를 덮어쓴다"는 흥미로운 관찰을 제시.
4. RoboTwin에 사전학습-후학습 분리 프로토콜을 새로 구성해 재현 가능한 평가 제공.

## 10. 약점 및 한계

1. **규모가 작다**: 실물 과제당 10–20 trial, 단일 로봇·단일 백본(π0.5). 통계적 불확실성 보고 없음.
2. **수동 개입**: self-demo 프롬프트 선정과 위험 행동 필터링을 사람이 한다. 규모 확장 시 병목.
3. 새 물체 일반화(T_NO)는 RoboTwin에서 Multi-Task ES(57.0)보다 오히려 낮다(44.5).
4. 실패 행동도 함께 증류된다: 바구니 가장자리에 그리퍼 걸기, 놓은 뒤 다시 집기 반복 등.
5. RoboTwin 결과는 표준 RoboTwin 2.0 리더보드 프로토콜이 아닌 저자 구성 벤치마크.

## 11. 재현 및 확장 아이디어

- self-demo를 성공 여부로 가중하거나 VLM 판정으로 자동 필터링해 수동 개입 제거.
- 여러 라운드 반복하는 data flywheel(개선된 정책으로 다시 self-demo 수집).
- RL 기반 last-mile 자기개선(π0.7류, latent-space RL)과 결합.
- GR00T, OpenVLA-OFT 등 다른 백본으로 일반성 검증.

## 12. 총평

"기본 VLA가 실패하더라도 그 롤아웃은 쓸모 있는 prior를 담고 있다"는 단순한 관찰을 VLA 후학습의 망각 문제 해결책으로 연결한 실용적인 연구다. 실물과 시뮬 모두에서 지시 따르기와 사전학습 행동을 지켜내면서 새 과제 샘플 효율까지 높였다. 다만 실험 규모가 작고 self-demo 큐레이션이 수작업이어서, 대규모 자동화된 파이프라인으로 확장될 수 있는지는 후속 연구의 몫이다.

**한 문장 요약**: 동결 π0.5의 온라인 자기 롤아웃을 전문가 데이터와 1:1로 섞어 전체 파인튜닝하면, 사전학습 데이터 없이도 새 로봇에서 파지 능력을 얻으면서 지시 따르기와 기존 행동을 유지할 수 있다.

<!-- VERIFIED: pdf -->
