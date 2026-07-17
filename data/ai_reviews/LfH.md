# LfH: Learning More from Less — Reinforcement Learning from Hindsight

## 1. 개요 (Overview)
LfH(Learning from Hindsight)는 VLA(Vision-Language-Action) 모델의 강화학습(RL) 사후학습(post-training)에서 샘플 효율을 끌어올리기 위한 방법이다. 로봇 롤아웃은 수집 비용이 크고 느리며, 조작(manipulation) 과제는 대개 희소 보상(sparse reward)만 제공한다. 약한 초기 정책은 학습 초기에 거의 모든 롤아웃에 실패하므로 학습 신호가 없다. LfH의 핵심 관찰은 "한 지시(instruction)에 대한 실패는 다른 지시에 대한 성공"이라는 점이며, 사후재라벨링(hindsight relabeling)을 언어 수준에서 GRPO에 도입한다. MIT, MIT-IBM, Stanford, UC San Diego 공동 연구(arXiv 2607.09042, 2026-07-10)이다.

## 2. 문제 정의 (Problem Statement)
GRPO는 그룹 내 상대적 이점(advantage)으로 정책을 갱신한다. 그런데 모든 궤적이 동일한 보상(특히 전부 0)을 받는 그룹은 이점 분산이 0이라 폐기된다. 약한 초기 정책에서는 대부분의 그룹이 all-zero가 되어 학습이 거의 진행되지 않는다. 즉 병목은 "폐기되는 실패 그룹으로부터 신호를 회복하는 것"이다.

## 3. 핵심 아이디어 (Key Idea)
실패 궤적을 그것이 실제로 달성한 과제로 재라벨링한다. "전자레인지를 닫아라"는 명령에 실패하고 컵을 집었다면, 그 궤적을 "컵을 집어라(g')"의 성공 사례로 전환한다. VLA는 언어에 조건화되어 지시 간 일반화가 가능하므로, 언어로 재라벨링된 롤아웃이 원래 RL이 버렸을 데이터에서 유용한 지도(supervision)를 제공한다.

## 4. 방법론 (Methodology)
LfH는 저신호 그룹(평균 보상 < eta)에만 활성화된다. 두 단계로 사후 라벨 그룹을 만든다.
- 지시 재라벨링: 실패 궤적 중 앵커 i*를 균등 샘플링하고, VLM 재라벨러 M_psi가 RGB 관측·행동으로부터 사후 지시 g'를 생성한다(식 4). 먼저 앵커가 유의미한 행동을 담는지 분류해 "무의미(uninteresting)" 궤적은 버린다.
- 보상 재라벨링: 공유 지시 g' 아래 그룹의 모든 궤적을 재채점하여 R̃_i ∈ {0, 0.5, 1}을 부여(식 5). 공유 지시가 필수인 이유는 GRPO 이점이 그룹 상대적이기 때문이다.

## 5. GRPO 통합 (Integration with GRPO)
사후 그룹 G̃에 대해 그룹 정규화 이점 Ã_i를 계산(식 6)하고, 궤적은 g로 샘플됐으나 g'에서 최적화하므로 hindsight policy gradient 방식의 중요도 보정(식 7)을 적용한다: 비율 r̃ = pi_theta(a|o,g') / pi_theta_old(a|o,g). 최종 목적은 L_LfH = L_GRPO + lambda * L_H-GRPO(식 9)로, 명령 그룹은 P_g 정렬을 유지하고 사후 항은 저보상 그룹에서 신호를 회복한다.

## 6. 실험 설정 (Experimental Setup)
기본 초기화는 RLinf-Pi05-LIBERO-SFT 체크포인트(4개 LIBERO 스위트에 few-shot SFT). VLM 재라벨러는 Qwen3-VL-235B-A22B-Thinking-FP8이며, 궤적은 전역 카메라 비디오 프레임 시퀀스로 표현된다. 구현은 RLinf 프레임워크 기반이다.

## 7. 주요 결과 (Main Results)
OOD LIBERO-PRO(task perturbation) 과제에서 LfH는 GRPO의 최종 성능을 약 5스텝만에 달성(GRPO는 약 30스텝 소요), 즉 약 5배의 샘플 효율 향상을 보였다. 또한 dense progress reward 기반 GRPO+RoboMETER를 능가한다. GRPO/RoboMETER는 궤적 그룹의 20-40%만 유지하는 반면 LfH는 약 70-80%를 유지한다. 예시 과제(전자레인지 닫기)에서 초기 정책은 0% 성공이나 LfH는 약 60% 성공에 도달한다.

## 8. 일반성 및 전이 (Generality)
동일 프로토콜로 GR00T(200스텝), OpenVLA-OFT(60스텝)를 LIBERO-PRO의 Spatial/Goal/Object 스위트에서 미세조정했을 때도 LfH가 일관되게 샘플 효율을 개선한다. 즉 이점이 pi0.5 백본을 넘어 전이된다.

## 9. 실제 로봇 실험 (Real-Robot)
실제 Franka FR3에서 조작 물체, 서랍 2개, 2단 랙 환경으로 검증했다. pi0.5를 10개 언어조건 과제(과제당 20개 SpaceMouse 시연)로 SFT한 뒤, 초기 성공률 0%인 held-out 과제("초록 컨테이너를 그릇에 넣기")에서 RL 미세조정한다. 128 롤아웃에서 LfH는 GRPO의 약 2배 성공률, 160 롤아웃에서 LfH 56% vs GRPO 22%로 실제 환경에서도 샘플 효율 우위를 보인다.

## 10. 분석 및 절제 실험 (Analysis & Ablations)
Rephrase-only(원 지시의 패러프레이즈 + 원 보상)와 Reward-only(원 지시 유지 + 보상만 재라벨)를 비교했을 때, 어느 한쪽만으로는 성능이 크게 개선되지 않았다. LfH의 이득은 지시와 보상을 함께 재라벨링하는 데서 온다. 흥미롭게도 목표 과제와 의미적으로 무관한 사후 지시조차 학습을 돕는데, 이는 대비적(contrastive) grounding 신호로 정책이 목표 지시를 무관 행동과 구분하도록 돕는다는 해석이 제시된다.

## 11. 한계 (Limitations)
LfH는 수집된 데이터에 존재하는 행동에만 의존한다. 정책이 유의미한 행동을 전혀 생성하지 못하면 재라벨링할 대상이 없다. 또한 VLM 재라벨러의 지시·보상 판정 품질에 의존하며(약 20%의 그룹은 무의미로 폐기됨), 대형 VLM(235B) 추론 비용이 수반된다.

## 12. 총평 (Assessment)
LfH는 "롤아웃 한 개당 더 많은 유용 비트"라는 관점에서, 실패를 부정 예시가 아닌 인접 과제에 대한 재사용 가능한 경험으로 전환하는 개념적으로 깔끔한 접근이다. GRPO 상단에 최소 침습적으로 얹히고 백본(pi0.5/GR00T/OpenVLA-OFT)과 실제 로봇으로 전이되는 점이 강점이다. 다만 절대 벤치마크 점수가 아닌 정규화 gain 위주로 보고되어 표준 리더보드와 직접 비교는 제한적이며, 대형 VLM 재라벨링 비용이 실용적 고려사항이다. VLA RL 사후학습의 샘플 효율 문제에 대한 설득력 있는 기여로 평가된다.

<!-- VERIFIED: pdf -->
