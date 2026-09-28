# Prism-GRPO: Faster VLA Policy Optimization via Splitting Same-outcome Groups

> **한 줄 요약**: 이진 성공 보상 GRPO에서 전부 성공/전부 실패인 그룹은 advantage가 0이라 버려지는데, 여기에 궤적 수준 실행 품질 점수 q∈[0,1]를 λ<1 가중으로 더해(success + λq) "모든 성공이 모든 실패보다 높다"는 순서를 유지한 채 동일 결과 그룹을 품질 스펙트럼으로 쪼개 학습 신호를 살려내는 방법. OpenVLA-OFT를 RoboTwin 2.0 네 과제에서 RL 후학습해, Binary GRPO의 목표 성공률에 최대 56% 적은 롤아웃으로 도달하고 reward hacking(shove-cheat)도 억제한다.

- **arXiv**: 2608.17423v1 (2026-08-18, cs.RO)
- **소속**: Purdue University, AWS AI
- **코드**: 공개 정보 없음

---

## 1. 배경 및 동기

GRPO는 critic 없이 그룹 내 상대 보상으로 advantage를 만들기 때문에 VLA RL(SimpleVLA-RL 등)에 널리 쓰인다. 그러나 이진 성공 보상 하에서는

- 그룹의 모든 롤아웃이 성공하거나 모두 실패하면 보상이 전부 같고, advantage=0, gradient=0.
- Dynamic sampling이 이런 "degenerate group"을 버리고 재샘플링한다.
- 학습 초기처럼 대부분 실패할 때 특히 흔해서, 비싼 로봇 롤아웃 예산이 대량으로 낭비된다.

LLM 추론 쪽에서는 RL-ZVP(엔트로피), CAST(self-teacher), LLM-as-a-Verifier(외부 검증기 logit) 등으로 zero-variance 그룹을 살리려는 시도가 있었다. 로봇에서는 **실행 자체가 자연스러운 신호**를 준다: 같은 결과라도 의도치 않은 접촉, 물체 교란, 움직임의 매끄러움이 다르다. 이는 "얼마나 진행했나(task progress)"가 아니라 "어떻게 수행했나"를 기술하므로 과제 간 재사용이 쉽다.

## 2. 문제 정의

- 과제 T가 장면 분포 x∼Scene(T)를 유도하고, 정책 π_θ가 궤적 τ를 생성.
- success(τ)∈{0,1}, 품질 q(τ)∈[0,1] (클수록 좋음).
- 목표: Binary GRPO보다 **같은 목표 성공률에 더 적은 롤아웃으로** 도달하고, 그 시점에서 품질도 같거나 높을 것.

## 3. 방법: 품질 증강 보상

R_combined(τ) = success(τ) + λ·q(τ), 0 < λ < 1 (기본 λ=0.2).

- 실패는 [0, λ], 성공은 [1, 1+λ] 구간 → λ<1이면 두 밴드가 분리되어 **성공 우위(success dominance)** 보존.
- 품질은 같은 결과 클래스 내부 순서만 바꾼다.
- **RLOO(leave-one-out) advantage** 사용: A_i = R_i − (1/(G−1))Σ_{j≠i} R_j. 표준 GRPO의 std 정규화는 같은 결과 그룹에서 λ의 스케일을 상쇄해 버리므로, RLOO로 원래 품질 차이 크기를 보존한다.
- 같은 결과 그룹에서는 결과 항이 상쇄되어 A_i = λ(q_i − mean_{j≠i} q_j).

📌 [Figure 2 삽입] — 동일 결과 그룹이 품질 스펙트럼으로 분할되는 그림

## 4. 이론적 결과

- **Theorem 1**: 장면 성공률 p, 그룹 크기 G에서 정보 있는 그룹당 기대 롤아웃 수 C_combined(p) ≤ C_binary(p). 연속 품질 점수면 동점 확률이 0이라 C_combined = G. 예: p=0.1 또는 0.9에서 Binary GRPO는 G=8일 때 약 1.76배, G=4일 때 2.91배 롤아웃 필요.
- **Proposition 1**: 단일 softmax 결정에서 성공·품질 값의 정책 가중 상관 ρ(s) ≥ (κ_π−1)/(κ_π+1)이면 두 gradient가 정렬(내적 ≥ 0). κ_π는 정책 분포의 조건수. 경계가 tight하며 bounded conditioning이 필요조건임도 보인다.
- **Proposition 2**: 정렬 조건 하에 g_success + λg_quality는 성공·품질 목표 모두의 1차 상승 방향.
- 해석: 포화된 장면에서 품질만으로 학습되는 "구조된 그룹"도 정렬이 성립하면 과제 수준 성공 목표를 전진시킨다. 단, 모집단 수준 정렬은 가정으로 남기고 Appendix H에서 상관 진단으로 경험적 확인.

## 5. 품질 신호

- **Prism-Peak (기본, GT-Max Force)**: 비목표 물체에 가해진 최대 충격량.
- **Prism-Count**: 비목표 접촉 횟수.
- **Prism-VLM-Contact**: Qwen3-VL-235B zero-shot judge가 8프레임을 보고 충돌 여부 판정 (256 궤적에서 정확도 78.9%, contact F1 69.0%, precision 90.9%, recall 55.6%).
- **Prism-Flips / Prism-MeanFlips / Prism-Jerk**: 관절 방향 반전 수, 청크 평균 반전, 최대 급격 액션 변화.
- 임계값 T는 SFT 정책 분포로 보정하며, T를 3배 이상 바꿔도 비슷한 성능.

## 6. 실험 설정

- **벤치마크**: RoboTwin 2.0 (도메인 랜덤화), 과제 4개 — Lift Pot, Move Can Pot, Handover Block, Beat Block Hammer.
- **정책**: OpenVLA-OFT, 이산 액션 헤드(DoF당 256 bin, 스텝당 14 토큰, 청크 25 스텝). 모든 방법이 SimpleVLA-RL의 공개 SFT 체크포인트에서 출발.
- **학습**: G=8, 스텝당 512 롤아웃, lr 5×10⁻⁶, clip (0.2, 0.28), KL/엔트로피 보너스 없음, 온도 1.6. 고정 증분 대신 adaptive gap-fill 재샘플링. 설정당 5 seed.
- **자원**: 과제당 8×H100-80GB, RL 1회 약 12–16시간(롤아웃 생성이 8–12시간).
- **베이스라인**: Binary GRPO(SimpleVLA-RL 설정), Binary RLOO, Random quality, RL-ZVP(액션 청크 스텝 단위 엔트로피로 이식).
- **지표**: 성공률, calibrated 성공률(비목표 접촉 수로 할인), Max-Force Quality, Sum-Impulse Quality.

## 7. 주요 결과

- 동일 롤아웃 예산에서 Binary GRPO 대비 **22–56% 롤아웃 절감**, calibrated 성공률도 같은 경향. 최대 이득은 Lift Pot: Binary GRPO는 초기 배치의 70%를 버리고 이후 20%에서 안정되는 반면, Prism-GRPO는 초기 폐기율 약 0%, 이후 약 14%.
- 목표 성공률에 도달한 시점의 실행 품질도 더 높고, 학습에 쓰지 않은 Sum-Impulse Quality도 개선.
- 절대 성공률 곡선은 그림(Fig. 3)으로만 제시된다. 참고로 Appendix E에 따르면 Binary GRPO 재현에서 Lift Pot seed별 최고 성공률 평균은 63.6%로, SimpleVLA-RL이 보고한 64.1%와 일치한다.

**λ 민감도 (Table 1, CS = Binary GRPO calibrated 성공률 도달 시 롤아웃 절감 %, Q = 그 시점 품질 이득 point)**

| 과제 | 지표 | λ=0.2 | 0.5 | 0.9 | 1 | 2 |
|---|---|---|---|---|---|---|
| Lift Pot | CS | 56 | 49 | 45 | 25 | 53 |
| Lift Pot | Q | 11 | 14 | 15 | 14 | 26 |
| Move Can Pot | CS | 41 | 48 | 41 | 3 | × |
| Move Can Pot | Q | 7 | 10 | 13 | 16 | × |

Random quality는 Lift Pot에서 모든 λ에서 목표 미도달, Move Can Pot에서 λ=0.2/0.5일 때만 CS 24/33, Q 1/0. λ≥1이면 성공 우위가 깨져 불안정(λ=2의 Lift Pot 결과는 이상치로 해석).

## 8. Reward hacking과 실기

**Shove-cheat**: Move Can Pot의 성공 판정기는 최종 기하만 보므로, 캔을 들지 않고 팔로 냄비를 캔 쪽으로 밀어도 성공 처리된다. 시뮬레이션에서 RL-ZVP는 seed별 0.6–20.0%, Binary GRPO 0.7–7.0%, Prism-GRPO는 0.6–1.3%로 억제.

**실기 (Table 2, Piper, 25회 zero-shot sim-to-real, 시뮬 성공률 약 65% 체크포인트 중 shove-cheat 비율이 가장 높은 것 선택)**

| | Binary GRPO | RL-ZVP | Prism-GRPO |
|---|---|---|---|
| Clean Success | 4/25 (16%) | 2/25 (8%) | **6/25 (24%)** |
| Shove-Cheat | 1/25 (4%) | 5/25 (20%) | **0/25 (0%)** |

저자들 스스로 clean success 차이(2회)는 결론적이지 않다고 인정하고, 핵심은 shortcut 미발생이라고 해석한다.

## 9. 어블레이션

- **품질 신호 종류 (Fig. 6)**: 충돌 계열 38–56% 절감, 매끄러움 계열 25–44% 절감. VLM 판정은 가장 약하지만 여전히 이득.
- **품질 항 적용 범위**: 모든 그룹에 적용해야 함 — 동일 결과 그룹이나 전부 실패 그룹에만 적용하면 성공률이 거의 절반.
- **그룹 크기**: G=8에서 이득 최대, G=16에서 감소, G=32에서 거의 동일 (큰 G에선 degenerate 그룹 자체가 드묾).
- **λ decay**: 일관된 개선 없음 → 고정 λ.
- **advantage 추정기**: RLOO가 표준 group-normalized GRPO보다 안정적.
- **오버헤드**: 시뮬레이터 품질 +0.04%, VLM 품질 +4.8% wall-clock.

## 10. 강점

- 아이디어가 극도로 단순(보상 한 줄 + RLOO)하면서, 성공 우위 보존·샘플 효율 비악화에 대한 증명을 붙였다.
- 5 seed, 다양한 품질 신호, λ/G/추정기/적용 범위 어블레이션까지 실험 설계가 촘촘하다.
- "품질 신호가 reward hacking을 억제한다"는 부수 효과를 시뮬과 실기 양쪽에서 보여 준 점이 흥미롭다.
- 롤아웃 수를 비용 축으로 두고 모든 폐기·잉여 롤아웃까지 계산에 넣어 공정성을 확보했다.

## 11. 약점 및 한계

- **절대 성공률 표가 없다.** 주요 결과가 학습 곡선과 "롤아웃 절감 %"로만 제시되어, 다른 RoboTwin 2.0 모델과 직접 비교하기 어렵다.
- 과제 4개, 단일 정책(OpenVLA-OFT 이산 헤드)에서만 검증. flow-matching 정책(π0 계열)으로의 일반화는 미확인.
- 기본 품질 신호가 시뮬레이터 접촉 로그(특권 정보)에 의존. VLM 대체 신호는 recall 55.6%로 약하다.
- 성공–품질 gradient 정렬은 증명이 아니라 가정이며, 품질이 성공과 충돌하는 과제(예: 의도적 접촉이 필요한 과제)에서는 성립하지 않을 수 있다.
- 실기 평가는 과제 1개, 25회로 통계적 힘이 약하다.
- 코드 미공개.

## 12. 총평

Prism-GRPO는 새 VLA 아키텍처가 아니라 **VLA RL 후학습 레시피**다. OpenVLA-OFT를 실제로 RL로 학습해 정책 가중치를 바꾸고 정량 결과를 제시하므로 자체 학습 정책을 가진 논문으로 볼 수 있다. 기여의 본질은 "이진 보상 GRPO가 버리는 데이터를, 성공 순서를 깨지 않는 범위에서 실행 품질로 살려낸다"는 것이며, 롤아웃 비용이 병목인 로봇 RL에서 실용적 가치가 크다. 다만 벤치마크 리더보드 관점에서는 절대 성공률이 그림에만 있어 트래커에 직접 올릴 수치는 롤아웃 절감률과 실기 결과에 한정된다.

**한 문장 요약**: success + 0.2·quality 한 줄과 RLOO만으로 같은 결과 GRPO 그룹을 되살려 롤아웃을 최대 56% 아끼고 shove-cheat를 0으로 만든다 — 다만 절대 성공률은 곡선 속에만 있다.

<!-- VERIFIED: pdf -->
