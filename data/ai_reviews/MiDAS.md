# MiDAS: Minimal-Data Adaptation Strategy

**논문**: Adaptation of Generalist Robot Policies with Minimal Data (arXiv:2608.11363)
**저자**: Shreyas Kowshik\*, Sreyas Venkataraman\*, Leo Wang, Niharika Pant, Max Simchowitz†, Aviral Kumar† (Carnegie Mellon University, \*동등 기여, †동등 지도)
**공개일**: 2026-08-11
**웹사이트**: minimal-data-adaptation.github.io

---

## 1. 한 줄 요약

사전학습된 generalist VLA(pi0.5)를 **단 1개의 시연**과 자율적 온라인 상호작용만으로 새로운 태스크에 적응시키는 offline-to-online RL 레시피. LIBERO-Long에서 91.2%, RoboCasa-365에서 89.3%를 달성하며, 저자들은 이를 "단일 시연으로부터의 신뢰할 만한 로봇 정책 적응을 보인 최초의 사례"로 주장한다.

## 2. 문제 정의: Minimal-Data Adaptation (MDA)

로봇 학습의 궁극적 목표는 태스크별 인간 데이터 수집을 넘어 **자율적 상호작용을 통한 개선**이다. 그러나 완전 자율 학습은 현재 정책으로는 어렵다. 희소 보상(sparse reward)과 약한 zero-shot 탐색 때문에 로봇이 처음부터 성공 행동을 발견할 확률이 낮기 때문이다.

저자들은 그 중간 지점인 **MDA 체제**를 정의한다: 사전학습 정책이 K개(가능한 한 K=1)의 시연으로 시작해 이후 자율 온라인 상호작용으로 태스크를 학습하는 설정이다. 이는 완전 자율 개선의 가장 가까운 tractable proxy이며, "최소한의 인간 지도가 자율 학습을 부트스트랩할 수 있는가"를 묻는다.

핵심 관찰: 사전학습된 pi0.5의 **zero-shot 성공률은 평가한 모든 태스크에서 정확히 0.0%**다. 즉 온라인 RL을 그대로 붙이면 보상 신호가 전혀 없어 학습이 시작조차 되지 않는다. 시연 1개는 이 "탐색 부트스트랩" 문제를 풀기 위한 최소한의 앵커다.

## 3. 방법: 2단계 레시피

**Stage I — 최소 시연 Behavior Cloning.**
pi0.5를 K개 시연에 대해 flow matching 목적함수로 파인튜닝하여 pi_base^K를 얻는다. VLM 백본은 LoRA로, action head는 전체를 학습한다. OpenPI 공식 코드베이스로 16,000 gradient step. 결과 정책은 "태스크를 대략적으로 시도"하지만 신뢰성 있게 완수하지는 못한다(Fig. 2: Both Moka Pots 태스크에서 올바른 물체에 접근하나 두 번째 pot의 방향 변화에서 파지 실패).

**Stage II — Value-based Residual RL.**
pi_base^K를 **동결**하고, 그 표현 위에 경량 residual actor-critic을 희소 보상 온라인 RL로 학습한다. 학습 파라미터 수, 메모리, 연산량이 크게 줄어든다.

Residual 파라미터화:
```
pi_res_theta(·|s_t, a_base) = tanh(N(mu_theta(s_t, a_base), sigma^2_theta(s_t, a_base)))
```
중요한 설계 선택은 이것이 **가산적(additive)이 아니라는 점**이다. 일반적 residual 형태 `a_exec = a_base + Delta_theta`는 실행 행동을 base 제안의 고정된 근방으로 제한한다. MiDAS는 실행 행동 `a_exec`를 직접 예측하므로, 상태에 따라 이동량이 이질적일 수 있고 base 정책의 support 밖으로도 나갈 수 있다. 최소 시연 체제에서는 base 정책의 support에 최적 행동이 있다는 보장이 없기 때문에 이 자유도가 필수적이다.

정책 개선은 **PA-RL** 구조를 따른다: 행동 후보를 샘플링하고, `a* <- a + eta * grad_a Q_psi(s, a)`로 critic에 대해 gradient ascent를 수행한 뒤, 최적화된 청크를 residual 정책에 distill한다. 이 policy-agnostic 구조 덕분에 flow 정책이든 discrete-action 정책이든 동일하게 적용 가능하다.

## 4. 최소 시연이 야기하는 문제와 대응

시연이 극소수일 때 세 가지 문제가 발생하며, 각각에 대응하는 설계가 있다.

1. **Critic/actor 초기화용 전문가 궤적 부족** → **Offline warmup**. 시연 데이터와 base 정책의 자율 rollout(성공 여부 무관)로 B_warm 버퍼를 만들고, residual actor가 frozen base의 제안 a_base를 재현하도록 학습(L_warm)하면서 critic을 TD로 보정한다. 무작위 초기화된 residual actor는 base와 무관한 행동을 만들어 초기 탐색을 불안정하게 만든다.
2. **자율 rollout의 희소한 성공(2-5%)** → **Success buffer 오버샘플링**. critic/actor 업데이트 배치를 `(1 - rho_succ) * B_warm + rho_succ * B_succ`로 구성(예: 균일 샘플 대비 성공 전이 13개 추가 샘플링)하여 보상이 관측된 상태 쪽으로 PA-RL의 elite selection과 gradient ascent를 유도한다.
3. **Base support에 최적 행동이 없을 수 있음** → 위의 비가산적 residual 파라미터화.

## 5. 주요 결과 (Table 1, Table 5)

**LIBERO-Long (10개 장기 태스크, 1 demo, 3 seeds, 태스크당 50 rollout)**

| 방법 | 평균 성공률 (%) |
|---|---|
| Zero-shot (사전학습 pi0.5) | 0.0 |
| BC | 33.5 ± 1.8 |
| DSRL (Diffusion-Steering RL) | 33.5 ± 0.8 |
| Filtered BC | 39.0 ± 5.9 |
| DICE-RL | 89.3 ± 1.8 |
| **MiDAS** | **91.2 ± 4.2** |

태스크별 MiDAS: Alph.Soup+Cr.Cheese 99.3, Alph.Soup+Tom.Sauce 82.7, Black Bowl→Bottom Drawer 98.7, Book→Caddy 98.7, Both Moka Pots→Stove 76.0, Cr.Cheese+Butter→Basket 96.0, Moka Pot→Stove 95.3, White Mug+Choc.Pudding 96.7, White Mug+Plates 79.3, Yellow-White Mug→Microwave 89.3.

**RoboCasa-365 (모바일 매니퓰레이터, 1 demo)**

| 방법 | 평균 성공률 (%) |
|---|---|
| Zero-shot | 0.0 |
| BC | 22.2 ± 3.4 |
| DSRL | 19.1 ± 3.2 |
| Filtered BC | 34.0 ± 10.6 |
| **MiDAS** | **89.3 ± 8.7** |

태스크별: Banana Fridge Drawer→Shelf 93.3, Hot Dog Counter→Cabinet 100.0, Mug→Coffee Machine+Start 74.7. 네 번째 태스크 Cup+Bowl→Dishwasher+Close(1100 스텝 지평)는 MiDAS 포함 **모든 방법이 0.0%**이며 평균에서 제외되었다 — 저자들이 명시적으로 인정한 한계다.

DICE-RL은 LIBERO에서만 비교되며 MiDAS와 경쟁적이다(10개 중 5개에서 표본 효율이 더 좋고, 1개에서 열세, 4개에서 동등). 저자들은 DICE-RL이 frozen base 위의 경량 residual actor, 오프라인 데이터 혼합, value 기반 개선 등 MiDAS가 중요하다고 본 요소를 독립적으로 포함하고 있다는 점을 강조한다.

## 6. Stage I 스케일링 (Table 16)

BC만 수행했을 때 LIBERO-Long 평균: K=1 → 33.5%, K=3 → 56.3%, K=5 → 78.1%. 즉 MiDAS는 **시연 1개(33.5%)를 5개(78.1%)보다 훨씬 높은 91.2%로** 자율 상호작용만으로 끌어올린다. 이것이 논문의 핵심 경제성 주장이다.

## 7. 일반화 분석 (LIBERO-PRO 기반)

LIBERO-PRO의 섭동 프로토콜 중 4개 축(task 섭동 제외)을 사용하고, 위치 섭동은 저·중·고 3단계(5cm/0.1rad, 10cm/0.15rad, 20cm/0.25rad)로 자체 정의했다. 섭동 평가는 태스크당 50 rollout, 3 seed.

**(Q1) 관측 수준 강건성은 동결된 VLM 백본에서 상속된다.**
- 언어 패러프레이즈: pi1_base 34.7% → pi1_RL **90.2%** (ID 33.5 → 91.2). 사실상 손실 없음.
- 시각(색/텍스처/크기): pi1_base 31.3% → pi1_RL **82.0%** (ID 38.9 → 88.7).

residual 모듈은 지각·언어 용량을 거의 추가하지 않으므로, 이 강건성은 사전학습 표현에 이미 존재하던 불변성으로 설명된다. 즉 온라인 RL은 관측 수준 일반화를 희생하지 않고 제어를 개선한다.

**(Q2) 상태 수준 일반화는 "새로운 행동이 필요한가"에 달려 있다.**
- Shape change(같은 범주의 기하학적으로 다른 변형): pi1_base 6.3% → pi1_RL **29.0%** (ID 25.8 → 92.5). 파지 affordance가 전이되는 경우 개선.
- Object swap(두 물체의 초기 위치 교환): pi1_base 0.0% → pi1_RL **0.0%** (ID는 38.9 → 88.7로 강함). 완전 실패.
- Object change(범주가 다른 물체로 교체, Book→Caddy): pi1_base 0.0% → pi1_RL 2.0%.

결론은 정직하다. 상태 공간의 다른 지점에서 **같은 행동**을 실행하면 되는 경우엔 온라인 RL이 도움이 되지만, 다른 affordance/조작 전략이 필요한 경우엔 시연 1개와 그 주변 상호작용만으로는 추론 불가능하다.

**(Q3) 커리큘럼이 격차를 메운다.** 온라인 RL 중 reset 분포의 폭을 점진적으로 넓히면(Appendix F), 추가 시연 없이 50개 시연으로 학습한 정책과 유사한 위치 일반화 강건성을 회복한다(Fig. 8). 시연의 폭을 늘려도(Appendix G) 유사한 효과가 있다.

## 8. 실제 로봇 검증 (Table 3)

Bimanual **YAM** 플랫폼, pi0.5를 base로 동일한 MiDAS 프로토콜(K=1), 태스크당 15 rollout(물체 위치·방향을 바꾼 ID/OOD 혼합).

| 태스크 | BC (1 demo) | MiDAS |
|---|---|---|
| 초록 블록→우측 용기, 파랑 블록→좌측 용기 | 40.0 | **67.0** |
| 칼과 도넛을 접시에 올리기 | 27.0 | **80.0** |
| 평균 | 33.5 | **73.5** |

T1의 40% 초기 성공률은 pi0.5 사전학습 분포와의 중첩 때문으로 추정된다. 더 어려운 T2는 27% → 80%로, **5-6시간의 자율 상호작용** 후 달성되었다. 온라인 RL은 base 정책 rollout에서 관찰되지 않던 질적으로 새로운 교정 행동을 만들어냈다고 보고된다.

## 9. 왜 baseline들이 실패하는가 (Support 분석)

DSRL은 pi_base^K의 행동 분포 **내부에서** 초기 노이즈를 조작해 조향하는 방법이다. 저자들은 LIBERO-Long의 결정적 상태에서 여러 행동 샘플 클라우드 간 최소 거리를 측정(Table 2)하여, 최적 행동이 base 정책의 **유효 support 밖**에 있음을 보인다. 따라서 support 내부에서의 sharpening이나 steering으로는 도달할 수 없고, **행동 분포를 확장할 수 있는** 온라인 적응이 필요하다 — MiDAS의 비가산적 residual이 정확히 그것을 한다. 이 분석은 DSRL이 BC와 동일한 33.5%에 머무는 이유를 설명한다.

추가 대조군으로 (a) 사전학습 ResNet 특징 위의 flow-matching 정책과 (b) privileged state 입력 flow-matching 정책을 평가했는데, 셋 다 저조했다(ResNet flow 정책은 LIBERO-Long 10개 중 3개에서만 0이 아닌 성공). 즉 VLA 특징 자체가 가치 학습에 중요하다.

## 10. Ablation (Appendix E)

Both Moka Pots→Stove 태스크에서 세 가지 구현 선택을 절제했다.
1. **Residual actor 입력에서 base 행동 제거**: 기본 레시피와 거의 동등. 저자 해석 — action head의 주 역할은 태스크 관련 warmup 경험을 생성하는 것이며, 그 경험이 확보된 후에는 사전학습 VLA 특징 위의 온라인 가치 학습만으로 최종 성능의 상당 부분을 회복할 수 있다.
2. **Success buffer 비활성화**: 희소 성공 전이의 오버샘플링 필요성 검증.
3. **지속적 success-only BC 페널티 추가**: value 기반 개선을 제약하는지 검증.

1번 결과는 흥미로운 자기 비판이다 — 논문이 강조한 "a_base 조건화"가 이 태스크에서는 결정적이지 않았다.

## 11. 강점과 한계

**강점**
- Zero-shot 0%에서 시작해 시연 1개 + 자율 상호작용으로 90%대에 도달하는, 명확하고 재현 가능한 문제 설정과 결과.
- 시뮬레이션 두 벤치마크 + 실제 bimanual 하드웨어까지 일관되게 검증.
- 일반화 실패 사례(object swap 0%, LoadDishwasher 0%)를 숨기지 않고 보고.
- Base 정책 동결로 학습 파라미터·메모리·연산이 크게 절감되어 실제 하드웨어에서 6시간 온라인 학습이 가능.

**한계**
- **희소 보상 성공 판정기(reward function)가 필요**하다. 시뮬레이터는 이를 제공하지만, 진정한 자율 학습에서 이 신호를 어디서 얻을지는 다루지 않는다. "최소 데이터"라는 주장의 회계에서 이 부분이 빠져 있다.
- 장기 지평 실패: 1100 스텝의 LoadDishwasher는 모든 방법이 0%. 희소 보상 + 장기 지평 조합은 미해결.
- 상태 shift 일반화 실패: object swap에서 pi1_RL도 0.0%. 학습된 것은 여전히 좁은 리셋 분포 주변의 교정 행동이다.
- RoboCasa 평균이 4개 중 3개 태스크만으로 계산됨(저자 명시). 4개 전체 평균은 67.0%가 된다.
- LIBERO 4개 표준 suite 중 Long 하나만 평가 — 표준 libero_avg와 직접 비교할 수 없다.
- DICE-RL(89.3)과의 격차가 1.9%p이고 3 seed 표준편차가 4.2인 점을 고려하면, LIBERO에서의 우위는 통계적으로 강하지 않다. 저자들도 DICE-RL이 5/10 태스크에서 표본 효율이 더 좋다고 인정한다.
- 실제 로봇 평가는 태스크 2개 × 15 rollout으로 표본이 작다.

## 12. VLA-Tracker에서의 위치

MiDAS는 새로운 VLA **아키텍처**가 아니라 기존 VLA(pi0.5)를 위한 **적응 레시피**다. 따라서 리더보드의 libero_avg(91.2)는 4개 suite 평균이 아니라 LIBERO-Long 단일 suite 수치이며, 표준 LIBERO 평균 항목들과 직접 비교해서는 안 된다. `action_head_category`는 base 정책 pi0.5와 Stage I 목적함수를 따라 `flow_matching`으로 분류한다(Stage II residual actor 자체는 tanh-Gaussian 회귀 헤드).

이 논문의 진짜 가치는 리더보드 순위가 아니라, **"사전학습된 generalist 정책이 있으면 시연 1개로 온라인 RL을 부트스트랩할 수 있다"**는 체제 자체를 확립한 데 있다. 데이터 수집 비용이 VLA 배포의 주된 병목인 상황에서, DSRL/PA-RL/DICE-RL 계열의 residual RL 흐름을 최소 데이터 극단까지 밀어붙인 참조점으로 읽는 것이 적절하다.

<!-- VERIFIED: pdf -->
