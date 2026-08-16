# VANE: Reliable Test-Time Training for VLA Models via Future Visual Representation Prediction

> **한 줄 요약**: QwenPi(Qwen3-VL 기반 flow-matching VLA)의 state-grounding proxy를 frozen V-JEPA 2 미래 latent 예측(WPI)으로 대체해 QwenWPI를 학습하고, context-routed latent prompt bank(MoLP)와 "attention 이벤트에서 shadow 업데이트 제안 → 미래 관측으로 검증 → 원자적 commit/rollback"(AGV-TTT)을 결합하여 SimplerEnv WidowX 71.2%를 달성한 신뢰 가능한 test-time training 프레임워크.

- arXiv: 2608.09448v2 (2026-08-12) · 소속: CUHK-Shenzhen, BUPT, CASIA/UCAS, Li Auto
- 판정: **VLA 산출물 있음** — 정책 자체를 로봇 데이터로 학습(QwenWPI)하고, 배포 시 latent-prompt 가중치를 실제로 갱신한다. frozen 정책 위의 순수 래퍼가 아니다.

---

## 1. 배경 및 동기

VLA backbone을 새 태스크·시각 환경에 적응시키려면 보통 action label이 붙은 시연과 반복 fine-tuning이 필요하고, 결과적으로 태스크마다 따로 튜닝된 정책 뭉치가 생긴다. 반면 실제 배포는 **라벨 없는 시각 관측 스트림**을 공짜로 만들어낸다. Test-time training(TTT)은 이 스트림에서 self-supervised 목적함수로 소수 파라미터만 갱신하는 대안이며, TTT-VLA는 latent prompt만 적응시켜 frozen 정책에 연결하는 방식을 보였다.

문제는 신뢰성이다. 저자들의 통제 실험에서 latent-prompt TTT는 태스크마다 결과가 들쭉날쭉했다: 한 설정을 개선하는 업데이트가 다른 설정을 악화시키고, 검증되지 않은 온라인 업데이트가 배포 정책의 성능을 떨어뜨릴 수 있다. 논문은 TTT가 **선택적으로 제안되고 회귀가 억제될 때** "reliable"하다고 정의하고, 세 가지 설계 난점을 제시한다.

---

## 2. 세 가지 설계 난점

1. **적응 공간(what to adapt)**: 단일 공유 prompt는 이질적인 태스크 보정 방향을 한 파라미터 블록에 뒤섞는다. 반대로 완전 독립 prompt는 태스크 간 재사용 가능한 구조를 버린다.
2. **Proxy 목적함수(what signal)**: 배포용 proxy는 (a) expert action·reward·성공 라벨 없이 관측 가능해야 하고 (b) 정책과 관련된 적응 신호를 줘야 한다. State grounding은 (a)는 만족하지만 저차원·embodiment 특화이며 현재 프레임과 동기화되어, **상호작용 이후 장면이 어떻게 변하는지 설명하지 않고도** 최적화될 수 있다.
3. **폐루프 결합(when/whether to deploy)**: 미래 표현 예측으로 바꾸면 목표값이 지연 도착하고, 더 심각하게는 후보 prompt를 즉시 배포하면 **그 업데이트가 자신을 심판할 미래 관측 자체를 바꾼다**.

---

## 3. MoLP — Context-Routed Mixture of Latent Prompts

공유 latent-prompt bank $P = \{P_i\}_{i=1}^{E}$를 두고, 최종 projected VLM 표현의 마지막 유효 토큰 $c_t$(현재 이미지 + instruction 요약)에서 선형 router가 선택 확률을 낸다.

$$q_t = \mathrm{softmax}(W_r c_t + b_r),\quad S_t = \mathrm{TopK}(q_t, K),\quad p_t = \sum_{i \in S_t} \frac{q_{t,i}}{\sum_{j \in S_t} q_{t,j}} P_i$$

- bank 크기 **8**, **top-2** routing, 텐서 shape **[8, 16, 1024]**, 각 성분은 $\mathcal{N}(0, 0.02^2)$로 독립 초기화.
- **태스크 ID를 외부에서 받지 않는다.** bank는 태스크/embodiment로 분할되지 않아 관련 태스크가 성분을 재사용할 수 있다.
- routing weight는 제어 관측마다 재계산 후, 해당 action chunk를 생성하는 flow-matching 반복 동안 고정.
- 학습 시 router·prompt·정책이 함께 학습되고, **test time에는 router가 얼고 prompt bank만** 최적화된다.

---

## 4. WPI — World-Predictive Interface

Frozen V-JEPA 2 인코더 $f_\omega$가 현재/지연 시각 토큰 $Z_t = f_\omega(o_t)$, $Z_{t+k} = f_\omega(o_{t+k})$를 뽑는다. Latent-Action DiT 쿼리는

$$Q^{(0)}_\tau = [\,p_t \mid E_z(Z_t) \mid E_a(A_\tau, \tau)\,]$$

로 구성되어 시각 latent 토큰과 action 토큰이 shared self-attention으로 정보를 교환한다. 즉 예측이 현재 이미지만이 아니라 **정책의 action context에 조건화**된다. $Z_t$로 초기화된 슬롯이 지연 표현 $\hat{Z}_{t+k}$로 디코딩되고

$$L_{wpi} = \frac{1}{NC}\big\|\hat{Z}_{t+k} - \mathrm{sg}(Z_{t+k})\big\|_F^2$$

로 학습된다. 미래 offset $k = 8$, V-JEPA 2는 계속 frozen. **픽셀이 아니라 latent를 예측**한다는 점이 핵심 — 픽셀 생성 없이 의미 구조를 보존한 미래 목표를 얻는다.

QwenPi의 state-grounding 분기를 WPI로 교체한 변형이 **QwenWPI**이며, VLM과 action 정책 모듈은 그대로 유지된다(proxy interface만 교체 → 효과 분리). 저자들은 QwenWPI가 독립적인 world model이 아니며 action 정책을 대체하지 않는다고 명시한다.

학습 목적함수: $L_{train} = L_{act} + 0.1 L_{wpi} + 0.01 L_{lb}$ (action은 flow matching, $A_1 \in \mathbb{R}^{16\times7}$, $u \sim \mathrm{Beta}(1.5, 1.0)$, $\eta = 0.999$).

---

## 5. AGV-TTT — Attention-Gated & Validation-Driven 업데이트

**(1) 내생적 이벤트 탐지.** live prompt로 $a_t$를 정상 추론한 뒤, 그 forward pass에서 (i) action-to-VLM cross-attention, (ii) action-to-predictive-latent attention을 수집한다. 채널 $c$, denoising step $u$, DiT layer $l$에 대해

$$\Delta^c_t = \frac{1}{|U||L|}\sum_{u,l} \tfrac{1}{2}\big\|a^c_{t,u,l} - a^c_{t-1,u,l}\big\|_1$$

이 최근 robust baseline 대비 이상치일 때 이벤트가 발생한다. **탐지가 $a_t$ 생성 이후에 일어나므로, 제안이 자신을 촉발한 action에 소급 영향을 줄 수 없다.**

**(2) Attention-focused shadow 제안.** 고-attention 시각 토큰 집합 $K_t$와 정규화 가중치 $F_{t,n}$으로 focused loss $L_{focus,t} = \sum_{n \in K_t} F_{t,n} e_{t,n}$를 만들고, **shadow copy**에 한 스텝만 적용해 $P_{cand}$를 얻는다. live prompt와 optimizer state는 즉시 복원되므로 후보는 검증 전에 제어에 영향을 줄 수 없다. 선택 토큰과 가중치는 사이클 내내 detach·고정.

**(3) 미래 검증과 원자적 배포.** 이후 $H_v$개의 유효 WPI 쌍에서 old/candidate를 **동일 입력·동일 focus 가중치·동일 flow-matching 난수**로 평가하고

$$R_{focus} > 0,\quad R_{global} \ge 0,\quad \sum_{j=1}^{H_v} \mathbb{I}[R^{(j)}_{focus} > 0] > \frac{H_v}{2}$$

를 모두 만족할 때만 prompt bank와 optimizer state를 **원자적으로 commit**, 아니면 둘 다 폐기(rollback). 검증 프레임 중 앞의 $H_v - 1$개 action은 항상 old prompt로 생성되어 제안 데이터/검증 증거/배포 사이의 인과 분리를 유지한다.

> ❓ **예상 질문**: attention 이벤트가 "좋은 업데이트 시점"이라는 보장이 있는가?
> **답변**: 없다. 저자 스스로 Fig. 2 시각화는 단일 rollout 예시일 뿐 detector의 precision/recall을 입증하지 않는다고 못 박고, attention은 **언제 시도할지**의 라벨-프리 단서로만 쓰며 **유익한지**는 §4.2.4 검증 ablation으로 따로 검증한다.

---

## 6. 진단 실험 1 — 태스크 의존적 prompt 간섭 (Table 1)

고정 checkpoint의 state-grounded 단일 prompt Offline TTT. No TTT 대비 성공률 변화:

| 적응 subset | Stack | Carrot | Spoon | Eggplant | Avg. |
|---|---|---|---|---|---|
| Task-specific prompts | +2.5 | +3.1 | 0.0 | +2.6 | +2.0 |
| Carrot + Eggplant | +5.2 | +10.4 | +1.1 | 0.0 | **+4.1** |
| Carrot + Spoon + Eggplant | −2.1 | +16.7 | −2.1 | +3.1 | **+4.1** |
| Stack + Eggplant | +4.1 | +11.5 | **−6.2** | +1.0 | +2.6 |
| 4-task 공유 prompt | −0.6 | +2.1 | **−5.7** | +1.0 | **−0.8** |

비적응 정책 64.1% → 태스크별 개별 prompt 66.1% → 4태스크 공유 prompt **63.3%**. 즉 공유 prompt는 비적응보다도 나빠질 수 있다. 이것이 MoLP의 동기이며, MoLP 자체의 효능은 §4.2.3에서 따로 측정된다.

---

## 7. 진단 실험 2 — attention 재분포와 표현 분포 이동 (Table 2)

Stack Blocks rollout에서 36개 DiT layer 전반의 action-to-VLM attention은 gripper–물체 접촉/그립 폐쇄 및 post-grasp 전환 근처에서 뚜렷하게 재분포한다(이벤트 단서의 근거).

RT-1/Bridge 학습 참조 분포(1,000 프레임) 대비 RBF-MMD:

| Test suite | PCA-95 (681-D) | Raw standardized (2560-D) |
|---|---|---|
| WidowX | 0.330 [0.326, 0.338] | 0.348 [0.343, 0.358] |
| Google Robot | 0.262 [0.259, 0.269] | 0.262 [0.260, 0.271] |

WidowX가 두 특징 공간 모두에서 더 멀다. 저자들은 이를 **분포적 맥락**으로만 쓰며 TTT 이득 예측에 사용하지 않는다고 명시한다(500회 episode-level bootstrap CI).

---

## 8. 주 결과 — SimplerEnv WidowX (Table 3)

| Method | Carrot | Eggplant | Spoon | Cube | Overall |
|---|---|---|---|---|---|
| π0 | 36.1 | 81.9 | 45.8 | 26.4 | 47.6 |
| CogACT | 37.5 | 91.7 | 58.3 | 20.8 | 52.1 |
| TTT-VLA (π0.5 + SG-LP + TTT) | 74.5 | 76.0 | 71.0 | 48.0 | 67.4 |
| QwenPi + SG-LP + TTT | 58.0 | 95.6 | 83.9 | 18.2 | 63.9 |
| QwenPi + MoLP + AGV-TTT | 64.6 | 97.1 | 84.9 | 28.4 | 68.8 |
| QwenWPI + FP-LP + TTT | 63.1 | 90.2 | 73.9 | 44.9 | 68.0 |
| **QwenWPI + MoLP + AGV-TTT (VANE)** | 68.8 | 94.5 | 77.3 | 44.3 | **71.2** |

QwenPi 계열 63.9 → 68.8, QwenWPI 계열 68.0 → 71.2, 공개된 TTT-VLA 67.4 대비 **+3.8점**. 다만 각 쌍이 prompt 구조와 업데이트 프로토콜을 동시에 바꾸므로 성분 귀속은 §4.2.3에서 별도로 한다(저자 명시).

---

## 9. 주 결과 — Google Robot (Table 4/5)

3-태스크 범위(TTT-VLA 공개 비교와 동일 scope, VM/VA 6개 평균):

| Method | Overall |
|---|---|
| π0-FAST (fine-tuned) | 60.5 |
| TTT-VLA (π0.5 + SG-LP + TTT) | 66.3 |
| QwenPi + SG-LP + TTT | 71.7 |
| **QwenPi + MoLP + AGV-TTT** | **76.7** |
| QwenWPI + FP-LP + TTT | 73.6 |
| QwenWPI + MoLP + AGV-TTT | 72.4 |

4-태스크 완전 breakdown(Put in Drawer 포함, VM/VA 8개 평균):

| Method | Overall |
|---|---|
| QwenPi + SG-LP + TTT | 70.7 |
| QwenPi + MoLP + AGV-TTT | 71.7 |
| QwenWPI + FP-LP + TTT | 64.6 |
| **QwenWPI + MoLP + AGV-TTT** | **70.4** |

QwenWPI 계열은 3-태스크에서는 73.6 → 72.4로 **떨어지지만**, Put in Drawer를 포함하면 64.6 → 70.4로 크게 오른다(해당 태스크 VM/VA가 19.0/56.0 → 58.8/70.4). 즉 이득이 **태스크 구성에 의존**한다. 저자들은 Google Robot을 "우월성의 증거"가 아니라 **범위·경계 테스트**로 취급한다.

---

## 10. 성분 통제 실험 (Table 6/7)

WidowX (4 checkpoint 평균):

| Proxy | Prompt | No TTT | Offline | Online | AGV |
|---|---|---|---|---|---|
| State | Single | 64.3 | 63.9 | 64.7 | 65.9 |
| State | MoLP | 67.5 | 68.1 | 67.8 | 68.8 |
| WPI | Single | 67.5 | 68.0 | 67.5 | 69.0 |
| WPI | MoLP | 70.7 | 69.4 | 70.3 | **71.2** |

Google Robot (4-태스크):

| Proxy | Prompt | No TTT | Offline | Online | AGV |
|---|---|---|---|---|---|
| State | Single | 73.6 | 70.7 | 74.0 | **74.0** |
| State | MoLP | 72.3 | 71.7 | 71.6 | 71.7 |
| WPI | Single | 64.2 | 64.6 | 64.7 | 65.3 |
| WPI | MoLP | 70.1 | 70.2 | 70.5 | 70.4 |

- **Prompt 구조**: WidowX에서 MoLP는 매칭된 모든 단일-prompt 설정을 개선(state +2.9~+4.2, WPI +1.4~+3.2). No-TTT 열에서도 개선된다는 점이 중요 — routed prompt는 **배포 최적화 이전에 이미 학습된 정책을 바꾼다**. Google에서는 WPI 행만 일관되게 개선.
- **Proxy**: WidowX에서 WPI가 8개 매칭 비교 전부에서 state grounding 초과(No-TTT 격차부터 이미 +3.2). 그러나 각 모델의 자기 No-TTT 기준 **증분** 이득이 커지는 것은 아님. Google에서는 8개 비교 중 **한 번도** 초과하지 못함.
- **프로토콜**: WidowX에서 AGV-TTT는 모든 proxy–prompt 행에서 최고이며, 네 행 모두를 No TTT 위로 올린 **유일한** 프로토콜(+1.6/+1.3/+1.5/+0.5). Google에서는 State–MoLP를 오히려 악화시킨다.

---

## 11. 미래 검증의 필요성과 최적화 비용 (Table 8/9)

단일 prompt QwenWPI 기준:

| Online protocol | Event gate | Future validation | Success |
|---|---|---|---|
| Online TTT | – | – | 67.5 |
| Event + Accept-All | ✓ | – | 67.3 |
| AGV-TTT | ✓ | ✓ | **69.0** |

Accept-All(동일 detector + shadow 제안, 검증 없이 전부 commit)은 67.3으로 Online TTT보다도 **낮다**. 검증을 붙여야 69.0 → **이벤트 희소성만으로는 이득이 설명되지 않고, +1.7점은 순수하게 미래 검증의 기여**다.

| Protocol | Backward | Validation forward | Success |
|---|---|---|---|
| Online TTT | 45,824 | – | 67.5 |
| AGV-TTT | 612 | 4,831 | 69.0 |

backward 호출 **98.7% 감소(약 75×)**, 적격 프레임의 1.3%에서만 최적화, 제안의 **19.4%만 수용**. 저자들은 이를 wall-clock speedup으로 환산하지 않고 호출 수만 보고한다(정직한 태도).

---

## 12. 종합 평가

**강점**
- **폐루프 인과 분리 설계가 정교하다.** 이벤트 탐지가 action 생성 이후에 일어나고, 후보는 shadow에서만 갱신되며, 검증 프레임 대부분의 action은 old prompt가 만든다 — "업데이트가 자신의 심판 데이터를 조작하는" 문제를 구조적으로 차단한다. 게다가 old/candidate 비교에서 **동일 난수·동일 focus 토큰**을 강제해 페어링 노이즈를 제거했다.
- **음성 결과를 숨기지 않는다.** Google Robot 3-태스크에서 QwenWPI 계열이 오히려 하락하는 것, WPI가 Google 8개 비교에서 전패하는 것, AGV가 State–MoLP를 악화시키는 것을 모두 표로 드러내고 "범위 테스트"로 규정한다.
- **Ablation이 주장과 정확히 대응한다.** Accept-All 대조군은 "attention gating이 아니라 validation이 이득의 원천"이라는 구체적 주장을 분리해 검증한다.
- 성공률이 아니라 **4개 연속 checkpoint 평균**을 보고하고, checkpoint 선택에 평가 성공률을 쓰지 않았다고 명시.

**한계**
- **평가 범위가 SimplerEnv 시뮬레이션 2개 suite뿐**이다. 실로봇 검증, LIBERO/CALVIN 등 타 벤치마크 없음.
- **핵심 이득의 상당 부분이 TTT가 아니라 사전학습에서 온다.** WidowX WPI–MoLP는 No TTT에서 이미 70.7이고 AGV는 +0.5만 더한다. 논문 제목이 TTT를 앞세우지만, 절대 성능의 주역은 WPI proxy와 MoLP 구조(학습 시 이득)다.
- **cross-system 비교(vs TTT-VLA)가 backbone·학습 데이터·적응 방식을 동시에 바꾼다.** +3.8/+10.4점 격차는 방법 효과로 귀속될 수 없다(저자도 인정).
- Attention 이벤트 detector에 **정량 평가가 전혀 없다**(precision/recall 미측정). 임계값·robust normalization은 Appendix C에 있으나 민감도 분석은 없다.
- **재현성**: 코드/체크포인트 공개 언급 없음. 파라미터 수, 추론 주파수, 학습 GPU 자원 미보고.
- $H_v$(검증 horizon)와 $k=8$ 미래 offset에 대한 sweep이 없어, 지연 목표 설정의 민감도를 알 수 없다.

**결론**: "언제 제안할지"는 정책 내부 신호(attention)로, "배포할지"는 외부 미래 증거로 나누어 결정한다는 분업이 이 논문의 진짜 기여다. Accept-All 대조군이 그 분업의 후반부가 필수임을 깔끔하게 보여준다. 다만 절대 성능 향상의 주된 원천은 배포-시 적응이 아니라 WPI/MoLP라는 **학습-시 인터페이스 변경**이며, 평가가 시뮬레이션 2개 suite에 갇혀 있어 일반성은 아직 열린 문제다.

<!-- VERIFIED: pdf -->
