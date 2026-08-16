# SLIM-0.5B: Learning Action-Grounded Predictive Latents for Robot Manipulation

> **한 줄 요약**: 대형 VLM backbone도, 픽셀 수준 미래 생성도 없이 **관측 latent와 연속 action token의 상호작용**에 제어 연산을 몰아넣은 0.47B급 정책. Stage 1에서 inverse/forward dynamics를 짝지은 masked trajectory prediction으로 "action-grounded predictive latent"를 학습하고, Stage 2에서 flow matching 정책만 남겨 배포한다 (LIBERO 97.5, zero-shot LIBERO-Plus 77.45, CALVIN ABC→D 4.556, 60.6 ms 지연).

- **arXiv**: 2608.09771v1 (2026-08-10, cs.RO)
- **저자/소속**: Jingkai Wang¹²*, Zihan Tang³²*, Gu Zhang³, Mingyu Cao², Jingjiao Zhao⁴², Xiansheng Chen², Pengwei Wang², Jiapeng Chen¹, Lemao Liu¹, Dejing Dou¹ — ¹Fudan University, ²Beijing Academy of Artificial Intelligence(BAAI), ³Tsinghua University, ⁴Renmin University of China (*동등 기여)
- **Project page**: https://kzz1031.github.io/slim-project-page/

---

## 1. 배경 및 동기

논문은 현재 language-conditioned manipulation을 지배하는 두 패러다임의 비용 구조를 문제 삼는다.

1. **VLA 패러다임**: 지각·명령 이해·행동 생성을 하나의 큰 multimodal backbone에 넣는다. 강력한 semantic prior와 open-domain 언어 이해를 얻지만, **행동과 그 행동이 유발하는 관측 변화의 관계는 action supervision을 통해 암묵적으로만** 학습된다. 게다가 매 제어 step마다 VLM 전체를 통과해야 한다.
2. **World model / WAM 패러다임**: 미래 이미지·비디오·joint video-action 궤적을 예측한다. "행동은 그것이 야기한 결과에 근거해야 한다"는 동기는 제어와 잘 맞지만, 배경 텍스처·조명·사소한 시각 변화처럼 **다음 행동 결정에 무관한 appearance 디테일에 용량을 쓴다**. 추론 시 future generation이나 world-model rollout을 돌리면 제어 루프에 추가 계산·지연이 붙는다.

여기서 저자들이 잡는 설계 지점은 명확하다. **semantic task conditioning은 유지하되, action–observation 상호작용을 명시적으로 모델링하는 compact 표현**. 이 표현이 갖춰야 할 성질을 저자들은 *action-grounded*라 부르며, 양방향으로 정의한다.

- 관측된 transition을 **어떤 행동이 설명하는지** 추론할 수 있어야 하고,
- **행동이 유발하는 latent transition을 예측**할 수 있어야 한다.

핵심 전환은 이것이다: **미래 예측을 pixel-generative model의 출력이 아니라, latent 공간에서 정책 표현을 형성하는 학습 신호로만 사용**한다.

---

## 2. 방법론 — 문제 정식화

언어 명령 `y`는 T5-small로 인코딩되어 조건 `ℓ = g_lang(y)`가 된다. 시각 관측 `o_t`는 visual encoder(DINOv2 초기화)를 통해 `N_z`개의 observation-latent token으로 매핑된다.

```
Z_t = g_ψ(o_t) ∈ R^{N_z × d_z}
```

정책은 action chunk `A_t = (a_t, ..., a_{t+H-1})`를 예측한다. **action token은 연속 embedding이며 어떤 이산화도 적용하지 않는다.** 미래 latent는 online 버전 `Z_{t+H} = g_ψ(o_{t+H})`와 EMA copy 기반의 안정적 타깃 `Z̄_{t+H} = g_ψ̄(o_{t+H})` 두 가지가 있고, `Z_t`와 `Z_{t+H}`는 **같은 관측-latent 공간에 살며 시간적 역할만 다르다**. 이 동일 공간 설계가 뒤의 forward-dynamics 목적함수를 "정책이 실제로 소비하는 공간"에서 성립시키는 열쇠다.

---

## 3. Latent Interaction Policy — MoT backbone

backbone은 compact **Mixture-of-Transformers(MoT)**로, 두 개의 상호작용하는 stream을 유지한다.

- **Observation stream**: 각 observation-latent token을 공유 hidden dim `d = 768`로 선형 사영. proprioceptive state `q_t`는 별도 선형 사영으로 single state token이 되어 stream 앞에 붙는다. 여기에 더해 `N_z`개의 **learned future-slot embedding** `M_z`를 유지한다(예제 간 공유, 위치별로는 구별됨).
- **Action stream**: 3-layer MLP encoder가 각 연속 action vector를 `d`로 사영하고, flow timestep의 sinusoidal embedding과 concat한 뒤 action token으로 매핑.

각 MoT block 안에서 두 stream은 stream-specific Q/K/V를 만들고 **shared joint-attention**으로 상호작용한다. 출력은 다시 observation-side / action-side로 분리되어 **per-stream language cross-attention**과 FFN을 거친다. 즉 **언어는 task condition으로만 주입되고, 주 연산은 observation-action 상호작용에 예약**된다.

future 위치에 무엇이 채워지는지가 branch를 규정한다.

| Branch | future 슬롯 | action stream |
|---|---|---|
| action-masked (IDM) | 사영된 clean future latent `Z_{t+H}` | noised chunk `Ã^τ_t` |
| future-latent-masked (FDM) | learned mask embedding `M_z` | clean chunk `A_t` |
| policy (Stage 2) | `M_z` | noised chunk `Ã^τ_t` |

flow sampling 동안 observation latent와 언어 context는 고정되고 정책은 action stream만 반복 갱신한다. 저자들의 표현대로 SLIM은 "large language-centric decoder"보다 **제어 관련 latent와 action token 위의 compact interaction model**처럼 동작한다.

---

## 4. Action-grounded 학습 — 2단계 레시피

### Stage 1: masked trajectory prediction

JEPA 계열의 representation-space prediction 원리를 따르되, **추가 annotation도 픽셀 재구성도 없이 로봇 궤적 자체가 supervision**이다.

**(a) Inverse-dynamics 목적 (action-masked)**. `Z_t`, clean `Z_{t+H}`, `q_t`, `ℓ`에 조건화하여 masked action chunk를 conditional flow matching으로 복원한다.

```
ε ~ N(0, I),  τ ~ U(0,1),  Ã^τ_t = (1-τ)ε + τA_t,  V*_t = A_t - ε
L_IDM = E ‖ v_θ(Ã^τ_t, τ | Z_t, Z_{t+H}, q_t, ℓ) - V*_t ‖²
```

**(b) Forward-dynamics 목적 (future-latent-masked)**. future 슬롯을 `M_z`로 대체하고, `Z_t`와 clean `A_t`로부터 미래 관측 latent를 예측한다. 타깃은 **stop-gradient EMA encoder 출력** `Z̄_{t+H}`이고 L1으로 감독한다.

두 목적은 표현을 양방향으로 묶는다. 행동은 관측 변화를 설명해야 하고, 미래 관측 latent는 행동으로부터 예측 가능해야 한다. objective별 attention mask가 정보 흐름을 통제한다(Fig. 2c). 손실 가중은 `λ_IDM = 0.125`, `λ_FDM = 1`.

### Stage 2: flow-matching policy

배포용 정책은 현재 관측·proprioception·언어만 쓴다. future 슬롯은 `M_z`로 채워지고 미래 관측 latent는 **입력으로도 타깃으로도 등장하지 않는다**.

```
L_FM = E ‖ v_θ(Ã^τ_t, τ | Z_t, M_z, q_t, ℓ) - V*_t ‖²,   L_stage2 = L_FM
```

핵심은 이것이다. **observation-side hidden state에는 Stage 1이 형성한 predictive slot이 여전히 남아 있고**, action stream은 MoT joint attention을 통해 flow sampling 중 그 hidden state에 접근한다. 따라서 추론 시 행동 생성은 **명시적 future-latent 예측이 아니라 암묵적 future-prediction 구조**에 의해 유도된다 — 이것이 SLIM이 계산적으로 저렴한 이유다.

---

## 5. 실험 설정

- **벤치마크 3종**: LIBERO(-10/Object/Spatial/Goal), LIBERO-Plus(camera/robot/language/light/background/noise/layout 7종 perturbation), CALVIN ABC→D. 지표는 앞 둘이 success rate, CALVIN은 average sequence length.
- **핵심 통제**: LIBERO/LIBERO-Plus용 SLIM은 **원본 LIBERO 데이터로만** 학습한다. Stage 1은 LIBERO-90 + 4개 target suite 혼합, Stage 2는 4개 target suite만. LIBERO-Plus는 **동일 체크포인트를 무적응 zero-shot**으로 10,030개 perturbation case에 평가. CALVIN은 ABC 학습 / 미관측 D 평가.
- **평가 규모**: task당 50 rollout(4 suite 합 2,000 rollout), CALVIN은 표준 1,000개 5-instruction chain.
- **입출력**: workspace + wrist 2개 RGB view, 224×224. 7차원 action, horizon은 LIBERO 8 / CALVIN 12, 추론 시 flow sampling 4 step.
- **P.T. 지표**: 표에 "추가 embodied policy/world pretraining" 여부를 명시하며, 일반 vision-language/video backbone 초기화만으로는 카운트하지 않는다. **SLIM은 P.T. = ×.**

---

## 6. LIBERO 및 zero-shot LIBERO-Plus 결과 (Table 1, Table 5)

원본 LIBERO suite별 결과 (Appendix Table 5, %):

| Method | P.T. | Size | Long | Spatial | Object | Goal | Overall |
|---|---|---|---|---|---|---|---|
| OpenVLA | ✓ | 7B | 53.7 | 84.7 | 88.4 | 79.2 | 76.5 |
| OpenVLA-OFT | ✓ | 7B | 94.5 | 97.6 | 98.4 | 97.9 | 97.1 |
| π0 | ✓ | 3.3B | 85.2 | 96.8 | 98.8 | 95.8 | 94.1 |
| π0.5 | ✓ | 3.3B | 92.4 | 98.8 | 98.2 | 98.0 | 96.9 |
| NORA | ✓ | 3B | 74.6 | 92.2 | 95.4 | 89.4 | 87.9 |
| WorldVLA | × | 7B | 59.0 | 85.6 | 89.0 | 82.6 | 79.1 |
| UnifiedVLA | ✓ | 8.5B | 94.0 | 95.4 | 98.8 | 93.6 | 95.5 |
| RIPT-VLA | ✓ | 7B | 93.8 | 99.0 | 98.6 | 98.6 | 97.5 |
| Fast-WAM | × | 6B | 95.2 | 98.2 | 100.0 | 97.0 | 97.6 |
| VLA-JEPA | ✓ | 3B | 95.8 | 99.6 | 96.2 | 97.2 | 97.2 |
| **SLIM** | × | **0.47B** | 94.4 | 99.4 | 99.4 | 96.8 | **97.5** |

zero-shot LIBERO-Plus (Table 1, %):

| Method | Size | Camera | Robot | Language | Light | Background | Noise | Layout | **Overall** |
|---|---|---|---|---|---|---|---|---|---|
| OpenVLA | 7B | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | 7B | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.3 | 69.6 |
| π0 | 3.3B | 61.0 | 40.8 | 63.7 | 89.3 | 84.1 | 80.1 | 75.9 | 69.3 |
| NORA | 3B | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| WorldVLA | 7B | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 11.0 | 38.0 | 25.0 |
| UnifiedVLA | 8.5B | 1.8 | 46.2 | 69.5 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| RIPT-VLA | 7B | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |
| Fast-WAM | 6B | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 50.0 |
| VLA-JEPA | 3B | 64.2 | 67.7 | 88.1 | 91.8 | 93.4 | 65.8 | 83.9 | **79.5** |
| **SLIM** | **0.47B** | **70.73** | 36.90 | 87.57 | **94.75** | 92.01 | **86.07** | 83.21 | **77.45** |

읽어야 할 지점:

- SLIM은 **7B RIPT-VLA와 동일한 원본 LIBERO 97.5**를 0.47B로 달성하며, 추가 embodied pretraining도 없다.
- LIBERO-Plus overall 77.45는 **camera(70.73), light(94.75), noise(86.07)에서 표 전체 1위**를 포함한 결과다. 1위 VLA-JEPA(79.5)는 6배 크고 P.T.도 받았다.
- **약점은 robot initial-state perturbation(36.90)**으로 π0(40.8), Fast-WAM(44.5), VLA-JEPA(67.7)보다 낮다. 시각 latent 중심 설계가 로봇 자세 자체의 분포 이동에는 상대적으로 취약함을 시사한다.

---

## 7. CALVIN ABC→D 결과 (Table 2)

| Method | P.T. | Size | Task1 | Task2 | Task3 | Task4 | Task5 | **Avg. Len** |
|---|---|---|---|---|---|---|---|---|
| RoboFlamingo | × | 3B | 82.4 | 61.9 | 46.6 | 33.1 | 23.5 | 2.47 |
| OpenVLA | ✓ | 7B | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.27 |
| RoboDual | ✓ | 7.02B | 94.4 | 82.7 | 72.1 | 62.4 | 54.4 | 3.66 |
| UnifiedVLA | ✓ | 8.5B | 98.9 | 94.8 | 89.0 | 82.8 | 75.1 | 4.41 |
| FLOWER | ✓ | 0.95B | 99.4 | 95.8 | 90.7 | 84.9 | 77.8 | 4.53 |
| Seer-Large | ✓ | 0.57B | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.28 |
| GR-MG | × | – | 96.8 | 89.3 | 81.5 | 72.7 | 64.4 | 4.04 |
| GR-1 | × | 0.20B | 85.4 | 71.2 | 59.6 | 49.7 | 40.1 | 3.06 |
| UniPi | × | – | 56.0 | 16.0 | 8.0 | 8.0 | 4.0 | 0.92 |
| VPP | ✓ | 1.5B | 96.5 | 90.9 | 86.6 | 82.0 | 76.9 | 4.33 |
| DreamVLA | ✓ | – | 98.2 | 94.6 | 89.5 | 83.4 | 78.1 | 4.44 |
| **SLIM** | × | **0.47B** | **99.3** | **96.7** | **92.3** | **87.1** | **80.2** | **4.556** |

표 내 전 항목 최고치다. 특히 **후반 task(3/4/5)에서 격차가 벌어진다** — Task 5에서 SLIM 80.2 vs DreamVLA 78.1, FLOWER 77.8. LIBERO 계열 밖의 장기 horizon 언어 조건부 task composition에서도 latent interaction 정책이 작동함을 보여준다.

---

## 8. 실로봇 평가 (Figure 6)

5개 task(당근을 그릇에 담기, 접시 3개 쌓기, 토스터에서 토스트 꺼내 접시에 놓기, 블록 쌓기, 화이트보드 닦기), task당 150 demo(총 750), **다섯 개의 개별 정책이 아니라 하나의 multi-task 정책**으로 학습. 각 task-condition 쌍당 10 trial, task별 부분 점수(0.5/1.0 milestone) 평균×100.

| 설정 | SLIM | π0.5 | Fast-WAM |
|---|---|---|---|
| Nominal | **86.0** | 72.0 | 71.0 |
| Distractor | **63.0** | 54.0 | 39.0 |
| Lighting | **73.0** | 47.0 | 48.0 |
| Background | 49.0 | **54.0** | 2.0 |
| **Average** | **67.8** | 56.8 | 40.0 |

nominal·distractor·lighting 3개 설정에서 최고이며, background shift에서만 π0.5(54)에 근소하게 뒤진다(49). Fast-WAM은 background에서 2.0으로 사실상 붕괴하는데, WAM 계열이 appearance 변화에 얼마나 민감한지를 보여주는 대비 사례다.

---

## 9. Ablation과 표현 진단 (Figure 7, Table 3)

세 가지 질문으로 구성된다.

1. **Stage 1이 최종 정책을 개선하는가?** LIBERO-Plus와 CALVIN 양쪽에서 Stage-2-only 대비 **일관되게 향상**.
2. **IDM:FDM 손실비 민감도**: **0.125:1이 최적**. 중간 비율에서는 비교적 안정적이고 양 극단에서 열화된다.
3. **EMA 타깃 encoder의 효과**: LIBERO-Plus **66.82% → 77.45%**, CALVIN avg length **4.382 → 4.556**.

Table 3의 진단이 특히 설득력 있다. 최종 20개 Stage-1 evaluation probe 평균:

| Variant | Eff. rank ↑ | Top-1 ↓ | Top-5 ↓ | Cosine ↓ | Future-latent MSE ↓ |
|---|---|---|---|---|---|
| EMA | **61.28** | **0.097** | **0.362** | **0.071** | 0.245 |
| No EMA | 13.95 | 0.395 | 0.674 | 0.352 | **0.166** |

EMA를 제거하면 **future-latent MSE는 오히려 0.245 → 0.166으로 내려간다**. 그러나 effective rank가 61.28 → 13.95로 붕괴하고, token cosine similarity가 0.071 → 0.352로, top-1 energy concentration이 0.097 → 0.395로 치솟는다. 즉 **낮은 오차는 예측이 좋아진 것이 아니라 타깃이 degenerate low-rank로 무너진 결과**다. 이것이 LIBERO-Plus 77.45 → 66.82 하락과 동반된다. JEPA 계열의 고질적 collapse 문제를 지표로 정면 진단한, 이 논문에서 가장 잘 설계된 분석이다.

**정성 분석(Fig. 8)**: MoT에서 action-to-observation attention을 뽑아 patch score를 이미지 평면에 사영. SLIM은 조작 대상 물체·gripper·둘의 접촉 영역을 일관되게 따라가는 반면, Stage 1 없는 정책은 attention이 산만하고 로봇 몸체나 무관한 장면 영역에 지배되는 경우가 많다. 저자들은 이를 **causal attribution이 아니라 qualitative probe**라고 명시적으로 한정한다.

---

## 10. 효율성 (Table 4)

단일 NVIDIA H100 80GB, PyTorch eager, BF16, batch 1, 2개 224×224 입력. 20 warmup + 200 synchronized call을 3개 프로세스에서 반복(600 pooled measurement).

| Model | Latency (ms) ↓ | Peak VRAM (GiB) ↓ | GFLOPs/chunk ↓ |
|---|---|---|---|
| Fast-WAM | 360.6 | 13.63 | 2090.07 |
| π0.5 | 193.1 | 7.94 | 4714.59 |
| **SLIM** | **60.6** | **4.26** | **490.73** |

π0.5 대비 **3.19× 빠르고**, Fast-WAM 대비 **5.95× 빠르다**. pooled p95 지연은 61.6 ms로 분산도 작다. 메모리는 π0.5 대비 1.86×, Fast-WAM 대비 3.20× 절약. FLOPs는 π0.5 대비 9.61×, Fast-WAM 대비 4.26× 적다.

**단, 저자들이 명시한 caveat**: native horizon이 다르다(SLIM 8-step chunk / 4 sampling step, π0.5 10-step / 10 step, Fast-WAM 32-step / 10 step). 따라서 보고된 값은 **실행된 action 1개가 아니라 replanning call(=action chunk) 1회 기준**이다. chunk당 action 수로 나누면 Fast-WAM의 상대적 불리함은 줄어든다.

**파라미터 구성(Table 6)**: DINOv2-B/14 86.58M(18.34%), MoT trunk 377.96M(80.05%), action/state interface + prediction head 7.60M(1.61%), 학습 가능 총 472.14M. 여기에 Stage 1에서만 인스턴스화되는 frozen EMA encoder 86.58M을 더하면 558.72M. **frozen T5-small은 0.47B 카운트에서 제외**되어 있다는 점은 비교 시 유의할 부분이다.

---

## 11. 강점과 한계

**강점**

- **파라미터 효율의 명확한 증명**: 0.47B로 7B급 LIBERO 성능과 CALVIN SOTA를 동시에 달성. P.T. 없이.
- **미래 예측을 학습 신호로만 쓰는 설계**: 추론 경로에서 future latent가 완전히 사라지므로, "world model의 grounding 이득"과 "world model의 추론 비용"을 분리하는 데 성공했다. 60.6 ms / 4.26 GiB / 490 GFLOPs라는 삼중 절감이 그 대가다.
- **collapse 진단의 엄밀함**: MSE 하락과 성능 하락이 함께 나타나는 상황을 effective rank·energy concentration·cosine으로 해부한 Table 3은, JEPA류 목적함수를 로보틱스에 쓸 때의 표준 진단 프로토콜로 삼을 만하다.
- **양방향 결합의 단일 backbone화**: DeFI처럼 forward/inverse를 별도 모듈로 두지 않고, 제어에 쓰는 그 MoT 안에서, 제어에 쓰는 그 latent 공간에서 처리한다.

**한계**

- **단일 규모만 검증**. 저자들 스스로 결론에서 인정하듯 모델 용량·pretraining 데이터·embodiment 다양성 축으로의 scaling 특성은 전혀 확립되지 않았다.
- **robot initial-state perturbation 36.90**은 표에서 하위권이다. 관측 latent에 proprioception이 단일 state token으로만 들어가는 구조와 관련이 있어 보이나 논문은 이를 분석하지 않는다.
- **open-domain 언어 능력 포기**: 언어는 frozen T5-small의 cross-attention 조건으로만 들어간다. LIBERO/CALVIN 수준의 정형 명령에서는 충분하지만, VLM backbone이 제공하던 open-vocabulary 일반화·상식 추론은 설계상 제거되었다. LIBERO-Plus language perturbation 87.57은 좋지만 이는 어디까지나 같은 task 집합의 표현 변주다.
- **레이턴시 비교의 chunk 단위 문제**: §10 caveat대로 per-action 기준으로 정규화하면 배수는 달라진다.
- **코드/체크포인트 미공개** (project page만 제공).

---

## 12. 종합 평가

SLIM의 기여는 새로운 손실 함수가 아니라 **"어디에 계산을 둘 것인가"에 대한 재배치**다. VLA는 제어 연산을 open-domain VLM에, WAM은 픽셀 미래 생성에 두었다. SLIM은 둘 다 거부하고 **관측 latent ↔ action token 상호작용**에 둔다. 그리고 미래 예측은 그 상호작용 공간을 조각하는 데만 쓰고 추론에서는 버린다.

이 논지가 설득력 있는 이유는 세 축의 결과가 서로를 보강하기 때문이다. (a) 성능 — LIBERO 97.5 / CALVIN 4.556, (b) 강건성 — zero-shot LIBERO-Plus 77.45와 실로봇 조명·distractor 우위, (c) 비용 — 60.6 ms, 4.26 GiB. 셋 중 하나만 있었다면 흔한 결과였겠지만, 0.47B에서 셋이 동시에 성립한다는 점이 논문의 실질이다. 특히 Fast-WAM(6B)이 실로봇 background shift에서 2.0으로 붕괴하는 대비는, 픽셀 수준 미래 예측이 강건성 측면에서도 반드시 이득이 아님을 보여준다.

남는 질문은 저자들도 지목한 scaling이다. VLM backbone을 버림으로써 얻은 효율이, cross-embodiment 다양성과 open-vocabulary 요구가 커질 때도 유지되는지는 미검증이다. 다만 **"제어에 필요한 것은 open-domain semantics가 아니라 action-observation transition의 compact 표현"**이라는 가설을 이 정도 예산으로 이만큼 밀어붙인 사례로서, 경량 VLA·edge 배포·실시간 제어를 고민하는 쪽에서는 반드시 참조할 설계다.

---

<!-- VERIFIED: pdf -->
