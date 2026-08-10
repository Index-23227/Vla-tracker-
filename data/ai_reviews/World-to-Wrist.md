# World-to-Wrist (W2-VLA): Task-Conditioned Future Wrist Modeling for Fine-Grained Robot Manipulation

**arXiv**: 2608.05369 (2026-08-05) · **소속**: HKUST, NUS, NTU, 우한대, 중산대, 시안전자과기대, 동남대 · **공동 1저자**: Yuhao Pan, Haosong Peng · **교신**: Wenchao Xu · **프로젝트 페이지**: https://yyyyu120.github.io/W2-VLA/

> **한 줄 요약**: main view와 wrist view를 대등한 시각 입력으로 취급하던 관행을 깨고, wrist를 "action-proximal" 채널로 재정의한다. VLM이 문맥화한 16개 latent modeling token을 고정 길이 task-conditioned 인터페이스로 삼아 V-JEPA 2.1 latent 공간에서 **미래 wrist latent**를 예측하고, 이를 flow-matching action head의 조건으로 주입한다. LIBERO 98.5%, RoboTwin 2.0 Easy 60.71% / Hard 18.21%, 실기 3개 태스크 전 항목 SOTA, 그러면서 87.43 Hz.

---

## 1. 배경 및 동기

기존 multi-view VLA(SmolVLA, DexVLA, OpenVLA-OFT 등)는 여러 카메라 뷰를 **병렬 시각 토큰**으로 인코딩하거나 융합할 뿐, wrist 관측이 갖는 특수한 역할을 명시적으로 모델링하지 않는다. 그러나 두 뷰의 정보 성격은 다르다.

- **Main view**: 장면 배치, 객체 정체, 목표 관계, 태스크 진행도 등 **전역 태스크 문맥**.
- **Wrist view**: end-effector 주변에서 빠르게 변하는 **gripper–object 상호작용**.

plug insertion처럼 fine-grained·contact-rich한 태스크는 전역 이해와 국소 상호작용 동역학의 긴밀한 협응을 요구한다. 저자들은 여기서 두 번째 관찰을 더한다. wrist의 역할이 action-proximal이라면, **현재 wrist 관측만이 아니라 그 근미래 전개까지 모델링**해야 한다는 것이다.

기존 future-predictive 계열(CoT-VLA, DreamVLA, VLA-JEPA, WoG, Being-H0.7)은 시각 subgoal, 미래 관측, 혹은 통합 정책 표현을 예측 타깃으로 삼는다. 이 타깃들은 대체로 **전역 장면 또는 통합 표현 위에 정의**되어 있어 end-effector 주변의 시간적 변화를 분리해내지 못한다.

## 2. 핵심 문제 정의: wrist 예측의 task-ambiguity

Wrist-centered 예측을 단독으로 세우면 곧바로 모호성 문제에 부딪힌다. 유사한 wrist history가 여러 그럴듯한 미래에 대응하기 때문이다.

- approach 상태 → grasping일 수도, pushing일 수도 있다.
- aligned 상태 → 추가 조정, insertion, release 중 무엇이든 될 수 있다.

즉 **wrist history는 현재 국소 상호작용의 증거**를 제공하지만, **어느 전이가 task-relevant인지는 전역 문맥이 결정**한다. 이것이 "VLM 전역 문맥 → 미래 wrist 예측"을 잇는 compact interface가 필요한 이유이며, 논문 제목 *World-to-Wrist*가 가리키는 경로다.

## 3. 방법론 심층 분석

### 3.1 Task-Conditioned Latent Modeling Interface (§3.1)

제어 스텝 t에서 지시문 ℓ, main-view 관측 I^m_t, wrist-view 관측 I^w_t가 주어진다. K개의 전용 latent modeling token ⟨q_1⟩...⟨q_K⟩를 프롬프트 뒤에 이어 붙인다.

```
p_t = [Prompt(ℓ) | ⟨q_1⟩ | ... | ⟨q_K⟩]
S_t = F_θ^VLM(O_t, p_t)[P_q] ∈ R^{K×d}
```

`S_t`는 해당 토큰 위치에서 추출한 **최종 레이어 hidden state**이며, 이것이 VLM과 wrist predictor 사이의 고정 길이 인터페이스다. 학습 시 `S_t`는 세 방향의 gradient를 동시에 받는다: (1) W2-CoT 보조 감독, (2) 미래 wrist 예측, (3) action 생성. 각 목적함수가 서로 다른 계산 경로를 통해 같은 인터페이스를 형성한다는 점이 설계의 핵심이다.

### 3.2 Task-Conditioned Future Wrist Prediction (§3.2)

동결된 V-JEPA 2.1 인코더 `E_φ`가 wrist history clip과 (학습 전용) future clip을 latent로 매핑한다: `Z^w_hist = E_φ(I^w_hist)`, `Z^w_fut = E_φ(I^w_fut)`. 다중 wrist view는 각 latent time step 내에서 view 축으로 concat된다.

Predictor는 `S_t`를 V-JEPA latent time step 수만큼 복제하고, 인터페이스 상태와 history 토큰을 공유 hidden space로 각각 사영한 뒤, 각 스텝에서 인터페이스 토큰을 wrist latent 앞에 prepend하여 **bidirectional Transformer**로 처리한다.

```
Ẑ^w_fut = G_ψ(Z^w_hist, S_t)
L_wrist = || Ẑ^w_fut − sg(Z^w_fut) ||_1
```

stop-gradient `sg(·)` 덕분에 이 손실은 predictor를 직접 감독하고, **인터페이스에는 conditioning 경로를 통해서만** 학습 신호를 준다. RGB 재구성이 아니라 wrist-view 상태 전이를 모델링한다는 점이 중요하다.

이어서 Q-Former 스타일 adapter `A_ω`가 M개 learnable query로 밀집한 예측 latent를 압축한다.

```
C^w_t = A_ω(Q^w, sg(Ẑ^w_fut))
```

여기서도 stop-gradient가 걸려 있어 **action 목적함수가 adapter 경로를 타고 wrist predictor를 갱신하지 못한다**. adapter 자체는 학습 가능하다. 즉 예측기와 정책이 서로를 오염시키지 않도록 두 지점에 의도적으로 gradient 차단막을 세운 구조다.

### 3.3 W2-CoT: 구조화 주석 합성과 보조 감독 (§3.3, Appendix A)

주석은 세 필드로 구성된다.

| 필드 | 내용 |
|---|---|
| **Subtask** | 현재 조작 단계와 태스크 내 진행도 |
| **Reasoning** | 상태–행동·시각 증거가 뒷받침하는 로봇 중심 물리 전이 (approach-to-contact, grasp stabilization, transport, alignment, release) |
| **Wrist** | wrist-local 증거 (타깃 근접, 손끝 접촉, 파지 안정성, 객체 운동, 정렬, 배치 안정성, gripper–object 분리) |

Bimanual 데이터에서 Wrist는 `left=...; right=...` 순서 형식을 따르며 각 팔은 자기 gripper 증거만 기술한다.

합성 파이프라인은 4단계다.
1. **Pre-generation grounding**: gripper openness, end-effector 운동, action 변화로 approach/grasp/transport/release 경계 후보를 잡고, 경계 주변 시각 keyframe으로 객체 정체·접촉·배치를 보강한다.
2. **Structured VLM proposal**: offline VLM 주석기(Qwen3-VL)가 지시문 + 동기화된 main/wrist 프레임 + 상태–행동 증거로 구조화 제안을 생성한다.
3. **Post-VLM consistency projection**: 결정론적 검증 — 열린 gripper는 grasping/carrying으로 라벨될 수 없고, release는 선행 holding을 요구하며, carrying은 안정 파지 증거를 요구하고, per-arm wrist locality와 시간 순서 일관성을 강제한다. 짧은 jitter/retry가 허위 단계를 만들지 못하게 억제한다.
4. **Language normalization → frame-level 전파**: 검증된 세그먼트 라벨을 정규화해 프레임 단위로 확산.

학습 시에는 이 주석 시퀀스에 대한 next-token prediction 손실 `L_cot`을 부과한다. phase·target·contact·confidence 등 내부 필드는 메타데이터로만 남고 감독 타깃에는 렌더링되지 않는다.

### 3.4 학습 목적함수와 추론 (§3.4, B.3)

```
H^act_t = F_θ^VLM(O_t, p_t)[P_act],  P_q ⊆ P_act
C_t = [H^act_t | C^w_t]
Â_t = Π_η(C_t)                       (DiT flow matching)
L_act = E_{τ,ε} || D_η(X_τ, τ, C_t) − V*_t ||²
L = L_act + λ_cot·L_cot + λ_wrist·L_wrist,  λ_cot = 0.1, λ_wrist = 0.2
```

`H^act_t`에는 시각·지시문·latent modeling 상태가 포함되고 **CoT 주석 상태는 제외**된다. 학습과 추론에서 동일한 state selection을 쓴다.

추론 경로: 관측 + wrist history + 지시문 → VLM 1회 forward로 `H^act_t`와 `S_t` 확보 → wrist predictor가 미래 latent 예측 → adapter가 `C^w_t` 생성 → flow 적분으로 action chunk. **미래 wrist 관측도, autoregressive CoT 디코딩도 필요 없다.**

## 4. 구현·학습 세부사항

| 항목 | 값 |
|---|---|
| 코드베이스 | StarVLA |
| VLM | Qwen3-VL-4B-Instruct |
| Wrist 인코더 | V-JEPA 2.1 ViT-L/384 (frozen) |
| Wrist predictor | 4-layer bidirectional Transformer |
| Adapter 출력 | 32 action-context token |
| Latent modeling token | K = 16 |
| 총 파라미터 | 약 4.97B |
| Action chunk | 8 (LIBERO) / 16 (RoboTwin 2.0, 실기) |
| 해상도 | main·wrist 모두 224×224 |
| Optimizer | AdamW (β=0.9/0.95, ε=1e-8, wd 1e-8, clip 1.0) |
| LR | Qwen3-VL 1e-5, JEPA predictor 5e-5, adapter 5e-5, action head 1e-4 (cosine, warmup 5000, min 1e-6) |
| LIBERO 학습 | 60K steps, 4×A100, per-device batch 16, h=8 |
| RoboTwin 2.0 학습 | 100K steps, 8×B200, per-device batch 16, h=16 (격프레임 샘플링 → 실제 8입력/8예측) |

에피소드 시작·끝에서 history/future 프레임이 부족하면 경계 이미지를 padding하고, JEPA 예측 손실에 discount 계수를 적용해 학습 강도를 조절한다.

## 5. 실험 설정

- **LIBERO**: 4 suite(Spatial/Object/Goal/Long), suite당 10 태스크. 40 태스크 1,693 궤적으로 **단일 multi-task 정책** 학습, 태스크당 50 에피소드 → suite당 500, 총 2,000 trial.
- **RoboTwin 2.0**: clean 학습셋 2,500 demo(태스크당 50)로 단일 multi-task 정책. clean(Easy)/domain-randomized(Hard) 각각 태스크당 100 trial.
- **실기**: CoBoT Magic(Mobile ALOHA 계열) 플랫폼, 3개 태스크 × 100 teleop 궤적 = 300. 태스크당 standard 30 trial + OOD 30 trial(테이블 clutter / 조명 변화 / 배경 변화 균등 분할). binary success와 stage-wise progress score(만점 R = 4, 3, 3) 병행 보고.

## 6. 주요 결과

### LIBERO (Table 1, 성공률 %)

| Method | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| GR00T-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| MemoryVLA | 98.4 | 98.4 | 96.4 | 93.4 | 96.7 |
| VLA-JEPA | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 |
| DeepThinkVLA | 96.6 | 99.0 | 96.4 | **96.2** | 97.0 |
| StarVLA | 97.8 | 98.8 | 97.4 | 92.0 | 96.5 |
| **W2-VLA** | **99.6** | **99.8** | **99.2** | 95.2 | **98.5** |

평균 98.5%로 최강 베이스라인 대비 +1.3pp. Spatial/Object/Goal 1위, Long은 95.2로 2위권.

### RoboTwin 2.0 (Table 2, 평균 성공률 %)

| Method | Easy | Hard |
|---|---|---|
| π0 | 46.42 | 16.34 |
| RDT | 34.50 | 13.72 |
| Diffusion Policy | 28.06 | 0.64 |
| UP-VLA | 52.92 | 15.16 |
| StarVLA-OFT | 50.38 | – |
| StarVLA-GR00T | 48.80 | – |
| **W2-VLA** | **60.71** | **18.21** |

Easy에서 UP-VLA 대비 +7.79pp, StarVLA-OFT 대비 +10.33pp, StarVLA-GR00T 대비 +11.91pp. Hard에서 π0 대비 +1.87pp, UP-VLA 대비 +3.05pp. 절대 수치가 낮은 Hard 구간에서도 상대 우위가 유지된다는 점이 domain randomization 강건성의 근거다.

### 실기 (Fig. 3, 성공률 % / progress score)

| Task | π0 | VLA-JEPA | W2-VLA |
|---|---|---|---|
| Table Cleaning (standard) | 36.7 | 53.3 | **63.3** |
| Occluded Placement (standard) | 60.0 | 80.0 | **86.7** |
| Bimanual Plug Insertion (standard) | 26.7 | 30.0 | **60.0** |
| **standard 평균** | 41.11 | 54.44 | **70.00** |
| Table Cleaning (OOD) | 26.7 | 36.7 | **50.0** |
| Occluded Placement (OOD) | 46.7 | 66.7 | **73.3** |
| Bimanual Plug Insertion (OOD) | 3.3 | 10.0 | **33.3** |
| **OOD 평균** | 25.57 | 37.78 | **52.22** |

standard에서 VLA-JEPA 대비 +15.56pp, π0 대비 +28.89pp. OOD에서 VLA-JEPA 대비 +14.44pp. 가장 큰 격차는 Bimanual Plug Insertion OOD(33.3 vs 10.0)로, contact-rich·fine-grained 상황에서 wrist 예측의 이득이 가장 크다는 논문 주장과 정합적이다. Progress score도 전 태스크·전 조건 1위이며, Plug Insertion에서 1.86→2.60(standard), 1.53→2.27(OOD). 실패 분석에 따르면 정책들은 초기 파지 단계는 통과하지만 최종 정렬/삽입에서 실패하며, W2-VLA는 그 접촉 민감 단계에 **더 자주 도달**한다.

## 7. Ablation 분석 (Table 4)

**(a) 컴포넌트 기여도**

| Configuration | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| W2-VLA | 99.6 | 99.8 | 99.2 | 95.2 | **98.5** |
| w/o Wrist Predictor | 98.6 | 99.6 | 98.2 | 93.6 | 97.5 |
| w/o W2-CoT | 99.0 | 99.2 | 98.8 | 95.0 | 98.0 |

Wrist Predictor 제거 시 −1.0pp이며 하락 폭이 가장 큰 곳은 Long(95.2→93.6)이다. 미래 wrist 예측이 **시간적으로 확장된 조작**에 특히 유효함을 시사한다. W2-CoT 제거는 −0.5pp.

**(b) 인터페이스 설계와 추론 지연**

| Idx | Decode CoT | Wrist Pred. | Latent Tokens | Latency (ms) | Avg. (%) |
|---|---|---|---|---|---|
| 1 | ✔ | ✘ | N/A | 1550.77 | 97.6 |
| 2 | ✔ | ✔ | N/A | 1615.27 | 98.1 |
| 3 | ✘ | ✔ | 4 | 98.58 | 98.0 |
| 4 | ✘ | ✔ | 8 | 102.15 | 98.1 |
| 5 | ✘ | ✔ | 32 | 148.69 | 98.4 |
| **W2-VLA** | ✘ | ✔ | **16** | **110.58** | **98.5** |

명시적 CoT 디코딩은 action chunk당 1.5초 이상을 요구한다. 고정 latent 인터페이스는 **지연을 한 자릿수 이상 줄이면서 동등하거나 더 높은 성공률**을 낸다. 토큰 수를 4→32로 늘려도 일관된 개선은 없고 지연만 증가하며, 16이 sweet spot이다.

**(c) 미래 예측 타깃**

| Idx | Main-view | Wrist-view | Latency (ms) | Avg. (%) |
|---|---|---|---|---|
| 1 | ✔ | ✘ | 102.49 | 97.7 |
| 2 | ✔ | ✔ | 132.76 | 98.0 |
| **W2-VLA** | ✘ | ✔ | 110.58 | **98.5** |

wrist-only가 최고다. 고정 main 카메라는 정적 장면 비중이 커서 미래 latent가 action 유발 국소 변화를 덜 강조하고, 두 뷰 동시 예측은 비용을 늘리면서 서로 다른 예측 신호 간 간섭으로 wrist 감독을 희석한다는 해석이다.

## 8. 정성 분석

Final language-transformer layer의 post-softmax self-attention을 head·latent token 전체에 대해 평균해 시각화하면, latent modeling token이 단계별로 **stage-relevant 영역**에 주의를 옮긴다. LIBERO-10에서는 approach/grasp 시 파지 가능 영역 → transport/alignment 시 든 객체와 목표 영역으로 이동하고, 실기 Table Cleaning에서는 활성 객체와 gripper를 따라가다 wiping 구간에서 cloth–stain 접촉부에 집중한다. 이 시각화를 위해서만 언어 헤드로 Subtask/Reasoning/Wrist를 디코딩하며, 정책 추론 경로에는 포함되지 않는다.

## 9. 효율성

16-step action chunk를 183 ms에 생성 → **87.43 Hz**의 action-generation rate. LIBERO 설정의 chunk 지연은 110.58 ms. "명시적 CoT를 학습 시에만 쓰고 추론 시에는 latent로 대체한다"는 설계가 성능과 실시간성을 동시에 취한 지점이다.

## 10. 강점

1. **문제 정식화의 참신함**: multi-view VLA에서 wrist를 "또 하나의 뷰"가 아니라 action-proximal 예측 타깃으로 격상시키고, 그 예측의 task-ambiguity를 전역 문맥 조건화로 해소하는 논리가 일관적이다. Ablation (c)가 이 선택을 직접 검증한다.
2. **인터페이스의 경제성**: 고정 길이 16 토큰이라는 좁은 통로가 CoT 감독·미래 예측·행동 생성 세 목적함수의 교차점 역할을 한다. 명시적 CoT 대비 14배 이상의 지연 감소를 성능 손실 없이 얻었다.
3. **gradient 격리 설계**: `L_wrist`의 타깃 stop-gradient와 adapter 입력의 stop-gradient로 예측기·정책 간 간섭을 차단한 것은 latent world model 계열에서 흔한 표현 붕괴(collapse) 위험에 대한 구조적 방어다.
4. **주석 파이프라인의 물리적 검증**: VLM 제안을 그대로 쓰지 않고 gripper 상태 일관성·release 선행조건·wrist locality·시간 순서를 결정론적으로 검사한다. VLM 주석 노이즈를 다루는 실무적으로 유용한 레시피다.
5. **실기 평가의 설계**: 세 태스크가 각각 long-horizon, occlusion 하의 global-to-local grounding, fine-grained bimanual을 겨냥하고 OOD 조건까지 대칭적으로 평가한다. progress score 병행 보고로 부분 성공까지 드러낸다.

## 11. 한계 및 비판적 검토

1. **LIBERO 포화**: 98.5%는 상위 베이스라인 대비 +1.3pp이며 Spatial/Object/Goal은 이미 99% 대다. 이 구간에서의 순위는 시드·평가 프로토콜 편차에 취약하다. 신뢰구간이나 다중 시드 보고가 없다.
2. **Long suite 열위**: 정작 "미래 예측이 long-horizon에 유리하다"는 서사와 달리 Long은 95.2로 DeepThinkVLA(96.2)·VLA-JEPA(95.8)에 밀린다. Ablation에서 wrist predictor 제거 시 Long 하락 폭이 가장 크다는 결과와 함께 놓으면, 이득은 실재하나 절대 상한은 다른 요인이 결정한다는 해석이 필요하다.
3. **RoboTwin 2.0 Hard의 절대 수치**: 18.21%는 최고 기록이지만 여전히 매우 낮다. 도메인 랜덤화 하에서 wrist latent 예측이 얼마나 신뢰 가능한지에 대한 별도 분석(예: 예측 latent 오차와 성공률 상관)이 본문에 없다.
4. **비교 대상의 백본 비대칭**: Qwen3-VL-4B + V-JEPA 2.1이라는 최신 구성이 4.97B 규모로 동원되었다. π0(약 3B)나 StarVLA 계열과의 격차 중 어느 정도가 인터페이스 설계에서, 어느 정도가 백본 세대 차이에서 오는지 분리되지 않는다. 동일 백본에서 wrist-branch만 켜고 끈 ablation은 있으나 백본 통제 비교는 없다.
5. **W2-CoT 주석 비용**: 오프라인 VLM 주석기 + 4단계 검증 파이프라인이 필요하다. 새 데이터셋마다 태스크별 물리 제약을 다시 설계해야 하며(실기 주석은 RoboTwin 스타일 인터페이스 + 태스크별 제약), 이 비용은 논문에 정량화되어 있지 않다. 다만 CoT 제거 시 손실이 −0.5pp에 그친다는 점은 비용 대비 이득이 크지 않을 가능성을 시사한다.
6. **wrist 카메라 의존성**: 방법 전체가 wrist view 존재를 전제한다. wrist 카메라가 없거나 시야가 심하게 가려지는 embodiment로의 일반화 경로가 논의되지 않는다.
7. **공개 범위**: 프로젝트 페이지는 있으나 본문에 코드/체크포인트 공개 명시가 없어 재현성 확인이 어렵다.

## 12. 종합 평가 및 시사점

W2-VLA의 기여는 "무엇을 예측할 것인가"라는 질문에 대한 **국소화된 답**이다. 전역 장면이나 통합 정책 표현을 예측 타깃으로 삼던 흐름(DreamVLA, VLA-JEPA, WoG)에서 한 걸음 좁혀, end-effector 주변 동역학만 분리해 예측하고 그 모호성은 VLM 전역 문맥으로 해소한다. Ablation (c)에서 main-view 예측(97.7)이 wrist-only(98.5)보다 낮다는 결과는 이 논문에서 가장 정보량이 큰 단일 수치다. "더 많이 예측할수록 좋다"가 아니라 **action-relevant한 것만 예측해야 좋다**는 방향성을 제시하기 때문이다.

두 번째 시사점은 **latent 인터페이스가 명시적 CoT의 실용적 대체재**라는 실증이다. 1550 ms → 110 ms의 지연 차이는 실기 배포에서 결정적이며, 성능은 오히려 동등 이상이다. reasoning-enhanced VLA 계열이 추론 시 언어 디코딩을 유지할 필요가 있는지 재고하게 만든다.

한계는 명확하다. LIBERO는 이미 포화되어 변별력이 떨어지고, RoboTwin Hard의 절대 성능은 실용 수준과 거리가 멀며, 백본 세대 효과가 통제되지 않았다. 그럼에도 실기 Bimanual Plug Insertion에서 OOD 33.3% vs VLA-JEPA 10.0%라는 3배 이상의 격차는 시뮬레이션 지표만으로는 보이지 않던 차이를 드러낸다. 후속 연구에서 확인할 가치가 있는 지점은 (a) 동일 백본 통제 하의 wrist-branch 순효과, (b) wrist latent 예측 오차와 태스크 성공의 정량적 상관, (c) wrist 카메라가 부재하거나 저품질인 embodiment로의 확장이다.

<!-- VERIFIED: pdf -->
