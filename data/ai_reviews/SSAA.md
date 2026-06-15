# SSAA: See Selectively, Act Adaptively — Dual-Level Structural Decomposition for Bimanual Robot Manipulation

> **한 줄 요약**: 양손 조작에서 "어떤 시점을 볼 것인가"(View-Selective Visual Router)와 "어떻게 행동할 것인가"(Interaction-Aware Action MoE)를 명시적으로 분리한 dual-level decomposition 프레임워크. pi0.5 베이스 위에 LoRA 3-expert MoE를 얹어 RoboTwin 2.0 6개 task에서 monolithic baseline 대비 시뮬레이션 +27.7%, 실세계 +43.3%의 성공률 향상을 달성.

---

## 1. 배경 및 동기

### 양손 조작의 두 가지 이질성
- **지각적 이질성 (Perceptual heterogeneity)**: task stage와 manipulation context에 따라 좌·우 wrist view의 관련성이 시간적으로 변동
  - 예: 한 팔이 grasp 수행 중일 때 해당 wrist view는 결정적, 반대쪽 wrist view는 task-irrelevant
- **상호작용 이질성 (Interaction heterogeneity)**: stage별로 양팔이 독립(independent) 또는 협동(coordinated) 모드로 전환
  - 예: 순차적 배치(독립) → 물체 전달(협동) → 회전 배치(혼합)

### 기존 monolithic VLA의 구조적 한계
- 단일 shared representation과 action pathway로 모든 시각 입력·행동 패턴을 처리
- View 관련성을 명시적으로 모델링하지 않아 task-irrelevant 시각 단서가 noise로 작용
- 독립/협동 행동이 같은 head 안에서 entangle → mode-specific 학습 신호 간섭
- 두 이질성이 dynamic하게 결합(coupling)되는데, 한쪽만 다루면 실패

### 핵심 질문
- **시각 선택과 행동 분해를 명시적·동시적으로 모델링하면 양손 manipulation의 강건성을 높일 수 있는가?**

---

## 2. 방법론 심층 분석

### 2.1 전체 구조

베이스 정책: **pretrained pi0.5** [11]. 두 가지 모듈을 삽입:
1. **VSR**: VLM 입력 단계 전에 wrist-view token을 reweight
2. **IAMoE**: action expert(flow matching velocity field) 를 3-expert MoE로 확장

관측 $O_t = (\{I_t^v\}_{v \in \{E,L,R\}}, \ell, s_t)$ → H-step action chunk $A_t = [a_t, \ldots, a_{t+H-1}]$ (양팔 joint + gripper) 예측.

### 2.2 View-Selective Visual Router (VSR)

- ViT가 외부/좌·우 wrist view를 각각 $F_t^E, F_t^L, F_t^R$로 인코딩
- 언어+state 토큰 $C_t$와 $F_t^E$로 routing context $z_t^{vis}$ 구성 (wrist view는 의도적으로 제외)
- $F_t^E$에는 **attention pooling**, $C_t$에는 **mean pooling** 적용 후 결합
- MLP + sigmoid → relevance weight $w_t = [w_t^L, w_t^R] \in [0,1]^2$
- 재가중 wrist token: $F_t^{'L} = w_t^L F_t^L$, $F_t^{'R} = w_t^R F_t^R$ → 사전학습 VLM에 입력

> ❓ **왜 wrist token을 routing context에서 제외했는가?**
> 모든 3개 image stream을 routing에 포함하면 hard setting에서 routing이 불안정 → 학습 환경 특이적 shortcut에 의존. 외부 view + language만이 global scene/task conditioning을 제공.

### 2.3 Interaction-Aware Action MoE (IAMoE)

Conditional flow matching의 velocity field $u_\tau$를 3개의 LoRA expert로 분해:
- **Coordinated expert** $P_{coord}$: 전체 action space
- **Left/Right arm-wise expert** $P_L, P_R$: action mask $M^L, M^R$로 해당 팔 부분공간에만 작용

$$\hat{u}_\tau^{coord} = P_{coord}(h_t^{coord}),\ \hat{u}_\tau^L = M^L \odot P_L(h_t^L),\ \hat{u}_\tau^R = M^R \odot P_R(h_t^R),\ \hat{u}_\tau^{ind} = \hat{u}_\tau^L + \hat{u}_\tau^R$$

**Action Router**: $z_t^{act}$에서 hard one-hot $m_t \in \{[1,0], [0,1]\}$ 예측
$$\hat{u}_\tau = m_t^{coord} \hat{u}_\tau^{coord} + m_t^{ind} \hat{u}_\tau^{ind}$$

- 학습: straight-through Gumbel-Softmax (forward hard, backward soft)
- 추론: $\arg\max$로 결정, **생성 과정 내내 mode 고정**(action chunk 일관성 유지)

### 2.4 학습 목적함수

$$\mathcal{L}_{total} = \mathcal{L}_{main} + \lambda_{vis} \mathcal{L}_{vis} + \lambda_{act} \mathcal{L}_{act} + \lambda_{aux} \mathcal{L}_{aux}$$

- $\mathcal{L}_{main}$: backbone과 동일한 flow matching loss (router-conditioned)
- $\mathcal{L}_{vis}$: VSR BCE supervision (KNN-기반 semi-auto label $y_t^{vis} \in \{0,1\}^2$)
- $\mathcal{L}_{act}$: Action Router CE supervision ($y_t^{act}$)
- $\mathcal{L}_{aux}$: branch-wise auxiliary loss — 모든 expert가 충분한 학습 신호 받도록 (학습 중 점진적 감소)

$$\mathcal{L}_{aux} = \mathbb{E}[\|\hat{u}_\tau^{coord} - u_\tau\|^2 + \|\hat{u}_\tau^L - M^L \odot u_\tau\|^2 + \|\hat{u}_\tau^R - M^R \odot u_\tau\|^2]$$

---

## 3. 데이터 전략

| 환경 | Task 수 | Demo/task | Eval 조건 |
|------|--------|-----------|----------|
| RoboTwin 2.0 (sim) | 6 (S1–S6) | 50 | 100 rollouts, easy + hard |
| Real-world | 3 (R1–R3, 6 stage each) | 40 | 10 trials, easy + hard (hard는 baseline/Ours만) |

- Router supervision label은 **KNN 기반 semi-automatic**: human-in-the-loop seeding 후 nearest-neighbor 전파
- 학습 데이터는 모두 easy setting에서 수집 → hard에서의 generalization을 별도 평가

### Task 카테고리 설계 (의도된 ablation)
- **지각 heterogeneity 강함**: S1 (Stack Bowls Three), S2 (Blocks Ranking Size)
- **상호작용 heterogeneity 강함**: S3 (Lift Pot), S4 (Place Bread Skillet)
- **양쪽 모두 강함**: S5 (Handover Block), S6 (Put Bottles Dustbin)

---

## 4. 실험 결과 심층 분석

### RoboTwin 2.0 시뮬레이션 (Fig. 4)

| Method | S1 E/H | S2 E/H | S3 E/H | S4 E/H | S5 E/H | S6 E/H | Overall |
|--------|--------|--------|--------|--------|--------|--------|---------|
| Baseline (pi0.5) | 67/36 | 27/10 | 97/47 | 63/8 | 69/9 | 46/24 | **41.9%** |
| w/o IAMoE (VSR만) | 71/45 | 50/47 | 95/70 | 65/29 | 75/21 | 51/32 | **54.3%** |
| w/o VSR (IAMoE만) | 78/50 | 49/33 | 99/75 | 70/38 | 86/33 | 65/40 | **59.7%** |
| **Ours (Full)** | **85/68** | **63/55** | **97/83** | **78/47** | **93/46** | **71/49** | **69.6%** |

- VSR 단독으로 S2 hard에서 **+37%p** (지각 heterogeneity 가설 검증)
- IAMoE 단독으로 S3/S4 hard에서 **+28%/+30%p** (상호작용 heterogeneity 가설 검증)
- 결합 시 S5/S6에서 single-module 대비 추가 이득 (hard에서 +17.3/+13.2%p)

### 실세계 (Fig. 3, cumulative stage-wise success)

| Task | Method | S1 | S2 | S3 | S4 | S5 | S6 |
|------|--------|----|----|----|----|----|----|
| R1 (Pack and Place) | Baseline | 70 | 60 | 60 | 60 | 50 | 40 |
| | Ours | 100 | 100 | 100 | 100 | 100 | **90** |
| R2 (Handover and Pour) | Baseline | 100 | 80 | 80 | 80 | 60 | 60 |
| | Ours | 100 | 100 | 100 | 100 | 100 | **100** |
| R3 (Rotate and Place) | Baseline | 100 | 70 | 70 | 70 | 40 | 30 |
| | Ours | 100 | 100 | 100 | 100 | 90 | **80** |

- 평균 final stage 성공률: Baseline 43.3% → Ours 90% (+46.7%p, 논문 표현 +43.3%)
- Hard setting에서도 R1/R2/R3 각각 baseline 대비 **+30/+40/+50%p**

### 실패 사례 분석
- **w/o IAMoE 실패**: coordinated stage에서 양팔 trajectory 비정렬 → handover/box 운반 실패
- **w/o VSR 실패**: arm-role assignment 오류, target-position inference 부정확
- **w/o VSR S6 hard 실패**: 시각 모호성으로 다음 active arm을 구분 못해 두 팔이 동시 진행 → action-level decomposition만으로는 부족함을 입증

---

## 5. 강건성/일반화

### Q3: Hard setting 성능 (Fig. 5)
- Ours가 simulation hard에서 baseline 대비 **+35.7%p**
- Hard에서의 큰 이득은 monolithic 정책이 visual ambiguity를 implicitly 해결해야 하는 부담을 explicit routing이 덜어준다는 해석을 뒷받침
- Router가 easy 분포에 단순 fit한 것이 아님 (shortcut 의존 부정)

### 한계
- Human-in-the-loop label에 의존 (view 관련성은 camera placement/object config에 따라 변동)
- Independent vs. coordinated 이분법으로 단순화 (loose/tight coordination 미구분)
- Hard setting은 "training과 다른 환경"이지만 robot embodiment·매우 큰 distribution shift는 미검증

---

## 6. 아키텍처 디자인 결정의 정당성

| 결정 | 근거 |
|------|------|
| pi0.5를 base로 채택 | open-world generalization을 갖춘 최신 generalist VLA |
| VLM **앞단**에 VSR 삽입 | backbone 내부 attention에 의존하지 않고 입력 단계에서 명시적 modulation |
| Wrist token만 reweight (external 그대로) | external view = global context 보존 (BFA [20] 인사이트) |
| Routing context에서 wrist token 제외 | hard setting에서 shortcut 의존 방지 |
| LoRA 기반 expert 분해 | shared backbone 유지 + mode-specific specialization (parameter 효율) |
| Hard one-hot routing (STE Gumbel) | action chunk 내 일관성 유지, soft mix가 야기하는 mode collapse 방지 |
| Action mask $M^L, M^R$ | arm-wise expert가 자기 팔 부분공간에만 신호를 갖도록 강제 |
| Auxiliary loss 점진 감소 | 초기 모든 expert 학습 보장 → 후기 router 결정에 expert가 적응 |

---

## 7. 강점

1. **인과적 분해**: "what to see" / "how to act"를 perception/action 두 level에서 모듈로 분리 — 해석 가능하며 ablation으로 각 모듈의 역할 검증 완료
2. **Pretrained VLA 보존**: pi0.5 파이프라인을 거의 그대로 유지, LoRA + 경량 router만 추가 → 학습 비용 절감
3. **명시적 supervision**: router에 explicit label 제공으로 학습 안정 (Gumbel STE만으로는 어려운 hard routing 학습)
4. **이중 평가**: 시뮬레이션(통제된 heterogeneity 패턴) + 실세계(long-horizon stage 전환) 모두에서 일관된 이득
5. **Hard setting에서 더 큰 이득**: monolithic 정책의 implicit reasoning 부담을 explicit module이 덜어준다는 가설을 직접 검증

---

## 8. 약점 및 비판적 검토

> ❓ **KNN 기반 label이 정말 scalable한가?**
> 저자도 limitation에서 인정. VLM 기반 자동 라벨링을 future work로 제안. 50 demos × 6 task = 300 trajectory 정도는 manageable하지만, large-scale에서는 병목.

> ❓ **Independent vs. coordinated 이분법의 한계**
> 실세계 bimanual은 loose/tight/stabilizing 등 continuum. 본 논문은 loose를 coordinated에 묶음. 더 세분화한 hierarchical routing이 future direction.

> ❓ **Parameter-matched baseline이 본문에 없다**
> Appendix C 언급. LoRA expert + router 추가 파라미터의 이득이 단순 capacity scaling 효과가 아닌지 검증 필요.

> ❓ **Hard setting이 "unseen environment"이지만 같은 task**
> Task-level/embodiment-level transfer는 미검증. 더 큰 distribution shift에서의 router 거동 미지수.

> ❓ **추론 시 mode 고정의 trade-off**
> Stage 전환이 chunk 중간에 일어나면? H가 짧으면 자주 re-plan으로 보완되지만, hard one-hot이 stage transition 모호 구간에서 jitter를 일으킬 위험.

---

## 9. 관련 연구와의 위치

| 연구 | 차별점 |
|------|--------|
| pi0.5 [11] | 본 논문의 base. Monolithic VLA |
| RDT-1B [15] | Diffusion 기반 bimanual generalist (구조 분해 없음) |
| InterACT [27] | Inter-dependency aware action chunking (action-level만, view-level 없음) |
| VoxAct-B [31] | Acting/stabilizing 역할 사전 정의 (정적 decomposition) |
| BFA [20] | Multi-view best-feature-aware fusion (perception만, action 분해 없음) |
| Selective Perception [24] | Task-aware multimodal attention (perception만) |
| DriveMoE [34] | 자율주행 VLA용 MoE (도메인 다름) |
| SkillVLA [25] | Dual-arm combinatorial skill 재사용 (skill 단위 분해) |

**SSAA의 위치**: perception과 action 양 level에서 **동시·dynamic** decomposition을 수행하고 두 routing을 명시적 supervision으로 학습. 정적/단일level 접근의 한계를 직접 겨냥.

---

## 10. 재현성 평가

- **공개**: 코드 미공개(논문 기준), pi0.5 base 또한 비공개 가중치
- **재현 가능성**: 학습 디테일(LoRA 30k step, batch 16, 50 demos)은 명시. Router architecture(MLP+sigmoid)와 pooling 방식 공개 → 동등 구조 재구현 가능
- **데이터**: RoboTwin 2.0은 공개 benchmark이므로 시뮬레이션은 재현 가능. 실세계 task는 독자 수집
- **하이퍼파라미터**: $\lambda_{vis}, \lambda_{act}, \lambda_{aux}$ 구체 값은 본문에 없음 (Appendix 필요)

---

## 11. 향후 연구 방향

1. **자동 router label 생성**: VLM 기반 scene understanding으로 view relevance / interaction mode 자동 annotation
2. **연속/계층적 interaction mode**: loose vs. tight coordination 분리, continuous coordination score
3. **Chunk 내 dynamic mode 전환**: stage transition을 chunk 경계와 비동기로 처리
4. **Cross-embodiment generalization**: humanoid, mobile bimanual 등 다양한 plataform에서 검증
5. **Longer-horizon**: 6-stage 이상 task, hierarchical routing과의 결합
6. **Routing interpretability**: VSR/Action Router의 결정을 attention heatmap·시간축으로 시각화

---

## 12. 종합 평가

**핵심 기여**: Bimanual manipulation의 두 이질성(perceptual, interaction)을 단일 framework에서 **명시적·dynamic**으로 분해. pi0.5 위에 가벼운 router + LoRA expert만 얹어 sim +27.7%, real +43.3% 향상.

**왜 중요한가**: Monolithic VLA가 scale로 모든 것을 해결한다는 가정에 대한 구조적 inductive bias의 가치를 입증. 특히 hard setting에서의 큰 이득은 explicit decomposition이 단순한 capacity 추가가 아니라 **task structure를 반영하는 학습 신호 분리**임을 시사.

**적용 범위**: Bimanual long-horizon manipulation. Stage 전환과 view relevance 변동이 잦은 task일수록 이득이 큼. 단일 stage 단순 task에는 overkill.

**점수 (5점 만점)**:
- 새로움: **4.0** (dual-level + dynamic routing 결합은 신선, 개별 요소는 기존 연구 차용)
- 실험 엄밀성: **4.2** (sim/real 양쪽, 의도된 ablation 카테고리, hard setting 평가)
- 실용성: **3.7** (KNN 라벨링 의존, code 미공개)
- 명확성: **4.3** (문제 정의·분해 동기·아키텍처 흐름 명료)
- **종합: 4.05 / 5.0**

**한 줄 결론**: 양손 조작에서 "분해는 무엇을·어디서·어떻게" 해야 하는가에 대해, perception/action 양 level의 dynamic decomposition이라는 설득력 있는 답을 제시한 논문.

<!-- VERIFIED: pdf -->
