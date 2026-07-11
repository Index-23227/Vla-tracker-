# FabriVLA: A Lightweight Vision-Language-Action Model for Precise Multi-Task Manipulation

> **한 줄 요약**: InternVL3.5-1B 백본(14 layer 유지) + flow-matching action head에 **zero-init gated self-attention**과 **shallow VLM layer fusion(layer 6 + 14)**을 결합, 단일 스테이지 joint training만으로 Meta-World MT50에서 tier-average 90.0% / overall 92.0%를 달성한 0.89B 경량 VLA.

---

## 1. 배경 및 동기

- 대형 VLA(RT-2, OpenVLA, π₀ 등)는 수십억 파라미터로 강한 성능을 내지만 **추론 지연과 연산 비용**이 실시간 로봇 제어에 부담.
- Evo-1 등 경량 VLA 계열이 1B 미만 규모로도 경쟁력 있는 성능을 입증 → FabriVLA는 Evo-1에서 영감을 받아 **1B급 VLM 위에서 정밀 멀티태스크 조작**을 목표로 설계.
- 핵심 질문: (1) 액션 토큰 간 시간적 의존성을 어떻게 안정적으로 학습에 주입할 것인가? (2) 정밀 조작에 필요한 **저수준 공간 정보**를 어떻게 action head에 공급할 것인가?

## 2. 방법론 심층 분석

### 2.1 아키텍처 개요
- **VLM 백본**: InternVL3.5-1B, 입력 이미지 448×448, transformer **14개 layer만 유지**. corner RGB view 단일 카메라 입력.
- **State encoder**: 로봇 상태 s ∈ R²⁴ (EE pose, gripper, object positions) → 2-layer MLP(ReLU) → 1024차원 state token, VLM context 앞에 prepend.
- **Action encoder**: noisy action x_t ∈ R⁵⁰ˣ²⁴ → shared MLP(24→1024→1024→1024, ReLU) + horizon positional encoding → 50개 action token.
- **Action head**: FabriVLA TransformerBlock ×8 (attention head 8개, embedding 1024).

### 2.2 Gated Self-Attention (핵심 기여)
각 블록에서 액션 토큰이 학습 가능한 게이트 g(0으로 초기화)를 통해 self-attention:

```
A' = A + g · SelfAttn(LayerNorm(A))
```

- 초기에는 g=0으로 self-attention이 완전히 꺼져 cross-attention-only transformer와 동일 → **부드러운 최적화 경로** 확보.
- 학습이 진행되며 게이트가 열려 50-step 호라이즌 내 액션 토큰 간 의존성을 점진적으로 학습.
- 이후 cross-attention(context C 대상) → time-embedding이 더해진 FFN(1024→4096 GELU→1024) 순으로 처리, 마지막에 LayerNorm + sequence-pool + 2-layer MLP로 velocity v ∈ R⁵⁰ˣ²⁴ 출력.

### 2.3 Shallow VLM Layer Fusion
- 최종 layer(14)는 고수준 semantics, 중간 layer(6)는 객체 경계·상대 위치 같은 **공간 디테일**을 보존.
- 두 layer의 토큰을 feature 차원으로 concat 후 학습 가능한 W ∈ R²⁰⁴⁸ˣ¹⁰²⁴로 projection(**[I | 0] 초기화** → 초기에는 deep-only baseline과 기능적으로 동일).
- 추가 파라미터 약 2.1M으로 매우 경량.

### 2.4 Flow Matching
- 노이즈 ε ~ U([-1,1]⁵⁰ˣ²⁴), 선형 보간 x_t = (1−t)ε + t·a, t ~ **Beta(2,2)** (중간 영역 집중 샘플링).
- 목표 velocity는 a − ε, loss는 L(θ) = E‖v_θ − (a−ε)‖².
- 추론: uniform noise에서 시작해 **N=50 Euler step**으로 적분, denormalize 후 유효 4차원 Meta-World 액션을 receding-horizon 방식으로 실행.

## 3. 데이터 전략

- 학습 데이터: **Evo-1 공개 Meta-World demonstration dataset** — MT50 태스크당 50 trajectory, 총 2,500 trajectory.
- **로봇 데이터 사전학습 없음**(Robo-Pre. = No). VLM은 공개 pretrained checkpoint에서 로드, action head는 random init.
- 데이터 증강: random cropping, rotation, color jitter.

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|---|---|
| 총 학습 스텝 S | 100,000 (평가에는 **93k checkpoint** 사용) |
| Batch size B | 40 (NVIDIA RTX PRO 6000 ×5) |
| Learning rate η | 2.0×10⁻⁵, linear warmup 1,000 steps + cosine decay |
| Optimizer | AdamW, weight decay 10⁻⁴, grad clip 1.0 |
| 정밀도 | BF16 + **DeepSpeed ZeRO-2 FP32 master weights** (없으면 VLM 업데이트가 BF16 양자화 노이즈에 묻힘) |
| 프레임워크 | HuggingFace Accelerate |

## 5. 실험 설계 및 평가 프로토콜

- 벤치마크: **Meta-World MT50** (50개 조작 태스크, easy/medium/hard/very hard 4 tier).
- 태스크당 10 episode, 랜덤 초기 상태, **environment seed 4042**, 400-step horizon.
- 지표: **tier-average**(4개 tier 점수의 평균)와 **overall episode-level**(500 episode 중 성공 비율).

## 6. 실험 결과 심층 분석

**Table 1 (MT50) 비교:**

| 모델 | Params | Robo-Pre. | Easy | Med. | Hard | V.Hard | Tier Avg. |
|---|---|---|---|---|---|---|---|
| TinyVLA | 1.3B | No | 77.6 | 21.5 | 11.4 | 15.8 | 31.6 |
| π₀ | 3.5B | Yes | 71.8 | 48.2 | 41.7 | 30.0 | 47.9 |
| SmolVLA | 2.3B | No | 87.1 | 51.8 | 70.0 | 64.0 | 68.2 |
| RoboTron-Mani | 4B | No | 85.5 | 67.7 | 76.7 | 81.0 | 77.7 |
| Evo-1 | 0.8B | No | 89.2 | 76.8 | 77.2 | 79.2 | 80.6 |
| Evo-Depth | 0.9B | No | 83.1 | 84.7 | 87.3 | 82.4 | 84.4 |
| LA4VLA | 1B | MixPT | 88.9 | **94.5** | 66.7 | **100.0** | 87.5 |
| **FabriVLA** | **0.89B** | No | **95.0** | 88.2 | 86.7 | 90.0 | **90.0** |

- 0.89B, 로봇 사전학습 없이 **tier-average 1위(90.0%)**, overall episode-level 92.0%. Easy 최고, Medium/Hard/V.Hard는 2위.
- **Task-demand 분석(Table 2)**: planar sliding(95.7%), articulated contact(95.7%), insertion(93.3%)에 강하고, tool-mediated(83.3%), coarse reaching/transport(85.0%), grasp & place(86.0%)가 상대적 약점 — 국소 공간 정렬보다 광역 transport와 시간 조율이 필요한 tool use가 더 어려움을 시사.
- **학습 동역학(Figure 4)**: 20k→93k까지 꾸준히 상승, 93k가 피크이며 100k에서는 소폭 하락(후기 열화).

## 7. Ablation 분석

- **Shallow layer fusion (Figure 5)**: deep-only(layer 14) 대비 layer-6 fusion이 **모든 tier에서 개선**, tier-average 82.9→90.0 (+7.1pp), overall 86.8→92.0 (+5.2pp).
- **Action head 모듈 누적 ablation (Figure 6, 50k steps + frozen VLM 조건)**: Base 48.5/55.4 → **+SA 57.7/66.9**(결정적 기여) → +TR 51.2/57.6(하락) → +TC 51.4/59.4(SA 단독보다 낮음). 세 모듈 모두 zero-init gated residual이라 초기화 시 base head와 정확히 동일 → 공정 비교. **출시 모델은 SA만 활성화.**

## 8. 관련 연구 비교

- **Evo-1**: 직접적 영감의 원천이자 학습 데이터 출처. FabriVLA는 gated SA + layer fusion으로 tier-average +9.4pp.
- **π₀ / Diffusion Policy 계열**: flow matching/diffusion 기반 연속 액션 생성 흐름을 계승하되 1B급 소형 백본으로 축소.
- **LA4VLA**: MixPT 사전학습으로 medium/very hard에서 강하지만 hard(66.7%)의 큰 편차 — FabriVLA는 tier 간 균형(86.7~95.0)이 강점.
- **TinyVLA, SmolVLA**: 같은 경량 노선이나 MT50에서 큰 격차로 하회.

## 9. 한계 및 미해결 문제

- **시뮬레이션 단일 벤치마크**: Meta-World MT50 결과만 보고, 실제 로봇 실험·LIBERO/CALVIN 등 교차 벤치마크 부재.
- **단일 카메라(corner view)·단일 시드(4042)·태스크당 10 episode** 평가 → 통계적 신뢰구간 미보고.
- Ablation은 50k step + frozen VLM 조건이라 최종 세팅에서 각 모듈 기여가 동일하게 재현되는지는 미확인.
- Tool-mediated·broad transport 태스크 약점의 원인 분석 및 해결책은 향후 과제.
- 코드/체크포인트 공개 여부가 논문에 명시되지 않음.

## 10. 총평

zero-init gating이라는 단순하고 재현 가능한 설계 원칙(SA 게이트, [I|0] fusion projection, gated residual ablation 슬롯)을 일관되게 적용해, 0.89B 모델로 MT50 tier-average 1위를 달성한 깔끔한 엔지니어링 논문. 이론적 신규성보다는 **경량 VLA 설계의 실용적 레시피**(FP32 master weights의 중요성, Beta(2,2) 시간 샘플링, 93k 체크포인트 선택 등)를 상세히 공개한 가치가 크다. 다만 실기기 검증과 다중 벤치마크 부재로 일반화 주장은 제한적.

## 11. 실무 적용 관점

- 1B급 VLM + flow-matching head 조합은 **RTX 5장 규모**로 100k step 학습이 가능해 중소 연구실/기업(실제로 Youibot 산업 로봇 팀 참여)에서 재현 가능한 스케일.
- BF16 joint training 시 **FP32 master weights 필수**라는 실무 팁은 소형 VLM full fine-tuning 전반에 적용 가능.
- Zero-init gated residual 패턴은 기존 학습 파이프라인을 깨지 않고 모듈을 추가하는 안전한 방법으로 다른 VLA에도 이식 용이.

## 12. 🔥 예상 날카로운 질문 모음

| 질문 | 답변 포인트 |
|---|---|
| Meta-World만으로 "정밀 멀티태스크 조작" 주장이 가능한가? | MT50은 50개 태스크·4 tier로 다양성은 있으나 시뮬레이션 단일 벤치마크. 저자들도 task-demand 분석으로 보완했지만 실기기·교차 벤치마크 검증은 부재. |
| Evo-1 데이터로 학습해 Evo-1을 이기는 것이 공정한가? | 동일 데이터·동일 벤치마크 조건이므로 아키텍처 기여(gated SA + layer fusion)의 직접 비교로는 오히려 공정. 단, baseline 수치는 각 논문에서 발췌한 것으로 평가 시드/에피소드 수가 다를 수 있음. |
| gated SA가 왜 TR·TC보다 효과적인가? | SA는 모든 블록 내부에서 50개 액션 토큰 간 전역 의존성을 학습하는 반면, TR은 per-step 선형 잔차, TC는 k=3 국소 스무딩에 그침. 장호라이즌(50 step) 시간 구조에는 전역 attention이 유리. |
| 93k 체크포인트 선택은 test-set 선택 아닌가? | 평가 벤치마크 성능 곡선(Figure 4)으로 체크포인트를 골랐으므로 사실상 evaluation-based model selection — 별도 validation 분리가 없어 낙관적 편향 가능성 있음. |
| layer 6 선택 근거는? | 논문은 "empirically chosen"이라고만 명시 — 시각 디테일 보존과 충분한 transformer 처리 사이의 균형. layer sweep 결과는 미제시. |
| 50 Euler step 추론이 실시간 제어에 충분히 빠른가? | 논문은 inference latency/Hz를 보고하지 않음. receding-horizon으로 50-step chunk를 재계획하므로 실효 빈도는 미확인 — 경량성 주장 대비 아쉬운 부분. |

<!-- VERIFIED: pdf -->
