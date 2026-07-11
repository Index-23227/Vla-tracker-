# XS-VLA: Coupling Coarse-grained Spatial Distillation with Latent Flow Matching for Lightweight Robotic Control

> **한 줄 요약**: 0.25B 초경량 VLA를 위한 2단계 "distill-then-control" 프레임워크. (1) Qwen3-VL-4B 교사 모델이 LIBERO 장면 이미지에 grasp keypoint 2개와 9개 이산 방향 영역(top/top-left/.../bottom-right) 기반 coarse-grained 공간 설명을 자동 주석 → SmolVLM2-0.25B를 spatial instruction tuning하여 "spatial blindness"를 치유(SmolVLM2-PD-0.25B). (2) 이 grounded backbone 위에 ACT식 CVAE(latent style 변수 z)와 flow matching을 결합한 **Latent Flow Matching** 정책을 end-to-end로 학습. LIBERO 평균 **90.0%**로 <0.5B 신규 SOTA — Vanilla SmolVLA 2.25B(88.8%), OpenVLA 7B(76.5%)를 상회하고, LIBERO-Long에서 0.25B baseline 대비 **+23.0%p**(63.0→86.0%), SmolVLA-PD 대비 3.2배 실행 속도(58 s/epoch), 추론 GPU 메모리 약 1.6GB.

---

## 1. 배경 및 동기

### 추론 능력 vs 제어 주파수의 트레이드오프

- 접촉을 동반하는 manipulation 제어 루프는 10–50Hz를 요구하지만, 7B+ LVLM은 표준 하드웨어에서 1Hz 미만으로 동작 — closed-loop 저수준 제어에 부적합.
- Jetson Orin급 엣지 디바이스를 겨냥한 경량 VLM(SmolVLM2, MobileVLM, TinyLLaVA)은 속도는 확보하지만 **"spatial blindness"** — 객체 존재는 인식하되 좌표를 정밀 인코딩하지 못하는 결함 — 을 보인다. "left of", "behind" 같은 상대 공간 전치사를 이해하지 못해, 대상은 맞게 식별하고도 옆의 빈 공간을 집는 실패 모드가 발생.

### 저자의 가설

- 소형 VLM의 한계는 아키텍처 크기의 본질적 제약이 아니라 **학습 데이터 분포의 결과**다. 좌표 토큰 출력이 가능한 Qwen3-VL-4B를 교사로 삼아 공간 지식을 0.25B 스케일로 "다운로드"할 수 있다.
- 추가 문제: 혼합된 human demonstration 데이터의 높은 행동 다양성(multimodality)은 결정론적 정책 학습을 열화시킴 → CVAE + Flow Matching의 생성적 정책으로 대응.

## 2. 방법론 심층 분석

### 2.1 Stage 1: Qwen3-VL-4B 기반 공간 시맨틱 생성 (Sec III-A)

- 2단계 파이프라인: (i) Qwen3-VL-4B가 LIBERO 이미지 $I_i$에서 instruction 기반 프롬프트로 **grasp keypoint 2개를 정수 픽셀 좌표로 예측**, 원본에 오버레이하여 주석 이미지 $\tilde{I}_i$ 생성. (ii) $\tilde{I}_i$ + 추가 프롬프트를 다시 Qwen3-VL-4B에 입력, **9개 이산 방향 영역**(top, top-left, top-right, center, center-left, center-right, bottom, bottom-left, bottom-right)으로 표현된 공간 설명 $T_i$ 생성 → 학습 튜플 $(\tilde{I}_i, T_i)$.
- 사람 주석 없이 확장 가능한 grounded spatial supervision 구축이 핵심.

### 2.2 Stage 2: 0.25B backbone 미세조정 (Sec III-B)

- SmolVLM2-0.25B(SigLIP + SmolLM)를 사전학습 가중치에서 초기화, 연속 bbox 좌표 회귀 대신 **방향 지향 이산 예측 과제**로 정식화. 목적함수는 표준 autoregressive CE: $L_{FT} = -\sum_t \log P(y_t | y_{<t}, \tilde{I}_i; \theta)$.
- 결과물 SmolVLM2-PD-0.25B를 추론 속도를 위해 **16 layer로 truncation** 후 정책과 결합.

### 2.3 Stage 3: Latent Flow Matching 정책 (Sec III-C)

- **CVAE (latent intent)**: BERT식 Transformer 인코더 $E_\phi$가 [CLS] + proprioceptive state $c_t$ + GT action chunk $A_t$를 처리(입력은 computation graph에서 detach, $d_{model}=256$, pre-norm, sinusoidal PE, time-major). [CLS] 출력을 대각 가우시안 $q_\phi(z|A_t, c_t)$로 사영, reparameterization으로 z 샘플. z는 모션의 고수준 "style"을 인코딩. **추론 시 인코더 우회, z = 0 또는 prior N(0,I)에서 샘플**.
- **Interleaved attention action expert**: SmolVLA 아키텍처 계승. 대부분 layer는 prefix(이미지/텍스트 임베딩 + MLP 사영 state + 사영된 z)에서 suffix(noisy action 토큰 + flow time 임베딩)로 단방향 cross-attention; N layer마다 joint self-attention을 삽입해 action 표현이 backbone에서 decoupling되는 것을 방지(causal masking).
- **Robust flow matching 목적함수**: $A_t^\tau = \tau\epsilon + (1-\tau)A_t$, 벡터장 $u = \epsilon - A_t$를 회귀. 시연 outlier에 강건하도록 MSE 대신 **Huber loss** $H_\delta$. 총손실 $L = \lambda_{FM} L_{FM} + \lambda_{KL} L_{KL}$, posterior collapse 방지를 위해 $\lambda_{KL}$은 **첫 10,000 step 선형 warmup**. flow time $\tau$는 Beta 분포에서 샘플.
- Vision encoder, language backbone, action expert, CVAE 전체를 **end-to-end 공동 최적화**.

## 3. 데이터 전략

- Stage 1 증류 데이터: 시뮬레이션 LIBERO 이미지에 대해 교사 모델이 전자동 생성 — 로봇 데이터셋의 고비용 인간 주석을 우회하며 "balanced dataset"으로 학습.
- 정책 학습: SmolVLA와 **동일한 Lerobot-Libero 데이터셋**, 160,000 steps — backbone 효과의 통제 비교 가능.
- 실기: XLerobot에서 **3명의 서로 다른 시연자**가 텔레오퍼레이션으로 수집한 100개 demo. 접근 각도/전달 타이밍/속도가 다른 다중 전문가 데이터로 multimodality 처리 능력을 의도적으로 시험.

## 4. 실험 설계

- **환경**: LIBERO(MuJoCo), 4개 suite — Spatial / Object / Goal / Long(LIBERO-100 중 장기 10 task). task당 10 evaluation episode 평균 SR + epoch당 실행 시간(completion time) 보고.
- **Baseline**: Diffusion Policy, Octo(0.09B), OpenVLA(7B), SpatialVLA(4B), TraceVLA(7B), ThinkAct(7B), FPC-VLA(7B), Dita(0.33B*, 전체 0.64B), Vanilla SmolVLA 3개 스케일(0.25B/0.5B/2.25B). 외부 결과는 원 논문/SmolVLA 논문에서 인용.
- **통제 ablation**: SmolVLA-PD-0.25B(증류 backbone + 표준 SmolVLA 아키텍처), XS-VLA w/o backbone pre-training(공간 증류 없이 Latent FM만).
- **실기**: XLerobot 양팔 당근 전달(왼손 파지→오른손 전달→목표 배치), 10 trial, partial credit(파지 0.5 + 전달·배치 0.5). ACT(80k steps), SmolVLA-0.5B(60k steps), XS-VLA(20k steps) 비교.

## 5. 주요 결과

| Model | Spatial | Object | Goal | Long | Avg | Time/Epoch |
|---|---|---|---|---|---|---|
| Vanilla SmolVLA (0.25B) | 87.0 | 93.0 | 88.0 | 63.0 | 82.8 | - |
| Vanilla SmolVLA (0.5B) | 90.0 | 96.0 | 92.0 | 71.0 | 87.3 | - |
| Vanilla SmolVLA (2.25B) | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 | - |
| XS-VLA w/o backbone PT (0.25B) | 94.0 | 83.0 | 88.5 | 84.0 | 87.4 | 14 s |
| SmolVLA-PD (0.25B) | 86.5 | 96.5 | 89.5 | 81.0 | 88.8 | 186 s |
| **XS-VLA (0.25B)** | 90.0 | **96.0** | 88.0 | **86.0** | **90.0** | **58 s** |

- **<0.5B 신규 SOTA**: 82.8%(기존 SOTA인 SmolVLA-0.25B) → **90.0%**. 2.25B SmolVLA(88.8%), OpenVLA-7B(76.5%), FPC-VLA-7B(86.9%), ThinkAct-7B(84.4%)를 모두 상회. 단 Dita(w/ wrist, 92.3%)에는 미달하되 LIBERO-Long에서는 Dita(w/ wrist, 83.6%) 대비 +2.4%p.
- **LIBERO-Long +23.0%p**(63.0→86.0): Latent Flow Matching이 장기 horizon에서 compounding error/drift를 억제하는 강력한 prior로 작용.
- **실기(Table II)**: XS-VLA 7.5 > ACT 7.0 > Vanilla SmolVLA-0.5B 6.5 (10 trial 합산). CVAE가 3인 시연자의 상충하는 전략을 z 공간의 별개 모드로 인코딩, mean-seeking jitter를 회피.
- 추론 메모리 약 **1600M(1.6GB)** — OpenARM, PiPER 하드웨어에도 배포.

## 6. Ablation 분석

- **공간 증류의 기여** (Vanilla 0.25B → SmolVLA-PD): 82.8→88.8%로 단일 최대 상승. LIBERO-Object +3.5%p(유사 객체 구분 — 잘못된 instance 파지 실패 해소), LIBERO-Long +18.0%p(63.0→81.0; 공간 grounding이 배경 텍스처/distractor 과적합을 막는 시각적 prior 역할).
- **Latent Flow Matching의 기여** (Vanilla 0.25B → XS-VLA w/o backbone PT): 82.8→87.4%. LIBERO-Long +21.0%p(63.0→84.0). 결정론적 head의 시연 평균화(jitter)를 연속 action 분포 모델링으로 대체.
- **효율**: XS-VLA w/o backbone PT는 14 s/epoch vs SmolVLA-PD 186 s(13.3배). 최종 XS-VLA는 58 s로 SmolVLA-PD 대비 **3.2배** 단축 — 정확도 손실 없이 더 부드럽고 정밀한 궤적.
- **시너지**: 두 구성요소 결합 시 90.0% / Long 86.0% — 증류는 기하·지각 정밀도를, Latent FM은 이를 안정적 제어 궤적으로 변환.

## 7. 관련 연구와의 위치

- **경량 VLA 계열**: SmolVLA의 아키텍처(interleaved attention, layer truncation)를 직접 계승하되, (i) 공간 증류 backbone, (ii) CVAE+FM head 두 축으로 개선. Dita(0.33B*/0.64B)와 동급 경량 경쟁.
- **공간 표현 강화 VLA**: SpatialVLA(4B), FPC-VLA, TraceVLA가 대형 모델에서 공간/포인트클라우드 표현을 탐구한 것과 달리, **0.25B 스케일에서 교사-학생 증류로** 공간 prior를 주입 — SpatialVLM의 문제의식을 경량화 맥락으로 이식.
- **생성적 정책**: Diffusion Policy의 multimodality 해법과 π0 계열 flow matching의 효율을 잇고, ACT의 CVAE를 결합한 하이브리드. 증류(VL2Lite 등)와 정책 학습을 하나의 파이프라인으로 묶은 것이 차별점.

## 8. 강점

1. **통제된 ablation 설계**: 동일 데이터(Lerobot-Libero)·동일 아키텍처 골격에서 backbone과 policy head 기여를 분리(SmolVLA-PD, w/o backbone PT) — 90.0%의 출처가 명확.
2. **주석 비용 제로의 증류 파이프라인**: 교사 모델 2-pass(keypoint→방향 설명)로 사람 개입 없이 확장 가능. 연속 좌표 회귀 대신 9-영역 이산 예측이라는 과제 단순화가 0.25B 용량에 적합.
3. **실기 + 다중 전문가 시연 검증**: multimodality 처리라는 주장(CVAE)을 3인 시연자 실기 셋업으로 직접 검증. 20k steps만으로 80k steps ACT를 상회.
4. **엣지 실용성**: 1.6GB 추론 메모리, 3.2배 실행 속도 — 경량 VLA의 배포 장벽을 실질적으로 낮춤.

## 9. 약점 및 한계

1. **속도 수치의 모호성**: abstract는 3.2배, ablation 본문은 14 s vs 186 s로 13.3배를 병기 — 최종 XS-VLA(58 s) 기준 3.2배가 정확하나 서술이 혼재. epoch당 실행 시간의 측정 조건(하드웨어)도 미명시.
2. **suite별 trade-off**: w/o backbone PT 변형은 Object에서 83.0%로 오히려 baseline(93.0%)보다 하락 — Latent FM 단독이 일부 suite를 해칠 수 있음이 충분히 분석되지 않음.
3. **2D 교사의 한계**: 저자 인정 — depth ambiguity에 취약하고, goal-conditioned task의 고유 기하 객체에서 여전히 어려움(Goal 88.0%는 baseline 대비 개선 없음).
4. **실기 규모**: 단일 task(당근 전달) 10 trial, 점수 차 0.5는 통계적 유의성 논증 부재.
5. **평가 episode 수**: task당 10 episode로 표준 LIBERO 프로토콜(50 rollouts) 대비 적어 분산이 클 수 있음.

## 10. 실용적 시사점

- 소형 VLM의 공간 결함은 **데이터 문제**이며, 대형 grounding 모델의 자동 주석으로 저비용 치유 가능 — 온디바이스 VLA 구축의 일반 레시피.
- coarse-grained(9-영역) 감독만으로도 manipulation에 충분한 grounding 신호가 됨 — 정밀 bbox 회귀가 필수라는 통념에 반례.
- CVAE latent를 조건으로 한 flow matching은 다중 시연자 데이터의 style 분리라는 실무적 문제(텔레옵 데이터 혼합)에 직접 적용 가능.
- 16-layer truncation + 0.25B로 Jetson급 배포와 10Hz+ 제어 루프 대응 가능성.

## 11. 예상 질문과 답변

| # | 질문 | 답변 |
|---|---|---|
| 1 | 증류 데이터가 LIBERO 이미지 기반인데 LIBERO 평가는 in-domain 아닌가? | 맞다. 공간 증류 감독이 평가 도메인과 동일 분포 — 교차 도메인 일반화 주장은 실기(XLerobot/OpenARM/PiPER) 정성 결과에 의존하며, 정량 out-of-domain 벤치마크는 없음. |
| 2 | 9-영역 이산화로 충분한가? 정밀 파지는 연속 좌표가 필요하지 않나? | backbone은 방향 prior만 제공하고 정밀 좌표는 flow matching 정책이 시연에서 학습 — grounding은 "어느 쪽인가"를, 제어는 "정확히 어디인가"를 담당하는 분업. |
| 3 | 추론 시 z=0으로 두면 CVAE의 의미가 없지 않나? | 학습 시 z가 style 분산을 흡수해 flow matching이 단일 모드 회귀에서 벗어나는 것이 핵심(ACT와 동일 논리). 추론에서는 prior 샘플로 모드 선택 가능. |
| 4 | 3.2배 vs 13.3배 speedup의 진실은? | 14 s는 backbone 미증류 변형, 58 s가 최종 모델 — SmolVLA-PD(186 s) 대비 3.2배가 최종 모델 기준 공정 수치. 13.3배는 ablation 변형의 수치. |
| 5 | Dita(w/ wrist) 92.3%에 뒤지는데 SOTA 주장이 성립하나? | 논문의 주장은 명시적으로 "<0.5B 중 SOTA"이며 Dita는 전체 0.64B(*DiT 부분만 0.33B). Long에서는 Dita(w/ wrist)도 상회(+2.4%p). |
| 6 | VLA pretraining(action trajectory) 없이 이 성능이 나온 이유는? | 공간 증류가 시각적 prior를 대체 공급하고, LIBERO 단일 도메인 학습이라 대규모 action pretraining의 이득이 제한적인 설정. cross-embodiment 일반화는 미검증. |
| 7 | Huber loss와 Beta 분포 τ 샘플링의 기여는 분리 검증되었나? | 아니다 — 개별 ablation 없음. π0 계열 관행의 채택으로 보임. |
| 8 | Goal suite가 개선되지 않는 이유는? | 저자 스스로 unique object geometry 의존 task의 어려움과 2D 교사의 depth ambiguity를 한계로 지목. |

## 12. 결론

XS-VLA는 "경량 모델의 spatial blindness는 용량이 아니라 데이터의 문제"라는 가설을 0.25B 스케일에서 설득력 있게 입증한 논문이다. Qwen3-VL-4B의 grounding 능력을 9-영역 이산 방향 예측이라는 압축된 과제로 증류하고, ACT의 CVAE와 flow matching을 결합한 Latent Flow Matching 정책으로 다중 시연 multimodality를 처리하여, LIBERO 평균 90.0% — 2.25B SmolVLA와 7B OpenVLA를 넘는 <0.5B 신규 SOTA — 를 달성했다. 특히 LIBERO-Long +23.0%p와 두 구성요소의 시너지를 보인 통제 ablation, 3인 시연자 실기 검증이 방법론적 강점이다. 다만 증류 데이터와 평가 도메인의 중첩, task당 10 episode의 제한된 평가 규모, speedup 수치 서술의 혼재, Goal suite 정체는 후속 검증이 필요하며, cross-domain/cross-embodiment 일반화가 자연스러운 다음 과제로 남는다.

<!-- VERIFIED: pdf -->
