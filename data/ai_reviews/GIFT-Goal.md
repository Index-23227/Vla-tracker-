# GIFT: Goal-Injected Fine-Tuning for Efficient Manipulation Policy Adaptation

> **한 줄 요약**: 이미지 편집 모델로 초기 관측에서 목표 이미지를 한 번 생성하고, 목표-관측 특징 차이를 zero-initialized convolution으로 기존 VLA의 관측 토큰에 더하는 ControlNet식 플러그인 미세조정. 단 1 epoch 학습으로 CogACT의 SIMPLER Google VM/VA를 +6.0/+13.4, WidowX를 51.3→67.7, OpenVLA의 LIBERO를 76.5→81.2로 끌어올림.

- **arXiv**: 2609.07006v1 (2026-09-07, cs.CV)
- **소속**: 난징항공항천대학(NUAA) 인공지능학원, 교육부 뇌-기계 지능기술 중점실험실
- **트래커 표기**: 동명 논문(2609.04193 "GIFT: Guided Intermediate Feature Training", 트래커명 GIFT-Feature)과 구분하기 위해 **GIFT-Goal**로 등록. SIMPLER 헤드라인은 CogACT+GIFT, LIBERO 수치는 OpenVLA+GIFT(LIBERO에서 평가된 유일한 베이스).

---

## 1. 배경 및 동기

SuSIE, GHIL-Glue처럼 생성 모델로 목표/하위 목표 이미지를 만들어 저수준 정책을 유도하면 강건성이 올라간다는 것이 알려져 있다. 그러나 (1) 이런 방법 다수는 사전학습 VLA 백본을 활용하지 않고, (2) CoT-VLA처럼 목표 생성을 VLA에 통합하는 방법은 대규모 재학습이 필요하며 사전학습된 추론 경로를 흔들 수 있다. 저자들은 "기존 VLA에 목표 이미지 조건을 싸고 안전하게 붙이는 방법"을 목표로 한다.

## 2. 문제 정의

기존 VLA는 â_t ~ π_ϕ(a_t | o_t, l). GIFT는 두 단계로 나눈다: (1) 편집 모델 p_θ가 초기 관측 o_0와 지시 l로 목표 이미지 ĝ를 한 번 생성, (2) 소수의 파라미터만 추가된 π_ϕ'가 â_t ~ π_ϕ'(a_t | o_t, l, ĝ)로 행동 예측. 핵심 제약은 사전학습 정책을 망가뜨리지 않으면서 짧은 학습으로 목표 조건을 흡수하는 것.

## 3. 방법

- **Semantic Contrast**: 공유 이미지 인코더로 목표 특징 F_g와 현재 관측 특징 F_t를 뽑고 F_Δ = F_g − F_t를 사용. 고정된 F_g를 그대로 넣으면 상수 오프셋이 되어 표현력이 제한되기 때문.
- **ZeroConv 주입**: E_t = ZeroConv(Proj_g(F_Δ)) + Proj_o(F_t). 초기에는 0이라 원 정책과 동일하게 동작하고, 학습이 진행되며 목표 신호가 점진적으로 섞인다.
- **학습 대상**: 목표 투영기(관측 투영기로 초기화), zero conv, 행동 헤드만 갱신. 나머지 VLA는 동결.
- **로봇 목표용 이미지 편집**: FLUX.1-Fill-dev를 LoRA(r=32)로 미세조정. "왼쪽은 현재 관측, 오른쪽은 {지시} 완료 후 관측"이라는 side-by-side in-context 프롬프트를 사용하고, 관측 이미지의 inversion latent와 노이즈를 이어 붙여 denoise. 초기 25% step 동안 z = α·z_g + (1−α)·z_o(α=0.5)로 관측 구조를 주입해 장면 일관성을 유지.

## 4. 데이터

- SIMPLER Google Robot 평가 모델은 Fractal, WidowX 평가 모델은 BridgeData V2로 미세조정. LIBERO는 LIBERO 데이터로 OpenVLA만 미세조정.
- 편집기 학습: BridgeData V2 90% + Fractal 90% + LIBERO 30%(첫 프레임 → 마지막 프레임, 지시문).
- 실물: Synria Alicia-D 6-DoF + 그리퍼, "pick up <obj>", "put <obj> into basket" 과제당 30개 시연(30 Hz, 관절각).

## 5. 구현 세부

- 모든 VLA: 1 epoch, 배치 256, 고정 lr 1e-5, FSDP, 4×L20. 1 epoch 소요 시간은 예: CogACT Fractal 43.2h, OpenVLA LIBERO 9.6h.
- 추가 파라미터: π0.5 +20M, CogACT/OpenVLA +72M, SpatialVLA +10M. 학습 파라미터 100M~630M.
- 오버헤드: ZeroConv 2–4 ms/step, 편집 1회 약 4.2 s. 에피소드 기준 +22.7%(CogACT) ~ +63.7%(π0.5).
- 실물: π0.5를 LoRA로 30k step(배치 20, 1×L20, 약 40시간) 먼저 적응한 뒤 GIFT를 3인칭 카메라 분기에만 1 epoch(약 2시간) 추가 학습.

## 6. 실험 설정

- SIMPLER Google Robot VM/VA 4과제(Table 1), WidowX VM 4과제(Table 2, 과제당 3회 평가로 기술), LIBERO 4 suite(Table 3).
- 베이스 3종(OpenVLA, SpatialVLA, CogACT)에 각각 GIFT 적용해 쌍 비교. 외부 기준선 RT-1, RT-1-X, RT-2-X, Octo, SuSIE, GHIL-Glue, TraceVLA, Diffusion Policy.
- 절제: 관측 구조 주입(정성), Semantic Contrast와 FiLM/cross-attention 대체(Figure 7b), 목표 이미지 품질(Table 4), 편집 시드 안정성(Table 7), 연산 오버헤드(Table 8).
- 실물: ID와 저조도 OOD, 과제·설정별 30회.

## 7. 주요 결과

**Table 1 – SIMPLER Google Robot 평균 (%)**

| 모델 | VM Avg | VA Avg |
|---|---|---|
| OpenVLA → +GIFT | 34.3 → 45.1 | 39.3 → 48.0 |
| SpatialVLA → +GIFT | 52.5 → 62.9 | 50.3 → 57.5 |
| CogACT → +GIFT | 74.5 → **80.5** | 61.3 → **74.7** |

CogACT+GIFT VM 과제별: Grasp 91.0, Move 89.9, Drawer 78.6, PutInDrawer 62.6. VA: 87.6 / 86.9 / 64.7 / 59.7. OpenVLA는 GIFT 후에도 PutInDrawer 0%.

**Table 2 – WidowX VM (%)**: CogACT 51.3 → **67.7**(spoon 80.6, carrot 63.9, stack 43.1, eggplant 83.3), SpatialVLA 33.3 → 43.4, OpenVLA 4.2 → 10.4. GHIL-Glue 40.3, SuSIE 31.9.

**Table 3 – LIBERO (%)**: OpenVLA 84.7/88.4/79.2/53.7 (76.5) → OpenVLA+GIFT 89.5/90.6/88.6/56.0 (**81.2**).

**Table 4 – 목표 품질**: LIBERO-Long에서 degraded 24.0, SuSIE 20.9, Turbo 55.7, GIFT 편집기 56.0, GT 목표 56.5. SIMPLER Stack은 SuSIE 0.0 vs GIFT 편집기 41.1 vs GT 52.5.

**실물 (π0.5 → +GIFT)**: Task1 56.7 → 71.7, Task2 36.7 → 43.3, Task1-OOD 33.3 → 51.7, Task2-OOD 23.3 → 36.7.

## 8. Related Work 상의 위치

목표 이미지 조건화(SuSIE, GHIL-Glue, Gen2Act, GoalVLA)와 VLA 내부 시각적 CoT/월드모델 통합(CoT-VLA, WorldVLA, UP-VLA) 사이에 위치한다. 전자처럼 외부 생성기를 쓰되, ControlNet의 zero-conv를 차용해 기존 VLA를 거의 그대로 두고 조건만 덧붙인다는 점이 차별점이다. 같은 달의 GIFT-Feature(중간 특징 보조 감독)와는 이름만 같고 접근이 전혀 다르다.

## 9. 강점

1. 서로 다른 백본(Prismatic, PaliGemma2)과 헤드를 가진 3개 VLA에서 일관된 이득을 보여 플러그인 성격이 설득력 있다.
2. 1 epoch, 소수 모듈 학습으로 비용이 작고, zero-init 덕에 학습 안정성이 좋다는 주장이 cross-attention 대체 실패(거의 0%)와 대비된다.
3. 목표 이미지 품질 절제(Table 4)로 "좋은 편집기가 필요하다"는 점을 정량화했고, GT 목표 상한과의 격차가 작다.
4. 연산 오버헤드를 FLOPs·메모리·지연·에피소드 비용까지 투명하게 보고.

## 10. 약점 및 한계

1. 베이스 모델의 LIBERO 성능이 낮은 OpenVLA(76.5%) 하나뿐이어서, 최신 강한 정책(95%+)에서도 이득이 유지되는지 알 수 없다.
2. WidowX는 "과제당 3회 평가"로 기술되어 표본이 작고, 본문의 Task 3 기준값(16.4%)이 표(15.0%)와 다르다.
3. Figure 7(b)의 절제 수치(57.3/54.7)와 본문 서술(57.5 → 67.7)이 조금 어긋난다.
4. 목표 이미지는 에피소드 시작 시 한 번만 생성되므로, 장기·다단계 과제에서는 중간 목표 갱신이 불가능하다(LIBERO-Long 56.0%에 그침).
5. 저자 스스로 인정하듯 기존 정책 능력 밖의 과제는 해결하지 못한다(OpenVLA PutInDrawer 0%).
6. 실물은 2과제, 과제당 30회로 규모가 작다.

## 11. 재현 및 확장 아이디어

- π0.5/OpenVLA-OFT 등 강한 베이스에서 LIBERO·RoboTwin 성능 변화 측정.
- 하위 목표를 주기적으로 재생성하는 multi-goal 버전으로 장기 과제 개선.
- 편집 대신 빠른 비디오/이미지 예측 모델로 4.2 s 지연 감소.
- zero-conv 주입을 다른 조건(깊이, 궤적 스케치)에도 적용하는 일반 조건화 어댑터로 확장.

## 12. 총평

GIFT-Goal은 "생성된 목표 이미지를 기존 VLA에 가장 싸게 붙이는 법"에 대한 실용적 답을 제시한다. zero-conv와 목표-관측 차이 특징이라는 단순한 설계로 여러 베이스에서 일관된 향상을 얻었다는 점이 가장 큰 가치다. 다만 강한 최신 베이스와 장기 과제에서의 검증, 평가 표본 크기가 부족해 절대 성능보다는 효율적 적응 기법으로 보는 것이 맞다.

**한 문장 요약**: 편집 모델이 만든 목표 이미지와 관측의 특징 차이를 ZeroConv로 기존 VLA에 주입하고 1 epoch만 미세조정해 CogACT의 SIMPLER(VM 80.5, VA 74.7, WidowX 67.7)와 OpenVLA의 LIBERO(81.2)를 개선한 플러그인 목표 조건화 기법.

<!-- VERIFIED: pdf -->
