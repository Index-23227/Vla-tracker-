# ART: Evolve Vision-Language-Action Model into an Agent with On-the-fly Tool-use

> **한 줄 요약**: π0-FAST(3B)에 **tool-use 토큰**(시각 향상, depth/검출 affordance, 카메라 회전·줌·자세 리셋 등)과 추론 토큰을 주입하되, 이를 **동적으로 켜고 끄는 LoRA**로만 학습해 원래 VLA의 action 생성 능력은 보존하는 tool-injection 미세조정 프레임워크. 30K 합성 tool-use 궤적(AT 데이터셋)으로 1 epoch 학습해 교란된 LIBERO(AT LIBERO) 평균 75%(π0-FAST 39%), Astribot S1 실제 로봇 62%(π0-FAST 43%).

- **arXiv**: 2608.14047 (v3: 2026-09-24, cs.RO)
- **소속**: Astribot, Tsinghua University, Juxi Tech, IntelliFusion, CUHK
- **백본**: π0-FAST 3B
- **코드**: 공개 정보 없음

---

## 1. 배경 및 동기

VLA 연구는 두 갈래다. **모듈형**(RoboTool, RoboScript, CaP 등)은 기능을 도구 API로 묶어 유연하지만 손으로 만든 함수에 묶여 정교한 행동이 어렵다. **end-to-end**(OpenVLA, ECoT, π 계열)는 정밀한 action을 내지만 새 상황에 비싼 사후학습이 필요하고 catastrophic forgetting에 취약하다. LIBERO-Plus가 보였듯 조명, 노이즈, 시점, 초기 자세 교란에 현 VLA는 크게 무너진다.

핵심 질문: **end-to-end VLA의 정밀 action 능력을 유지하면서 외부 도구를 즉석에서 활용할 수 있는가?**

## 2. 문제 정의

action 공간을 A* = A × R × T로 확장한다(R: 언어 추론, T: 도구 사용). 올바른 embodied action은 원시 관측의 교란과 무관하게 향상된 관측에만 의존해야 한다는 가정 아래 목적함수를

P_θ,1(a_t | o_{1:t}) · P_θ,2(a_r,t, a_t,t | õ_{1:t})

로 분해한다. 앞쪽은 일반 VLA 목적, 뒤쪽이 "tool injection" 목적이다.

## 3. Tool Injection 구조

- **Tool token**: 도구 상태는 이진(on/off)이므로 a_t,t ∈ {0,1}^n. RT-2/OpenVLA처럼 어휘의 마지막 N개 토큰을 재활용해 next-token CE로 학습.
- **Adaptive LoRA**: 백본은 고정하고 LoRA만 tool-use 추론에 학습. 추론 시 LoRA를 켠 상태로 추론·도구 토큰을 생성 → 도구 실행으로 관측 향상 → **LoRA 출력을 마스킹**한 원래 VLA가 FAST action 토큰 생성.
- **Action chunk 대응**: 도구 추론은 매 H step(chunk 단위)마다 한 번 수행하고 선택된 도구가 다음 H step 관측에 적용.

## 4. 도구 정의

관측의 세 모달리티에 대응:
1. **Visual enhancement** 10종: 저조도 향상, 디노이징, jitter 보정, deblur 등.
2. **Affordance enhancement**: depth 추정(Metric3D), 객체 검출.
3. **Embodiment enhancement**: 헤드 카메라 회전·줌, 로봇 팔 초기 자세 리셋.

## 5. AT 데이터셋 구축

T3-Agent에서 영감을 받은 3단계 파이프라인으로 기존 LIBERO, Bridge v2, DROID 데이터를 확장:
1. **Task generation**: 시각/embodiment 교란을 무작위 적용, GPT로 단순 지시를 affordance 추론이 필요한 지시로 변환("접시 위에" → "서랍 왼쪽 30 cm").
2. **Tool chain generation**: 교란 해소에 필요한 도구 시퀀스 기록.
3. **Trajectory generation**: GPT가 도구 사용 이유를 포함한 장기 추론 궤적 생성.

새 action 데이터 수집 없이 30K tool-use 궤적을 확보한다.

## 6. 실험 설정

- 1 epoch, 8×A800, batch 24, lr 5e-5, 1k warmup.
- **AT LIBERO**: 조명·노이즈, 위치 기반 지시, 카메라·초기 자세 교란을 넣은 LIBERO. 표준 LIBERO 수치와는 비교 불가.
- **Astribot S1**: 16 DoF 양팔 휴머노이드. 16k pick-and-place 궤적(물체 80종, 용기 10종)으로 baseline FAST를 먼저 학습.

## 7. 주요 결과

**Table 1: tool-use 효과**

| 모델 | AT LIBERO V / A / E / Avg | Astribot S1 V / A / E / Avg |
|---|---|---|
| OpenVLA | 20 / 10 / 7 / 12 | 5 / 5 / 10 / 6.7 |
| π0 | 65 / 15 / 10 / 30 | 40 / 30 / 40 / 37 |
| π0-FAST | 60 / 12 / 45 / 39 | 40 / 30 / 60 / 43 |
| **ART-FAST** | **81 / 62 / 82 / 75** | **70 / 55 / 70 / 62** |

**Table 2: ECoT 대비 (affordance 과제)** — 시각 교란 有: ART 72 vs ECoT 12, 無: ART 62 vs ECoT 58.

**Table 3: 같은 데이터로 도구 추론 없이 사후학습한 FAST 대비** — Vision 71→81, Embodiment 61→62, Affordance 65→82.

## 8. Related Work 상의 위치

TIGeR(기하 계산 도구), VLA²(웹 지식 검색)처럼 end-to-end VLA에 도구를 결합한 선행 연구가 있지만 특정 과제에 한정된다. ART는 다중 모달 도구를 장기 다중 과제 궤적에 걸쳐 쓰는 방향이며, ECoT류 "추론만" 모델과 달리 추론 결과가 관측 자체를 바꾼다는 점이 차별점이다.

## 9. 강점

- LoRA 마스킹으로 tool 추론과 action 생성을 깔끔히 분리 → 원래 action 품질 보존과 forgetting 회피를 구조적으로 보장.
- 기존 데이터만으로 tool-use 데이터를 합성하는 저비용 파이프라인.
- 도구 추가가 모듈식으로 확장 가능.
- 동일 데이터 end-to-end 사후학습 대비(Table 3)로 "데이터 증가 효과"와 "도구 효과"를 일부 분리.

## 10. 약점 및 한계

- 평가가 전부 **자체 제작 교란 벤치마크**(AT LIBERO, AT Astribot)이며 표준 LIBERO/LIBERO-Plus 수치가 없다.
- 과제당 trial 수, seed, 분산 등 통계 정보가 부족하고 결과가 모두 정수 % 단위.
- Table 3에서 Embodiment는 61→62로 사실상 차이 없음.
- 교란이 도구가 해결하도록 설계된 유형과 정확히 일치 → 도구 목록 밖 교란에 대한 일반화는 불명.
- 외부 도구(Metric3D, 검출기, 이미지 향상) 호출에 따른 지연·추론 비용 분석 부재.
- 본문이 부록의 데이터셋 통계를 언급하지만 제공 PDF에서는 확인되지 않는다.

## 11. 재현 및 확장 아이디어

- LIBERO-Plus 공식 교란 세트에서 π0-FAST vs ART 비교.
- flow-matching 백본(π0, π0.5)으로 LoRA tool 분기 이식.
- tool 선택을 RL로 최적화해 GPT 합성 궤적 의존도 감소.
- 도구 호출 빈도/지연과 성공률 간 trade-off 측정.

## 12. 총평

ART는 "모듈형 vs end-to-end" 이분법 사이에서, **LoRA로 분리된 도구 추론 두뇌**를 VLA에 덧붙이는 실용적 설계를 제시한다. π0-FAST 위에 LoRA를 실제로 학습한 자체 정책이며(백본 고정이어도 학습된 LoRA가 정책의 도구 사용 행동을 결정), 시뮬레이션과 실제 휴머노이드에서 정량 결과를 보고하므로 트래커 대상이다. 다만 모든 평가가 도구에 맞춰 설계된 자체 교란 벤치마크라는 점에서 수치의 외부 비교 가능성은 낮다.

**한 문장 요약**: 흐린 눈엔 안경을, 헷갈리는 지시엔 depth 센서를 — VLA가 스스로 도구를 골라 관측을 고친 뒤 원래 실력대로 행동하게 하자.

<!-- VERIFIED: pdf -->
