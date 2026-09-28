# PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control

> **한 줄 요약**: 사전학습 MLLM(Qwen3.5-9B, "Ponder")의 네이티브 causal context를 에피소드 메모리로 그대로 쓰고, 연속 벡터 "cognition"과 그 나이(age)를 비동기로 π0.5/GR00T 액션 모델("Pounce")에 전달하는 이중 시스템을 end-to-end로 공동 학습. RoboMME 60.83%(1× data, FrameSamp+Modul 44.51%), 9× data 75.54%, RoboCasa-DC 12.5%, 실물 ALOHA 4과제 평균 60.98%.

- **arXiv**: 2608.24115 (v1 2026-08-25, v2 2026-09-17)
- **소속**: MAUM.AI, 서울대학교, Georgia Tech — Suhwan Choi, Jaeyoon Jung (공동 1저자), Sungkyung Kim, Yunsung Lee, Youngjae Yu
- **백본**: Ponder = Qwen3.5-9B, Pounce = π0.5(3.6B) 또는 GR00T N1.5(3B)

---

## 1. 배경 및 동기

가려진 물체, 이전 사건, 지시 대상, 시연된 절차처럼 현재 관측에 없는 정보가 필요한 조작 과제가 많다. MLLM은 긴 시각 이력과 few-shot 예시를 통합하는 능력이 있지만 VLA는 이를 에피소드 메모리로 쓰지 않고, 기존 연구는 전용 메모리 모듈·검색·시연 인코더·계획기를 따로 설계한다. RoboMME에서 π0.5는 이력 없이 17.93%, 과거 행동을 넣어도 19.73%로 사람(90.50%)과 격차가 크다. 저자들은 "사전학습 MLLM의 causal context 자체를 에피소드 컨텍스트 엔진으로 쓸 수 있는가"를 묻는다.

## 2. 문제 정의

- Ponder(System 2): 지시, 선택적 시연 D_1:n, 누적 관측 O_1:t, 자체 생성 텍스트를 append-only 컨텍스트로 유지하고 매 질의마다 K개 carrier 토큰의 최종 hidden state를 cognition C_t로 출력.
- Pounce(System 1): 지시, 현재 관측, proprioception + 최신 cognition과 그 age Δ를 받아 h-step 액션 청크 예측.
- 두 시스템은 독립 클록에서 동작하며, 인터페이스는 cognition + age만 통과.

## 3. 방법

- **Carrier 토큰**: 학습 가능한 입력 임베딩의 carrier K개(학습·평가는 K=1)를 매 질의 끝에 추가, 최종 hidden state를 풀링 없이 cognition으로 사용.
- **선택적 grounding**: 전이 토큰 T_Yes/T_No 예측 → 전이 시 서브골 텍스트 S_t 생성, 시연 에피소드에서는 첫 실행 전이 때 demonstration reasoning DR 생성.
- **Pounce 조건화**: P_j = [e_Δ, W_c·C̃_1..K] 프리픽스. 준비된 cognition이 없으면 학습된 null cognition(0 초기화).
- **비동기 스케줄**: latest-ready 규칙으로 가장 최근 완료된 Ponder 질의를 선택하고, 원 관측 시점 기준 age를 계산.
- **추론 최적화**: 사전할당 StaticCache KV 캐시로 새 토큰만 인코딩(16K 컨텍스트 한도), Pounce는 fused Triton 커널로 p50 142 → 25ms; cognition-only 갱신 p50 78ms.

## 4. 학습 절차

- 두 시스템을 사전학습 체크포인트에서 초기화하고 bridge 사전학습 없이 **end-to-end 공동 학습**. L = w1·L_fm + w2·L_ground.
- action gradient는 cognition을 통해서만 Ponder로 흐르며, 공동학습 shortcut을 막기 위해 0.5배 스케일.
- RoboMME: 4,320 episode-step, lr 2e-5, (w1,w2)=(1,0.1), SigLIP 동결·PaliGemma LM과 action expert 학습, 20-step 청크. 학습 중 Pounce 평균 100ms 간격, Ponder 평균 1s, 지연 300ms 시뮬레이션.
- RoboCasa-DC: 4,000 step, lr 1e-5, Ponder 전체 미세조정(LoRA 없음), 액션 감독만(w2=0), cognition dropout 15%.

## 5. 데이터

- **RoboMME**: 16과제(Counting / Permanence / Reference / Imitation 각 4개). base 1,587 에피소드, 9×는 새로 수집한 14,400 에피소드. grounding 주석은 시뮬레이터에서 도출, DR은 템플릿 생성.
- **RoboCasa-DC**: Category-Balanced / Cross-Embodiment Demonstration 설정, 학습 19과제·held-out 5과제, 약 1,900 학습 / 500 held-out 시연-실행 쌍.
- **실물**: Trossen ALOHA Stationary(오른팔만), 710 텔레오퍼레이션 에피소드(사람 주석 서브골 포함, DR 없음), held-out 59 장면.

## 6. 실험 설정

- RoboMME 기준선: π0.5(±과거 행동), SAM2Act+, SimpleSG/GroundSG+QwenVL, MemER, FrameSamp+Modul(대부분 RoboMME 논문 보고치), 9× FrameSamp+Modul은 저자 재학습.
- RoboCasa-DC 기준선은 SeeTraceAct 논문 보고치(Vid2Robot, UniSkill, ViVLA, SeeTraceAct).
- 평가: RoboMME 3회 × 과제당 50 에피소드(1 Hz 동기, 지연 300ms 고정), RoboCasa-DC 5회 × 50, 실물은 사람 판정 2분 제한.

## 7. 주요 결과

**Table II – RoboMME (%)**

| Method | Counting | Permanence | Reference | Imitation | Avg |
|---|---|---|---|---|---|
| π0.5 | 28.78 | 17.00 | 17.16 | 8.78 | 17.93 |
| MemER | 48.83 | 53.16 | 38.00 | 29.50 | 42.38 |
| FrameSamp+Modul | 65.22 | 25.11 | 36.33 | **51.39** | 44.51 |
| **PonderPounce (1×)** | **74.67** | **62.83** | **72.17** | 33.67 | **60.83** |
| FrameSamp+Modul (9×) | 86.00 | 24.50 | 58.00 | 63.00 | 57.88 |
| **PonderPounce (9×)** | 81.33 | 80.17 | 92.67 | 48.00 | **75.54** |
| Human | 88.50 | 91.00 | 93.00 | 89.50 | 90.50 |

Permanence·Reference에서 격차가 가장 크고(9×에서 +55.67, +34.67pp), Imitation과 9× Counting은 FrameSamp+Modul이 우세.

**Table IV – RoboCasa-DC**: 12.5±0.9% (SeeTraceAct 11.6, UniSkill 11.2), 추론 시 cognition을 null로 바꾸면 8.6%.

**Table IX – 실물 ALOHA**: PutFruits 11/14, TrackCube 10/16, RepickBlock 6/14, DrawPattern 9/15, 평균 60.98% (FrameSamp+Modul 40.67%, π0.5 23.99%).

**통제 실험**
- 실행 이력 제거(동일 9B Ponder·감독): 60.83 → 26.21% (Table V).
- DR 제거 48.21%, LM-head grounding 전부 제거 27.96%, 별도 학습 Ponder가 서브골 텍스트만 넘기는 방식 59.96% (Table III).
- Ponder 0.8B 54.12% vs 9B 60.83%, 무작위 초기화 9B 0.00% (Table VI).
- 서브골 전이 때만 cognition 전달: age를 300ms로 고정하면 1.83%, 실제 age 제공 시 22.42% (Table VII) → 매 질의 갱신과 age 신호가 중요.

## 8. Related Work 상의 위치

- 전용 메모리(FrameSamp+Modul, SAM2Act+, MemER, MemoryVLA, MEM, RoboTTT)와 달리 **MLLM의 KV 캐시 컨텍스트 자체를 메모리로** 사용.
- ICRT처럼 시연과 롤아웃을 causal context에 두지만, 별도 사전학습 MLLM이 cognition을 만들고 액션 모델이 소비하는 구조.
- Helix류 비동기 연속 채널 이중 시스템을 계승하되, System 2가 최신 관측만이 아니라 전체 에피소드 이력·시연을 유지하고 cognition age를 명시적으로 전달. OpenHelix가 지적한 채널 붕괴 문제를 grounding 감독과 gradient 스케일링으로 완화.

## 9. 강점

1. **통제 실험이 탄탄함**: 동일 백본·감독에서 이력만 제거한 대조, System 2 스케일링, 갱신/age 개입으로 주장을 직접 검증.
2. 메모리 의존 과제에서 큰 향상(RoboMME 1× +16.32pp, 9× +17.66pp).
3. 인터페이스 고정으로 System 2만 교체·확장 가능(0.8B → 9B), 두 종류 액션 백본(π0.5, GR00T N1.5)에서 동작.
4. 실제 비동기 실행 지연(표 VIII)과 추론 최적화 수치를 상세히 보고.
5. 실물에서 시뮬레이션과 일관된 경향(Permanence 과제 TrackCube 최대 향상).

## 10. 약점 및 한계

1. **grounding 주석 의존**: LM-head grounding 없이 학습하면 27.96%로 모든 기준선보다 낮아진다 — 이득의 상당 부분이 시뮬레이터 유래 전이·서브골·DR 감독에 기대고 있다.
2. **Imitation 약세**: 1 Hz 갱신의 거친 시간 해상도로 경로 모방 과제에서 FrameSamp+Modul에 뒤진다.
3. **비용**: 9B MLLM + 3.6B 액션 모델, 실물에서 H100 두 장. 저자도 컨텍스트 엔진의 학습·추론 비용을 한계로 명시.
4. **RoboCasa-DC 절대 성능이 낮음**(12.5%)이며 기준선과의 차이(+0.9pp)가 작고 기준선 수치는 타 논문 보고치.
5. 각 구성은 단일 체크포인트로 평가, 실물은 과제당 14–16 trial로 소규모.
6. 표준 벤치마크(LIBERO, SimplerEnv, CALVIN) 결과가 없어 일반 조작 성능 비교 불가. 코드 공개 언급 없음.

## 11. 재현 및 확장 아이디어

- 과거 이미지·생성 텍스트·이전 cognition 각각의 기여를 분리하는 이력 구성요소 절제(저자 제안).
- 적응적 cognition 전달 주기, 증류된 소형 Ponder로 비용 절감.
- VLM 자동 라벨링으로 grounding 주석 비용을 낮추는 실험.
- 16K 이상 긴 에피소드(모바일 조작, 다중 방 탐색)에서 컨텍스트 한도 문제 검증.

## 12. 총평

PonderPounce는 "메모리 모듈을 새로 설계하지 말고 사전학습 MLLM의 컨텍스트를 그대로 쓰라"는 명확한 주장을 비동기 이중 시스템으로 구현하고, 이력 제거·스케일링·갱신 개입 등 설득력 있는 통제 실험으로 뒷받침한다. 다만 성능이 grounding 감독에 크게 의존하고, 계산 비용이 크며, 모방형 과제와 표준 조작 벤치마크에서의 검증은 부족하다.

**한 문장 요약**: Qwen3.5-9B의 causal context를 에피소드 메모리로 재활용해 π0.5 액션 모델과 비동기 공동 학습한 이중 시스템으로 RoboMME 60.83%(9× 75.54%)와 실물 60.98%를 달성한 메모리 의존 조작용 VLA.

<!-- VERIFIED: pdf -->
