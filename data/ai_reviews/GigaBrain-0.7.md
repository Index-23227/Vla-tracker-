# GigaBrain-0.7: Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture

> **한 줄 요약**: PaliGemma2(3B) + 0.5B Action Expert의 dual-stream MoT VLA(System 1)에 서브태스크 계획(System 2)과 GigaWorld-1 기반 World Value Model(System 3: 서브골 이미지 + 가치/advantage)을 결합하고, 16종 로봇·37,257시간 궤적과 약 2.7억 VL 샘플로 one-stage 사전학습한 GigaAI의 파운데이션 모델. RoboTwin 2.0 Co-Train Overall 67.35(Hard 67.9, π0.5 46.0), EBench SR 33.30, 실물 오프라인→온라인 RL로 4개 과제 평균 30.0% → 100%.

- **arXiv**: 2608.15875v1 (2026-08-16, cs.RO)
- **소속**: GigaAI (GigaBrain Team)
- **코드**: https://github.com/open-gigaai/giga-brain-0 (학습 코드·가중치 공개 예정으로 표기)
- **백본**: PaliGemma2 3B + Action Expert 0.5B (System 1), GigaWorld-1 확장 World Value Model (System 3)

---

## 1. 배경 및 동기

VLA는 데이터·모델 규모를 키우면서 실물 실행과 다운스트림 적응 능력이 좋아졌지만, 로봇 데이터는 embodiment·행동 공간·실행 조건이 제각각이라 정렬 없이 섞으면 오히려 간섭이 생긴다. 또 대부분의 VLA는 "관측 → 행동"의 반응형 매핑에 머물러 미래 상태 예측이나 진행도 평가 기능이 약하다. 저자들은 GigaBrain-0(월드 모델을 데이터 엔진으로), GigaBrain-0.5M*(RAMP로 월드모델 조건 정책 학습)의 뒤를 이어, 데이터 큐레이션·사전학습·계획·예측·실행·경험 기반 개선을 하나의 학습 시스템으로 묶는 것을 목표로 한다.

## 2. 문제 정의

- 입력: 현재 관측 + 짧은 시각 이력, 언어 지시, proprioceptive state + Robot ID, System 2의 CoT/서브태스크, System 3의 서브골 이미지 g_t와 advantage 조건 A_t ∈ {0,1}.
- 출력: flow matching으로 생성되는 연속 action chunk (훈련 중에는 이산 action 토큰 NTP 경로도 병행).
- 목표: 16개 로봇 형태에 걸친 단일 정책이 zero-shot, 언어 추종, 후학습 성공률, 경험 기반 개선 모두에서 π0.5 등 기존 모델을 넘도록 하는 것.

## 3. 방법: 3-System 아키텍처

- **System 1 (Action & Control)**: PaliGemma2(3B)와 Action Expert(0.5B)를 MoT로 결합. VL 스트림은 자기 토큰에만 causal attention, Action Expert는 VL+AE 전체에 양방향 attention, FFN은 분리. 시각 인코더에 Temporal-Spatial Block을 넣어 과거 프레임 정보를 현재 프레임 토큰으로 융합한 뒤 과거 토큰은 버린다(토큰 수는 단일 프레임 수준 유지). Action Expert 본체는 embodiment 간 공유, 입출력 사영만 embodiment별. 회전은 6D 표현.
- **System 2 (Understanding & Planning)**: 현재·과거 관측을 해석해 진행도를 추적하고 긴 지시를 계층적 프롬프트로 서브태스크로 분해.
- **System 3 (Prediction & Evaluation)**: GigaWorld-1을 로봇 비디오로 추가 사전학습 → MoT World Value Model로 확장해 (a) 서브태스크 horizon 길이의 미래 영상을 생성해 마지막 프레임을 서브골 이미지로, (b) 진행도 값 V_t를 추정해 이진 advantage A_t로 변환해 System 1 프롬프트에 주입. 후학습 동안 System 3는 동결.
- **Soft Knowledge Insulation**: FM 손실의 VLM 방향 gradient를 α_KI(0<α_KI<1)로 감쇠, NTP 감독은 그대로 — 완전 차단(KI)이 아닌 부분 적응.

## 4. 학습 절차

1. **VLA 사전학습 (System 1+2 공동)**: L_VLA = L_NTP + L_FM. NTP 타깃은 서브태스크 설명, 이산 action, 캡셔닝·VQA·공간추론·grounding·affordance 등. 이력 프레임 랜덤 드롭, 태스크/서브태스크 지시 혼합 감독, embodiment별 유효 마스크로 무효 차원 제외.
2. **World model 사전학습**: 로봇 중심 비디오 사전학습 → 가치 학습(궤적 완료 주석 기반 진행도).
3. **월드모델 조건 후학습**: 동결된 System 3 출력을 조건으로 System 1·2만 최적화. 추론 시 positive progress 조건으로 생성.
4. **경험 강화 학습**: HIL 개입 데이터로 오프라인 advantage-weighted 정제 + 오류 인지 진행도 판별기, 이후 온라인에서 서브태스크 성공 라벨 + 판별기로 dense reward를 만들어 경량 actor-critic 루프 실행. 모든 단계에서 동일한 System 1 정책을 유지.

## 5. 데이터

- 궤적 코퍼스 37,256.98시간: 실제 로봇 20,535.65h(55.12%, 16종, 1,810,101 에피소드), UMI 8,251.83h(22.15%), EGO 2,862.36h(7.68%), 시뮬레이션 1,453.92h(3.90%), 월드모델 생성 4,153.22h(11.15%) (Table 1).
- VLM 이미지-텍스트/QA 271,976,674 샘플.
- LeRobot v3.0 변환, embodiment 간 state-action 정규화, LLM 기반 지시 재작성·서브태스크 주석(Qwen3.6-27B), 다단계 품질 관리.

## 6. 실험 설정

- 스케일링 연구: VLM 백본(PaliGemma2 3.5B / Qwen3.5 5B / Gemma 4 8.5B), VLM–Action Expert 결합 방식(dual stream / last-layer / multi-layer cross attention), 데이터 규모·출처.
- 시뮬레이션: RoboTwin 2.0(공식 Co-Train: 50개 과제 × 과제당 clean demo 50개로 단일 정책), EBench, RoboColiseum(Real2Sim2Real).
- VL 평가: MiMo-Embodied 13개 벤치마크.
- 실물: AgileX PiPER/PiPER-X, 사내 Maker H01 휴머노이드에서 언어 추종 6과제, 복합 조작 5/7과제, 경험 RL 4과제.

## 7. 주요 결과

**Table 9 – RoboTwin 2.0 (Co-Train, %)**

| Model | Easy | Hard | Overall |
|---|---|---|---|
| X-VLA | 68.0 | 20.9 | 44.45 |
| X-WAM | 70.0 | 25.8 | 47.90 |
| π0.5 | **70.7** | 46.0 | 58.35 |
| **GigaBrain-0.7** | 66.8 | **67.9** | **67.35** |

Easy에서는 π0.5보다 낮지만 domain-randomized Hard에서 21.9pt 앞서 Overall 1위.

**Table 10 – EBench**: SR 33.30 / Score 46.1 (π0.5 28.08 / 42).
**Table 11 – RoboColiseum**: 지시 추종 .8166, 공간 추론 .4729, 강건성 .6800, 일반 조작 .6092로 4개 차원 모두 1위.
**Table 8 – MiMo-Embodied (MiMo 데이터 투입 전)**: Spatial .5215, Affordance .3669, Overall .4621 (G0.5-base .3916). MiMo 데이터를 섞은 최종 체크포인트는 0.5704지만 held-out이 아니라고 저자 스스로 명시.

**Table 6/7 – 실물 후학습 평균 (%)**

| 설정 | π0.5 | GigaBrain-0.7 |
|---|---|---|
| PiPER 언어 추종 (6과제) | 88.8 | **91.5** |
| H01 언어 추종 (6과제) | 75.2 | **84.2** |
| PiPER 복합 조작 (5과제) | 76.6 | **84.9** |
| H01 복합 조작 (7과제) | 45.2 | **74.1** |

**Table 12 – 경험 기반 RL**: Link Installation 20→40→100, Gift Box Packing 80→90→100, Cable Tie 0→40→100, Bearing Installation 20→60→100 (SFT → Offline RL → Online RL). 평균 30.0 → 57.5 → 100.

**구조 절제 (Table 4/5)**: PaliGemma2 dual stream이 Clean Desk 50 / Fruit 88 / Shirt Folding 30%로, last-layer cross attention(20/40/0)보다 훨씬 강하지만 추론 0.221s vs 0.073s. Gemma 4(8.5B)는 구조화된 과제에서 강하나 셔츠 개기 0%.

## 8. Related Work 상의 위치

- π0/π0.5/π0.7, GigaBrain-0, Wall-OSS-0.5, HyVLA-0.5, Xiaomi-Robotics 계열과 같은 "병렬 VLM–Action Expert MoT" 계보.
- π0.5의 단계적 레시피(사전학습은 이산 토큰, 후학습에 FM 도입)와 달리 NTP+FM을 사전학습 전 구간에서 공동 최적화 — GigaBrain-0, Wall-OSS-0.5와 같은 방향.
- 월드 모델을 데이터 엔진(GigaBrain-0)·조건 신호(GigaBrain-0.5M*)로 쓰던 흐름을 서브골 이미지 + 가치의 이중 인터페이스로 확장.

## 9. 강점

1. **규모와 이질성**: 37K시간, 16개 embodiment, UMI/EGO/월드모델 데이터까지 통합한 학습과 출처별 기여(Fig. 8c) 분석.
2. **Hard 설정 강건성**: RoboTwin 2.0 Hard 67.9는 Easy(66.8)와 거의 같아 domain randomization에 대한 강건성이 두드러진다.
3. **휴머노이드에서의 큰 격차**: Maker H01 복합 조작 평균 74.1 vs π0.5 45.2.
4. **라이프사이클 통합**: 사전학습→월드모델 조건 후학습→오프라인/온라인 RL이 동일한 System 1 정책 위에서 이어진다.
5. 리더보드 스냅샷과 접근 날짜를 부록에 명시하고, MiMo 오염 가능성을 스스로 구분 보고.

## 10. 약점 및 한계

1. **비교 공정성**: 실물 평가는 사내 플랫폼(H01) 중심이며, 기준선(π0.5, G0.5, Xiaomi-Robotics-1)의 후학습 레시피 동일성 여부가 명확하지 않다. G0.5의 H01 결과(평균 8.3)처럼 극단적으로 낮은 수치는 이식 문제일 가능성이 있다.
2. **RL 결과의 표본 크기**: Table 12의 100%들은 과제당 시도 횟수가 명시되지 않아(퍼센트가 10/20 단위로 보임) 통계적 신뢰도가 제한적. 온라인 단계는 사람 교정이 포함된다.
3. **System 3 절제가 그림 중심**: 서브골·가치 조건의 효과는 Fig. 11 수치 위주이며 과제 3개, 규모가 작다.
4. **"창발"의 근거**: 검증 손실 감소와 일부 과제 성공률 상승을 capability emergence로 해석하나, 정량적 임계 현상 분석은 부족.
5. 표준 LIBERO/SimplerEnv/CALVIN 결과가 없어 기존 리더보드와의 직접 비교가 어렵다. 가중치는 "공개 예정" 상태.

## 11. 재현 및 확장 아이디어

- System 3 없이(=System 1+2만) RoboTwin/EBench 성능을 보고해 월드 모델의 순기여를 분리.
- Soft KI의 α_KI 스윕과 KI(완전 차단)·무차단과의 VL 성능/조작 성능 트레이드오프 정량화.
- EGO/UMI 비율을 더 늘린 스케일링 곡선, embodiment 수에 따른 전이 곡선.
- 경험 RL을 더 많은 과제·시도 수로 확장하고, 사람 교정량 대비 개선 효율을 보고.

## 12. 총평

GigaBrain-0.7은 "큰 MoT VLA + 계층 계획 + 월드 가치 모델 + 경험 RL"을 하나의 파이프라인으로 엮은 산업형 파운데이션 모델 보고서다. RoboTwin 2.0 Hard와 휴머노이드 복합 조작에서 π0.5 대비 뚜렷한 우위를 보이지만, 핵심 실물 평가가 사내 플랫폼 중심이고 구성요소별 순기여 분석은 제한적이다.

**한 문장 요약**: 37K시간·16개 embodiment로 one-stage 사전학습한 3-System VLA로 RoboTwin 2.0 Co-Train Overall 67.35(Hard 67.9)와 H01 휴머노이드 복합 조작 74.1%를 달성한 대규모 시스템 논문 — 개별 설계의 순효과 검증은 상대적으로 얇다.

<!-- VERIFIED: pdf -->
