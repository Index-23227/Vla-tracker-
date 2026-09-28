# ProWAM: Learning to Use Imagination — Progress-Conditioned Future Utilization for World Action Models

> **한 줄 요약**: World Action Model(WAM)이 상상한 미래 video latent를 실행 단계와 무관하게 고정적으로 쓰는 문제를, 자기지도 학습된 진행도 표현(SS-DTPE)과 그에 조건화된 계층적 attention 변조(HPIM)로 해결. Fast-WAM 계열 Wan2.2-5B 기반 joint WAM 위에서 LIBERO 평균 99.1%, RoboTwin 2.0 평균 93.4%(embodied 사전학습 없음), 실제 Galaxea R1 Lite 장기 과제 72%.

- **arXiv**: 2609.06578v1 (2026-09-06, cs.CV)
- **소속**: Harbin Institute of Technology (Shenzhen), Great Bay University, UESTC
- **백본**: Wan2.2-5B video DiT + 1B action expert (MoT, shared attention)
- **코드**: https://github.com/JiuTian-VL/ProWAM

---

## 1. 배경 및 동기

VLA는 현재 관측에서 행동으로 직접 사상하며, WAM은 여기에 미래 시각 동역학 예측을 결합해 "상상한 미래"를 행동 생성의 단서로 쓴다. 기존 WAM 연구는 미래를 *어떻게 잘 모델링할지*(표현, 공간 grounding, 메모리, 효율)에 집중했지만, 상상한 미래를 *언제·얼마나 쓸지*는 거의 다루지 않았다. 저자들은 탐색·이동·접근 단계에서는 미래 단서가 유용하지만, 파지·배치 같은 접촉 단계에서는 즉각적인 관측 피드백이 더 중요하다는 직관에서 출발한다.

## 2. 진단 실험: 상상의 비균일한 효용

Galaxea R1 Lite에서 5개 장기 과제(각 100 trial, 총 500 trial)로 Fast-WAM 방식의 joint WAM을 분석한다.
- **Finding 1 (inter-progress)**: action stream이 미래 latent에 접근하지 못하게 한 action-only 변형은 search/transit·approach 단계 실패 비율이 더 높다. 즉 미래 latent는 접촉 전 단계에서 특히 유용하다.
- **Finding 2 (intra-progress)**: 미래 latent를 near/mid/long horizon 그룹별로 교란하면, 접촉 전 단계는 mid·long horizon에, 접촉 단계는 near-future에 더 민감하다.
- 결론: **실행 진행도(progress)**가 상상 활용을 조절하는 핵심 중간 신호다.

## 3. 전체 구조

Fast-WAM을 따라 Wan2.2-5B의 video DiT, text encoder, VAE를 재사용하고 1B action expert(hidden 1024, action horizon 32)를 붙여 Mixture-of-Transformers(공유 attention)로 구성한다. 학습 목적은 L_act + L_vid(rectified flow matching). 여기에 progress-first 설계로 (1) SS-DTPE가 진행도 토큰을 만들고 (2) HPIM이 action→future attention 경로만 변조한다. 다른 attention 경로는 그대로 둔다.

## 4. SS-DTPE (Self-Supervised Dual-Temporal Progress Encoder)

- **Short-term Action-Observation Encoder**: 언어 조건 하에서 직전 행동과 현재 관측의 상호작용을 모델링해 "직전 행동이 무엇을 바꾸었는가"를 진행도 단서로 포착.
- **Long-term Progress Aggregator**: Mamba 기반 recurrent 경로(상태 차원 16)가 압축된 진행 이력을 유지해 시각적으로 비슷하지만 단계가 다른 상태를 구분하고, Progress Attention이 이 기억으로 단기 증거를 질의한다. 진행도 토큰은 8개.
- **자기지도 목적**: 수동 진행도 주석 없이 시연의 의미적 진행(Semantic Progress Alignment, 두 성질 i/ii)과 시간 순서(Temporal Structure Learning, 대조 학습·단조 순서 항)를 활용.

## 5. HPIM (Hierarchical Progress-Conditioned Imagination Modulation)

- **Inter-Progress Global Gate** d_t ∈ [0,1]: 실행 단계별로 미래 의존도를 조절. gripper 상태 전이·접촉 변화 등으로 자동 검출한 keyframe 앵커 주변은 0(보수적), 나머지는 1로 약한 라벨을 만들어 BCE로 감독(λ_gate = 0.05).
- **Intra-Progress Future-Latent Relevance** ω: 토큰 단위 라벨 없이 WAM 목적을 통해 end-to-end로 학습, 같은 단계 내 개별 미래 latent의 기여를 차등화.
- 층별 스케줄(중간층 강조)로 attention log-bias를 적용.

## 6. 학습 및 추론

- Stage 1: SS-DTPE 단독 학습, 5 epoch, H100 4장.
- Stage 2: SS-DTPE 고정 후 WAM+HPIM을 L_act + L_vid + λ_gate L_gate로 10 epoch, H100 8장. AdamW lr 1e-4.
- 추론: 에피소드 시작 시 진행 메모리 초기화, 매 재계획 스텝마다 관측·직전 행동·지시로 진행도 갱신 → HPIM 변조 → receding-horizon으로 action chunk 실행. 오프라인 진행도 주석이나 캐시 불필요.

## 7. 주요 결과

**LIBERO (Table 1, 40 task 2000 trial)**

| 방법 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| MaskWAM | 98.8 | 100.0 | 98.2 | 96.4 | 98.4 |
| **ProWAM** | **99.6** | **100.0** | **98.8** | 97.8 | **99.1** |

**RoboTwin 2.0 (Table 2)**: Clean 93.9 / Randomized 92.8 / Avg 93.4. FlowWAM(92.5), LingBot-VA(92.2), Fast-WAM(91.9)보다 높고, embodied 사전학습 없이 달성.

**VLABench (Table 3)**: SR 68.4 / IS 85.4 / PS 81.2 (π0.5: 65.4/80.4/77.8, Fast-WAM*: 58.4/81.2/74.5).

**RoboEval (Table 4)**: SR 0.375 / TP 0.494 (Fast-WAM* 0.313/0.404, ACT 0.278/0.454).

**Mikasa-Robo (Table 5)**: 평균 50.8% (MemoryVLA++ 44.4%, MemoryVLA 41.2%), InterceptMedium 68%.

**실제 로봇**: 장기 과제 평균 full-task 성공률 Galaxea R1 Lite 72%(π0.5† 64, Fast-WAM† 56), AgileX Cobot Magic 71%(Fast-WAM† 57). 일반화·강건성 과제에서는 69% / 68%.

## 8. Ablation

- **구성 요소 (Table 8)**: baseline 대비 SS-DTPE만 추가 시 LIBERO-Long 94.8→96.2, HPIM만 96.6, 전체 97.8. 실제 Long./Gen.은 56/58 → 72/69.
- **Aggregator (Table 9)**: 없음 < LSTM < Transformer < Mamba; 실제 Long.에서 58→60→62→72.
- **자기지도 목적 (Table 10)**: 모두 제거 시 LIBERO-Long 94.4, 실제 54/57 → 전체 97.8, 72/69.
- **HPIM (Table 11)**: w/o Inter. Long 95.4 / 실제 58, w/o Intra. 95.8 / 62.
- **진행도 소스 (Table 14)**: 정규화 시간만 쓰면 LIBERO-Long 95.2, 실제 56/59.
- **Gate 감독 (Table 15)**: 감독 제거 시 실제 59/61, 라벨 20% 교란에도 68/66 유지.

## 9. 강점

- "미래를 얼마나 믿을 것인가"라는 WAM의 활용 측면 문제를 진단 실험으로 먼저 보이고 방법을 설계한 논리 흐름이 명확하다.
- 5개 시뮬레이션 벤치마크 + 2개 실제 플랫폼이라는 폭넓은 평가, 특히 진행도 지표(PS, TP)와 메모리 벤치마크에서 이득이 뚜렷하다.
- 변조는 action→future attention 경로에만 적용되어 기존 WAM 백본을 보존하며, 진행도 주석이 필요 없다.
- 절제 실험이 매우 체계적이다(구성 요소, aggregator, 목적 함수, 경로, 진행도 소스, 약한 라벨 강건성).

## 10. 약점 및 한계

- LIBERO는 이미 포화 영역(99.1 vs 98.5)이라 차이가 작고, 단일 수치의 seed 분산이 보고되지 않는다.
- gate 약한 라벨은 gripper 전이·접촉 등 휴리스틱에 의존하며, 과제별 저차원 단서가 필요할 수 있다.
- 실제 로봇 baseline 대부분이 저자 재현(†)이고, 추론 지연·연산량(5B video DiT + 1B expert + 인코더) 보고가 부족하다.
- 진단 분석(Fig. 2)은 그림 위주이며 정량 표로 제시되지 않는다.

## 11. 재현 및 확장 아이디어

- 코드 공개됨: Fast-WAM 체크포인트에 SS-DTPE/HPIM을 추가하는 형태로 재현 가능성이 높다.
- 진행도 토큰을 VLA(비-WAM)의 memory/subgoal 모듈에 이식하거나, RL 보상 shaping의 진행도 추정기로 재사용하는 방향.
- 약한 gate 라벨 대신 VLM 기반 단계 분할이나 contact force 센서를 쓰는 변형.
- 불확실성 추정(미래 예측 분산)과 진행도 gate를 결합해 "신뢰할 수 없는 상상"을 명시적으로 걸러내는 확장.

## 12. 총평

ProWAM은 WAM 연구의 초점을 "더 좋은 미래 예측"에서 "미래 예측의 적응적 활용"으로 옮긴 논문이다. 진행도 표현이라는 명시적 중간 변수를 도입해 LIBERO 99.1%, RoboTwin 2.0 93.4%, Mikasa-Robo 50.8% 등 폭넓은 벤치마크에서 최고 수준을 달성했고, 특히 장기·메모리 의존 과제에서 설득력 있는 개선을 보였다. 연산 비용과 약한 라벨 설계의 일반성은 추가 검증이 필요하지만, WAM 계열의 실용적인 다음 단계로 평가할 만하다.

<!-- VERIFIED: pdf -->
