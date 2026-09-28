# FlashVLA — Streaming Action Decoding for Fast and Asynchronous VLA Inference

- arXiv: 2608.27384 (2026-08-27)
- 소속: UC San Diego, MIT (Zekai Li, Jiaming Tang, Zhijian Liu)
- 코드: https://github.com/z-lab/flashvla
- 참고: 트래커의 Realtime-VLA-FLASH(2605.13778, speculative inference)와는 다른 논문이다.

## 1. 한 줄 요약

FlashVLA는 flow-matching VLA의 행동 디코딩을 "청크 하나를 10스텝으로 따로 디노이징"하는 방식에서, **서로 다른 노이즈 수준의 청크 N개를 버퍼에 두고 chunk-wise causal attention으로 함께 한 스텝씩 전진시키는 스트리밍 디코딩**으로 바꾼다. π0.5를 multi-buffer joint fine-tuning으로 재학습해, LIBERO 비동기(d=1)에서 성공률 96.9 → 97.8%와 2.43배 per-step 속도 향상, RoboTwin 2.0 동기 평균 86.0 → 90.5%를 보고한다.

## 2. 문제 설정

π0.5 추론 시간의 75%가 행동 디코딩(10회 순차 디노이징)이다. 동기 실행은 청크 경계마다 로봇이 멈추고, 비동기 실행은 오래된 관측으로 예측해 lookahead가 길수록 예측-실행 불일치가 커진다. 기존 효율화(경량 백본, 양자화, 토큰 가지치기)는 한 패스를 싸게 만들 뿐 루프를 유지하고, 비동기 기법(VLASH의 미래 상태 조건, RTC)은 불일치를 사후에 보정한다. 저자들은 두 문제의 공통 원인을 "청크를 고립적으로 디코딩한다는 점"으로 진단한다.

## 3. 핵심 아이디어

긴 영상 생성의 streaming chunk-wise diffusion(Diffusion Forcing, StreamDiT, MAGI-1)을 행동 디코딩에 이식한다. 버퍼 안에서 노이즈 수준이 단조 증가하도록(τ1<…<τN) 배치하면, 디노이징 진행 순서와 실행 순서가 일치해 매 패스마다 가장 깨끗한 청크가 하나씩 튀어나온다. 또한 더 노이즈가 큰(미래) 청크가 덜 노이즈가 큰(곧 실행될) 청크를 attend하므로, 미래 상태 예측기 없이도 궤적 연속성이 구조적으로 확보된다.

## 4. 아키텍처와 스트리밍 추론

τ_i = 0.001 + 0.999·(i−1+u_i)/N, u_i ~ Beta(1.5,1.0)로 계단형 노이즈 수준을 샘플링한다. 버퍼 내 chunk 단위 causal mask(노이즈 큰 청크 → 작은 청크 방향만 허용)를 적용하고, action expert의 다단계 timestep 조건을 FiLM(scale/shift/gate)으로 강화한다. 추론은 cold start(N−1 패딩 + 가우시안 1개로 시작, N−1 패스 동안 안전한 기본 행동 실행) 후 steady streaming(한 패스로 모든 청크 1스텝 전진 → 가장 깨끗한 청크 pop·실행 → 새 노이즈 청크 push)으로 진행한다. CUDA Graph, linear 패킹, max-autotune 컴파일을 함께 쓴다. LIBERO는 청크 10 × 버퍼 4, RoboTwin 2.0은 20 × 4이며, 버퍼 총 길이를 사전학습 모델의 원래 청크 길이(π0.5는 50)에 맞추는 것이 좋다고 한다.

## 5. 학습: Multi-Buffer Joint Fine-tuning

사전학습 VLA는 부분적으로 채워진 버퍼나 chunk-wise causal 패턴을 본 적이 없다. 한 관측 o_t에 대해 가능한 N개의 버퍼 상태(j개의 실제 청크 + N−j 패딩)를 한 샘플에 packing하고, 버퍼 간 attention을 마스크로 차단해 관측 인코딩은 한 번만 계산한다. 목적함수는 모든 구성에 대한 표준 flow-matching 손실의 합이다. LIBERO: LR 1e-4 cosine, 1K warm-up, 50K 스텝, GPU당 배치 32, 8×H200. RoboTwin 2.0: LR 5e-5, 글로벌 배치 256, 100K 스텝. 일부 데이터에서 action expert norm 층이 불안정해 이를 재초기화했다(RoboTwin 50-task). 즉 가중치를 실제로 갱신한 자체 정책이다.

## 6. 비동기 실행 결과 (Table 1, LIBERO d=1)

| 방법 | Spatial | Object | Goal | Long | Avg | Time/Step (ms) |
|---|---|---|---|---|---|---|
| π0.5 (동기) | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | 53.8 |
| + VLASH (d=1) | 98.8 | 99.2 | 96.7 | 94.4 | 97.2 | 46.8 (1.15×) |
| + VLASH (d=4) | 92.5 | 96.9 | 93.3 | 89.6 | 93.1 | 32.8 (1.64×) |
| + StreamingVLA (d=1) | 96.6 | 96.6 | 95.4 | 91.0 | 94.9 | 31.6 (1.70×) |
| **+ FlashVLA (d=1)** | 98.8 | 99.6 | 97.6 | 95.4 | **97.8** | **22.1 (2.43×)** |

본문에 따르면 d=1–4에서 LIBERO 97.5–98.3%를 유지하며 2.43–2.62× 속도 향상, RoboTwin 2.0(50 task)에서는 d=1 90.6%(π0.5 동기 86.0, VLASH 86.9), d=4에서도 89.8%다. RoboTwin의 per-step 속도 향상(1.08–1.10×)이 작은 이유는 시뮬레이터 렌더링이 스텝 시간을 지배하기 때문이다(정책 추론 8.1/47.4 ms).

## 7. 동기 품질과 장기 과제 (Table 3, 4)

| | LIBERO Avg | Long | RoboTwin Clean | Random | Avg |
|---|---|---|---|---|---|
| π0.5 | 96.9 | 92.4 | 86.1 | 85.8 | 86.0 |
| **+FlashVLA** | **97.9** | **96.2** | **90.8** | **90.2** | **90.5** |

RoboTwin 2.0 horizon 그룹(Table 4, clean/random 평균): short 93.9 → 93.2, medium 81.1 → 85.4, **long 53.0 → 89.6(+36.6)**. 저자들은 버퍼가 최근 정제된 청크의 짧은 이력을 담는 "청크 수준 메모리" 역할을 한다고 해석한다.

## 8. 지연·반응 속도·아키텍처 일반화 (Table 2, 5)

per-invocation 지연(같은 시스템 최적화 적용): RTX 4090 2뷰 45.8 → 26.7 ms, 3뷰 55.4 → 36.8 ms; RTX 5090 2뷰 37.0 → 20.3 ms(~50 Hz), 3뷰 44.8 → 27.1 ms. Realtime-VLA(29.2/38.9/26.6/34.2 ms)보다 모든 설정에서 빠르다. 30 Hz 목표에서 TTFA 37.1 ms(FASTER 62.1, π0.5 80.0), 기대 TTR 70.4 ms(FASTER 112.1). 교차 아키텍처: SmolVLA는 LIBERO 80.1 유지(d=1 79.5), 추론 19.7 → 10.1 ms; LingBot-VLA는 RoboTwin 2.0 85.2 → 88.6(d=0) / 89.3(d=1), 70.6 → 25.1 ms.

## 9. 실로봇과 어블레이션

7-DoF Franka + RTX A4000, 30 Hz, 비동기 2스텝 지연. pick-and-place(단기), 화이트보드 닦기(중기), 테이블 정리(장기) 각 50개 시연, 15회, 3점 루브릭. 평균 점수 FlashVLA 84.4% vs π0.5 동기·RTC 80.0%, naive async 75.6%. 성공 시 완료 시간은 동기 대비 평균 1.3×, RTC 대비 1.2× 빠르다(과제별 수치는 그림에만 있음). A4000에서 추론 67.3 ms. 어블레이션(부록): causal mask를 제거하고 버퍼만 유지하면 비동기 성공률이 약 10pt 하락, 청크 크기 C=10이 최적, 버퍼 길이 N∈{4,5,6}에는 둔감하다.

## 10. 강점

- 지연과 비동기 불일치라는 두 문제를 하나의 구조 변경으로 해결하는 명료한 논지와, 이를 검증하는 causal-mask 어블레이션.
- 속도만 올리는 것이 아니라 동기 성공률도 올리며, 특히 장기 과제에서 큰 향상을 보인다.
- π0.5, SmolVLA, LingBot-VLA 세 아키텍처에 적용되어 flow-matching VLA 일반에 대한 드롭인 가능성을 보였다.
- 비교 대상 모두에 같은 시스템 최적화(CUDA Graph, 커널 퓨전)를 적용해 공정성을 확보했다.

## 11. 한계

- 스트리밍 방식은 사전학습 목적과 달라 미세조정이 필수이며, 학습 불안정(norm 층 재초기화)이 발생할 수 있다.
- 에피소드마다 N−1 스텝 cold-start가 필요해 매우 짧은 과제에서는 부담이 된다.
- RoboTwin long-horizon +36.6pt라는 큰 향상의 원인 해석(청크 메모리)은 가설 수준이며, π0.5 베이스라인의 청크 설정(50)과 FlashVLA의 설정 차이가 교란 요인일 수 있다.
- 실로봇 과제별 수치와 여러 지연 스윕 결과가 그림으로만 제시된다.

## 12. VLA-Tracker 관점 평가

제목은 추론 프레임워크처럼 보이지만, π0.5(및 SmolVLA, LingBot-VLA)의 action expert를 새 목적함수로 실제 미세조정해 자체 정책을 만들고 정량 결과를 보고하므로 수록 대상이다(EMS 선례와 동일). 트래커에는 동기 설정(Table 3)을 LIBERO 헤드라인(97.9)과 RoboTwin v2 평균(90.5)으로, 비동기 d=1 결과(97.8, 2.43×)는 별도 블록으로 기록했다. 기존 Realtime-VLA-FLASH와는 방법·저자·논문이 모두 다르다. 실시간 제어·비동기 실행 계열(RTC, VLASH, StreamingVLA, FASTER)의 비교 기준점으로 유용하다.

<!-- VERIFIED: pdf -->
