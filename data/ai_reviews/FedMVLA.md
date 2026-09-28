# FedMVLA: Modality-Decoupled Federated Learning for Privacy-Preserving Embodied Intelligence in 6G

> **한 줄 요약**: VLA의 vision / language / action 경로가 파라미터 규모·프라이버시 노출·업데이트 동역학·압축 내성에서 서로 다르다는 점에 착안해, 모달리티별로 aggregation(MAFA), DP 예산(MAPA), 압축(MACO), 6G 전송 slice를 분리한 federated VLA fine-tuning 프레임워크. 시뮬레이션 기반 16-client 연합 조작 실험에서 84.8% 성공률(FedAvg 62.6%, FedVLA 74.6%), uplink payload 95.6% 절감.

---

## 1. 배경 및 동기

- 6G 네트워크(URLLC, eMBB, edge intelligence)는 이질적 로봇들이 협업하는 대규모 embodied intelligence의 인프라로 기대된다.
- 로봇별 로컬 데이터(영상·지시문·궤적)는 분산돼 있고 대역폭이 크며 privacy-sensitive → 중앙 수집·재학습이 비현실적 → Federated Learning(FL)이 자연스러운 선택.
- 그러나 기존 FL(FedAvg/FedProx/SCAFFOLD)과 embodied FL(FedVLN, FedVLA)은 VLA를 하나의 monolithic 파라미터 벡터로 취급한다. 저자들은 세 경로가 **파라미터 규모, 프라이버시 노출, 업데이트 동역학, 압축/섭동 내성**에서 근본적으로 다르다고 지적(Table I).

## 2. 방법론 심층 분석

### 2.1 MAFA (Modality-Aware Federated Aggregation)
- 모달리티마다 다른 aggregation topology 사용: action 업데이트는 embodiment 간 유사도가 거의 0, 그룹 내 ≈0.6 → embodiment 그룹별 집계; vision gradient는 3개의 scene cluster 형성; language gradient는 0.8 이상 정렬 → 전역 집계.

### 2.2 MAPA (Modality-Aware Privacy Allocation)
- 총 DP 예산 ε를 모달리티별로 배분. 배분은 privacy sensitivity와 **실측 perturbation tolerance**(Fig. 4a) 두 요소를 결합한 최적화로 결정. ε=1에서 action 61%, vision 31%, language 8%.

### 2.3 MACO (Modality-Aware Communication Compression)
- Vision: top-k sparsification + error feedback (공간적 중복 활용), Language: 4-bit quantization, Action: FP32 유지(정밀도 critical).

### 2.4 Modality-sliced transport
- Action stream은 보호된 5 MHz URLLC slice(repetition coding), vision/language는 100 MHz eMBB slice. 지연된 vision/language 업데이트는 다음 동기화 시점으로 defer(staleness 기록), 지연된 action 업데이트는 해당 라운드에서 drop.

## 3. 데이터 전략

- 16 clients, 4 embodiments(7-DoF Franka Panda, Sawyer; 6-DoF UR5e, Jaco), 4 task family(tabletop pick-and-place, bin packing, stacking, drawer manipulation).
- Client당 500 trajectories, object category 비중첩 + 서로 다른 scene texture → 강한 non-IID.
- Scaling 실험은 총 8,000 demonstrations 고정, 4 clients/1 cell → 128 clients/8 cells.

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|---|---|
| Vision encoder | SigLIP 400M, LoRA r=16 (4.0M trainable) |
| Language backbone | LLaMA-2 7B, LoRA r=8 (4.2M trainable) |
| Action head | diffusion-based 50M, conditioning/output layer 0.3M만 학습, chunk 16 |
| Trainable 합계 | 8.5M (monolithic FP32 upload 34 MB/round) |
| FL 설정 | 100 rounds, local 5 epochs |
| 무선 | 3GPP TR 38.901 InF-SL, 7 GHz, Rayleigh block fading, 23 dBm, HARQ ≤4 |

## 5. 실험 설계 및 평가 프로토콜

- 비교: FedAvg, FedProx, SCAFFOLD, FedVLN-P(partial aggregation 이식), FedVLA(expert-driven aggregation). 동일 backbone·데이터 분할·local budget·라운드·paired channel trace.
- 평가: client당 50 rollouts(sweep 지점은 20), 3 seeds, 95% CI. N=128에서는 embodiment 층화 32-client subset.
- 기본: 평균 uplink SNR 10 dB, jammer off, DP off.
- **주의**: 표준 공개 벤치마크(LIBERO 등)가 아닌 저자 구성의 시뮬레이션 연합 조작 설정이며, 대부분 결과가 Figure로 제시되고 본문에 일부 수치만 명시됨.

## 6. 실험 결과 심층 분석 (PDF 본문 수치 인용)

| 항목 | 값 |
|---|---|
| FedMVLA 최종 성공률 | **84.8%** |
| FedAvg | 62.6% (−22.2 pp) |
| FedVLA | 74.6% (−10.2 pp) |
| 이상 채널 대비 열화 | FedMVLA 1.3 pp / FedAvg 3.4 pp |
| 128 clients 확장 시 FedAvg 대비 격차 | 17.8 → 28.8 pp |
| Pulsed jammer 하 FedMVLA 최대 하락 | 1.9 pp (monolithic 전송은 최악 56.4%) |
| DP ε=0.5 | MAPA 72.7% / sensitivity-only 58.1% / uniform 35.6% |
| Payload | 1.5 MB/round (FedAvg 34 MB 대비 95.6% 절감) |
| p95 round-critical uplink | ≈1.5 s (FedAvg 136 s) |
| 송신 에너지 | ≈0.3 J vs 2.7 J |

- 단일 모달리티 노이즈 주입(z=4): language −1.1, vision −9.6, action −16.0 pp → 모달리티 비대칭성이 설계 가정이 아닌 측정값임을 보여줌.
- 압축 마이크로벤치(All-FP32 85.4% 기준): vision top-1% −0.3 pp, language 4-bit −0.4 pp, action 4-bit −10.8 pp(EE RMS 오차 0.87 mm).

## 7. Ablation 분석

- Canonical 지점에서 MAFA→global averaging −4.4 pp, MACO→iso-bitrate uniform compression −5.3 pp, sliced→monolithic transport −3.2 pp.
- MAPA ablation은 Fig. 4b의 uniform-DP 곡선(ε=0.5에서 35.6%).
- k=0.005에서 uniform 전송은 빠르지만 성공률 53.8%로 하락, MACO는 지연·성능 모두 유지.

## 8. 관련 연구 비교

- FedAvg/FedProx/SCAFFOLD: 모델 전체를 동일하게 취급.
- FedVLN: navigation용 partial aggregation. FedVLA: expert-driven aggregation. 두 방법 모두 모달리티별 privacy·압축·전송을 분리하지 않음.
- OpenVLA: backbone 설계의 기반(SigLIP + LLaMA-2), 여기에 diffusion action head를 분리 학습.

## 9. 한계 및 미해결 문제

- 표준 VLA 벤치마크나 실로봇 평가가 없어 다른 VLA와 정량 비교가 불가능.
- 시뮬레이터·태스크 세부사항이 제한적으로 기술되며 FedProx/SCAFFOLD/FedVLN-P 수치는 그림에만 존재.
- 학습 파라미터가 8.5M(LoRA + head 일부)으로 매우 작아, 대형 VLA의 full fine-tuning FL로의 확장성은 미검증(저자도 언급).
- 무선 채널은 모델링된 substrate이며 실제 6G 배치 실험이 아님.

## 10. 총평

통신/신호처리 관점의 magazine-style 논문으로, "VLA는 모달리티마다 다른 FL 처리가 필요하다"는 직관을 측정(노이즈 내성, gradient 기하)과 시스템 설계로 일관되게 연결한 점이 강점이다. 실제로 VLA를 LoRA로 연합 학습하고 성공률을 보고하므로 학습된 정책을 산출하지만, 평가 설정이 비표준 시뮬레이션이라 manipulation 성능 자체보다 FL·통신 효율성 기여로 읽는 것이 적절하다.

## 11. 🔥 예상 날카로운 질문 모음

1. 시뮬레이터와 태스크 정의는 무엇이며, 표준 벤치(LIBERO, CALVIN)에서도 FedAvg 대비 22 pp 격차가 유지되는가?
2. Action head의 0.3M만 학습하는데 embodiment 간 action 업데이트 유사도가 0에 가깝다면, 그룹별 집계는 사실상 로컬 학습과 무엇이 다른가?
3. MAPA의 perturbation tolerance 측정 자체가 privacy leakage를 유발하지 않는가?
4. 중앙집중 학습(upper bound) 대비 성능은?
5. 4-bit action 양자화의 10.8 pp 하락은 chunk 16과 어떤 관계가 있는가?

## 12. 재현성 및 후속 연구 제안

- 코드/데이터 공개 언급 없음. 무선 파라미터와 학습 하이퍼파라미터는 상세하나 시뮬레이션 환경이 모호.
- 후속: (1) 표준 벤치 + 실로봇 연합 학습 검증, (2) full fine-tuning 규모에서 MACO/MAPA 효과, (3) 비동기 FL과 client dropout 대응, (4) 모달리티별 membership/gradient inversion 공격 평가.

<!-- VERIFIED: pdf -->
