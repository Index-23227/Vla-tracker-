# Drop-Then-Recovery: How Redundant Are Vision-Language-Action Models?

> **한 줄 요약**: 사전학습 VLA에서 transformer block을 물리적으로 제거한 뒤 recovery fine-tuning으로 회복 가능성을 측정하는 DTR 프로토콜과 one-shot virtual-gate 중요도 메트릭 GateProbe를 제안 — LLM 블록 절반을 제거해도 OpenVLA-OFT의 LIBERO 성능이 95.0%→98.3%로 오히려 향상되며, 언어 백본 2개 블록만 남겨도 baseline 수준을 유지함을 4개 아키텍처·시뮬/실로봇에서 입증.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- VLA 모델은 사전학습 VLM에서 **거대 언어 백본을 그대로 상속** — 하지만 로봇 지시문은 "pick up the red cup" 수준으로 짧고 템플릿화되어 있어 용량 과잉 가능성
- 기존 layer-dropping 메트릭(cosine similarity, magnitude, PPL 등)은 **제거 직후 성능 하락(importance)**만 측정하고 **fine-tuning 후 회복 가능성(recoverability)**을 예측하지 못함
- LLM/VLM 압축은 recovery 없이 평가하는 drop-only 방식이 통용되지만, VLA는 작은 action 오차가 long horizon에서 누적되어 task 붕괴로 이어지므로 부적합

### 핵심 질문
- **closed-loop 로봇 제어에 VLA 모델의 어느 부분이 실제로 필요한가?** — 파라미터 수가 아니라 recovery 후 task success로 측정해야 함
- 언어/비전/액션 컴포넌트 간 redundancy는 대칭적인가?

---

## 2. 방법론 심층 분석

### DTR (Drop-Then-Recovery) 프로토콜 — 2단계
1. **Stage 1 (Drop)**: 중요도 메트릭 I와 목표 제거 수 K가 주어지면, 가장 덜 중요한 K개 블록을 물리적으로 제거 (residual short-circuit: h_i = h_{i-1}, 파라미터 θ_i 삭제) → 진짜 작은 dense 모델
2. **Stage 2 (Recovery)**: 제거된 모델을 downstream task에서 fine-tuning (action loss: 연속 액션은 MSE, diffusion 계열은 flow-matching)

핵심 구분: **importance**(제거 직후 하락) ≠ **recoverability**(fine-tuning 후 회복) — zero-shot으로 크게 무너져도 쉽게 회복되는 블록이 존재.

### GateProbe — virtual gate sensitivity
- 각 블록의 residual branch에 가상 스칼라 게이트 α_i 도입: h̃_i = h_{i-1} + α_i·F_i(h_{i-1})
- 중요도 = task loss의 게이트 민감도: I_gate(B_i) = E_x |∂L/∂α_i|_{α_i=1}
- chain rule로 모델 수정 없이 계산: **⟨∂L/∂h_i, F_i(h_{i-1})⟩** — downstream gradient와 블록 residual 기여의 내적
- 비용: calibration set에서 forward+backward 1회 (π0.5 18블록, H200 1장, 64배치 기준 **24.9초**)

### π0.5 같은 joint-attention 구조의 특수 처리 (Appendix B)
- 언어 블록 drop 시 K/V projection + input LayerNorm은 유지(action expert의 cross-attention 필요) → 블록당 약 75%만 제거, 유지된 K/V는 recovery 중 cross-attention adapter로 재활용됨

---

## 3. 데이터 전략
- 시뮬레이션: LIBERO (4 suites × 10 tasks, 20 trials/task), LIBERO-Plus (perturbation 확장), RoboTwin 2.0 (양팔, 7개 대표 task)
- 실로봇: 창고 소포 분류 — teleoperation (Meta Quest 3, 10Hz)으로 **~110K 프레임 (~600 grasps)** 수집
- GateProbe는 dataset-specific profiling 사용 (LIBERO/LIBERO-Plus/RoboTwin에서 선택 블록이 다름)

---

## 4. 시스템/학습 세부사항
- 평가 모델 4종: **π0.5** (PaliGemma 18층 + Gemma action expert 18층, flow matching), **OpenVLA-OFT** (Llama-2-7B 32층, SigLIP, MLPResNet regression head), **Lingbot-VLA** (Qwen2.5-VL-3B 36층 + flow-matching decoder, ~4B), **GigaBrain-0** (PaliGemma2-3B + Gemma2 expert 26층, diffusion head, ~3.5B)
- 기본 학습: bsz 32 × 30K steps; FLOPs-matched 실험은 bsz 64 + step 수 조정 (Drop-16은 59.8K steps)
- OpenVLA-OFT의 action head는 블록 drop 대신 hidden dim 축소(4096→2048/256)로 압축 (전체의 ~0.7%에 불과)
- 실로봇 배포: UFACTORY xArm 850 + Gripper G2, wrist RealSense D435 + 3인칭 카메라, **NVIDIA Jetson Thor**에서 π0.5 구동

---

## 5. 실험 설계 및 평가 프로토콜
- **컴포넌트별 redundancy**: Vision/Language/Action 각각에 Drop Half(홀수 블록 제거)와 Keep 2(첫·끝 블록만 유지) 적용 — 메트릭 편향 배제
- **granularity**: 블록 전체 vs MHA만 vs MLP만
- **메트릭 비교**: 8종 (GateProbe, Taylor, IGIA, Fisher, Hessian trace, CosSim, CosSim-contig, PPL, Magnitude)을 Drop-9/12/16/17에서 비교
- 실로봇: 3 runs × 20 grasps, OOD 6종 (조명 3종, novel object, 컨테이너 방향/제거)

---

## 6. 실험 결과 심층 분석

### 컴포넌트 redundancy 비대칭 (Table 1, LIBERO)
| 모델 | 설정 | Spatial | Object | Goal | Long | Avg |
|------|------|---------|--------|------|------|-----|
| OpenVLA-OFT | Baseline | 97.2 | 98.4 | 95.6 | 88.6 | **95.0** |
| OpenVLA-OFT | Drop Half **Language** (55.5% size) | 99.0 | 100.0 | 97.8 | 96.4 | **98.3** |
| OpenVLA-OFT | Keep 2 Language (16.6% size) | 97.2 | 99.0 | 95.4 | 88.6 | **95.1** |
| OpenVLA-OFT | Drop Half Vision | 82.6 | 99.0 | 77.2 | 76.8 | 83.9 |
| π0.5 | Baseline | 96.6 | 95.0 | 93.0 | 82.0 | **91.7** |
| π0.5 | Drop Half Language | 98.8 | 98.6 | 93.4 | 82.4 | **93.3** |
| π0.5 | Keep 2 **Action** | 3.6 | 40.8 | 16.0 | 44.4 | **26.2** (붕괴) |

- **언어 백본은 압도적으로 redundant, 액션/비전 경로는 압축 불내성** — π0.5 Keep 2 Vision 62.4%, Keep 2 Action 26.2%로 붕괴
- 4개 아키텍처 전부 Drop Half Language 후 baseline 동등 이상 (Table 7: GigaBrain-0 88.0→88.0, Lingbot-VLA 82.8→83.7)

### 실용적 이득 (Table 4, 5)
- FLOPs-matched: π0.5 Drop-12 (47.5% size)가 93.7%로 최고 (+2.0); Drop-17(언어 블록 1개!)도 91.0% 회복
- DTR-16: **1.64× task speedup, 메모리 14.4→8.36GB (-42%)**, LIBERO-Goal 100% — 하드웨어 무관 dense 가속
- zero-shot block drop은 SR 붕괴로 step 수 폭증 → task speedup 0.72×(오히려 느려짐) — **recovery는 선택이 아니라 필수**

### 실로봇 (π0.5, 창고 소포 분류)
- Env 1: Drop-9 65.0% > Full 63.3%; Env 2: Full 75.0%, Drop-9 71.7%, Drop-16 66.7% — 시뮬 패턴 재현
- OOD에서는 격차 발생: green light에서 Drop-16 35% (Full 50%), 컨테이너 제거 시 40% (Full 60%)

---

## 7. Ablation 분석
- **Granularity (Table 2)**: OpenVLA-OFT에서 블록 전체 drop(98.3%)이 MHA만(91.9%), MLP만(65.6%)을 크게 상회 — 블록 단위가 압축률·성능 모두 최적
- **메트릭 비교 (Table 3, π0.5)**: GateProbe가 4개 drop 수준 모두 최고/차상위; 극한 압축에서 격차 확대 (Drop-16에서 Taylor 85.2 vs GateProbe/Fisher 92.2, +3.9~+4.3). 비-gradient 메트릭(CosSim/Magnitude/PPL)은 일관되게 열세
- **LIBERO-Plus perturbation 분해 (Table 6)**: drop 후 최대 하락은 Language 카테고리(-5.1)가 아니라 **Robot(초기 포즈 변경, -10.6)** — 제거된 언어 블록이 언어 이해보다 **물리적 일반화**에 기여했음을 시사
- RoboTwin 2.0: Easy는 -0.6에 그치나 Hard는 -6.6 — 난이도 높을수록 redundancy 감소

---

## 8. 관련 연구 비교
- **ShortGPT/SLEB/layer-dropping 계열**: drop-only 평가 — VLA에서는 오차 누적 때문에 recovery 없는 평가가 부적합함을 보임
- **VLA 효율화 (EfficientVLA, BitVLA, MoLe-VLA, SpecPrune-VLA, QVLA, RLRC)**: 양자화/토큰 프루닝/layer skipping 등과 직교(orthogonal); DTR은 dense 모델을 산출해 특수 커널 불필요
- **동시기 연구 (Jabbour et al. 2025, Grant et al. 2026)**: VLA redundancy 존재는 확인했으나 "얼마나, 어디까지 회복 가능한가"의 한계선을 체계적으로 측정한 것이 본 논문의 차별점

---

## 9. 한계 및 미해결 문제
- LIBERO 포화(saturation)가 개선폭의 일부를 설명 — "redundancy"가 모델의 문제인지 **벤치마크의 언어 압력 부족** 문제인지 분리 곤란 (저자들도 인정)
- OOD/perturbation 하에서 dropped 모델의 강건성 저하 — 배포 환경의 robustness 요구에 따라 압축 실익이 달라짐
- 실로봇 OOD 평가는 단일 run 20 grasps로 통계적 신뢰도 제한
- 새 아키텍처 설계 지침("언어 용량을 task 난이도에 맞춰라")은 제시하나 구체적 설계는 future work
- venue 미정 (Preprint)

---

## 10. 총평
컴프레션 논문이자 벤치마크 비판 논문. "LLM 블록 절반을 제거하면 성능이 오른다"는 결과는 현재 VLA 벤치마크가 언어 grounding을 거의 시험하지 않는다는 강력한 증거이며, GateProbe는 25초짜리 one-shot 메트릭으로 실용성이 높다. Drop 후 학습(throughput 증가)과 하드웨어 무관 추론 가속이라는 이중 이득 구조가 깔끔하고, 4개 아키텍처 + 실로봇(Jetson Thor 배포)까지 검증 범위가 넓다. 단, 개선의 상당 부분이 LIBERO 포화와 compute 재배분 효과라는 점, OOD 강건성 손실이 존재한다는 점은 유의해야 한다.

---

## 11. 🔥 예상 날카로운 질문 모음
1. Drop Half가 baseline을 능가하는 것은 같은 budget에서 더 많은 step을 돌 수 있어서인가, 아니면 정말 regularization 효과인가? (Table 1은 같은 bsz/steps인데도 +3.3 — 어떻게 설명?)
2. GateProbe는 1차 Taylor 근사인데 Drop-16/17 같은 극한 설정에서 왜 Taylor(parameter-space)보다 강건한가?
3. LIBERO-Plus에서 Language perturbation보다 Robot perturbation 하락이 큰 이유 — 언어 블록이 사실상 시각-운동 표현을 담고 있다는 뜻인가?
4. π0.5의 K/V 유지 방식은 사실상 cross-attention adapter 재학습인데, 이를 "블록 제거"로 부르는 것이 공정한가?
5. 언어 지시가 복합적/조합적인 벤치마크(예: 긴 명령, 조건부 지시)에서도 같은 redundancy가 유지될 것으로 예상하는가?
6. GateProbe의 calibration set 의존성 — dataset-specific profiling이 필요하다면 새 도메인마다 재계산 비용은?

---

## 12. VLA-Tracker 관점에서의 의의
- **분류**: 새 정책 모델이 아닌 **효율화 방법론(efficiency-method)** — headline 스코어는 DTR을 OpenVLA-OFT에 적용한 결과(LIBERO 98.3)로, base model 대비 +3.3
- 트래커 내 actquant, DeeR-VLA, DepthCache 등 효율화 계열과 같은 범주; layer-dropping + recovery fine-tuning이라는 새 축 추가
- "현 벤치마크가 언어 grounding을 under-test한다"는 주장은 LIBERO 중심 리더보드 해석에 중요한 caveat — LIBERO 고득점이 언어 이해력을 보증하지 않음
- 코드 공개: https://github.com/s1ghhh/VLADrop

<!-- VERIFIED: pdf -->
