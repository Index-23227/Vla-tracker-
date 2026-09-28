# IMLE-VLA — Fast Single-Step Action Generation for Vision-Language-Action Policies

**arXiv**: 2609.10915 · **기관**: Simon Fraser University, University of Pennsylvania, Amii · **날짜**: 2026-09-10

## 1. 한 줄 요약
π0.5의 10-step flow-matching 액션 헤드를 조건부 IMLE(cIMLE)로 학습한 단일 스텝 생성기로 교체해, 추론 빈도를 15 Hz→55 Hz(3.67×)로 높이면서 LIBERO 평균 98.0%를 달성한 VLA.

## 2. 문제 설정
- Diffusion/flow-matching 액션 헤드는 청크 하나를 만들 때 여러 번의 순차 전방 패스(π0.5는 10 Euler step)가 필요하고, 동기 실행 시 로봇이 추론을 기다리며 "stop-and-go" 움직임을 보인다.
- 단순 단일 스텝 회귀(L1/L2)는 다봉 행동 분포의 평균으로 수렴하는 mode collapse 문제가 있다(OpenVLA-OFT는 7B 백본 용량으로 이를 상쇄).
- 증류 기반 가속(Shallow-π, consistency distillation)은 teacher 성능에 상한이 묶인다.

## 3. 핵심 아이디어
- cIMLE: 각 학습 샘플마다 노이즈 m개로 후보 청크 m개를 생성, 정답과 가장 가까운 후보만 골라(그래디언트 없음) 그 후보에 대해 L2 손실을 최소화.
- m=1이면 평범한 L2 회귀(조건부 평균)로 퇴화하지만, m>1이면 다른 후보들이 다른 모드를 덮을 수 있어 **구성적으로 mode coverage**를 보장.
- teacher 없이 전문가 행동에 직접 적합 → 증류 편향 없음.

## 4. 아키텍처
- π0.5의 VLM 백본(약 3B)은 그대로 동결, 액션 헤드 G_θ(f_VLM(o_t), z)만 교체. G_θ는 π0.5 액션 헤드 체크포인트로 초기화.
- z ~ N(0, I)는 액션 청크와 같은 차원, 한 번의 전방 패스로 청크 전체 생성.
- VLM 임베딩은 배치당 한 번 계산해 m개 후보가 공유.

## 5. 학습 목표
- L_cIMLE = (1/n) Σ ||Â_{i,j*(i)} − A_gt^(i)||², j*(i) = argmin_j ||Â_{i,j} − A_gt^(i)||².
- 샘플 인자 m=2 사용(m=5와 거의 동일, m=1은 급락 — Fig. 4, 수치는 그림).

## 6. 학습 절차
- VLM 동결, 액션 헤드만 cIMLE로 미세조정. 할당 단계는 배치·후보 차원 병렬 처리, 승자 후보만 다시 그래디언트 계산.
- 실로봇: π0.5와 IMLE-VLA 모두 DROID로 학습 후 제로샷 배치.
- 추론은 수신 지평(receding horizon): 한 번 추론 후 H개 행동을 개루프 실행.

## 7. 추론 속도 (Table I, III, IV; L40S)
| 시스템 | 추론 Hz | 배수 |
|---|---|---|
| π0.5 (JAX) | 15 | 1.00× |
| PyTorch + compile | 20 | 1.33× |
| Triton | 25 | 1.67× |
| **IMLE-VLA** | **55** | **3.67×** |
- 행동 처리량: H=10에서 550 Hz(3.67×), H=30에서 1,650 Hz(11.0×).
- LIBERO 성공 에피소드당 VLA 전용 wall-clock: π0.5 1.03 s → Ours(H=10) 0.28 s(3.6×) → Ours(H=30) 0.10 s(10.5×).

## 8. LIBERO 결과 (Table II)
| 모델 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| π0.5 (H=10) | 97.2 | 99.0 | 97.8 | 96.0 | 97.5 |
| π0.5 (H=30) | 95.0 | 98.0 | 96.2 | 95.2 | 96.1 |
| OpenVLA-OFT | 95.2 | 94.2 | 95.2 | 93.2 | 94.5 |
| CogVLA | 99.0 | 99.0 | 97.0 | 95.0 | 97.5 |
| Shallow-π0.5-L6 | 98.0 | 96.0 | 94.0 | 90.0 | 94.5 |
| **IMLE-VLA (H=10)** | 98.0 | 99.8 | 98.2 | 96.0 | **98.0** |
| IMLE-VLA (H=30) | 97.2 | 99.0 | 96.2 | 95.8 | 97.1 |
- LIBERO-Plus: π0.5 수준의 견고성 유지, OpenVLA-OFT 등은 심도가 올라갈수록 급락(Fig. 2, 수치는 그림에만 제시).

## 9. 실제 로봇 (Table V; Franka Panda, 과제당 20회)
| 과제 | Ours 성공 | π0.5 성공 | VLA-WC Ours / π0.5 (s) | Jerk 배수 |
|---|---|---|---|---|
| Pineapple in bowl | 19 | 15 | 18.56 / 123 | 2.7× |
| Swap pineapple & cube | 18 | 15 | 31.1 / 178.3 | 2.2× |
| Pineapple in cabinet | 15 | 12 | 99.4 / 389.4 | 3.0× |
| Pineapple on moving plate | 16 | 12 | 16.7 / 89.5 | 2.7× |
- π0.5는 H=8·15 Hz, IMLE-VLA는 H=12·55 Hz. 움직이는 접시 과제에서 짧은 재계획 주기 덕분에 목표 추적이 우수.

## 10. 어블레이션
- 실행 지평: H=10 98.0 → H가 커질수록 점진적 하락, H=30이 처리량–정확도 균형점(97.1, 같은 H의 π0.5 96.1보다 높음).
- 샘플 인자 m: m=1(L2 회귀)은 뚜렷한 하락, m=2 ≈ m=5.

## 11. 강점과 한계
**강점**
- 수정 범위가 액션 헤드에 한정된 drop-in 방식이며, 속도·정확도를 동시에 개선한 드문 사례.
- 증류 없이 mode coverage를 확보하는 간결한 목표.

**한계**
- VLM을 동결하므로 VLM 전방 패스 자체의 지연은 그대로이며, 속도 향상 상한이 VLM 비용에 묶인다.
- LIBERO-Plus와 어블레이션 수치가 그림으로만 제공되어 정량 비교가 어렵다.
- 실로봇 평가는 4과제×20회로 작고, 기준선은 π0.5 하나뿐.
- 추론 Hz 비교 중 일부(Shallow-π 등)는 H100 측정값을 인용해 하드웨어가 섞여 있다.

## 12. VLA-Tracker 관점 평가
π0.5 액션 헤드를 cIMLE 목표로 직접 미세조정해 자체 정책을 학습했으므로 ACCEPTED(단순 추론 가속 런타임이 아님). 액션 헤드는 노이즈→청크 단일 스텝 암묵적 생성기로 diffusion/flow/회귀 어디에도 정확히 속하지 않아 `action_head_category: other`로 분류했다. LIBERO는 논문이 보고한 H=10 평균 98.0을 `benchmarks.libero`에 등록했고, H=30 결과·기준선·속도·실로봇(성공 횟수/20)은 별도 블록으로 분리했다. LIBERO-Plus는 그림 전용이라 기록하지 않았다.

<!-- VERIFIED: pdf -->
