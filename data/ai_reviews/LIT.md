# LIT — Breaking the Vision–Action Shortcut: Latent Interface Training for Generalizable Robotics Foundation Models

- arXiv: 2609.12641 (v1, PDF 스탬프 2026-09-11)
- 저자/소속: Jianman Lin (South China University of Technology), Shailesh Shailesh·Jiafei Duan (National University of Singapore), Zhongyi Luo (Nanyang Technological University)
- 프로젝트: magiclab-nus.github.io/LIT (코드 공개 여부는 논문에 명시되지 않음)

## 1. 한 줄 요약

LIT는 VLA/WAM의 액션 전문가(action expert)가 학습 분포 안에서만 통하는 시각 단서(vision–action shortcut)에 기대는 문제를 줄이기 위한 **프레임워크 무관 2단계 학습 전략**이다. 1단계에서 이미지 없이 "청크 종단 SE(3) 자세"를 목표로 조건화한 행동 사전(prior)을 학습하고, 2단계에서 **자세 복원으로 감독되는 잠재 인터페이스 토큰**을 액션 전문가의 유일한 시각 조건화 경로로 둔다. π0.5·MolmoAct2·FAST-WAM·ImageWAM 네 구조 모두에서 LIBERO-Plus Overall이 3.87–10.70pt 오르고 LIBERO 평균은 유지 또는 상승한다.

## 2. 문제 설정

사전학습 VLM/비디오 백본 표현을 액션 전문가가 직접 참조하면, 시연 데이터의 시각적 다양성이 부족할 때 과제와 무관한 시각 단서와 행동 사이의 상관을 학습하기 쉽다. 카메라·조명·배경이 바뀌면 이 상관이 깨져 성능이 떨어진다. 표현 강화(SpatialVLA, TraceVLA 등)는 정보를 더할 뿐 "어떻게 쓰는지"를 제약하지 않고, LA4VLA·Qwen-VLA식 무이미지 사전학습은 청크별 명시적 공간 목표가 없으며 이후 시각 조건화를 제약하지 않는다는 것이 저자들의 진단이다.

## 3. 핵심 아이디어

"공간 목표"를 두 단계의 공통 매개로 삼는다. 각 시연 청크의 종단 로봇 상태 g_t = [위치(3); 축-각 회전(3); 그리퍼 관절(2)] ∈ R^8을 1단계에서는 **조건 입력**, 2단계에서는 **복원 목표**로 쓴다(추론 시에는 필요 없음). 그리고 2단계에서 시각 정보는 오직 잠재 토큰을 통해서만 액션 전문가에 들어가도록 경로를 제한한다.

## 4. 아키텍처

- **1단계**: 동결된 백본이 언어와 로봇 상태만 처리해 결합 층별 표현 H^sem을 만들고, 3층 GELU MLP(SE(3) 인코더)가 g_t를 목표 토큰으로 바꿔 이어 붙인다. 액션 전문가는 처음부터(scratch) 학습되며 각 구조의 고유 조건화 방식(π0.5의 공유 self-attention, MolmoAct2의 층별 KV cross-attention 등)을 그대로 쓴다.
- **2단계**: K=100개의 학습 가능 잠재 토큰이 층마다 self-attention → 의미(언어·상태) cross-attention → 시각 cross-attention 순으로 갱신되고(m개 연속 층이 인터페이스 파라미터 공유), 이 토큰이 해당 층의 액션 전문가를 조건화한다. 최종 토큰에서 MLP 디코더가 g_t를 복원한다.
- 추론 시에는 SE(3) 인코더와 자세 디코더를 떼어내고 잠재 인터페이스만 남긴다. 입력은 이미지·언어·로봇 상태뿐이다.

## 5. 학습과 추론

1단계: 백본과 모달리티 인코더 동결, 액션 전문가와 SE(3) 인코더만 flow matching(L_prior)으로 학습. 2단계: 1단계 액션 전문가로 초기화한 뒤 백본·인코더·액션 전문가·잠재 토큰·인터페이스·디코더 전체를 L_act + 0.3·L_pose로 **전체 미세조정**. 베이스라인과 LIT는 같은 사전학습 백본과 무작위 초기화 액션 전문가에서 시작하며 총 최적화 스텝 수를 맞춘다(MolmoAct2 기준 1단계 10K + 2단계 20K = 베이스라인 30K). 공개된 VLA/WAM 정책 체크포인트는 일부러 미세조정하지 않았으므로, 베이스라인 수치는 기존 논문 보고치와 직접 비교할 수 없다고 저자들이 명시한다.

## 6. 주요 결과 — LIBERO (Table I, 과제당 50 롤아웃)

| 구조 | 변형 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| π0.5 | Baseline | 88.60 | 93.40 | 89.20 | 79.80 | 87.75 |
| π0.5 | LIT | 90.20 | 98.80 | 93.40 | 84.80 | **91.80** |
| MolmoAct2 | Baseline | 93.00 | 97.80 | 95.40 | 87.80 | 93.50 |
| MolmoAct2 | LIT | 94.60 | 96.20 | 95.20 | 90.40 | **94.10** |
| FAST-WAM | Baseline | 98.20 | 100.00 | 97.00 | 95.20 | 97.60 |
| FAST-WAM | LIT | 98.80 | 99.80 | 98.40 | 95.40 | **98.10** |
| ImageWAM | Baseline | 98.40 | 100.00 | 97.60 | 96.40 | 98.10 |
| ImageWAM | LIT | 99.60 | 99.20 | 99.20 | 95.60 | **98.40** |

개별 suite에서는 소폭 하락도 있지만(MolmoAct2 Object −1.6, ImageWAM Long −0.8) 네 구조 모두 평균은 떨어지지 않았다.

## 7. 주요 결과 — LIBERO-Plus 제로샷 (Table II, 10,030 인스턴스)

| 구조 | Baseline Overall | LIT Overall | Δ | 가장 큰 개선 |
|---|---|---|---|---|
| π0.5 | 68.97 | 79.67 | +10.70 | Camera +22.01, Language +19.35 |
| MolmoAct2 | 63.62 | 71.92 | +8.30 | Sensor Noise +21.74, Layout +15.71 |
| FAST-WAM | 51.44 | 60.63 | +9.19 | Camera +27.43, Noise +19.19 |
| ImageWAM | 83.02 | 86.89 | +3.87 | Robot Init +12.38 |

28개 구조×교란 비교 중 26개에서 개선, 하락은 최대 2.11pt(MolmoAct2 Language 75.69→73.58)이다.

## 8. 실로봇 결과 (Sec. IV-D)

MolmoAct2 기반, 3개 과제(Keep LEGOs, Wipe trash, Transfer egg), 과제당 100 시연·총 300 시연으로 단일 다중과제 정책을 학습했다. 세 과제 합산 성공률은 ID 74.7→**88.0**, Lighting OOD 53.3→**70.0**, Camera OOD(상단 카메라만) 30.0→**46.7**, Distractors OOD 50.0→**63.3**. Transfer egg는 ID 52.0→92.0, Lighting·Distractors OOD에서 각각 30.0→90.0, 20.0→90.0; Wipe trash의 Camera OOD는 10.0→80.0으로 두드러진다. 평가 횟수는 ID 과제당 25회, OOD 조건별 과제당 10회로 작다.

## 9. 분석과 어블레이션 (Table III, MolmoAct2)

| 변형 | LIBERO Avg | LIBERO-Plus Overall |
|---|---|---|
| MolmoAct2 baseline | 93.50 | 63.62 |
| LIT w/o Stage 1 | 93.70 | 68.23 |
| LIT w/o pose supervision | 93.50 | 68.86 |
| LIT w/ direct visual access | 94.25 | 67.74 |
| LIT w/o Stage 1 & pose (토큰 집계만) | 93.45 | 65.70 |
| LA4VLA-inspired staged training | 93.75 | 65.46 |
| Baseline w/ pose supervision | 93.20 | 65.45 |
| **LIT** | 94.10 | **71.92** |

세 요소(행동 사전, 자세 감독, 제한된 시각 경로)를 하나씩 빼면 OOD가 3–4pt씩 떨어지고, 각 요소 단독(토큰 집계만, 단계적 학습만, 자세 감독만)은 +2pt 안팎에 그친다. 흥미롭게도 액션 전문가가 백본 시각 표현에 직접 접근하게 하면 ID는 오히려 가장 높지만(94.25) OOD는 67.74로 떨어져 "경로 제한"이 일반화의 핵심임을 보여준다. 학습 곡선상 1단계 flow loss는 10K에서 0.029, 2단계 자세 복원 손실은 20K에서 0.003이며, LIT의 2단계 action loss(20K에서 0.025)가 베이스라인 최종값(30K에서 0.033)보다 낮다. 주의맵 시각화와 반사실 개입(방해물 추가·블러·지시 교체)은 정성 분석이다.

## 10. 강점

- 네 가지 이질적인 구조(공유 self-attention VLA, 층별 cross-attention VLA, 학습 시에만 미래 예측하는 WAM, 추론 시에도 이미지 편집 디노이징하는 WAM)에 같은 전략을 적용해 일반성을 보였다.
- 베이스라인과 학습 데이터·총 스텝 수를 맞춘 쌍 비교라 개선 원인을 방법으로 돌리기 쉽다.
- "요소 단독으로는 부족하다"는 대안 설계 비교(LA4VLA식 단계 학습, 자세 감독만 추가)가 설득력 있다.
- 추론 시 추가 입력이 필요 없고 자세 인코더/디코더는 학습 전용이다.

## 11. 한계

- 모든 결과가 "공개 정책 체크포인트를 미세조정하지 않은" 설정이라, 베이스라인 수치가 공개 보고치보다 낮다(예: π0.5 LIBERO 87.75). 실제로 강한 사전학습 정책 위에서도 이득이 유지되는지는 검증되지 않았다.
- 파라미터 수, 학습 하드웨어, 잠재 인터페이스의 층 공유 간격 m 등 구현 세부가 본문에 부족하다.
- 실로봇 평가는 한 구조(MolmoAct2), 3개 과제, 조건당 10–25회로 통계적 힘이 약하고 과제별 수치 대부분이 그림에만 있다.
- 2단계 자세 복원은 종단 EE 자세를 목표로 하므로, 비정형 물체 조작·접촉 과제처럼 종단 자세만으로 목표가 규정되지 않는 경우의 효과는 불명확하다.

## 12. VLA-Tracker 관점 평가

사전학습 백본 위에서 액션 전문가를 처음부터 학습하고 2단계에서 전체 미세조정하는 자체 정책이므로 수록 대상이다. 트래커 headline(`benchmarks.libero`)은 실로봇과 모든 어블레이션에 쓰인 **MolmoAct2 + LIT**(LIBERO 94.10, LIBERO-Plus Overall 71.92)로 두었고, 논문 최고 LIBERO 평균인 ImageWAM + LIT(98.40)와 π0.5/FAST-WAM 변형, 짝을 이룬 베이스라인, 어블레이션은 별도 블록에 분리했다. LIBERO-Plus 제로샷 개선폭을 여러 구조에 걸쳐 보여 준 "학습 전략형" 논문으로, 강건성 축에서 참고 가치가 크다. 다만 베이스라인이 공개 체크포인트 미세조정이 아니라는 점을 순위 비교 시 감안해야 한다.

<!-- VERIFIED: pdf -->
