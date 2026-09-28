# PAVE — Predictive Alignment and Value-Guided Evolution for World-Action Policies

- arXiv: 2608.30378 (v1 2026-08-31, v3 2026-09-18; v3 PDF 제목 "PAVE: Separates Prediction from Preference in World-Action Policies")
- 소속: East China Normal University (Multi-Dimensional Information Processing Lab), EBKernel Technologies (Lab for Brain-Inspired Embodied Intelligence), Shanghai AI Laboratory (Botong Zhao, Fang Yu, Tim Yu, Senhua Zhu, Xinyuan Chen, Yue Lu)

## 1. 한 줄 요약

PAVE는 JEPA-WAM 계열의 world-action 정책(고정 V-JEPA 2.1 + LoRA Qwen2.5-0.5B + DiT flow matching)을 **실패를 포함한 혼합 품질 롤아웃으로 반복 개선(policy evolution)**하는 후학습 방법이다. 유효한 전이는 성공·실패와 무관하게 모두 예측 학습에 쓰고, 행동 학습에는 오프라인 critic이 매긴 상대적 품질 조건(positive/negative)을 붙인다. 로봇 사전학습 없이 LIBERO 98.9%, LIBERO-Plus 87.0%, RoboTwin 2.0 (20과제) Clean/Random 86.8/45.2%, 실로봇 S1에서 85.7/92.0%를 보고한다.

## 2. 문제 설정

경험 기반 후학습(RECAP, FlowPRO, RedFlow, ALOE 등)은 "어떤 경험을 남길까"에 집중한다. 저자들은 예측 목적과 행동 목적이 한 actor를 공유할 때 **같은 경험이 무엇을 감독해야 하는가**라는 "supervision-role mismatch"를 제기한다. 실패한 에피소드라도 관측된 미래는 유효한 예측 증거지만, 그 행동을 그대로 모방해서는 안 된다. 성공 필터는 쓸모 있는 전이를 버리고, 무조건 모방은 선호를 무시한다. 두 번째 문제는 고정 오프셋 미래 타깃이 에피소드 진행에 따라 남은 구간의 다른 비율을 덮고 끝에서 잘린다는 점이다.

## 3. 핵심 아이디어

세 가지 결합된 메커니즘이다.
1. **Trajectory-relative predictive alignment**: 로컬 타깃(δ=31) 외에, 남은 궤적의 고정 비율 ρ={0.25, 0.5, 0.75, 1.0} 지점의 joint current-future V-JEPA 임베딩을 horizon 임베딩이 붙은 예측기로 맞춘다. 초반에는 넓은 변화, 종료 근처에서는 세밀한 변화를 덮는다.
2. **Action-conditioned predictive critic**: 공유 fusion 표현 위에 분포형 return 헤드(201 bin)와, 실행된 행동 청크를 받아 미래 잠재를 예측하는 JEPA 헤드를 함께 학습한다.
3. **Episode-excluded quality conditioning + asymmetric reuse**: 2-fold 교차적합 critic으로 N-step advantage를 계산해 과제별 상위 30% 청크를 positive, 나머지를 negative로 라벨링한다. 두 조건 모두 actor 학습에 쓰이고, 배포 시에는 positive를 선택한다.

## 4. 아키텍처

- Actor: 고정 V-JEPA 2.1 ViT-L/16 (384², 뷰당 576×1024 토큰) → 고정 projector(896) → Qwen2.5-0.5B (LoRA rank 32, α=64, dropout 0.1). 입력 순서는 시각 토큰, 지시문, 품질 조건, 64개 행동 placeholder.
- 행동 디코더: 16층 DiT, flow matching velocity 예측, 행동 위치 상태 + 고유감각 + 32개 learnable future query 토큰 조건. H=20, Euler 4스텝.
- 예측 브랜치: 시각 위치 상태만 사용(인과 마스크 때문에 뒤의 지시/품질 토큰을 보지 못함), 기록된 행동 청크는 입력에서 제외해 flow matching 타깃 누설을 막는다.
- Critic: 고정 V-JEPA 2.1 + 고정 Gemma-2-2B(과제 텍스트, 256-bin 이산화 고유감각) → (Dv+Dl)→1024→512 fusion MLP → return 헤드 / JEPA 헤드. fusion 파라미터 3,934,720개.
- 배포 시 target encoder, 예측 헤드, critic 모두 제거. 온라인 파라미터 1.29B.

## 5. 학습과 추론

Actor 손실은 L_FM + 0.5·L_local + 0.5·L_MH. 두 손실 모두 Qwen LoRA를 갱신하지만 디코더와 learnable query는 행동 손실만, 보조 헤드는 예측 손실만 받는다. Critic 보상은 비종단 −1, 성공 종단 0, 실패 종단 −M_task이며 two-hot 분포 회귀로 학습. 기본 actor 60,000 업데이트 후 3라운드 진화(라운드당 actor 30,000, fold별 critic 8,000 업데이트), 데모:진화 데이터 샘플링 3:1, 조건 dropout 0.3. 시뮬레이션은 라운드당 학습 그룹·시드별 96 에피소드(누적 288), S1은 과제당 라운드 50 에피소드(누적 150). 라운드당 actor 31.20 GPU-hours, critic 8.40 GPU-hours(H200).

## 6. 주요 결과 (Tables 1–4, 시뮬레이션 5시드 평균)

| 벤치마크 | JEPA-WAM (demo-only, 로컬) | Local+QC (매칭 대조) | PAVE |
|---|---|---|---|
| LIBERO Avg | 96.6±0.4 | 97.5±0.3 | **98.9±0.2** |
| LIBERO-Plus Avg (zero-shot) | 79.0±0.8 | 82.0±0.7 | **87.0±0.7** |
| RoboTwin 2.0 20과제 Clean | 79.6±1.1 | 82.5±0.9 | **86.8±1.3** |
| RoboTwin 2.0 20과제 Random | 36.7±0.8 | 40.7±1.4 | **45.2±0.7** |

LIBERO 세부: Spatial 98.6 / Object 99.5 / Goal 98.9 / Long 98.5. LIBERO-Plus 세부: Camera 86.0, Robot 71.0, Language 80.5, Light 97.4, Background 97.4, Noise 91.2, Layout 85.7. 참고로 논문이 옮겨 적은 공개 수치는 π0.5+JEPA LIBERO 97.8 / LIBERO-Plus 86.3, π0.5 LIBERO-Plus 84.5.

실로봇 S1 (Table 4, 3시드 × 100 trials): 테이블 정리 85.7±3.5 vs Local+QC 80.0±1.0, 비커 붓기 92.0±2.0 vs 84.7±1.5.

## 7. 매칭 대조 실험 (Table 6)

A. 동일 풀 2×2 요인 설계 (LIBERO-Plus / RT Random): 로컬+무조건 80.0/38.4, 로컬+상대+무조건 81.6/39.8, 로컬+QC 82.0/40.7, 로컬+상대+QC 87.0/45.2. 두 요소가 따로는 작은 이득이지만 결합하면 큰 이득 — 상호작용 효과가 핵심이다.
B. 경험 사용 규칙: 성공만 예측 83.5/41.8, 양쪽 모두 positive 필터 82.8/40.7, 진화 중 예측 gradient 정지 83.1/41.5, 모든 유효 예측+positive만 행동 83.4/41.9, PAVE 87.0/45.2.

## 8. 진화 라운드와 조건 대조 (Tables 10, 11)

기본 actor 81.2/39.0 → 라운드 1 83.6/40.8 → 라운드 2 85.9/42.2 → 라운드 3 87.0/45.2 (LIBERO-Plus / RT Random). 같은 풀에서 무라벨 BC는 81.6/39.8로 거의 늘지 않는다. 한 모델의 테스트 조건만 바꾸면 null 82.9/41.6, negative 66.8/24.7, positive 87.0/45.2로, 조건 토큰이 실제로 행동 모드를 분리함을 보여준다.

## 9. 어블레이션 (Tables 12–19)

- 앵커 수 K: 1개 82.1/40.9, 2개 83.5/42.0, 3개 84.0/42.6, **4개 87.0/45.2**, 6개 85.7/43.0, 8개 84.4/42.5 — 커버리지와 해상도의 균형.
- 동일 타깃 예산에서 고정 오프셋 {16,32,64,128} 83.6/42.2, 무작위 상대 비율 84.2/42.8, 고정 상대 비율 87.0/45.2.
- Actor 시각 인코더: DINOv2 97.5/82.0/40.4, LingBot-Vision 97.8/83.1/41.7, V-JEPA 2.1 98.9/87.0/45.2 (LIBERO/LIBERO-Plus/RT Random).
- Critic: 스칼라 V 83.7/42.1 vs 분포형 V 87.0/45.2; Q−V 점수화 84.5/43.5; λ_J=0 85.5/44.3; RECAP 가치함수 84.7/43.1.
- 교차적합: in-sample 84.8/43.4, 2-fold 87.0/45.2, 5-fold 84.7/43.3, 동일 총 업데이트 2-fold 84.1/42.7.

## 10. 강점

- "예측 증거"와 "행동 선호"를 분리한다는 문제의식이 명확하고, 실패 데이터의 전이를 버리지 않는 설계가 설득력 있다.
- 동일 풀·동일 초기화·동일 업데이트 예산의 매칭 대조, 2×2 요인 설계, 조건 스왑, λ_J=0 대조 등 기여를 분리하는 실험 설계가 매우 꼼꼼하다.
- 시뮬레이션 5시드, 실로봇 3시드의 시드별 수치와 표본 SD를 모두 보고.
- critic과 예측 헤드가 모두 오프라인이라 온라인 비용이 매칭 JEPA-WAM과 거의 동일(p50 63.60 vs 63.40 ms, 10.60 GB).

## 11. 한계

- RoboTwin 2.0은 JEPA-WAM의 20과제 매니페스트로, 표준 50과제 프로토콜과 직접 비교할 수 없다. 공개 기준선 행은 다른 데이터 조건(롤아웃 없음)이라 공정 비교가 아니며 저자들도 이를 "참고 맥락"으로 명시한다.
- 개선은 동일 경험·동일 actor 업데이트 기준이며, 오프라인 계산량(critic 학습, fold별 critic)은 추가된다.
- 이진 라벨(상위 30%)은 클래스 내 품질 차이를 버리고, positive가 A>0이나 성공을 뜻하지 않는다. 안정적 단조 개선 보장이 없다.
- 실로봇은 2과제, 3시드로 규모가 작고 통계적 유의성은 주장하지 않는다.
- 코드·가중치 공개 언급이 없고, 논문 작성에 생성형 AI(Codex 포함)를 광범위하게 사용했다고 명시한다.

## 12. VLA-Tracker 관점 평가

PAVE는 자체 actor를 LoRA + DiT로 실제 학습하고 롤아웃으로 3라운드 후학습하므로 트래커 수록 대상이다. LIBERO Avg 98.9는 로봇 대규모 사전학습 없는 모델 중 최상위권이며, LIBERO-Plus 87.0은 논문에 인용된 π0.5+JEPA(86.3)보다 높다. 다만 이 수치는 데모에 더해 288개의 clean 학습 롤아웃을 사용한 결과라 순수 데모 학습 모델과는 조건이 다르다. RoboTwin 결과는 20과제 부분집합이므로 표준 robotwin_v2 순위가 아닌 별도 블록(robotwin_v2_jepawam_20task)으로 기록했다. 기여의 본질은 새 아키텍처가 아니라 world-action 정책을 위한 "혼합 품질 경험 재사용 레시피"이며, JEPA-WAM 및 RECAP 계열과 함께 비교해 볼 가치가 있다.

<!-- VERIFIED: pdf -->
