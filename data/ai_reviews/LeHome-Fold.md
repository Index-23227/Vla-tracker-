# Learning to Fold: prizewinning solution at LeHome Challenge 2026 (1st place online, 2nd offline)

> **한 줄 요약**: π0.5 기반 flow-matching VLA를 "정책이 곧 자기 자신의 value function"이 되도록 확장하고, AWR(샘플러 재가중)과 RECAP(advantage conditioning)을 HuggingFace Hub로만 통신하는 비동기 학습/롤아웃/DAgger flywheel에서 결합하여, ICRA 2026 LeHome 양손 의류 접기 챌린지에서 시뮬레이션 라운드 1위(62팀 중 79.63%), 실로봇 결선 2위를 차지한 경진대회 솔루션.

---

## 1. 배경 및 동기

- LeHome Challenge 2026[1]은 ICRA 2026 공식 경진대회로, 양팔 SO-ARM101(6-DOF×2, 12차원 joint action, 시뮬레이션 30Hz / 실환경 20Hz, 3개 RGB 카메라)로 테이블 위 의류 한 장을 접는 deformable-object manipulation 태스크다.
- 4종 의류(long/short-sleeved tops, long pants, shorts)를 keypoint 거리 조건으로 **이진(binary) 성공** 판정하며, 부분 점수가 없다(시뮬 라운드 기준).
- 핵심 난점 3가지: (i) 천은 변형이 심해 BC 단독으로는 취약, (ii) 보상이 sparse·binary라 중간 신호를 전부 엔지니어링해야 함, (iii) 리더보드 상당수가 학습 데이터 없는 unseen 의류라 일반화가 필수.
- 저자(Ilia Larchenko, 독립 연구자)는 BC만으로는 부족하다는 판단 하에 flow-matching VLA를 RL로 개선하는 레시피를 구축했다.

---

## 2. 방법론

### 학습 flywheel (Sec. 2)
세 개의 독립 컴포넌트가 오직 HuggingFace Hub로만 통신하는 비동기 루프:
- **Training worker** (GPU 1대, H200): 매 iteration마다 전체 롤아웃 데이터셋에 대해 advantage 재계산 → 약 1000 step 학습 → 약 500 step마다 체크포인트 업로드.
- **Rollout workers** (임의 개수, 주로 RTX PRO 6000): 최신 체크포인트를 받아 3-5개 병렬 Isaac Sim 인스턴스로 에피소드 수집, 예측 value와 함께 업로드. 데이터 수집 확장 = 머신 추가뿐.
- **Manual DAgger station**: 사람이 실패 상태를 teleop으로 교정.
- 동기화 barrier가 없어 trainer/worker가 각자 가진 최신 데이터·체크포인트로 진행한다.

### AWR + RECAP (Sec. 2.2-2.4)
- BC-pretrained 정책 개선의 두 방향: (a) 첫 시도를 더 깔끔히 완료, (b) 실패에서 복구. 저자는 주로 (a)에 집중.
- PPO/GRPO 같은 log-prob policy-gradient는 flow-matching VLA에 깔끔히 전이되지 않으며, "나쁜 행동을 밀어내는" 방식은 유효 action manifold 밖으로 예측을 밀어낸다. 반면 conditioning/reweighting 계열은 manifold를 벗어나지 않고 좋은 행동 쪽으로 확률질량을 재분배한다.
- **AWR (Sec. 2.3)**: loss 가중이 아니라 **샘플러**를 통해 적용 — `P(frame i) ∝ e^clip(A_i, -2, 2)`. 낮은 가중치 프레임은 거의 로드되지 않아 이미지 디코딩/배치 슬롯을 차지하지 않으므로 데이터 효율적. FM loss 자체는 unweighted MSE. auxiliary head 타깃은 importance weight w_i로 보정해 unbiased 유지.
- **RECAP-style conditioning (Sec. 4.3)**: advantage를 입력 conditioning으로 넣어 "좋은 행동만 예측"하도록 지시하며, inference 시 classifier-free guidance(CFG)를 해금. (Physical Intelligence π0.6 계열 아이디어.)

### 정책 = value function (Sec. 5)
- 단일 학습 query token이 image token만 읽는 cheap linear head들에 연결: success probability, task completion(progress), garment type, 현재 keypoint 거리, **30프레임 미래** keypoint 거리, 그리고 action-conditional success residual(Q-function 역할).
- value/Q/경량 world-model 대체물을 정책 안에 두어 별도 critic 없이 하나의 모델만 학습·서빙하고, 이 신호들이 action head와 표현을 공유.

### 보상·advantage (Sec. 6)
- 챌린지 자체 keypoint 조건으로부터 per-garment 중간 checkpoint를 만들어 binary 성공을 densify하되, 실패 시 모든 보상을 회수해 에피소드 return은 binary 유지.
- success head가 dampened(CUPED-style) baseline, completion head가 progress 신호를 제공하고 GAE로 per-frame advantage 결합 — 롤아웃이 오래될수록 outcome-only baseline으로 graceful하게 퇴화.

### Inference-time 최적화 (Sec. 7)
- 동일 체크포인트라도 execution length, playback speed, inpainting onset, guidance scale, noise temperature, best-of-N 후보 수(garment type별)에 따라 다르게 동작.
- 롤아웃 수집 중 **Thompson-sampling bandit**으로 온라인에서 저렴하게 탐색.

---

## 3. 아키텍처 상세

- **구조 (Sec. 4.1)**: SigLIP-So400m/14 image encoder(frozen) → Gemma-2B prefix transformer → Gemma-300M flow-matching action expert가 30-step(1s @ 30Hz) 12차원 joint-delta chunk 생성.
- prefix는 계층적 attention mask로 token group(images+query / state(12 joints, 256 bin discretize)+garment-type token / advantage / FAST tokens+FAST query)을 결합. 각 group은 자신과 이전 group만 보고, FAST token은 training-only로 action expert 실행 전 KV-cache에서 제거.
- BEHAVIOR-1K 2025 우승 솔루션[7](π0.5[8] 확장)에서 그대로 가져온 것(본 연구 기여 아님): 언어 입력 제거(tokenizer 없음), correlated FM noise(action covariance Cholesky, shrinkage β=0.5), correlation-aware soft inpainting, cross-layer KV-cache mixing, multi-sample(5) flow matching.
- **본 연구의 추가**: auxiliary head들(§5), garment-type input token(§4.2), advantage conditioning(§4.3), multi-signal AdaRMS conditioning(§4.4), exclusive self-attention(§4.5), smooth per-timestep action normalization(§4.6).

---

## 4. 실험 설정

- **온라인 라운드 (시뮬, 2026 2~4월)**: Isaac Sim/Isaac Lab, 62팀 공개 리더보드, 의류별 20 instance(seen 10 + unseen 10) 전체 성공률 랭킹.
- **실환경 결선 (2026.6 ICRA Vienna)**: 시뮬 상위 8팀이 실로봇으로 평가. 부분 성공 점수 부여, 심사위원 jury 채점, 의류별 5개(seen 3 + unseen 2). unseen 50% 보너스, 최대 1080점의 composite 점수.
- 핵심 프로토콜: 평가 시 garment 카테고리를 알려주지 않음 → 학습된 garment-type token + inference-time classifier bootstrap으로 추론.
- 하드웨어: 학습 H200 1대, 롤아웃 주로 RTX PRO 6000.

---

## 5. 주요 결과

- **온라인 라운드**: 62팀 중 **1위, 전체 성공률 79.63%** (2위 대비 +6.1점). per-type: long top 74.5% / short top 70.0% / long pants 80.5% / shorts 93.5%. short tops·long pants·shorts에서 단독 최고점.
- **실환경 결선**: **2위, 865점** (1위 sZs 895, 3위 Dum-E 762.5). sim → 자기 로봇 → 평가 로봇의 이중 전이를 1주 sprint로 처리.
- 실로봇 체크포인트 공개: huggingface.co/IliaLarchenko/lehome_real.

---

## 6. 강점

- **단일 모델이 정책+value+Q+경량 world-model** 역할을 모두 수행 → 별도 critic 학습/동기화/서빙 불필요, advantage·best-of-N 신호가 같은 forward pass에서 산출.
- flow-matching VLA에 적합한 RL 선택(conditioning/reweighting이 manifold를 벗어나지 않음)에 대한 설득력 있는 논증.
- AWR을 샘플러로 적용해 100% 배치 활용 + 데이터 효율 확보.
- HF Hub 기반 무barrier 비동기 분산 파이프라인은 단일 연구자가 다수 머신으로 확장 가능한 실용적 엔지니어링.

---

## 7. 약점·한계

- 저자 스스로 "통제된 실험이 아닌 엔지니어링 case study"라 명시 — 정식 ablation이 거의 없어 각 컴포넌트의 필요성 입증이 부족.
- DAgger 복구 전략이 시뮬에서 잘 안 됐고(§3.4) 큰 도움이 안 됨 — robustness ≠ 망가진 상태로부터의 복구.
- 시뮬 rendering artifact에 과적합 → 실환경 전이에 대량 augmentation 필요, 깔끔한 해결책 미발견.
- 모델이 태스크 규모 대비 과대(저자: 훨씬 작은 모델로도 가능). full-scale VLA RL 실험이 진짜 동기였음.

---

## 8. 핵심 기여 (정리)

1. flow-matching VLA를 위한 AWR(샘플러 적용) + RECAP advantage conditioning 결합.
2. HF Hub만으로 통신하는 비동기 분산 학습/롤아웃/DAgger 파이프라인.
3. 정책 내부 auxiliary head로 value/Q/progress/미래 keypoint를 예측해 정책을 자기 value function으로 사용.
4. Thompson-sampling bandit 기반 inference-time 하이퍼파라미터 온라인 최적화.
5. camera-alignment tooling + heavy augmentation + DAgger HIL 기반 sim-to-real 레시피.

---

## 9. 의의

- 단일 독립 연구자가 비교적 적은 컴퓨트(H200 1대)로 대규모 VLA를 RL fine-tune하여 ICRA급 경진대회에서 상위권을 달성한 사례로, flow-matching VLA + advantage 기반 RL의 실전 적용 가능성을 보여준다.
- "정책 = value function" 패턴은 별도 critic이 부담스러운 VLA RL 세팅에 재사용 가치가 있다.

---

## 10. 재현성

- 코드/체크포인트: 실로봇 체크포인트(lehome_real)가 HuggingFace에 공개. 챌린지 환경·에셋·success checker는 주최측 공개[1].
- 다만 ablation 부재와 경진대회 압박 하의 반복적 구축 특성상 완전한 레시피 재현은 상당한 엔지니어링을 요구.

---

## 11. 종합 평가

- 학술적 새로움보다 **기존 RL 아이디어(AWR/RECAP/DAgger/GAE)의 영리한 재조합 + 엔지니어링**이 강점인 prizewinning 솔루션. flow-matching VLA를 RL로 개선하려는 실무자에게 구체적이고 검증된 레시피를 제공한다는 점에서 가치가 높다.

---

## 12. 예상 질문과 답변

| # | 질문 | 답변 |
|---|------|------|
| 1 | 왜 PPO/GRPO 대신 AWR+RECAP인가? | log-prob policy-gradient는 flow-matching VLA에 깔끔히 전이되지 않고, "나쁜 행동 억제"는 예측을 유효 manifold 밖으로 민다. conditioning/reweighting은 manifold 안에서 좋은 행동으로 확률질량을 재분배 — 새로운 행동 탐색보다 한 번에 확실히 접는 본 태스크에 유리. |
| 2 | "정책이 자기 value function"이 실제로 어떻게 쓰이나? | 단일 query token이 success/completion/garment-type/keypoint 거리/30프레임 미래 거리/action-conditional success residual(Q)을 예측하고, 이들이 GAE advantage, live 실패 감지, best-of-N 후보 선택을 구동한다. 별도 critic 불필요. |
| 3 | 평가 시 의류 종류를 모르는데 어떻게 대응하나? | 학습된 garment-type input token + inference-time classifier bootstrap(§4.2, §7.6)으로 종류를 추론한다. |
| 4 | AWR을 loss가 아닌 샘플러로 적용한 이유는? | 기댓값상 동등하지만 더 데이터 효율적 — 낮은 가중 프레임은 거의 로드되지 않아 이미지 디코딩·배치 슬롯을 안 쓴다. 단, auxiliary head는 importance weight로 보정해 unbiased 유지. |

<!-- VERIFIED: pdf -->
