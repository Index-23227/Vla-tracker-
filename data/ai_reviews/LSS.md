# LSS: Reasoning Without Inference Cost — Latent Semantic Scaffolding for Robot VLA Policies

> **한 줄 요약**: 추론(reasoning)의 이점을 추론 시점이 아닌 학습 시점에 흡수하기 위해, VLA의 action token 표현을 물리적 추론 문장의 T5 임베딩에 정렬하는 보조 손실(LSS)을 인간 시연 사전학습 단계에 추가하고 배포 시 정렬 head를 버린다. 특히 각 token을 자신의 조작 단계(phase) 추론에 정렬하는 Dense LSS가 held-out 과제 전이에서 가장 좋다. H-RDT 기반 RoboTwin 2.0 3과제에서 adjust_bottle 70→90, shake_bottle 34→54, move_can_pot 9.1→26.

- **arXiv**: 2609.04893v1 (2026-09-04, cs.RO)
- **소속**: The Chinese University of Hong Kong, CLOVER Lab
- **백본**: H-RDT (DINO+SigLIP, T5-XXL, 2B diffusion transformer, flow-matching head)
- **코드**: 공개 정보 없음

---

## 1. 배경 및 동기

모방 학습으로 훈련된 VLA는 *무엇을* 할지는 배우지만 *왜* 하는지는 배우지 못한다. ECoT(텍스트 추론), CoT-VLA(미래 이미지), WorldVLA 같은 world action model은 추론이 조작 성능을 올린다는 것을 보였지만 매 제어 step마다 추론 비용을 다시 지불한다(WAM은 flow VLA 대비 4.8×~83× 느리다는 보고). 저자들은 "추론의 이점을 가중치에 굽고(bake) 런타임 비용은 없앨 수 있는가"를 묻는다.

## 2. 핵심 아이디어

추론 텍스트를 **입력이나 생성 목표가 아닌 정렬 목표**로 쓴다. 추론 문장은 정책을 조건화하지 않으며, 오직 보조 손실을 통해 backbone 표현을 형성한다. 정렬 head와 추론 인코더는 추론 시 제거되므로 배포 정책은 원래 base VLA와 동일한 계산 그래프를 가진다.

## 3. 백본 및 3단계 파이프라인

H-RDT: DINO+SigLIP 비전 인코더, 동결 T5-XXL 명령 인코더, 2B diffusion transformer, flow-matching action head. K = 16 action token(차원 2176)을 생성. 원래 2단계(인간 사전학습 → 로봇 미세조정) 사이에 **LSS 인간 사전학습 단계**를 삽입해 (i) 인간 사전학습 → (ii) LSS 인간 사전학습(48-D 인간 행동) → (iii) 로봇 미세조정(14-D)으로 구성.

## 4. Pooled LSS

LSAHead g_ϕ(2층 MLP, 2176→2176→4096, GELU)가 action token을 mean-pool한 벡터를 T5 공간으로 사상하고, 추론 문장의 mean-pooled T5 임베딩과 cosine 거리로 정렬. 총손실 L = L_flow + λ L_LSS, λ = 0.1(행동 예측을 덮어쓰지 않는 약한 보조 압력).

## 5. Dense LSS (phase-local 정렬)

추론을 approach / grip / rotate / withdraw 같은 순서 있는 phase로 분할하고 각 phase마다 한 문장 인과 추론과 프레임 범위를 둔다. 각 action token은 자신이 걸친 프레임의 phase에 배정되고(upsample_rate = 3), phase 경계에 걸친 약 5% token은 마스킹. 각 token을 자기 phase의 임베딩에 정렬한다. 목적 함수 형태는 동일하고 목표만 바뀐다. 단, phase 경계가 부정확하면 오히려 pooled보다 나빠질 수 있다는 점을 저자가 명시.

## 6. 데이터 및 실험 설정

- 인간 시연: Apple Vision Pro(48-D 손·손목) + 머리 장착 ZED 스테레오로 EgoDex 형식 수집, adjust_bottle 500 에피소드(좌/우 손 × Coke/Sprite 2×2 균형).
- 추론 문장: Gemini-3-flash가 인과 구조를 설명하도록 오프라인 생성, Dense용 phase 분할도 함께 반환. 텍스트에 대한 공식 품질 관리는 없음.
- 평가: RoboTwin 2.0 aloha-agilex 14-DOF, Stage 3는 모두 50 에피소드·22,000 step 동일 예산, 100 rollout.

## 7. 주요 결과

**Transfer (Table II)** — Stage-2 backbone별 동일 Stage-3 미세조정:

| 과제 | R1 H-RDT | R2 +인간 데이터 | R3 Pooled LSS | R4 Dense LSS |
|---|---|---|---|---|
| adjust_bottle (in-dist.) | 70 | 79 | 85 | **90** |
| shake_bottle (held-out) | 34 | 45 | 39 | **54** |
| move_can_pot (held-out) | 9.1 | 20 | 19 | **26** |

- Dense LSS가 모든 과제에서 최고. Pooled LSS는 학습 과제에는 도움이 되지만 shake_bottle에서 R2보다 낮아지는(45→39) 과특화를 보인다.
- R2~R4가 같은 인간 데이터를 쓰므로 R3/R4의 R2 대비 차이는 LSS에 귀속된다.

## 8. Ablation (Table I, adjust_bottle)

| Run | 메커니즘 | 성공률 |
|---|---|---|
| A | EgoDex 사전학습 + 미세조정 | 70 |
| B | AVP 운동학, 추론 없음 | 79 |
| C | 추론을 언어 입력으로 | 71 |
| D | 추론을 token 예측 손실로 | 74 |
| E | Pooled LSS | 85 |
| – | 자명한 지시문 텍스트에 정렬 | 58 |
| F | Run E Stage 2, 미세조정 시 backbone 동결 | 0 |
| G | Dense LSS | 90 |

- 정렬 *내용*이 중요하다(자명한 목표 58 vs 추론 85).
- 추론을 입력으로 주면 오히려 B보다 낮아진다(71 < 79).
- 표현 probe: Dense가 Pooled 대비 약 2배 phase 분리도(silhouette 0.047 vs 0.021, AVP-only 기준).

## 9. 강점

- 추론 효과와 추론 비용을 분리한다는 명확한 문제 설정과 **추론 비용 0**이라는 실용적 이점.
- 정렬 목표 내용, 입력 vs 손실 vs 정렬, 동결 여부 등 통제된 ablation ladder로 메커니즘을 분리.
- "정렬 입도(granularity)"가 전이의 핵심 레버라는 구체적이고 검증 가능한 주장.

## 10. 약점 및 한계

- Stage-2 정렬이 단일 인간 과제(병 조작)에만 적용되고, 평가도 RoboTwin 2.0의 3개 과제뿐이라 표준 전체 벤치마크 평균과 비교할 수 없다.
- 실제 로봇 검증이 없다(저자가 향후 과제로 명시).
- seed 반복·분산 보고가 없어 수 %p 차이의 통계적 유의성이 불분명하다.
- VLM 생성 phase 경계·추론 문장에 품질 관리가 없다.
- 추론을 가중치에 굽기 때문에 테스트 시 재계획 능력은 없다.

## 11. 재현 및 확장 아이디어

- H-RDT 공개 코드에 LSAHead와 cosine 정렬 손실만 추가하면 되므로 구현은 단순하다.
- 더 다양한 인간 과제·객체로 Stage 2를 확장해 전이 범위 측정.
- π0/π0.5 같은 VLM 기반 VLA의 action expert token에 동일 정렬을 적용하는 일반화.
- 텍스트 대신 미래 시각 특징(REPA 스타일)과 추론 텍스트를 함께 정렬하는 다중 teacher 변형.

## 12. 총평

LSS는 "추론은 생성하지 않아도 표현을 형성하는 것만으로 도움이 된다"는 흥미로운 가설을 작은 규모에서 깔끔하게 검증한 논문이다. Dense 정렬이 held-out 과제 전이를 개선하고 pooled 정렬은 과특화된다는 결과는 설득력 있지만, 평가가 3개 시뮬레이션 과제·단일 seed에 한정되어 있어 결론의 일반성은 더 넓은 벤치마크와 실제 로봇으로 확인되어야 한다. 비용 없는 추론 주입이라는 방향성 자체는 가치가 높다.

<!-- VERIFIED: pdf -->
