# EvoScene-VLA: Evolving Scene Beliefs Inside the Action Decoder for Chunked Robot Control

> **한 줄 요약**: 청크 단위(chunked) VLA 정책이 청크 내부에서 발생하는 접촉/가림/물체 이동을 반영하지 못하는 문제를, action decoder 내부에 들어가는 *recurrent scene prefix*로 풀어 RoboTwin 31개 태스크 평균 성공률을 87.2% → 89.1%(fixed), 86.1% → 88.5%(randomized)까지 끌어올린 연구.

---

## 1. 배경 및 동기

청크 단위 VLA 정책(예: 한 번의 VLM 호출로 여러 스텝의 action을 예측하는 방식)은 각 청크의 시작 시점에서 본 단일 시각 관측만을 조건으로 사용한다. 그러나 로봇 action 자체가 접촉, 가림(occlusion), 물체 이동을 유발하기 때문에, 다음 시각 업데이트가 도착하기 전에 이미 장면 기하(geometry)는 변해 있다. 기존 *Spatial VLA*는 현재 프레임의 기하만 강화하고, *Temporal VLA*는 과거 프레임을 누적할 뿐, **action에 의해 갱신되는 장면 사전(prior)을 청크 사이에 유지하는 메커니즘은 없다**는 것이 저자들의 진단이다.

---

## 2. 핵심 아이디어

- **Recurrent scene prefix**: 청크 사이를 가로질러 유지되는 geometry-aware scene state를 action decoder 내부에 둠.
- **VLM 호출 시 융합**: 매 호출마다 현재 관측에서 추출한 장면 정보와, 직전 청크에서 갱신된 사전(prior)을 결합.
- **이중 출력**: action decoder가 다음 action 청크와 함께 *compact scene update*를 동시에 출력. 이 업데이트가 다음 호출의 prior가 되고, 새로운 관측이 들어오면 VLM이 이를 보정.
- **학습 보조 모듈 (배포 시 제거)**:
  - *Scene Predictor* — 미래 scene-token 타깃 공급.
  - *Geometric Anchor* — frozen depth 및 3D teacher와 scene slot을 정렬.
- 결과적으로 매 제어 호출이 "최근 action의 영향 + 새 시각 증거"가 함께 반영된 prior에서 시작.

---

## 3. 방법론 요약

EvoScene-VLA는 VLM backbone + action decoder의 일반적인 청크 VLA 구조 위에, 청크 경계를 가로질러 흐르는 *scene prefix*라는 잠재 상태를 도입한다. 이 prefix는 VLM 입력의 한 축으로 들어가며, action decoder는 (1) 다음 action 청크, (2) 다음 청크의 prefix가 될 scene update를 함께 산출한다. 학습 시에는 Scene Predictor가 미래 scene-token에 대한 자기지도 타깃을 제공하고, Geometric Anchor가 frozen depth/3D teacher로부터 기하적 정합을 강제한다. 두 보조 모듈은 추론 시에는 제거되어 추가 비용 없이 prior가 흐른다.

---

## 4. 실험 결과 (abstract 보고치 기반)

| 평가 | Baseline | EvoScene-VLA |
|------|----------|--------------|
| RoboTwin 31 tasks · fixed | 87.2% | **89.1%** |
| RoboTwin 31 tasks · randomized | 86.1% | **88.5%** |
| Galaxea R1-Lite real robot | — | 모든 baseline 상회(정성적 보고) |

- 정확한 baseline 모델명 및 태스크별 분해 결과는 abstract에 미명시.
- Real robot 평가의 정량 지표(task별 성공률, 시도 횟수 등) abstract에 미명시.

---

## 5. 한계 및 의의

**의의**
- 청크 VLA의 본질적 약점("청크 내 환경 변화 무시")을 직접 짚고, **action decoder 안에 상태가 흐른다**는 비교적 단순한 설계로 RoboTwin 평균 +1.9~2.4%p의 이득을 보고.
- 보조 모듈을 배포 시 제거하므로 추론 비용 증가가 거의 없다는 점이 실용적.

**한계**
- Parameter count, backbone, latency, 학습 자원 등 구체 수치 abstract에 미명시.
- Baseline의 정체와 RoboTwin 버전(v1/v2) 명시 abstract에 미명시(평균 87% 수준 baseline은 RoboTwin2 류일 가능성).
- Scene prefix가 실제로 "기하 변화"를 표현하는지 검증할 시각화/probing 실험은 abstract 범위 밖.
- LIBERO/CALVIN 등 비-RoboTwin 시뮬레이터에서의 결과 부재.

<!-- VERIFIED: abstract-only -->
