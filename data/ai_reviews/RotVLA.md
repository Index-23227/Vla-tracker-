# RotVLA: Rotational Latent Action for Vision-Language-Action Model

> **한 줄 요약**: 기존 Latent Action Model(LAM)의 *이산 양자화* 인코더/디코더가 갖는 trivial reconstruction·표현력 제한·물리적 무의미성 문제를, **SO(n) 위의 연속 회전 잠재 액션**과 *triplet frame learning*, 그리고 flow-matching action expert로 풀어 1.7B 파라미터로 LIBERO 98.2%, RoboTwin2.0 89.6% / 88.5%(clean / randomized)를 달성한 VLA.

---

## 1. 배경 및 동기

VLA pretraining에서 heterogeneous 데이터(서로 다른 embodiment, 다양한 action space)를 다루기 위한 유력한 패러다임이 **Latent Action Model**이다. 그러나 기존 LAM은 대부분 *discrete quantization* 기반 encode/decode 파이프라인에 의존하며, 이는 (1) 디코더가 입력 프레임을 그대로 복원하는 trivial 행동, (2) 제한된 표현력, (3) **물리적으로 의미 있는 구조의 부재**라는 세 가지 문제를 동반한다. 저자들은 잠재 action을 "회전군 SO(n) 위의 연속 원소"로 다루는 것이 이 세 문제를 동시에 해결할 수 있다고 주장한다.

---

## 2. 핵심 아이디어

- **회전 잠재 액션 (Rotational Latent Action)**: latent action을 SO(n)의 원소로 모델링 → 연속성, 합성성(compositionality), 물리적 dynamics와 정합되는 구조적 기하 확보.
- **Triplet frame learning**: 의미 있는 temporal dynamics를 강제하고 degeneration(trivial reconstruction 류)을 회피.
- **VLM backbone + flow-matching action head**: 대규모 cross-embodiment 로봇 데이터 + 인간 영상으로 latent-action supervision 하에 pretrain.
- **Unified action expert**: 다운스트림에서는 flow-matching head를 *latent action + robot action을 함께 denoise*하는 expert로 확장. 잠재 action이 high-level planner처럼 작동하여 action 생성을 조건화.
- **규모**: 1.7B 파라미터, 1700+ 시간 pretraining 데이터.

---

## 3. 방법론 요약

RotVLA의 구조적 핵심은 **"이산 잠재 action 코드북"을 "SO(n) 연속 매니폴드"로 교체**한 것이다. SO(n) 원소는 그 자체로 합성(연쇄 곱)이 정의되고 연속 보간이 자연스럽기 때문에, manipulation처럼 회전 성분이 중요한 action 공간과 잘 정합된다. Triplet frame 구성(예: 앵커/양성/음성 프레임)은 시간 차이가 의미 있는 잠재 변화를 만들어내도록 학습 신호를 준다. Pretraining 단계에서 VLM은 이미지·언어를 인코딩하고, 별도의 flow-matching head가 latent action 분포를 학습한다. 다운스트림에서는 이 head를 "latent + 실제 robot action"을 한 번의 flow-matching으로 디노이즈하는 *unified action expert*로 키우며, 이때 latent action은 고수준 plan으로 기능한다.

---

## 4. 실험 결과 (abstract 보고치 기반)

| Benchmark | Setting | RotVLA |
|-----------|---------|--------|
| LIBERO | (suite 미명시; 평균으로 추정) | **98.2%** |
| RoboTwin2.0 | clean | **89.6%** |
| RoboTwin2.0 | randomized | **88.5%** |
| Real world | manipulation tasks | "consistently outperforming existing VLA models"(정성) |

- LIBERO suite별 분해(spatial/object/goal/long) abstract에 미명시.
- RoboTwin2.0 태스크별 분해 abstract에 미명시.
- 비교된 baseline 명칭 abstract에 미명시.

---

## 5. 한계 및 의의

**의의**
- LAM 계열의 고질적 문제(이산화, trivial reconstruction)를 **수학적으로 자연스러운 회전군**으로 우회한 점이 개념적으로 깔끔하다.
- 1.7B + 1700h라는 비교적 *컴팩트한* 규모로 LIBERO 98.2%·RoboTwin2 89%대를 보고하면서, latent를 단순한 표현이 아니라 *latent planner*로 활용하는 점이 실용적.
- Flow-matching head를 latent와 실제 action에 동시 적용하는 "unified action expert" 디자인은 reuse 측면에서도 매력적.

**한계**
- SO(n)의 n 선택, 회전 표현 방식(쿼터니언/Lie algebra/행렬 등), 그리고 어떤 manipulation 차원에 어떻게 매핑되는지 abstract에 미명시.
- Triplet 구성 방식(샘플링 전략, hard negative 사용 여부 등) abstract 범위 밖.
- LIBERO 98.2%의 split 단위 평균인지, 어떤 suite를 포함하는지 명시 없음.
- Real world 평가의 정량 지표·비교 baseline 미보고.
- Code/모델 공개 여부 abstract에 미명시.

<!-- VERIFIED: abstract-only -->
