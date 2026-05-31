# GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation

> **한 줄 요약**: 학습 가능한 GaussianDream Query를 인코더에 삽입해 현재 프레임의 3D Gaussian 장면과 단기 미래 진화를 동시에 예측하도록 보조 학습하고, 추론 시에는 prefix만 남겨 행동 생성에 조건으로 주입하는 feed-forward 3D Gaussian world-model plug-in. LIBERO 98.4%, RoboCasa Human-50 54.8%, 실로봇 50%를 기록.

---

## 1. 배경 및 동기

기존 VLA 정책은 사전학습된 vision-language 모델의 의미론적 prior를 행동 생성에 전이하는 데 성공해 왔지만, **명시적 3D 공간 정보, dense geometric supervision, 미래 환경 진화**가 부족하다. 이 세 가지는 정밀한 조작에 모두 중요하지만 일반적인 action-imitation 학습에서는 제대로 모델링되지 않는다. GaussianDream은 이를 **3D Gaussian world model**을 **plug-in**으로 붙여 보완하되, 추론 시 비용을 늘리지 않는 feed-forward 설계로 해결하고자 한다.

## 2. 핵심 아이디어

- **GaussianDream Queries**: 인코더에 학습 가능한 query를 추가해 현재 프레임의 3D 공간 구조와 단기 미래 진화를 잠재 prefix로 포착.
- **Dual auxiliary heads (학습 전용)**: prefix를 static reconstruction head와 future prediction head로 처리해 (a) 현재 3D Gaussian 장면 상태, (b) 미래 Gaussian 진화 상태를 각각 산출.
- **Multi-signal supervision**: 현재 분기는 RGB rendering + depth로, 미래 분기는 future RGB + depth + **pseudo 3D scene-flow** 신호로 감독.
- **Inference-time discard**: 추론 시 보조 head를 모두 버리고 학습된 prefix만 행동 생성에 조건으로 주입 — 테스트 시 Gaussian 재구성·미래 예측을 수행하지 않음.
- 결과적으로 **video-based world-model 방식보다 추론 효율이 높음**을 주장.

## 3. 방법론 요약

GaussianDream은 VLA 정책 위에 얹는 **plug-in** 형태로 설계된다. 인코더 안에 학습 가능한 GaussianDream Query를 두고, 이 query들이 입력 시각 토큰과 상호작용하면서 현재 장면과 미래 진화의 3D 구조 정보를 prefix로 압축한다. 학습 단계에서는 prefix를 두 보조 head로 분기시켜 한쪽은 현재 시점 3D Gaussian splat을 재구성(RGB+depth 감독), 다른 한쪽은 미래 Gaussian 진화를 예측(future RGB+depth+pseudo scene-flow 감독)한다. 추론 단계에서는 두 head를 떼어내고 학습된 prefix만 정책의 행동 생성에 conditioning으로 흘려보낸다. 이렇게 하면 3D 기하 정보를 implicit하게 정책에 주입하면서도 inference 시 추가 reconstruction/prediction cost를 발생시키지 않는다.

## 4. 실험 결과

- **LIBERO**: **98.4%** (suite별 분해는 abstract에 미명시).
- **RoboCasa Human-50**: **54.8%**.
- **실로봇 태스크**: **50.0%**.
- 비교 진술: "기존 3D-enhanced VLA 방법 대비 강한 정확도, video-based world-model 방식 대비 더 높은 추론 효율"이라고 명시. 구체 baseline 수치와 latency 수치는 abstract에 미명시.
- 모델 파라미터 수, 학습 데이터 규모는 abstract에 미명시.

## 5. 한계 및 의의

GaussianDream의 매력적인 설계 포인트는 **학습 시에만 3D Gaussian + 미래 예측이라는 무거운 신호를 감독에 사용하고, 추론 시에는 prefix만 남긴다**는 비대칭성이다. 이는 3D world-model의 장점을 취하면서 video-based prediction의 latency 부담을 피하려는 시도로, plug-in 구조 덕분에 다양한 VLA backbone에 이식 가능성이 있다. LIBERO 98.4%는 현 시점 상위권 수치이며, RoboCasa Human-50 54.8% 또한 robocasa 계열에서 의미 있는 결과다. 다만 abstract만으로는 (a) backbone과 action head의 정확한 형식, (b) LIBERO suite별 분해, (c) 어떤 3D-enhanced/video world-model baseline과 비교했는지, (d) 코드 공개 여부가 확인되지 않는다. pseudo 3D scene-flow의 생성 절차와 품질, 그리고 future head를 제거한 ablation도 본문 확인이 필요한 부분이다.

<!-- VERIFIED: abstract-only -->
