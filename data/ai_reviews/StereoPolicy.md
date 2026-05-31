# StereoPolicy: Improving Robotic Manipulation Policies via Stereo Perception

> **한 줄 요약**: 명시적 3D 재구성이나 카메라 캘리브레이션 없이, 사전학습된 2D 인코더로 좌/우 stereo 영상을 각각 처리한 뒤 Stereo Transformer로 융합하여 disparity와 공간 대응을 implicit하게 학습하고, diffusion 및 VLA 정책에 plug-in으로 결합해 RGB · RGB-D · point cloud · multi-view 베이스라인을 일관되게 능가하는 visuomotor 정책 프레임워크.

---

## 1. 배경 및 동기

- 최근 imitation learning은 **단안(monocular) 시각만으로** 다양한 객체를 조작하는 강력한 visuomotor 정책을 만들어냈으나, monocular 입력은 본질적으로 **신뢰할 만한 depth와 공간 인식**이 부족하다.
- 혼잡하거나 기하적으로 복잡한 장면, 정밀한 조작에서는 이 한계가 명백한 병목.
- 한편 RGB-D나 point cloud, multi-view 같은 대안은 **calibration, 3D 재구성, sensor 비용**을 요구해 scalability에 약점이 있다.

---

## 2. 핵심 아이디어

- 동기적으로 촬영된 **stereo image pair**를 직접 정책 입력으로 사용 → 명시적 3D 재구성 없이 기하 추론 강화.
- 두 영상에 각각 **사전학습된 2D vision encoder**를 적용해 강력한 representation을 확보.
- 두 영상의 feature를 **Stereo Transformer**로 융합하여 **공간 대응(spatial correspondence)과 disparity cue**를 implicit하게 포착.
- **Calibration-free**: 카메라 보정 없이도 동작해 실배포 부담 감소.
- Diffusion-based 정책 및 pretrained VLA 정책 모두에 **plug-in 형태**로 결합 가능.

---

## 3. 방법론 요약

- 입력: 동기화된 좌·우 RGB 이미지 한 쌍.
- 각 이미지는 동일한 사전학습 2D encoder(예: ViT 계열로 추정, abstract에 구체적 명시 없음)를 통과.
- 두 view의 patch token sequence를 Stereo Transformer가 cross-attention 등으로 융합 → 단일 visuomotor representation 산출.
- 산출된 표현은 그대로 diffusion policy head 또는 pretrained VLA policy head에 전달.
- 별도의 depth 예측, point cloud 생성, calibration 단계가 없음 — 모든 기하 추론이 융합 transformer 내부에서 implicit하게 일어남.

---

## 4. 실험 결과

- 시뮬레이션 3개 벤치마크에서 일관된 성능 향상:
  - **RoboMimic**
  - **RoboCasa**
  - **OmniGibson**
- 비교 베이스라인: **RGB, RGB-D, point cloud, multi-view** — 모두에 대해 일관된 우위.
- 실로봇 실험: **테이블탑** 및 **bimanual mobile manipulation** 환경에서 검증.
- 구체적 수치(success rate, 평균 등)는 abstract에 미명시.

---

## 5. 한계 및 의의

- **의의**:
  - "Stereo가 사실상 가장 저렴한 3D modality"임을 정책 학습 관점에서 재확인. **2D 사전학습 ↔ 3D 기하 이해**를 잇는 가벼운 다리.
  - Diffusion 정책과 VLA 정책 양쪽에 모두 결합 가능하다는 점은 본 프레임워크가 정책 아키텍처에 종속되지 않는 **모듈러 perception 컴포넌트**임을 보여준다.
  - Calibration 없이 동작 → 다양한 platform/robot에 빠르게 확장 가능.
- **한계**:
  - Abstract에는 **정량 수치가 전혀 보고되지 않아**, 향상폭(percentage point) 평가가 불가.
  - 사용된 backbone, 파라미터 수, 학습 데이터 규모 등 구현 세부 미공개.
  - Stereo baseline의 카메라 간 간격이 작은 경우 disparity 신호가 약해질 수 있는데, 이에 대한 sensitivity 분석은 abstract에서 확인할 수 없음.
  - Code release 여부도 abstract 시점에서는 명시되지 않음.
- 저자진(Li Fei-Fei, Jiajun Wu, Ruohan Zhang 등 Stanford 계열로 추정)의 후속 연구 및 코드 공개 여부가 이 모듈의 실용적 확산을 가를 것이다.

<!-- VERIFIED: abstract-only -->
