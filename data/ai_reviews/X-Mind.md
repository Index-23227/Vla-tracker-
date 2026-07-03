# X-Mind: Efficient Visual Chain-of-Thought via Predictive World Model for End-to-End Driving

**arXiv:** 2606.28758 · **Date:** 2026-06-27 · **Org:** XPeng Inc. (PWM Team)
**Project:** https://x-mind.github.io · **Code:** 미공개

## 1. 한 줄 요약
X-Mind는 Predictive World Model(PWM)을 외부 모듈이 아니라 LLM 백본 내부의 **Visual Chain-of-Thought**로 내재화한 end-to-end 자율주행 VLA로, 12프레임 미래를 96토큰의 추상 스케치로 압축(DC-AE)하고 Recurrent Block Diffusion(RBD)으로 단일 forward pass 안에서 denoising을 수행하여, 1.1x 추론 비용만으로 ADE Lat./Lon.을 0.2399/1.2979 → **0.1765/1.1849**로 개선한다.

## 2. 문제 정의
기존 주행 VLA는 perception→action 직접 매핑에 의존하는 반응형(reactive) 구조로, 물리 세계의 시공간적 전개를 예측하는 인지 능력이 없다. PWM을 결합하는 기존 접근은 (i) cascade 방식은 차량 온보드에서 감당 불가한 지연을 유발하고, (ii) 네트워크 말단에 보조 재구성 태스크로 붙이는 방식은 supervision이 깊은 LLM 레이어까지 역전파되지 못해 sparse GT trajectory에 의존하는 shortcut learning을 벗어나지 못한다.

## 3. 핵심 기여
1. **Visual CoT로서의 PWM 내재화**: 행동 생성 이전에 명시적 world rollout을 강제하여, 물리 기반의 dense 제약을 LLM 심층 feature에 직접 부여.
2. **Visual Thinking Representation (추상 스케치)**: BEV 레이아웃 + 주행 prior(신호등 상태, 내비게이션 의도, 속도 준수 바)를 단일 캔버스에 융합. 도메인 특화 DC-AE로 **12프레임 미래 rollout → 96토큰** 압축, long-context 병목 해소.
3. **Recurrent Block Diffusion (RBD)**: flow-matching의 반복 denoising을 LLM의 5개 블록 계층에 unroll하는 Layer Flow Matching(LFM)으로, 단일 forward pass 안에서 미래 생성 완료.

## 4. 아키텍처
- **입력 토큰**: text / ego status / multi-view(7카메라) / world model 토큰을 large drive model(LLM)에 통합.
- **추상 스케치**: BEV 물리 요소(ego 빨간 사각형, 주변 agent 노랑·파랑, 차선·경계) + 신호등 패널(좌상단) + cyan 내비게이션 경로 + 속도 준수 바(좌하단, 녹색=현재 속도/흰색=제한까지 여유/빨강=초과분).
- **LFM 학습**: 비균일 timestep 스케줄 {0, 0.1, 0.2, 0.4, 0.7, 1.0}을 특정 injection layer에 매핑, 각 레이어에서 노이즈-GT 선형보간 `h = EncProj((1−t_k)·ε + t_k·z1) + PE2D`로 hidden state 교체. 깊이가 깊어질수록 노이즈가 감소하는 미래 표현을 관찰 → 잠재공간 내 Visual CoT 실현.
- **추론**: z0~N(0,I)에서 시작, 각 블록 출력의 velocity로 Euler integration `z_{k+1} = z_k + (t_{k+1}−t_k)·v_k` (k=0..4), 최종 latent를 frozen DC-AE 디코더로 스케치 복원.
- **Planner**: 예측된 미래에 조건화된 inverse dynamics 플래너가 종방향 가속도 a_lon과 yaw rate ω_yaw에 L1 supervision을 받아 kinematic하게 실행 가능한 궤적 도출.

## 5. Action Head Category
**Inverse dynamics** — 행동은 내부적으로 생성된 미래 스케치에 조건화된 inverse dynamics planner가 파라미터화된 kinematic 제어량(a_lon, ω_yaw)으로 도출. 미래 생성 자체는 LLM 레이어에 unroll된 flow matching(RBD)이지만, 행동 헤드의 분류는 `inverse_dynamics`가 타당.

## 6. 학습 데이터 & 레시피
- **XPeng 내부 데이터**: 280,000시간, 34M 클립, 13.8T 토큰, 7카메라 서라운드 뷰(front fisheye/narrow, left/right front, left/right rear, rear), 도심 86.8% / 고속 13.2%. X-World 하드웨어 구성 및 X-Foresight 데이터 프로토콜 준수. **모든 실험은 전체의 1/8 subset** 사용.
- **손실**: L_total = λ_WM·L_WM + λ_plan·L_plan. L_WM = λ_flow·L_flow(K개 레이어 velocity MSE) + λ_img·L_img(랜덤 1개 레이어만 디코딩하여 MSE + LPIPS). L_plan은 kinematic 제어량 L1.

## 7. 장면 표현 비교 결과 (Table 1)
동일 백본·설정에서 미래 예측 타깃 비교 (ADE Lat./Lon. @6s, 상대 추론 비용):
- Base (world model 없음): 0.2399 / 1.2979, 비용 1.0
- Base + Image (3584 extra tokens): 0.2003 / 1.2456, 비용 **22.0**
- Base + 3DGS (3072 extra tokens): 0.1964 / 1.2247, 비용 19.0
- **Base + Sketch (96 tokens)**: **0.1765 / 1.1849, 비용 1.1**
→ 추상 스케치가 최고 정확도를 최소 토큰·최소 지연으로 달성. planning에 무관한 시각 디테일 제거의 효과.

## 8. Diffusion 아키텍처 비교 결과 (Table 2)
- Base: ADE 0.2399/1.2979 (FID 없음)
- Base + Sketch, Single-Step denoising: FID **67.30** (심각한 modality collapse), ADE 0.1783/1.1938
- **Base + Sketch, RBD**: FID **9.59**, ADE **0.1765/1.1849**, 추론 비용 동일(1.1)
→ denoising을 레이어에 분산하는 것만으로 FID 7배 개선. 정성 평가에서 RBD는 주야간 모두 선명·시간 일관적 스케치를 생성하고, GT에서 누락된 동적 객체의 운동까지 문맥으로 추론.

## 9. Reconstruction vs. Future Generation (Table 3)
스케치 타깃 ablation: 현재 프레임 재구성 FID 8.97 / ADE 0.1866/1.2132, 미래 1프레임 FID 9.05 / ADE 0.1840/1.2124, **미래 12프레임 FID 9.59 / ADE 0.1765/1.1849**.
→ FID는 재구성이 가장 좋지만 planning 성능은 12프레임 미래 생성이 최고. world model의 이득은 시각적 충실 재구성이 아니라 **예측적 생성 rollout**에서 나온다는 결론.

## 10. Ablation·분석 종합
- 스케치 GT는 신호등 위상 전환(정지→진행), 적응형 내비게이션 경로(cyan corridor), 속도 준수 바 등 구조화된 dense supervision을 제공 (Fig. 5).
- 정성 결과(Fig. 6): 전방 급제동 대응, 오프램프 차선 유지, 신호 준수 출발, 장애물 회피에서 world model 유무 대비 일관된 궤적 개선.
- RBD의 근거: (i) 추상 스케치는 고주파 정보가 제거되어 복잡한 diffusion 모델이 불필요, (ii) LLM의 얕은→깊은 레이어 추상화 위계가 coarse-to-fine denoising과 동형(isomorphism).

## 11. 한계 & 토의
- **재현 불가**: 전량 XPeng 사유 데이터(280K시간의 1/8), 코드·가중치 미공개, nuScenes/nuPlan 등 공개 벤치마크 결과 전무. 비교는 모두 내부 baseline 대비.
- **open-loop 평가만**: closed-loop 시뮬레이션 없음. ADE 개선이 closed-loop 안전성으로 이어지는지 미검증.
- **GT 스케치 의존**: 구조화 BEV 어노테이션이 필요해 스케일링 병목 — 저자들도 self-supervised 전환을 future work로 명시. 궤적-스케치 joint sampling도 미래 과제.
- 모델 파라미터 수, 절대 지연(ms), 차량 플랫폼 사양 미공개 (상대 비용 1.1x만 보고).

## 12. VLA-Tracker 관점에서의 의의
XPeng PWM 팀의 X-World(시뮬레이션), X-Foresight(joint vision-action 예측)에 이은 세 번째 축으로, **"세계 모델을 얼마나 싸게 VLA 내부에 넣을 수 있는가"**에 대한 답. dense 프레임(22x) 대신 96토큰 추상 스케치(1.1x)라는 표현 선택과, denoising을 LLM 깊이에 접는 RBD는 매니퓰레이션 VLA의 visual CoT(예: subgoal image 생성 계열)에도 이식 가능한 일반적 아이디어다. 조작(manipulation) 벤치마크가 전무한 주행 특화 모델이므로 LIBERO 등 리더보드에는 잡히지 않으며, real_world 항목으로만 추적된다.

<!-- VERIFIED: pdf -->
