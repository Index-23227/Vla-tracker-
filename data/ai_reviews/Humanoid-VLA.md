# Humanoid-VLA: 시각 통합 휴머노이드 제어 세미나 리뷰

## 1. 연구 배경 및 동기
Humanoid-VLA(Ding et al., 2025, arXiv:2502.14795)는 휴머노이드 로봇이 단순 모방(reactive mimicking)을 넘어 자율적 시각 인식 기반 객체 상호작용 및 환경 탐색을 수행하도록 하는 최초의 humanoid 전용 VLA 프레임워크다. Figure 1은 H2O, Exbody2 같은 reactive 방식과 본 모델의 차별점을 보여주며, 1인칭(egocentric) 시각 정보 통합이 universal humanoid control의 핵심임을 강조한다. 핵심 병목은 휴머노이드 데이터 부족 — AMASS 같은 motion capture 데이터는 1인칭 시각이 동기화되지 않고, teleoperation은 비용이 크다.

## 2. 방법론: 두 단계 학습 파이프라인
Figure 2의 구조는 (1) Language-Motion Pre-Alignment과 (2) Vision-Conditioned Fine-Tuning의 2단계로 구성된다. 첫 단계는 비-egocentric 인간 motion 데이터를 텍스트 caption과 정렬해 motion-language alignment을 학습한다. 두 번째 단계는 cross-attention 모듈로 1인칭 시각 정보를 융합한다. 식 (6)–(7)에서 Q는 언어 토큰(X_d^l W_Q), K/V는 시각 토큰(X_v^l W_K, X_v^l W_V)으로, copy & freeze된 transformer block에 cross-attention layer를 추가하는 parameter-efficient 설계이다.

핵심 기여 중 하나는 **Compositional Motion Quantization** (Figure 3, Section 3.2.1)이다. 신체를 5개 부위(좌측 다리, 우측 다리, 몸통, 좌측 팔, 우측 팔)로 분해하고 각 부위마다 독립적인 VQ-VAE encoder/decoder와 codebook(크기 1024)을 학습한다. 식 (4)는 reconstruction loss, embedding loss, commitment loss의 합 L_hvq를 정의하며, 이 분해 표현은 토큰 단위 motion 편집(예: 부위별 마스킹/재배치)을 가능하게 한다.

## 3. 데이터 자기지도 증강 전략
Table 1에 따르면 데이터셋은 motion capture 29K clips/4.1h, 온라인 영상 0.8M clips/7515.7h, 합성 데이터 100K clips/227.7h, 총 929K clips/7790.2h 규모로 이전 작업(Mao et al., 2024) 대비 25× 크다. Figure 3과 Table 6은 4가지 자기지도 증강 — `<Track>`(궤적), `<Time>`(지속시간), `<Occlusion>`(마스킹), `<State>`(상태) — 으로 motion-language QA 쌍을 자동 생성하는 전략을 보여준다. 예: "Missing left arm <Occlusion> motion data. Please complete the motion." 같은 prompt가 정답 motion과 결합된다. GPT-4를 통한 instruction rephrasing으로 N배 확장 가능하며, 원래의 59 subtasks가 N×로 증가한다.

## 4. 실험 결과: 운동 생성 품질
**Kinematic Fidelity (Table 3)**: HumanML3D에서 FID 0.467±.018(MDM 0.889 대비 -47.5%, T2M-GPT 0.531 대비 -12%), Diversity 4.585±.086. 자체 수집한 Humanoid-S 데이터셋에서는 FID 1.037±.147, DIV 4.466±.213로 모든 baseline을 능가한다. 평가는 Guo et al. (2022b)의 framework를 따라 15 universal joints에 대해 수행했다.

**Physical Plausibility (Table 2)**: IsaacGym 시뮬레이터에서 PPO 기반 whole-body controller로 평가, MPJPE_g는 모든 입력 조건에서 40mm 이하이며 Medium 난이도 D+T 조건에서 31.07mm로 최저, PA-MPJPE는 1.18mm, E_accel 27.84mm/s², E_vel 14.76mm/s. 입력 조합 D(설명), T(시간), A(절단 motion), Sn(상태)으로 Easy/Medium/Hard 3단계 평가했다.

**Ablation (Table 4)**: 저품질 video 데이터만으로도 FID 0.698 달성, video + 고품질 mocap 결합 시 0.467로 16% 개선되어 대규모 video 활용의 효용을 입증한다.

## 5. 실세계 시각 통합 평가
Section 4.2 및 Table 5에서 Unitree G1 로봇에 RGB egocentric 카메라를 장착해 8개 태스크 × 10회 평가:
- Turn to an object 10/10, Wave to people 10/10, Punch an obstacle 10/10
- Hold an object 9/10, Avoid an obstacle 9/10, Jump over an object 9/10, Kick a ball 9/10
- Dance with a partner 8/10

Figure 4는 "Kick Ball"과 "Avoid Obstacle" 시연을 보여주며, 시각 기반 자율 의사결정을 입증한다. 모델은 Llama3-70B backbone, learning rate 2e-5, cosine scheduler, batch size 4/device로 8×NVIDIA H100 GPU에서 216시간 학습되었다 (Section 4.1.1 implementation details).

## 6. 평가 및 한계
Humanoid-VLA는 humanoid 전용 최초의 통합 VLA로 의의가 크지만, Appendix E의 한계 인정이 솔직하다. 첫째, RL policy의 robustness 부족. 둘째, 고품질 humanoid 데이터 가용성 여전히 부족. 셋째, 현재 학습 방법론이 단순. 또한 표준 22 SMPL joints 대신 15개 universal joints만 사용해, 다운스트림 24-joint humanoid 매핑을 위한 Adam optimizer 기반 추가 최적화가 필요하다 (Appendix D). 

**YAML 점검**: open_source=false는 코드 미공개로 일치한다. action_head_category=other는 language-motion alignment + cross-attention 기반 토큰 예측이라는 점에서 적절하다. **YAML 불일치 가능성**: backbone="CLIP ViT (egocentric video encoder)"로 기재되어 있으나 논문은 visual encoder를 단순히 "Vision Encoder"로 부르고 CLIP을 명시하지 않으며, llm 필드도 단순히 "LLM"인데 실제로는 Llama3-70B(Section 4.1.1)이므로 업데이트 필요. parameters="not disclosed"도 Llama3-70B 기준 약 70B로 갱신 가능하다.

<!-- VERIFIED: pdf -->
