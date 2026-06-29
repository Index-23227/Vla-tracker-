# Reflective VLA: In-Context Action Consequences Make VLAs Generalize

저자: Qing Lian, Kent Yu, Lei Zhang (Futian Laboratory / IDEA / Visincept)
arXiv: 2606.25215 (2026-06-23) · [프로젝트 페이지](https://lianqing11.github.io/reflective-vla-page/)

## 1. 한 줄 요약
Reflective VLA는 과거의 (관측-행동-결과) 삼중항을 in-context 증거로 활용하여, 단일 프레임만으로는 식별 불가능한 배치 환경별 잠재 요인(카메라 기하, 캘리브레이션, 작동 편향)을 추론하고 테스트 타임 미세조정 없이 일반화하는 dual-system VLA이다.

## 2. 문제 정의
대부분의 VLA는 반응형(reactive)이다. 현재 명령과 관측에서 다음 행동 π(A_t | L, O_t)를 예측하며, 현재 관측이 행동에 필요한 상태를 완전히 명시한다고 암묵적으로 가정한다. 그러나 카메라-로봇 기하, 로봇 캘리브레이션, 작동 편향 같은 embodiment 고유 요인은 단일 관측에서 식별하기 어렵다. 잠재 변수 z에서 관측 O_t로의 매핑이 다대일(many-to-one)이므로, 단일 프레임에서 z를 추론하는 것은 ill-posed 문제이고, 정책은 훈련 환경을 암기할 수밖에 없어 배치 환경 일반화에 실패한다.

## 3. 핵심 아이디어
교차 환경 일반화를 인과 상호작용 삼중항에 대한 in-context learning(ICL) 문제로 정식화한다. 상호작용 피드백(관측-행동-결과 관계)은 환경이 명령에 어떻게 반응하는지를 드러낸다. 적응형 정책은 z에 대한 사후 분포의 주변화로 표현된다: π(A_t | L, O_t, H) = ∫ π(A_t | L, O_t, z) P(z | H) dz. 핵심은 맥락 H가 z에 대해 진단적(diagnostic)이어야 한다는 것이며, 이상적 Markov 가정 하에서 사후 분포가 P(z|H) ∝ P(z) ∏ P_env(O'_i | O_i, A_i, z)로 인수분해된다. 결과 항 O'를 포함해야만 응답 특성이 노출되므로, (O, A) 쌍만으로는 부족하다.

## 4. 방법: 아키텍처
- **삼중항 구성**: 각 삼중항 T_i = (L, τ_i, O_i, A_i, O'_i). 결과 관측은 다음 프레임이 아닌 청크 정렬 관측 O'_i = O_{i+C}로 정의하여 end-effector 변위와 잔차 제어 오차 같은 가시적 효과를 포착한다. 각 행동 청크는 학습된 투영 g_A를 통해 8개 토큰으로 임베딩된다.
- **공유 어텐션 dual-system**: 모든 관측 모달리티(3인칭/손목 이미지, proprioception)를 단일 VLM 토큰 시퀀스로 라우팅한다. 이미지는 시각 백본으로, proprioception과 과거 행동 청크는 2층 비선형 투영기로 토큰 공간에 매핑된다. flow-matching 연속 행동 expert가 suffix로 부착되어 모든 층에서 VLM prefix와 어텐션을 공유(MoT 스타일)하므로, 과거 관측/행동/결과/현재 관측에 직접 attend한다.
- **백본**: PaliGemma-3B 또는 Qwen3-VL-2B, 행동 expert hidden dim 1024.

## 5. 방법: 학습
- **Block-causal 학습**: in-context 조건화는 시퀀스 길이를 K배 늘린다. 각 위치를 독립 forward로 감독하면 비용이 O(K)가 되므로, LM 학습의 packed-sequence 아이디어를 차용해 block-causal 마스크로 K개 프레임을 단일 forward에서 공동 감독한다. 각 프레임이 동시에 후속 타겟의 맥락이자 자신도 예측 타겟이 된다.
- **마스크**: 쿼리 Â_{t_k}는 완료된 이전 삼중항과 현재 언어-prefix 관측에만 attend하고, 자신의 prefix 행동 토큰 A_{t_k}와 결과 O'_{t_k}(실행 후에만 관측됨), 후속 삼중항, 형제 쿼리 슬롯으로부터는 마스킹된다. 첫 타겟 t_1은 선행 삼중항이 없어 반응형 감독을, t_2..t_K는 점진적으로 긴 맥락의 ICL 감독을 제공한다.
- **스트라이드 랜덤화**: 인접 청크가 시간적으로 매끄러워 정책이 결과 대신 과거 행동에서 외삽하는 지름길을 막기 위해, 학습 시 backward spacing에 [0,15] 스텝의 랜덤 스트라이드를 추가한다.
- **하이퍼파라미터**: AdamW (β=0.9/0.95, wd 0.01, grad clip 1.0), bf16, DeepSpeed ZeRO-2. 행동 expert/투영기 lr 1e-4, VLM은 0.1배 + 초기 freeze. 8x H20 GPU. LIBERO Reflective는 50k steps K=8, SimplerEnv는 160k steps K=4.

## 6. 추론
배치 시 최근 K-1개 삼중항의 rolling FIFO 버퍼를 유지한다. 학습과 달리 각 삼중항은 정책 자신의 예측 청크 Â와 실제 도달한 관측 O_{t+C}를 저장하므로 실제 rollout을 반영한다. 각 스텝에서 새 관측(또는 실행 후 삼중항)만 인코딩하고, 과거 삼중항의 VLM-측 key/value는 한 번 캐싱되어 재사용되므로 스텝당 추론 비용이 거의 일정하게 유지된다.

## 7. 실험 설정
4가지 시뮬레이션 설정: (i) 표준 LIBERO (4 suite: Spatial/Object/Goal/Long), (ii) SimplerEnv-Bridge (BridgeData V2 -> ManiSkill2, real-to-sim), (iii) LIBERO-Plus (7개 perturbation 카테고리), (iv) LIBERO-Plus-Hard (저자 제안, Multi-camera shift와 Robot calibration shift). 주 베이스라인은 동일 백본/데이터/파라미터에 K=1만 설정한 재현 반응형 π0.5. 추가로 실세계 교차 카메라 일반화 실험(Agilex Piper 팔 + RealSense D435i).

## 8. 주요 결과
- **In-distribution (Table 1)**: LIBERO 97.6% 평균(SOTA, π0.5 96.9%·MemoryVLA 96.5% 상회, 반응형 베이스라인 대비 +0.7pp). SimplerEnv-Bridge 78.2% 평균(SOTA, 반응형 대비 +5.3pp). 세부: spoon 95.8 / carrot 83.3 / cube 79.2 / eggplant 54.2.
- **분포 이동 (Table 2)**: LIBERO-Plus 87.7% 평균(반응형 82.3%, OpenVLA-OFT 80.7%, MemoryVLA 81.5% 상회). 최대 향상은 Robot(+22.9pp), Background(+6.9pp), Noise(+5.2pp). LIBERO-Plus-Hard 68.8% 평균(반응형 대비 +4.2pp, MemoryVLA 대비 +8.1pp).
- **실세계**: 미관측 카메라 배치에서 반응형은 급락(box 32%, bowl 16%)하나 Reflective VLA는 76%/64% 달성(+44/+48pp). seen 배치는 소폭 향상에 그쳐, 맥락이 정상 성능을 해치지 않음을 확인.

## 9. Ablation
- **맥락 구성 (Table 3a, K 고정)**: 관측만(O) 추가는 개선 없음(72.3% vs 반응형 73.1%). 관측-행동(O,A)은 미미한 73.8%. 전체 (O,A,O') 구조가 77.8%로 +4.7pt, 특히 Rob. Calib†에서 55.2%->61.3%. 즉 결과 관측 O'가 핵심 적응 신호이며, 단순히 맥락 길이만으로는 불충분함을 입증.
- **맥락 길이 (Table 3b)**: K=1 73.1% -> K=2 75.3% -> K=4 76.7% -> K=8 77.8%. 대부분 이득이 K=4에서 포착되고 이후 수확 체감.
- **지연-정확도**: 캐싱 덕에 지연이 K에 sub-linear. K=8은 K=1 대비 1.43배(178 vs 124ms), K=4는 1.19배로 대부분 이득 회복.

## 10. 강점
- 핵심 가설(결과 관측이 일반화의 열쇠)을 history-only ablation으로 명확히 분리·검증한 점이 설득력 있음.
- 모델 크기 증가 없이 맥락 사용 방식만 변경하여 OOD 일반화를 개선(고정 비용).
- block-causal 마스크 + KV 캐싱으로 학습/추론 효율을 모두 확보하여 실시간 제어 가능.
- 진단적 벤치마크(LIBERO-Plus-Hard)를 직접 설계하여 행동-관측 매핑 변화를 표적화.

## 11. 한계
- 첫 청크는 반응형으로 예측하며, 결과가 청크 horizon 내에서 관측 가능하다고 가정(지연 효과·접촉 풍부 동역학은 더 긴 맥락 필요).
- 학습 메모리 제약으로 K를 8로 상한(추론 자체는 더 확장 가능).
- 과거 청크에서 외삽하는 지름길 위험(프레임 경계 토큰·스트라이드 랜덤화로 부분 완화).
- in-context 일반화가 데이터 다양성에 의존하므로 데이터 스케일링이 추가 이득에 필요.
- 실세계 연구는 테이블탑 교차 카메라 일반화·조건당 소수 시도로 제한.

## 12. 총평
Reflective VLA는 "맥락 길이를 늘리는 것"과 "행동-결과 결합 증거를 제공하는 것"을 명확히 구분하고, 후자가 교차 환경 일반화의 핵심임을 정밀한 ablation으로 입증한 점에서 개념적으로 깔끔하다. dual-system VLA에 ICL 관점을 도입하면서도 실시간 제어 비용을 유지하는 엔지니어링(block-causal + KV 캐싱)이 실용적이다. 테스트 타임 업데이트 없이 카메라/캘리브레이션 이동에 적응한다는 점은 배치 로봇 시스템에 직접적 가치가 있다.

<!-- VERIFIED: pdf -->
