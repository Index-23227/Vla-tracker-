# AdvDex: Learning Dexterous Manipulation from Human Demonstrations via Joint-Aligned Actions and Adversarial Learning

> **한 줄 요약**: 사람 손·로봇 dexterous hand·평행 그리퍼를 **SE(3) 손목 자세 + 15개 손가락 관절**의 공통 행동 공간(JAAS)으로 정렬하고, VLM 인지 토큰에 **Gradient Reversal Layer 기반 domain-adversarial 학습**을 걸어 embodiment 외형 단서를 억제한 VLM+DiT VLA. 자체 수집한 장갑·촉각 기반 인간 시연 데이터셋 OmniShare(100k+ 궤적)로 사전학습 후 1,000개 로봇 시연으로 post-train하여, Paxini DexH13 실기 5개 과제에서 π0.5·VITRA와 같거나 더 높은 성공률(예: Grasp Single 80%, Unseen Env 60% vs π0.5 45%), 인간 전용 과제의 zero-shot 전이(Press Button 70% vs π0.5 50%)를 보였다.

---

## 1. 배경 및 동기

- 로봇 teleoperation 데이터는 비싸고, 기존 foundation policy 대부분은 평행 그리퍼 중심이라 도구 사용 등 정교한 조작에 한계가 있다.
- 사람 손 데이터는 풍부하지만 dexterous hand(Wuji, Xhand, Shadow, DexH13 등)마다 운동학 구조·자유도·관절 제약·외형이 달라 **공유 행동 표현이 없다**.
- 공유 시각 인코더는 과제 관련 공간 정보와 embodiment 고유 외형(손 모양, 장갑 등)을 뒤섞어 embodiment identity를 지름길(shortcut)로 사용할 수 있다.
- 저자 주장: 행동 공간 정렬만으로는 시각 shortcut을 막지 못하고, 데이터를 단순히 섞는 것만으로는 행동이 비교 가능해지지 않으므로 **운동학 정렬 + embodiment-불변 표현 학습**을 함께 해결해야 한다.

## 2. 방법론 심층 분석

### 2.1 Joint-Aligned Action Space (JAAS)
- 공통 어휘: SE(3) 손목 자세(3D 이동 + 연속 3D 회전) + 손가락당 3개의 3-DoF Euler 관절(총 15 관절).
- 51-DoF MANO 인간 손 상태, 19-DoF dexterous hand, 7-DoF 팔 + 1-DoF 그리퍼를 **기능적 대응**으로 슬롯에 매핑. 그리퍼의 jaw는 두 개의 canonical 손가락 슬롯에 대응.
- 해당 embodiment에 없는 슬롯은 action loss 계산에서 마스킹 → 단일 action expert가 이질적 운동학을 학습.

### 2.2 Domain-Adversarial VLA
- 구성: VLM 백본 $E_\theta$(SigLIP 토크나이저, VITRA 방식의 compact cognition token $z_t$), DiT action expert $P_\phi$, 도메인 판별기 $D_\psi$.
- $P_\phi$는 $z_t$와 운동학 상태 $s_t$를 AdaLN으로 조건화해 JAAS 행동 chunk를 반복적으로 denoise.
- $D_\psi$는 $[z_t, s_t]$로부터 도메인(인간 장갑, dexterous hand, 그리퍼 등)을 분류하고, GRL을 통해 VLM에는 $-\lambda$배 gradient가 전달되어 embodiment 고유 외형 정보를 억제.
- 판별기를 상태로 조건화한 이유: 운동학으로 이미 설명되는 차이와 시각 토큰의 잔여 외형 단서를 분리하기 위함.
- 추론 시에는 판별기가 사용되지 않아 정책 인터페이스는 변하지 않음.

### 2.3 학습 목적함수
$\mathcal{L}_{final} = \mathcal{L}_{MSE}(\theta,\phi) + \lambda \mathcal{L}_D(\psi|\theta)$ — 표준 diffusion 노이즈 예측 MSE와 도메인 cross-entropy의 minimax. End-to-end 학습.

## 3. 데이터 전략

- **OmniShare**: 100k+ 궤적, 500+ 과제, 700+ 객체, 10k 시간 이상 영상(Fig. 2), 5개 실세계 도메인. Table 1의 비교에서 GigaHands(13.9k trajs) 등 기존 3D 양손 데이터셋보다 큰 규모와 dense text 주석을 표방.
- 수집: 마이크로초 동기화 센서 스위트, 29개 자기 회전 엔코더 장갑 + Hall-effect 촉각 어레이 + 다시점 카메라.
- 처리: ArUco + FoundationPose로 6D 손목/객체 자세 → 운동학·촉각 불일치를 동시 최소화하는 physics-aware 최적화로 MANO 리타게팅 → 거리 기반 감쇠 함수로 촉각 신호 보정 → JAAS 매핑.
- 사전학습 혼합: OmniShare : VITRA-1M 인터넷 영상 : OXE 부분집합 = 5:4:1. Post-train: 1,000개 로봇 teleop 시연(5개 과제).
- 누수 방지: Table 4 평가 과제·객체·지시문은 로봇 post-train 데이터에서 제외, VITRA-1M·OXE와의 지시문 비중복도 확인.

## 4. 실험 설계

- **손 동작 예측(Table 2)**: OmniShare-Unseen(200 궤적, 20개 신규 객체 범주), HOI4D(200 궤적). 지표 d_h-o(pre-grasp 손가락-객체 최소 거리), MPJPE, MWTE(모두 mm).
- **실기 조작(Table 3)**: Paxini Tora + 19-DoF DexH13. 과제 5개(Grasp Single Object, Multi-Object Grasp, Pour Water, Push Cube, Stack Bottle) + Unseen Objects/Unseen Environment. 과제당 20회(4개 작업 영역 × 5개 초기 배치).
- 공정성: AdvDex, π0.5, VITRA를 동일 로봇 시연·동일 행동 표현·동일 학습 스텝으로 post-train.
- **Zero-shot 전이(Table 4)**: 로봇 1,000 + 인간 1,000 궤적 co-training, 과제 집합 상호 배타. 평가 과제는 인간 데이터에만 존재.
- **Few-shot(Fig. 5)**: 0/5/20 시연 — 그림으로만 제시.

## 5. 주요 결과

### 손 동작 예측 (Table 2, mm, 낮을수록 좋음)
| Method | OmniShare d_h-o | MPJPE | MWTE | HOI4D d_h-o | MPJPE | MWTE |
|---|---|---|---|---|---|---|
| VITRA | 20.1 | 16.2 | 14.8 | 16.3 | 13.7 | 11.5 |
| **Ours** | **3.2** | **2.8** | **2.5** | **10.5** | **7.2** | **6.1** |

### 실기 조작 성공률 (Table 3, %)
| Method | Grasp Single | Multi-Obj | Pour | Push | Stack | Unseen Obj | Unseen Env |
|---|---|---|---|---|---|---|---|
| π0.5 | 75 | 60 | 55 | 85 | 60 | 35 | 45 |
| VITRA | 70 | 65 | 40 | 90 | 70 | 40 | 35 |
| **Ours** | **80** | **70** | **55** | **90** | **70** | **50** | **60** |

- 모든 열에서 baseline과 같거나 높음. 차이가 가장 큰 곳은 일반화 열(Unseen Obj +10, Unseen Env +15 vs 최고 baseline).

### Zero-shot 인간→로봇 전이 (Table 4, %)
| Method | Box Doll | Press Button | Move Bottle | Tool Use |
|---|---|---|---|---|
| π0.5 | 30 | 50 | 30 | 0 |
| VITRA | 40 | 45 | 25 | 20 |
| **Ours** | **60** | **70** | **45** | **30** |

## 6. Ablation 분석

- **Table 2**: w/o OmniShare(VITRA-1M+OXE만)는 VITRA와 비슷(19.8/15.7/14.2), OmniShare 추가(w/o Adv) 시 6.4/4.9/4.2로 크게 개선, adversarial 추가 시 3.2/2.8/2.5. HOI4D에서도 13.9/13.3/10.8 → 10.5/7.2/6.1.
- **Table 3**: w/o Pre-train(scratch)은 Grasp Single 35%, Unseen Env 0%로 붕괴 → 대규모 사전학습의 중요성. w/o OmniShare는 Unseen Obj 25/Env 20, w/o Adv는 Unseen Obj 15/Env 30 — 두 요소 모두 일반화 열에서 큰 하락.
- **Table 4**: Adv를 pre-train에서 제거(50/35/35/15)보다 post-train에서 제거(25/30/15/5)할 때 하락이 더 큼.
- **Fig. 6 t-SNE**: GRL 적용 시 embodiment 간 특징 분포가 더 겹침 — 정성적 증거.

## 7. 관련 연구와의 위치

- 인간 영상 기반 VLA(VITRA, EgoVLA, Being-H0, H-RDT)와 달리 **장갑·촉각 기반 고정밀 운동학**을 감독 신호로 사용.
- 리타게팅/잔차 학습(ManipTrans, HERMES)이나 정렬 fine-tuning 대신 canonical 공간으로 직접 사상해 인간·로봇 공동 학습.
- Domain-adversarial(GRL, Ganin 식) 아이디어를 VLA 인지 토큰에 적용한 사례로, cross-embodiment 시각 shortcut 문제를 명시적으로 다룬다.
- 아키텍처는 π0/Diffusion Policy 계열(VLM + diffusion expert) 위에 구축.

## 8. 강점

- 행동 공간 정렬과 시각 표현 불변성이라는 **두 병목을 함께** 다루며, 각 요소의 기여를 Table 2–4에서 일관되게 분리.
- 동일 데이터·스텝으로 π0.5, VITRA를 post-train한 비교 설정이 공정.
- 인간 전용 과제로의 zero-shot 전이 실험은 "인간 데이터가 실제로 새 기술을 가르치는가"를 직접 검증하는 설계.
- 누수 방지 조치(과제·객체·지시문 배제, 외부 데이터 지시문 비중복 확인)를 명시.

## 9. 약점 및 한계

- 실기 평가가 과제당 20회로 표본이 작고, 신뢰구간·통계 검정 없음. Pour Water처럼 baseline과 동률인 과제도 있다.
- 단일 로봇 플랫폼(DexH13)만 평가 — "cross-embodiment" 주장의 로봇 측 검증은 제한적(저자도 한계로 언급).
- 모델 크기, VLM 종류, 학습 스텝 등 구현 세부가 본문에 부족하며 코드 공개 언급 없음.
- JAAS는 embodiment별 동역학·접촉 제약을 모델링하지 않음.
- Few-shot 결과와 Fig. 1 평균 수치는 그림으로만 제시되어 정량 비교가 어렵다.
- t-SNE 겹침은 불변성의 정성적 신호일 뿐, 도메인 분류 정확도 등 정량 지표는 제시되지 않음.

## 10. 실용적 시사점

- dexterous hand 정책을 여러 하드웨어에 확장하려는 팀에게 **슬롯 마스킹 기반 공통 관절 어휘**는 구현이 간단하고 재사용성이 높은 설계.
- GRL은 추론 비용이 없고 학습 시에만 붙이므로, 인간+로봇 혼합 데이터를 쓰는 기존 VLA 파이프라인에 저비용으로 추가해볼 수 있다.
- 장갑+촉각 데이터 수집은 teleop보다 저렴하면서 비전 기반 손 추정보다 정확한 감독을 제공 — 데이터 투자 우선순위 결정에 참고할 만하다.

## 11. 예상 질문과 답변

**Q1. JAAS에서 그리퍼는 어떻게 표현되나?**
A. 1-DoF jaw 동작을 기능적으로 대응하는 두 개의 canonical 손가락 슬롯에 매핑하고, 나머지 슬롯은 loss에서 마스킹한다.

**Q2. 판별기에 상태 $s_t$를 넣는 이유는?**
A. 운동학 상태로 이미 설명되는 embodiment 차이는 제외하고, 시각 토큰에 남은 외형 단서만 제거하도록 유도하기 위해서다.

**Q3. adversarial 학습이 과제 성능을 해치지 않나?**
A. action denoising loss가 과제 완수에 필요한 정보를 유지하도록 강제하며, Table 2–4에서 Adv 추가가 모든 지표를 개선했다.

**Q4. OmniShare 없이도 효과가 있나?**
A. w/o OmniShare는 Table 2에서 VITRA 수준, Table 3 Unseen Env 20%로 떨어져, 고품질 인간 데이터가 핵심 기여 요소임을 보여준다.

**Q5. zero-shot 전이는 진짜 zero-shot인가?**
A. 평가 과제는 로봇 데이터에 전혀 없고 인간 데이터에만 존재하므로 "과제 수준" zero-shot이다. 다만 로봇 제어 경험 자체는 다른 과제에서 학습된다.

## 12. 결론

AdvDex는 dexterous manipulation의 cross-embodiment 문제를 "공통 관절 어휘(JAAS)"와 "embodiment-불변 시각 표현(GRL)"의 조합으로 정식화하고, 장갑·촉각 기반 대규모 인간 시연 데이터셋 OmniShare로 이를 뒷받침한 VLM+DiT VLA다. 손 동작 예측 오차를 VITRA 대비 수 배 줄였고(OmniShare-Unseen MPJPE 16.2→2.8 mm), 실기 과제에서 π0.5·VITRA와 같거나 높은 성공률, 특히 unseen 환경(60% vs 45%)과 인간 전용 과제 전이(Press Button 70% vs 50%)에서 뚜렷한 이점을 보였다. 다만 단일 로봇·과제당 20회 평가, 구현 세부 부족, 그림 기반 few-shot 결과는 후속 검증이 필요하다.

<!-- VERIFIED: pdf -->
