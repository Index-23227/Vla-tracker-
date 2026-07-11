# LaMem-VLA: Dual Latent Memory in Vision-Language-Action Models for Robotic Manipulation

**arXiv**: [2607.07608](https://arxiv.org/abs/2607.07608) · **기관**: Nanjing University of Science and Technology, Zhejiang University, National University of Singapore · **공개일**: 2026-07-08

---

## 1. 한 줄 요약

LaMem-VLA는 로봇의 과거 경험을 외부 메모리 뱅크가 아닌 VLA 모델의 잠재 임베딩 공간 내부(context-native latent memory)에서 저장·검색·소비하는 듀얼(단기 시각 + 장기 의미) 잠재 메모리 프레임워크로, LIBERO 5-스위트 평균 97.6%(MemoryVLA 대비 +1.1), SimplerEnv-Bridge 73.9%(CogACT 베이스라인 대비 +16.6)로 SOTA를 달성했다.

## 2. 문제 설정 및 동기

주류 VLA 모델은 마르코프 가정 하에 현재 관측만으로 행동을 예측하여, 이전 상태 전이·완료된 조작 단계·과업 진행 단계를 추론하지 못하는 temporal short-horizon bias를 겪는다. 기존 메모리 증강 접근은 (i) 관측 윈도우 확장(프레임 연결/비디오 입력) — 컨텍스트 길이에 비례하는 계산 비용과 고정된 메모리 상한, (ii) 외부 메모리 뱅크 검색(MemoryVLA, MemER 등) — 메모리가 모델의 네이티브 토큰 공간 밖에 저장되고 VLA 추론 이후 정책 측(policy-side) 보조 컨텍스트로만 소비되는 구조적 한계를 갖는다. 저자들은 "과거 경험을 VLA가 이미 지각·추론·행동하는 동일한 연속 잠재 공간 안의 기억으로 표현할 수 있는가"라는 질문을 던진다.

## 3. 핵심 기여

1. **패러다임 제안**: 과거 경험을 context-native latent memory로 취급 — 모델 임베딩 공간 내에서 저장·검색·소비되어 장면 지각, 지시 이해, 행동 의도 형성과 동일한 잠재 추론 과정에 참여.
2. **LaMem-VLA 프레임워크**: 로봇 이력을 단기 시각 vault와 장기 의미 vault라는 상보적 이중 메모리로 조직하고, 이중 스케일 메모리를 모델 추론에 직접 직조(weave).
3. **잠재 메모리 응축 메커니즘**: 검색된 과거 증거를 VLA 임베딩 공간과 호환되는 고정 길이 단기/장기 잠재 메모리 토큰으로 변환.

## 4. 방법론

네 개의 협조 모듈이 메모리 재구성과 행동 추론의 루프를 닫는다.

- **Curator (§3.3)**: Prismatic 7B(DINOv2+SigLIP 비전 인코더, LLaMA-7B, OXE 사전학습) 백본이 만든 표현을 두 vault로 분리. 단기 vault M_short는 SE-bottleneck 압축 모듈이 현재 시각 토큰을 압축한 key-value 쌍(key = mean-pool)을 저장하고, 장기 vault M_long은 학습 가능한 action query의 출력 은닉 상태 H_action을 그대로 축적. 용량 L=16 초과 시 시간적으로 인접한 유닛 중 코사인 유사도가 가장 높은 쌍을 평균하여 병합하는 중복 제거 전략 적용.
- **Seeker (§3.4)**: 시각·언어 토큰의 멀티모달 인지 상태에 학습 가능한 query slot을 붙여 transformer 기반 query builder B(masked attention, 2-layer)로 컨텍스트 인지 질의 q_t를 생성, 코사인 유사도로 각 vault에서 Top-K(K=8) 유닛을 검색. Top-K 연산은 gradient로 최적화되지 않음.
- **Condenser (§3.4)**: 검색된 증거 Z_short/Z_long를 학습 가능한 메모리 슬롯과 함께 경량 memory former(F_v, F_c)에 통과시켜 고정 길이 잠재 토큰 M_short(Ls=8), M_long(Ll=4)으로 재구성 — 주입되는 메모리 길이가 검색 크기와 무관해짐.
- **Weaver (§3.5)**: 학습 가능한 source embedding(b_s, b_l)을 더한 메모리 토큰을 시퀀스 앞에 접합해 S_t = [M_short; M_long; X_t; I; Q_action]을 구성. 메모리가 self-attention으로 관측·지시·행동 질의와 직접 상호작용하여 memory-grounded action token을 형성하고, 이것이 diffusion action expert(~300M, DiT 스타일, MSE noise prediction, 추론 시 DDIM 10 스텝)를 조건화해 16-스텝 7-DoF 액션 청크를 생성.

## 5. 실험 설정

- **구현**: 8x H800, PyTorch FSDP, 글로벌 배치 256, lr 2e-5. 입력은 단일 서드퍼슨 224x224 RGB + 언어 지시. 액션 청크 16.
- **SimplerEnv-Bridge**: Bridge v2로 50k 스텝 학습, 2.5k마다 검증해 최고 체크포인트 보고. WidowX 4개 태스크, 태스크당 24 trial.
- **LIBERO**: Franka, 5개 스위트(Spatial/Object/Goal/Long-10/Long-90). OpenVLA 프로토콜(태스크당 50 데모). Spatial/Object/Goal은 각각 20k 스텝 개별 학습, Long-10+Long-90은 40k 스텝 공동 학습. 태스크당 50 rollout.

## 6. 주요 결과

- **SimplerEnv-Bridge (Table 1)**: 평균 73.9% — CogACT 베이스라인(57.3) +16.6, π0(69.2) +4.7, MemoryVLA(71.9) +2.0, SemanticVLA(65.1) 상회. 태스크별 Spoon 83.3 / Carrot 75.0 / Stack Cube 41.7 / Eggplant 95.8.
- **LIBERO (Table 2)**: 5-스위트 평균 97.6% — Spatial 98.8 / Object 99.0 / Goal 97.2 / Long-10 95.8 / Long-90 97.0으로 전 스위트 최고. MemoryVLA(96.5) +1.1, CogACT(93.2) +4.4, 첫 4개 스위트 평균 97.7로 π0(94.2) +3.5 — 고유수용감각·손목 카메라 추가 입력 없이 달성. 특히 Long-10에서 MemoryVLA +2.4, Long-90에서 +1.4로 장기 과업 이득이 두드러짐.

## 7. Ablation 분석

- **듀얼 메모리 (Table 3)**: 둘 다 제거 시 SimplerEnv 57.3 / LIBERO-90 92.1로 최대 하락; 단기만 제거 65.6/95.4, 장기만 제거 64.6/94.8, 풀 모델 73.9/97.0 — 두 스트림의 상보성 입증.
- **잠재 통합 vs 정책 측 조건화 (Table 4)**: 동일 메모리를 외부 policy-side 조건으로 주면 71.9/94.8, 검색 원본을 그대로 조건화하면 69.8/95.1에 그침 — 이득이 "메모리 추가" 자체가 아니라 잠재-네이티브 통합에서 온다는 핵심 주장 검증.
- **검색 예산 K (Table 5)**: K=2→8에서 66.7→73.9(SimplerEnv)로 상승, K=12에서는 71.8로 하락 — 과도한 검색은 중복 증거와 응축 부담 증가.
- **메모리 토큰 수 (Fig. 3)**: Ls 2→16에서 61.4→65.6 상승 후 32에서 포화; (Ls, Ll)=(8, 4)를 성능-효율 균형점으로 채택.

## 8. 이전 연구와의 비교

관측 윈도우 확장 계열(Interleave-VLA, CronusVLA, HAMLET)은 컨텍스트 비용 증가와 고정 지평의 한계, 희소 이력 추상화 계열(TraceVLA, UniVLA, BPP)은 세밀한 지각 정보 손실, 외부 메모리 계열(MemoryVLA, MemER)은 정책 측 소비라는 구조적 분리가 문제였다. LaMem-VLA는 MemoryVLA와 동일한 CogACT-계열 백본(Prismatic 7B + diffusion expert) 위에서 메모리의 '소비 위치'만 잠재 공간 내부로 옮겨 SimplerEnv +2.0, LIBERO +1.1을 얻었다는 점에서, 메모리 통합 경로 자체가 성능 변수임을 보여주는 직접 비교로 읽힌다. MemGen 등 LLM 에이전트의 생성적 잠재 메모리 아이디어를 로봇 조작에 이식한 흐름이기도 하다.

## 9. 강점

- Table 4의 통제 실험이 논문의 핵심 주장(잠재-네이티브 > 정책 측 조건화)을 동일 메모리 내용으로 깔끔하게 분리 검증.
- 고정 길이 응축(Ls+Ll=12 토큰)으로 검색량과 무관한 유계(bounded) 컨텍스트 유지 — 윈도우 확장 계열의 비용 증가 문제를 구조적으로 회피.
- 추가 센서 입력(proprioception, 손목 카메라) 없이 π0 계열을 상회하고, 5개 스위트 전부에서 일관된 최고 성능.
- K, Ls/Ll, vault 용량 등 하이퍼파라미터 민감도를 체계적으로 보고.

## 10. 약점 및 한계

- **시뮬레이션 한정**: 저자 스스로 인정하듯 실기 로봇 실험이 전무 — 실세계 시각 노이즈에서 코사인 유사도 기반 검색과 인접 쌍 병합이 유지될지 미검증(실기 실험은 차기 버전 예고).
- 메모리 vault가 에피소드 내(intra-episode) 이력에 한정 — 에피소드 간 경험 축적이나 lifelong 학습으로의 확장은 다루지 않음.
- Top-K 검색이 미분 불가능해 seeker의 검색 품질이 end-to-end로 최적화되지 않음.
- 벤치마크가 LIBERO/SimplerEnv-Bridge 2개에 한정되고, CALVIN 같은 장기 연쇄 벤치마크 부재. 코드·프로젝트 페이지 미공개 상태.
- 인접 쌍 평균 병합은 단순하지만, 시각적으로 유사하나 의미적으로 구분되는 상태(예: 서랍 열기 전/후)를 뭉갤 위험이 분석되지 않음.

## 11. 향후 연구 방향

저자들은 실세계 로봇 플랫폼으로의 확장을 진행 중이라고 밝혔다. 그 외 자연스러운 방향으로는 (1) 에피소드 간 장기 메모리로의 확장(경험 재사용·스킬 축적), (2) Gumbel-softmax 등으로 검색을 미분 가능하게 만들어 seeker를 공동 최적화, (3) π0 류 flow-matching expert와의 결합, (4) 이동 조작·동적 환경에서 vault 갱신 전략의 강건성 검증이 있다.

## 12. 세미나 토론 질문

1. 장기 vault가 저장하는 것은 '행동 은닉 상태'인데, 이것이 정말 과업 진행(task progress)을 인코딩하는지 아니면 단순 행동 관성(action continuity)을 인코딩하는지 어떻게 구분 검증할 수 있는가?
2. Table 4에서 policy-side 조건화(71.9)가 MemoryVLA의 SimplerEnv 성적(71.9)과 정확히 일치하는데, 이 ablation이 사실상 MemoryVLA 재현이라면 나머지 +2.0의 이득은 얼마나 견고한가?
3. 인접 쌍 병합 기반 vault 압축은 시간적으로 떨어진 중복(예: 반복 방문한 장소)을 처리하지 못하는데, 비인접 병합이나 학습 기반 망각이 더 나은 대안이 될 수 있는가?
4. 고정 12개 메모리 토큰이라는 유계 인터페이스는 Long-90(90개 태스크)에서는 충분했지만, 수백 스텝의 실세계 장기 과업에서도 병목이 되지 않을까?

---

*이 리뷰는 arXiv 2607.07608 v1 PDF 전문을 직접 읽고 작성되었으며, 모든 수치는 논문 Table 1–5, Fig. 3 및 본문 §4에서 검증되었다.*

<!-- VERIFIED: pdf -->
