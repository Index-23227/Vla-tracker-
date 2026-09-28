# ContextFlow: In-Context Flow Matching for Robot Manipulation

> **한 줄 요약**: π0를 기반으로 in-context 시연(이미지·proprio·action 시퀀스)을 perceiver 방식의 멀티모달 압축기로 요약해 context expert에 넣고, flow-matching action expert가 연속 action chunk를 생성하게 한 in-context imitation learning 모델. LIBERO의 보지 못한 태스크 구성에서 평균 73.5%로 ICRT(38.5%)보다 35pt 높고, unseen 태스크 1-demo로 fine-tune한 π0(72.5%)와 비슷한 수준을 test-time 학습 없이 달성했다.

- **arXiv**: 2609.06852v1 (2026-09-06, cs.RO)
- **소속**: KAUST, UC Berkeley
- **프로젝트**: https://dingjiansw101.github.io/contextflow-page/
- **백본**: π0 초기화 SigLIP + 작은 Gemma 스타일 context expert/action expert

---

## 1. 배경 및 동기

in-context imitation learning은 몇 개의 시연만 보고 gradient 업데이트 없이 새 태스크를 수행하는 것을 목표로 한다. 기존 방법(ICRT, LipVQ-VAE, RICL 등)은 대부분 연속 행동을 토큰화하는 autoregressive 모델이라 (1) 초기 오류가 다음 토큰으로 누적되는 compounding error, (2) action tokenizer의 재구성 오류라는 두 문제를 안고 있다. 저자들은 이 문제가 학습/평가 태스크 구성이 다른 in-context 설정에서 특히 커진다고 보고(Fig. 1c/d), flow matching으로 연속 action chunk를 병렬 생성하는 설계를 제안한다.

## 2. 문제 정의

- 태스크 τ의 시연 d = [T, X, Q, A] (텍스트, 이미지 시퀀스, proprio, action)와 현재 관측 o_t = (I_t, q_t)가 주어질 때 π(a_t | o_t, d)를 학습 (a_t는 H 길이 action chunk).
- 학습은 seen 태스크의 관측 + 같은 태스크의 무작위 시연으로, 추론은 unseen 태스크 구성의 시연을 in-context로 주고 파라미터 갱신 없이 수행.
- 대상은 "관련 primitive 안에서의 보지 못한 구성"(새 물체·배치)이며 임의의 새 태스크는 아니다.

## 3. 방법

- **Conditional flow matching**: a^s = (1−s)ε + s·a, 목표 속도장 a − ε, s ~ Beta(1.5, 1). 추론은 Euler K=10 스텝.
- **ContextFlow-Plain**: context expert가 시연 + 현재 관측을 토큰화하고, action expert가 proprio와 noisy action 토큰을 받아 context 토큰에 attention하며 denoise (π0식 mixture-of-experts 구조).
- **Multimodal context compressor**: 시연을 시간축 subsampling(이미지 8프레임, 상태 128, 행동 128) 후, perceiver 스타일의 cross-attn/self-attn 교차 층으로 모달리티별 32개 latent 쿼리에 압축(이미지 4층, 상태/행동 각 2층; 이미지 압축기는 카메라 간 가중치 공유). 결과를 concat해 context expert 입력으로 사용.
- **Block-wise causal attention**: action 토큰은 context를 볼 수 있지만 반대는 불가 → context KV를 캐싱해 flow 스텝마다 재계산하지 않음.

## 4. 구현 세부

- SigLIP(224×224)과 action expert는 π0에서 초기화, context expert는 무작위 초기화.
- action expert는 LoRA(r=32, α=32), vision encoder·context expert는 full fine-tune.
- 20K iteration, batch 32, cosine LR (peak 2.5e-5 → 2.5e-6), 1,000 warm-up.
- 실물용 ALOHA 데이터셋 신규 수집: 3개 task suite 21개 구성 1,064개 + 학습 전용 bimanual 4개 태스크 254개 = 총 1,318 궤적, 25개 구성.

## 5. 평가 프로토콜

- **LIBERO**: Spatial/Object 각 suite의 10개 태스크를 8 seen / 2 unseen으로 고정 분할(학습에는 Goal·LIBERO-10 시연도 추가). unseen 태스크당 50 trial. **표준 LIBERO 프로토콜이 아니므로 표준 LIBERO 점수와 직접 비교 불가.**
- **기준선**: ICRT(AR), ContextAR(같은 비전 백본·π0 초기화의 AR 기준선), 텍스트만 받는 π0/OpenVLA-OFT, unseen 태스크 1-demo로 fine-tune한 π0(FT).
- **실물**: ALOHA(high + 두 wrist 카메라), unseen 구성 6개 × 10 trial, ICRT와 비교.

## 6. 실험 설계의 요점

가장 통제된 비교는 ContextFlow-Plain vs ContextAR로, 백본과 초기화가 같고 action 모델링 방식만 다르다. 여기에 압축기를 더한 ContextFlow, 그리고 "파라미터 적응(π0 FT)"과 "in-context 적응"을 나란히 두어, flow matching과 압축기 각각의 기여와 in-context 방식의 실용성을 분리하려 했다.

## 7. 주요 결과

**Table 1 – LIBERO unseen 구성 (%)**

| Model | Spatial S1 | S2 | Spatial Avg | Object O1 | O2 | Object Avg | Avg |
|---|---|---|---|---|---|---|---|
| OpenVLA-OFT | 94.0 | 32.0 | 63.0 | 0.0 | 34.0 | 17.0 | 40.0 |
| π0 | 52.0 | 0.0 | 26.0 | 46.0 | 80.0 | 63.0 | 44.5 |
| π0 FT (1 demo) | 94.0 | 32.0 | 63.0 | 80.0 | 84.0 | 82.0 | 72.5 |
| ICRT | 54.0 | 0.0 | 27.0 | 4.0 | 96.0 | 50.0 | 38.5 |
| ContextAR | 94.0 | 12.0 | 53.0 | 16.0 | 92.0 | 54.0 | 53.5 |
| ContextFlow-Plain | 90.0 | 20.0 | 55.0 | 92.0 | 54.0 | 73.0 | 64.0 |
| **ContextFlow** | 86.0 | 42.0 | 64.0 | 76.0 | 90.0 | 83.0 | **73.5** |

- Plain vs ContextAR: +10.5pt → flow matching의 이점. 압축기 추가로 +9.5pt 더.
- **Table 2 (압축기)**: 없음 64.0 → 이미지만 68.0 → 이미지+상태/행동 73.5.
- **Table 3 (모달리티 제거)**: 상태/행동 제거 29.0, 이미지 제거 45.0, 텍스트 제거 64.5 (full 73.5) — proprio/action이 가장 중요.
- **Table 4 (시연 길이)**: 이미지 8, 상태/행동 128이 최적(73.5); 256으로 늘리면 66.5.
- **Table 5 (action head)**: MLP L1 58.0, Diffusion(100k iter) 33.5, Flow matching 73.5.
- **Table 6 (누적 MSE, m²)**: ContextAR 4.7 → Plain 3.0 → ContextFlow 2.2.
- **Table 7 (학습 데이터 확장)**: LIBERO-90 추가 시 4개 suite unseen 평균 36.8 → 43.0 (Goal 0 → 29, LIBERO-10 0 → 3).
- **Table 8 (실물, 성공/10)**: ContextFlow 배 2, 오렌지 5, 바나나 4, 키위 4, 펜 뚜껑 열기 4, 달걀 상자 넣기 6; ICRT는 배 2/10 외 모두 0.

## 8. Related Work 상의 위치

- ICRT·LipVQ-VAE·RICL 같은 AR 기반 in-context imitation을 flow matching으로 옮긴 첫 시도에 가깝다.
- Instant Policy/IMOP처럼 3D 구조나 외부 분할에 의존하지 않고 RGB만 사용해 실행 가능한 행동을 바로 출력.
- VIMA는 멀티모달 프롬프트를 쓰지만 에피소드당 2–5개의 고수준 행동만 예측하는 반면, ContextFlow는 수백 스텝의 저수준 행동을 생성.

## 9. 강점

1. 동일 백본·초기화에서 AR vs flow matching을 비교한 통제된 실험 설계.
2. 압축기, 모달리티, 시연 길이, action head 등 절제 실험이 체계적이다.
3. π0 1-shot fine-tune과 동등한 성능을 test-time 학습 없이 달성해 in-context 방식의 실용성을 보여 줌.
4. 고주파 closed-loop 제어가 필요한 bimanual 과제(펜 뚜껑 열기, 달걀 상자)를 실물에서 시연하고 데이터셋까지 수집.

## 10. 약점 및 한계

1. LIBERO 평가가 suite당 unseen 2개 태스크뿐이라 분산이 크다(예: ContextFlow-Plain O1 92 vs ContextFlow O1 76처럼 태스크별로 순위가 뒤집힘).
2. 표준 LIBERO 프로토콜이 아니어서 다른 VLA 결과와 비교하기 어렵다.
3. LIBERO-Goal/LIBERO-10 unseen에서는 여전히 29%/3%로, 긴 horizon이나 목표가 달라지는 태스크로의 일반화는 약하다.
4. 실물 실험은 구성당 10 trial, ICRT와만 비교(π0 FT 등 강한 기준선 없음).
5. 모델 크기나 추론 지연이 본문에 수치로 제시되지 않았다(throughput은 보충자료).

## 11. 재현 및 확장 아이디어

- 여러 개의 시연을 동시에 넣는 multi-shot 설정, 시연 수에 따른 scaling 분석.
- 대규모 cross-embodiment 데이터로 context expert를 사전학습해 LIBERO-10류 장기 태스크 일반화 개선.
- 압축기 쿼리 수·층 수와 추론 비용의 trade-off 정량화.
- π0.5 등 더 강한 base와 결합하거나 언어 없는 순수 시연 프롬프트 설정 평가.

## 12. 총평

ContextFlow는 in-context imitation learning을 flow matching 틀로 옮기고, 긴 멀티모달 시연을 perceiver 압축기로 요약하는 간단한 설계로 AR 기준선 대비 큰 향상을 보였다. 통제된 비교와 절제 실험은 설득력 있지만, 평가 규모가 작고 LIBERO-Goal/10 같은 어려운 unseen 구성에서는 아직 성능이 낮다.

**한 문장 요약**: π0 기반 flow-matching VLA에 perceiver 시연 압축기와 context expert를 붙여, LIBERO unseen 구성에서 fine-tuning 없이 73.5%(ICRT 38.5%, π0 1-shot FT 72.5%)를 달성한 in-context 조작 정책.

<!-- VERIFIED: pdf -->
