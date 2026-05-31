# BlockVLA: Accelerating Autoregressive VLA via Block Diffusion Finetuning

> **한 줄 요약**: 자기회귀(AR) VLA 백본을 **block diffusion** 패러다임으로 fine-tuning해 블록 간에는 AR 의존성을, 블록 내에는 병렬 denoising을 유지함으로써 prefix KV-cache 재사용과 반복 denoising 비용 절감을 동시에 달성하는 정책. 표준 discrete diffusion 대비 **3.3× 추론 가속**.

---

## 1. 배경 및 동기

자기회귀 VLA 모델은 강력한 추론 능력을 보여주지만 토큰을 순차적으로 디코딩하기 때문에 **추론 latency가 크고 long-horizon에서 오차 누적이 증폭**될 수 있다. 대안으로 등장한 **이산 확산 언어 모델(dLLM)**은 토큰을 병렬로 정제(parallel refinement)할 수 있으나, (a) denoising function evaluation(NFE)이 반복적으로 필요하고, (b) 양방향 반복 디코딩에 표준 KV caching을 그대로 적용하기 어렵다는 문제로 로봇 실배포에서는 제한적이었다. BlockVLA는 이 두 패러다임의 장점을 **block 단위로 혼합**해 가속과 정밀성을 모두 잡고자 한다.

## 2. 핵심 아이디어

- **Block diffusion policy**: 토큰들을 블록으로 묶고, **블록 사이**는 자기회귀 의존성을, **블록 내부**는 병렬 denoising을 적용.
- **Global causal coherence + local parallelism**: 전역적 인과성(긴 시퀀스의 일관성)을 유지하면서도 블록 내에서는 dLLM식 병렬 정제로 throughput 확보.
- **Prefix KV-cache 재사용**: 완료된 블록까지의 KV를 dLLM에서도 활용 가능 — 표준 discrete diffusion에서 막혀 있던 KV caching의 적용 경로를 확보.
- **AR → diffusion 부드러운 전이**: 사전학습된 AR backbone을 block diffusion 정책으로 fine-tuning하는 형식이라, 처음부터 dLLM을 학습할 필요 없이 AR pretraining의 자산을 그대로 활용.

## 3. 방법론 요약

BlockVLA는 사전학습된 AR VLA backbone(예: OpenVLA류)에서 출발해, 행동/토큰 시퀀스를 일정 크기의 **블록**으로 분할한다. 학습 시에는 각 블록을 discrete diffusion 방식으로 마스킹/노이즈 처리하고 병렬 denoising을 학습하면서, 블록 간 순서에 대해서는 AR 손실(이전 블록을 condition으로 다음 블록을 생성)을 유지한다. 이로써 한 forward에서 한 블록 내의 모든 토큰을 동시에 정제할 수 있고, 이미 확정된 이전 블록들의 KV는 prefix cache로 재사용해 NFE의 비용을 크게 줄인다. 평가는 **LIBERO**와 **SimplerEnv**에서 광범위하게 수행되었다고 명시되어 있다.

## 4. 실험 결과

- **추론 가속**: 표준 discrete diffusion baseline 대비 **3.3×** 가속.
- 평가 벤치마크: **LIBERO**, **SimplerEnv** (구체적 suite별 점수는 abstract에 미명시).
- LIBERO suite별 absolute 성공률, SimplerEnv(Google Robot pick coke can 등) 절대 점수는 abstract에 미명시.
- AR baseline 대비 성능 보존/향상 폭, 블록 크기 ablation은 abstract에 미명시.
- 모델 파라미터 수, 코드 공개 여부는 abstract에 미명시.

## 5. 한계 및 의의

BlockVLA의 핵심 통찰은 **양방향 dLLM과 단방향 AR 사이의 trade-off를 "블록"이라는 granularity로 명시적으로 파라미터화**했다는 점이다. 블록 크기를 1로 두면 사실상 AR, 블록 크기를 전체 시퀀스로 두면 dLLM에 가까워지므로, 이 사이의 sweet spot을 찾는 일반적인 프레임워크 역할을 한다. KV-cache 재사용을 dLLM 정책 영역으로 확장한 점도 실용적 가치가 크다. 다만 abstract만으로는 (a) LIBERO/SimplerEnv 정확도가 AR baseline과 어떻게 비교되는지 (가속만 보고되고 정확도가 보고되지 않은 점은 보수적으로 해석할 필요), (b) 블록 크기와 NFE 횟수의 trade-off 곡선, (c) chunk-level/token-level 캐시 정책과의 비교, (d) 실로봇 검증 여부를 확인할 수 없다. 코드 공개도 abstract에서 확인되지 않아 재현성 평가는 본문/저장소 확인이 필요하다.

<!-- VERIFIED: abstract-only -->
