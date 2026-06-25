# NAC: Neural Action Codec for Vision-Language-Action Models

> **한 줄 요약**: Robot action chunk를 multi-channel 1D pseudo-waveform으로 보고 **neural audio codec (SoundStream/DAC/SNAC 계열의 multi-scale RVQGAN)을 그대로 가져와** action tokenizer로 재활용. 단, **mel-spectrogram loss를 완전히 제거**해야 동작 (mel을 쓰면 LIBERO-10 성공률이 0%로 붕괴) — 인간 청각이 아닌 *kinematic fidelity*에 맞춰 reconstruction은 time-domain MSE, 고주파 보존은 DAC discriminator의 adversarial loss, 신호 복원은 Vocos-style ISTFT head로 처리. Offset codebook 위에서 학습된 작은 autoregressive policy(NACPolicy)는 LIBERO-10 49.73% / RoboMimic 33.94% / Real-world 50%로 Bin, Diffusion Policy, FAST, VQ-VLA, OAT를 모두 능가. Action chunk 1개당 12개 token으로 압축해 Bin 대비 19×, FAST 대비 3× 짧은 시퀀스 길이를 달성.

---

## 1. 배경 및 동기 (Section 1)

- **VLA 모델의 핵심 병목**은 연속적인 로봇 action을 autoregressive sequence modeling이 다룰 수 있는 discrete token으로 어떻게 압축할지에 있다. 초기 OpenVLA는 per-dimension binning을 사용했지만 sequence가 prohibitively 길어졌고, FAST는 DCT 기반 frequency compression으로 sequence를 단축했지만 hand-designed frequency prior가 비선형 dynamics를 포착하기에 부족하다.
- 저자들은 **audio generation 도메인이 거의 동일한 문제**(고충실도 압축, 저지연, 복잡한 시간 분포)를 SoundStream/Encodec/DAC와 같은 *neural audio codec*으로 이미 해결했음을 관찰한다. 이들은 convolutional encoder-decoder + Residual Vector Quantization (RVQ) 구조이고 AudioLM, VALL-E 등 audio foundation model의 표준 front end가 되었다.
- 두 도메인의 차이: action은 30-60 Hz (audio는 16-48 kHz)로 훨씬 저주파, 그리고 multi-channel (7-14 joint/end-effector dim)이며 audio처럼 mel-frequency perceptual scale에 맞출 필요가 없다. 핵심 가설은 "*몇 가지 audio-specific 가정만 제거하면 multi-scale RVQGAN을 거의 그대로 transfer 가능*"이라는 것.

---

## 2. 문제 정의: Behavioral Cloning + Action Tokenization (Section 3.1)

- 입력: 시각 관찰 `o_img ∈ R^{Ho×Ncam×C×H×W}` (history Ho=2, multi-camera), task 조건 `l` (자연어 또는 discrete task UID).
- 출력: action chunk `a_{1:Ha} ∈ R^{B×Ha×Da}` (Ha=32, Da=7-14).
- Tokenizer `T`가 chunk를 discrete sequence `C = [c_1, ..., c_L]`로 매핑하고 policy는

  `max_θ E [Σ_i log π_θ(c_i | o_img, l, c_<i)]`  (Eq. 1)

  를 최적화. 추론 시 `T^{-1}: Ĉ_{1:L} → â_{1:Ha}` (Eq. 2)로 연속 action을 복원. 연속 action을 직접 regression하면 compounding error와 multimodal action distribution 모델링 어려움이 생기므로 discrete token화가 표현력을 높인다.

---

## 3. NAC Tokenizer: SEANet Encoder + MRVQ + Vocos Decoder (Section 3.2)

### 3.1 1D Signal Representation
- 연속 action chunk `a ∈ R^{B×Ha×Da}`를 시간·feature 축으로 flatten해 `a_flat ∈ R^{B×(Ha·Da)}`로 펴고, 다시 single-channel pseudo-waveform `w ∈ R^{B×1×L}` (L = Ha·Da)로 unsqueeze.
- 선택적으로 dataset statistics 기반 per-dimension linear normalization 적용.

### 3.2 SEANet Encoder
- 1D conv stack + strided downsampling (ratios `R = [r_1, ..., r_k]`, 총 hop length = ∏R).
- Residual block + ELU + weight normalization + reflection padding.
- 출력 latent `z ∈ R^{B×D_enc×T'}` (D_enc=512, T'=L/∏R).

### 3.3 Multi-Scale Residual Vector Quantization (MRVQ)
- nq stage RVQ, 각 codebook 크기 Vbins=1024.
- **차별점**: 표준 RVQ와 달리 각 stage 전에 *per-stage temporal pooling*을 적용 — 초기 stage가 더 넓은 시간 window에서 coarse 구조를 잡고, 후기 stage가 high-frequency residual을 잡도록 강제.
- `c_s = argmin_j ||pool(z_s) - e_j^{(s)}||_2` (Eq. 3), 잔차는 `z_{s+1} = z_s - upsample(e_{c_s}^{(s)})`로 propagate.
- Codebook은 첫 batch에서 k-means initialization, 이후 EMA update.

### 3.4 Commitment Loss (Eq. 4)
- `L_commit = ||z - sg(e_c)||_2^2`, encoder 출력이 codebook embedding과 너무 멀어지는 것 방지.
- **계수 λ_commit = 1000으로 매우 크게**: latent space를 엄격히 bound해 codebook collapse 방지.

### 3.5 Vocos-style Decoder + ISTFT Head
- Conv1D embedding → ResNet block (with attention) → ConvNeXt block stack.
- 마지막에 STFT magnitude와 phase를 예측한 뒤 **Inverse STFT (ISTFT)** 로 1D 신호 복원.
- ISTFT hop length를 encoder hop length와 동기화. ISTFT head를 linear head로 교체하면 LIBERO-10 성공률이 48.3 → 42.1%로 떨어진다 (Table 1c).

### 3.6 Discriminator + Adversarial Loss
- SNAC를 따라 MPD (Multi-Period), MRD (Multi-Resolution), DAC discriminator를 1D action 신호용으로 적응.
- **DAC discriminator가 최고**: 49.45 ± 2.02% (LIBERO-10), MPD 46.28%, MRD 45.68% (Table 1b).
- Discriminator를 제거하면 **성공률 0%**, MSE 0.35 — adversarial supervision이 rapid corrective motion / sharp velocity change의 고주파 detail 보존에 필수.

### 3.7 Generator Loss (Eq. 5)
`L_gen = L_reconst + λ_commit · L_commit + L_adv`
- L_reconst는 MSE, L1, 또는 unscaled spectrogram 가능. **Mel-spectrogram loss는 절대 사용하지 않음** (다음 섹션 참조).

---

## 4. 핵심 발견: Mel-spectrogram Loss 제거 (Section 4.2, Table 1a)

| Reconstruction Loss | LIBERO-10 Success (%) | MSE |
|---|---:|---:|
| L1 | 44.78 ± 2.48 | 0.002 ± 0.005 |
| **MSE** | **49.2 ± 1.54** | 0.0008 ± 0.0007 |
| DCT | 47.85 ± 1.18 | 0.0007 ± 0.0008 |
| **Mel Spectrogram** | **0 ± 0.11** | 0.038 ± 0.026 |
| Spectrogram (non-mel) | 48.3 ± 2.92 | 0.0002 ± 0.001 |

- Mel-spectrogram loss는 인간 pitch perception (Stevens et al. 1937)에 맞춰진 perceptual scale인데 robot action은 acoustic wave가 아니라 *kinematic signal* → mel을 쓰면 downstream policy가 완전히 collapse.
- 흥미로운 trade-off: **non-mel spectrogram이 reconstruction MSE는 가장 낮지만 (0.0002), downstream success는 MSE loss가 더 높음 (49.2 > 48.3)** → tokenizer는 단순 reconstruction error가 아닌 *downstream policy-friendliness* 기준으로 평가해야 한다.

---

## 5. NACPolicy: Offset Codebook 기반 Autoregressive BC (Section 3.3)

- Tokenizer가 nq scale을 갖는다면 policy 어휘 크기는 `|V| = nq × Vbins + 1` (BOS 포함).
- **Offset codebook**: scale s의 token ID는 `[s·Vbins, (s+1)·Vbins)` 구간으로 강제 → policy가 어느 scale을 생성 중인지 ID만으로 식별 가능.
- 학습 시 ground-truth code를 offset 후 flat sequence로 concat:

  `C_flat = [BOS, C_1^{(0)}, ..., C_{L_0}^{(0)}, C_1^{(1)}, ..., C_{L_1}^{(1)}, ...]`  (Eq. 6)

- 즉 모든 scale 0 token을 먼저 예측하고 그 다음 scale 1 token을 예측 (coarse-to-fine).
- 표준 causal cross-entropy로 학습. 추론은 Algorithm 2: BOS → autoregressive sampling → scale 별 segment partition → modulo arithmetic으로 code index 복원 → frozen detokenizer로 action chunk 복원 → receding horizon으로 첫 16 step만 실행.

---

## 6. 시뮬레이션 결과 (Section 4.3, Table 2)

| Environment | Bin | Diffusion Policy | FAST | VQ-VLA | OAT | **NAC** |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO-10 | 3.95 ± 0.8 | 25.48 ± 1.3 | 38.02 ± 1.3 | 10.85 ± 1.85 | 44.17 ± 1.2 | **49.73 ± 1.0** |
| RoboMimic | 7.56 ± 1.05 | 27.25 ± 1.87 | 28.38 ± 2.37 | 21.44 ± 1.45 | 31.94 ± 2.15 | **33.94 ± 1.86** |

- LIBERO-10에서 **FAST 대비 +11.71pt, OAT 대비 +5.56pt**. RoboMimic에서도 일관된 우위.
- 시사점: 압축률만 좋아져서는 부족하고, tokenizer가 **next-token policy learning에 적합한 구조**를 보존해야 함. NAC의 multi-scale RVQ + offset codebook이 hand-designed (FAST) 및 다른 learned tokenizer (VQ-VLA, OAT)보다 autoregressive control에 더 좋은 interface를 제공.

---

## 7. 실세계 결과 (Section 4.4, Table 2 + Appendix Table 6)

| Task | Bin | Diffusion | FAST | VQ-VLA | OAT | **NAC** |
|---|---:|---:|---:|---:|---:|---:|
| Weighing | 50 | 40 | 80 | 90 | 40 | 90 |
| Grapes | 0 | 30 | 80 | 30 | 80 | **100** |
| Marker | 0 | 30 | 60 | 30 | 50 | 50 |
| Two Blocks | 0 | 0 | 0 | 0 | 0 | **30** |
| Three Blocks | 0 | 0 | 0 | 0 | 10 | 0 |
| Chess | 0 | 0 | 10 | 0 | 40 | 10 |
| Place Stone | 0 | 0 | 10 | 50 | 30 | 40 |
| Fold Towel | 0 | 80 | 80 | 50 | 70 | 80 |
| **Total** | **6.25** | **22.5** | **40** | **31.25** | **40** | **50** |

- 8 physical task × 10 trial, 단일 정책이 모든 task를 다 잡지는 못하지만 평균적으로 NAC이 가장 안정적.
- 특히 **정밀한 localized correction**이 필요한 task (Grapes 100%, Two Blocks 30% — 다른 모든 방법 0%) 에서 두드러진 gain.

---

## 8. 압축률 / Latency (Section 4.5, Table 3)

| Method | Params (M) | Tokens / chunk | # of Bits | Enc (ms) | Dec (ms) | Total Recon (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Bin | 0.002 | 224 | 2240 | 0.045 | 0.039 | 0.079 |
| OAT | 65.2 | 12 | 120 | 0.931 | 2.392 | 3.347 |
| VQ-VLA | 65.6 | 12 | 120 | 7.086 | 4.045 | 11.049 |
| FAST | 0.000 | 36 | 360 | 0.170 | 0.110 | 0.290 |
| **NAC** | **63.0** | **12** | **120** | 1.270 | 2.183 | **3.536** |

- **Chunk 당 12 token** — Bin 대비 19×, FAST 대비 3× 압축. OAT와 동률이지만 simulation/real-world 성능이 더 높음.
- Total reconstruction latency 3.54 ms (RTX 4090). VQ-VLA(11 ms)보다 훨씬 빠르고 고주파 closed-loop control에 사용 가능한 수준.

---

## 9. 핵심 Ablation 정리 (Table 1 전체)

1. **Reconstruction Loss (Table 1a)**: Mel = 0%, MSE / Spectrogram / DCT / L1 ≈ 45-49%. → mel만 제거하면 audio codec backbone이 그대로 통함.
2. **Discriminator (Table 1b)**: None = 0%, MPD 46.3%, MRD 45.7%, **DAC 49.45%**. → 적대적 학습이 필수, DAC가 best.
3. **Decoder head (Table 1c)**: ISTFT 48.3% vs Linear 42.1%. → ISTFT 구조가 detail-sensitive trajectory 복원에 유의미한 기여.

세 component를 묶으면 SoundStream → DAC → SNAC로 이어진 audio codec 진화의 거의 모든 요소가 robot action에 그대로 transfer됨을 입증.

---

## 10. 한계 (Discussion, Limitations)

- **Sequence length 제약**: flatten한 1D 신호 길이 `Ha · Da`가 네트워크 downsampling ratio (K = ∏ratios × ∏vq_scales)로 나누어 떨어져야 함 → action horizon이나 action dim 변경 시 architecture 재설계 필요.
- **Detokenization의 explicit dim 의존성**: 복원된 1D 신호를 다시 (Ha, Da)로 reshape하려면 target action dimensionality를 명시적으로 알아야 함 → 임의 embodiment에 dynamic 적용 불가.
- 평가 범위: LIBERO에서 **LIBERO-10 (Long suite) 한 개만** 측정 (Spatial/Object/Goal 미보고). Cross-embodiment 일반화나 large-scale pretraining도 미보고.
- 비교 대상이 모두 *동일 protocol*에서 동일 backbone 위에 재학습된 결과지만, π0 / OpenVLA 등 대형 VLA와 직접 비교는 없음 — tokenizer-level 비교이지 frontier VLA와의 SOTA 경쟁이 아님.

---

## 11. 의의 (Section 5)

- **개념적**: VLA action tokenization을 *signal compression* 문제로 재정의하고, 이미 잘 푼 audio codec 영역의 30년치 engineering을 끌어옴 — mel만 빼면 transfer가 거의 trivial하다는 점이 본 논문의 가장 강한 메시지.
- **기술적**: (i) multi-scale RVQ + per-stage temporal pooling, (ii) ISTFT head로 phase까지 복원하는 Vocos backbone, (iii) DAC discriminator 기반 adversarial supervision — 이 세 가지를 합쳐 12-token 압축 + 49.7% LIBERO-10이라는 강력한 baseline 제공.
- **공학적**: 동일 token budget (12 tokens)을 쓰는 OAT보다 +5.6pt를 얻음 → tokenizer architecture 자체의 inductive bias가 여전히 큰 lever임을 보여줌. NACPolicy 자체는 매우 가벼운 4-layer Transformer임에도 강한 성능 — *좋은 token이 작은 policy를 가능케 한다*는 일반적 lesson.
- **향후 방향**: cross-embodiment shared token space (padding 회피), audio foundation model 수준의 massive corpus pretraining, high-frequency control 적용.

---

## 12. 평가

- **장점**:
  - Audio codec → action codec 전이의 **가장 깨끗하고 잘 ablation된 사례**. Mel을 빼야 한다는 단 한 개의 발견을 명확히 입증.
  - LIBERO-10 + RoboMimic + 8 실세계 task에서 5개 baseline (Bin, Diffusion Policy, FAST, VQ-VLA, OAT)을 동일 protocol로 비교 — 재현성과 공정성이 높음.
  - 압축 / latency / fidelity / downstream 성공률의 4축에서 모두 frontier에 위치.
  - Code, project page (`ahadjawaid.com/nac`) 공개 의지.
- **약점 / 의문**:
  - LIBERO를 **Long suite (LIBERO-10)만** 평가 — Spatial/Object/Goal까지 다 본 다른 모델과 직접 비교가 어려움. libero_avg를 4-suite mean으로 환산할 수 없음.
  - 절대 성공률 49.7%는 frontier OFT/π0 계열(95-98%)에 비해 낮음. NAC는 tokenizer 비교 baseline에서의 우위이지 SOTA VLA 자체는 아님.
  - Action codec / discriminator 등 audio domain technique을 그대로 transfer하면서 *왜 robot action에 적합한가*에 대한 spectral 분석은 부분적 (mel이 왜 깨지는지에 대한 경험적 증거만 있음).
  - Sequence-length 제약, embodiment dimensionality 명시 등 architecture 측면의 brittleness가 future cross-embodiment 확장을 막을 수 있음.
- **종합**: 새로운 SOTA VLA라기보다는 **action tokenization 분야의 강력한 새 baseline + cross-domain transfer의 모범 사례**. Audio codec 커뮤니티의 다음 발전 (예: 더 작은 SNAC, lower-bitrate codec)이 곧바로 robotics로 흘러들어올 통로를 열었다는 점에서 영향력이 클 것.

---

<!-- VERIFIED: pdf -->
