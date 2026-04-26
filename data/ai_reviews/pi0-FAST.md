# π0-FAST: Efficient Action Tokenization for Vision-Language-Action Models

> **한 줄 요약**: Discrete Cosine Transform(DCT) + Byte-Pair Encoding(BPE)으로 robot action chunk를 압축하는 FAST tokenizer를 도입, autoregressive VLA가 50Hz 고주파 dexterous task에서도 기존 binning 방식이 완전히 실패하는 영역을 학습 가능하게 만들고, π0 diffusion VLA 성능을 매칭하면서 **5배 빠른 학습**을 달성. 1M action sequence로 훈련된 universal tokenizer FAST+를 함께 공개.

---

## 1. 배경 및 동기 (Section I, IV)

- 기존 autoregressive VLA (RT-2, OpenVLA)의 action tokenization은 per-dimension·per-timestep binning (256 bins, Section III). 1초 H=50 chunk에 대해 D=14면 700 token 필요.
- **문제 (Section IV, Figure 3 case study)**: 합성 cubic spline 보간 task에서 sampling rate 25 → 800 timestep으로 늘리면 binning tokenizer의 MSE는 급격히 상승. Qualitative 시각화: 모델이 "단순히 첫 action을 복사" — 인접 token간 marginal information이 0에 수렴하기 때문.
- 핵심 통찰: "robot action signals need to be compressed before training, to reduce correlation between consecutive tokens" (Section I) — language의 BPE처럼 compression-based tokenization이 필요. 단, action은 연속이므로 frequency-domain 압축 (DCT)이 자연스러운 선택.

---

## 2. 방법론 (Section V, Algorithm 1, Figure 4)

### FAST Tokenization Pipeline (Section V-B, Figure 4)
1. **Normalize**: 각 dim별 1%/99% quantile로 [−1, 1] 정규화 (outlier robust).
2. **DCT**: 각 action dim별로 discrete cosine transform. Low-frequency가 전체 형태를, high-frequency가 sharp jump를 표현 (JPEG와 동일 원리).
3. **Quantize**: round(γ·C). γ는 lossiness vs compression rate trade-off (default γ=10).
4. **Flatten**: |A|×H DCT matrix를 column-first로 flatten (low-frequency components first across dims) — autoregressive sample이 전체 형태부터 예측하도록.
5. **BPE**: 1024 vocabulary로 byte-pair encoding. 0이 많은 sparse signal을 dense token sequence로.

### Universal Tokenizer FAST+ (Section V-C)
- 1M action chunk (single-arm·bimanual·mobile·humanoid 등) 위에서 BPE 학습.
- HuggingFace `AutoProcessor.from_pretrained("physical-intelligence/fast")`로 3줄 코드에 적용 가능.

### π0-FAST 통합 (Section VI-A)
- π0 backbone (PaliGemma 3B) 그대로, action expert 대신 FAST 토큰을 standard autoregressive next-token prediction으로 학습.
- 동일 vocabulary slot에 action token을 overwrite (RT-2/OpenVLA 방식).

---

## 3. 실험 결과

### Compression Ratio (Table I)
| Dataset | Frequency | Naive tokens | FAST tokens | Compression |
|---------|-----------|--------------|-------------|-------------|
| BridgeV2 | 5 Hz | 35 | 20 | 1.75× |
| DROID | 15 Hz | 105 | 29 | 3.6× |
| Bussing | 20 Hz | 140 | 28 | 5.0× |
| Shirt Fold | 50 Hz | 700 | 53 | **13.2×** |

특히 고주파 task에서 압축률이 급격히 증가하며, FAST는 frequency와 무관하게 **chunk당 약 30 token으로 수렴** — 신호 복잡도에 따라 적응적으로 token 수를 결정.

### Tokenizer 비교 (Section VI-B, Figure 6)
LIBERO / DROID / Table Bussing / T-Shirt Folding 4개 task에서 success rate / task progress:
- **Naive binning**: 20Hz/50Hz task에서 거의 0% — task progress 불가.
- **FSQ (Vector Quantization)**: 어느 정도 학습되나 dexterous task에서 FAST 대비 약함.
- **FAST**: 모든 task에서 강함, 특히 고주파에서 격차 큼.
- **FAST+** (universal): per-dataset FAST tokenizer와 거의 동등 → universal default로 사용 가능.

### OpenVLA + FAST (Section VI-D)
- T-Shirt Folding (50Hz)에서 OpenVLA + naive ≈ 0% → OpenVLA + FAST+ 큰 폭 향상. tokenizer가 model backbone과 독립적임을 입증.

### π0-FAST vs π0 Diffusion (Section VI-E, Figure 9)
- **소규모 dataset** (LIBERO, T-Shirt Folding <50h): 두 모델 동등.
- **대규모 dataset** (Table Bussing): π0-FAST가 **3배 적은 step**으로 동일 성능 도달.
- **DROID zero-shot**: π0-FAST가 language instruction을 더 잘 따름 (diffusion π0는 종종 language 무시).
- **Inference latency 단점**: π0 diffusion 100ms vs π0-FAST 750ms/chunk (Section VI-E) — 30~60 action token을 autoregressive decode + 2B language backbone 사용 (vs 300M action expert).

### Generalist Scaling (Section VI-F, Figure 11)
- 903M timestep + 9.1% open-source (BRIDGE/DROID/OXE)로 cross-embodied 학습.
- π0-FAST는 π0 diffusion의 zero-shot 성능을 매칭하면서 **5× fewer GPU hours** (Section VI-F).
- Laundry folding (Figure 10) 같은 가장 어려운 long-horizon dexterous task도 성공.

### DROID Zero-shot (Section VI-A, Figure 7)
- 3개 university campus (Berkeley/Stanford/U.Washington)의 unseen 환경에서 zero-shot table-top manipulation. 단순 picking, 캐비닛 열기, 수도꼭지 조작 등.
- 처음으로 DROID에서 **co-training/fine-tuning 없이 language prompt만으로 zero-shot evaluation** 성공.

### BPE Ablation (Section VI-D)
- BPE 제거 시 성능 하락 (T-Shirt Folding, Table Bussing). DCT만으로도 naive보다 좋지만, 0이 많은 sparse signal에서 BPE가 학습 신호를 살리고 inference 속도를 단축.

---

## 4. 한계

- **Inference speed**: π0 diffusion 대비 7.5× 느림. dynamic task에는 부적합. speculative decoding·quantization 등 LLM optimization 적용 필요 (Section VII).
- **Static manipulator만 검증**: humanoid·mobile robot은 offline tokenizer test만 했고 policy 평가 미실시.
- **DCT 가정**: action signal이 충분히 smooth해야 함. 매우 discontinuous한 control (예: contact-rich의 sudden grip)에서는 high-frequency component 손실 가능.
- **BPE vocab 의존**: BPE 1024 size가 default지만 더 큰 vocab의 효과 미검증.

---

## 5. 총평

FAST의 가장 큰 기여는 **autoregressive VLA가 diffusion VLA와 경쟁 가능하다**는 것을 입증한 것이다. Section VI-F의 5× compute 절감은 단순 효율 개선이 아니라 패러다임 선택 문제 — autoregressive는 LLM 인프라(speculative decoding, 양자화, KV cache, batch inference) 를 그대로 활용 가능. JPEG의 DCT라는 고전적 기법을 robotics에 응용한 점이 인상적이며, "BPE는 language tokenization의 핵심인데 왜 robot은 안 쓰지?"라는 단순한 질문에서 출발했다는 게 주목할 만하다. Figure 3의 case study (sampling rate ↑ → naive 실패)는 직관적이고 설득력 있다. 한편 inference latency 문제는 FAST의 **유일한 약점**이며, 후속작 π0.5는 "discrete pre-train + flow post-train"으로 이를 우회하는 전략을 채택했다.

---

## 6. 예상 질문

- **Q1**: 왜 DCT인가? FFT·Wavelet은?
  - A: Section V-A — DCT는 분석적·매우 단순·JPEG 등에서 검증된 압축률. Wavelet도 가능하지만 DCT의 단순성이 BPE와의 결합에 적합.
- **Q2**: BPE가 정말 필요한가?
  - A: Section VI-D ablation — BPE 없으면 0 token이 많아 학습 신호 dilute + 수백 token autoregressive decode로 inference 속도도 저하.
- **Q3**: FAST+ universal tokenizer는 dataset-specific tokenizer 대비 성능 손실 없나?
  - A: Figure 6 — 거의 동일. 1M chunk의 다양성으로 충분히 일반적인 BPE vocab을 학습.
- **Q4**: Diffusion vs Autoregressive — 결론은?
  - A: Section VII — "the jury is still out". FAST는 학습 효율·language following에서 우월, diffusion은 inference latency에서 우월. trade-off.
- **Q5**: 750ms/chunk inference로 50Hz control이 가능한가?
  - A: H=50의 1초 chunk를 750ms마다 생성하므로 receding-horizon control 가능 (action의 0~750ms 부분 실행 후 다음 chunk). 그러나 dynamic task에는 부족.
- **Q6**: BridgeV2 1.75× 압축은 작아 보이는데?
  - A: 5Hz는 이미 redundancy가 적어 압축 여지가 작음. FAST의 진가는 50Hz (13.2×).

<!-- VERIFIED: pdf -->
