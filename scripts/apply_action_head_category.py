#!/usr/bin/env python3
"""
Apply action_head_category to all model YAML files based on the taxonomy
defined in docs/action-head-taxonomy.md.

Classification is based on the core action generation mechanism,
not auxiliary components like reasoning, MoE, or RL fine-tuning.
"""

import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"

# ── Classification mapping ──────────────────────────────────────────────────
# model_name → category
# Categories: autoregressive, diffusion, flow_matching, discrete_diffusion,
#             regression, inverse_dynamics, hybrid, other

CLASSIFICATION = {
    # === autoregressive ===
    # Sequential token-by-token prediction (next-token, cross-entropy, causal)
    "OpenVLA":        "autoregressive",   # 256-bin tokenized, Llama-2 AR
    "RT-2-X":         "autoregressive",   # PaLI-X, tokenized actions
    "SpatialVLA":     "autoregressive",   # AR with 3D position encoding
    "GR-1":           "autoregressive",   # GPT-style, next-token + video
    "GR-2":           "autoregressive",   # GPT-style, scaled-up GR-1
    "VLA-Thinker":    "autoregressive",   # AR with dynamic visual CoT
    "ECoT":           "autoregressive",   # AR reasoning + AR action (same mechanism)
    "CoT-VLA":        "autoregressive",   # AR image tokens + AR action tokens
    "TraceVLA":       "autoregressive",   # AR with visual trace prompting
    "LLARVA":         "autoregressive",   # AR text-based action + visual trace
    "SimpleVLA-RL":   "autoregressive",   # AR base + GRPO RL (RL = training, not generation)
    "VLA-RL":         "autoregressive",   # AR base + online RL fine-tuning
    "X-VLA":          "autoregressive",   # AR with soft prompts
    "LLaVA-VLA":      "autoregressive",   # Discrete tokenizer with chunking
    "RoboVLM":        "autoregressive",   # VQ-VAE tokenization + AR prediction

    # === diffusion ===
    # Iterative Gaussian denoising (DDPM/DDIM, epsilon/score matching)
    "Diffusion Policy": "diffusion",      # Seminal work, DDPM U-Net/Transformer
    "CogACT":           "diffusion",      # DiT action module, DDIM inference
    "DexVLA":           "diffusion",      # 1B ScaleDP diffusion expert
    "Octo":             "diffusion",      # Diffusion head, 20 denoising steps
    "RDT-1B":           "diffusion",      # 1.2B diffusion transformer, bimanual
    "3D Diffuser Actor": "diffusion",     # 3D denoising transformer, keypose
    "GR00T-N1":         "diffusion",      # Dual-system, DiT action head (DDPM)
    "GR00T-N1.5":       "diffusion",      # DiT + FLARE objective
    "GR00T-N1.6":       "diffusion",      # Large DiT (32 layers)
    "GR00T-N1.7":       "diffusion",      # DiT variant
    "MemoryVLA":        "diffusion",      # DiT conditioned on memory tokens
    "3D-VLA":           "diffusion",      # Embodied diffusion for goal generation
    "TinyVLA":          "diffusion",      # Compact diffusion-based head
    "SuSIE":            "diffusion",      # Diffusion subgoal → diffusion low-level policy
    "TacVLA":           "diffusion",      # Diffusion with tactile gating
    "DiT4DiT":          "diffusion",      # Dual DiT (video + action), both diffusion
    "SAM2Act":          "diffusion",      # Heatmap prediction (denoising-like)

    # === flow_matching ===
    # ODE-based velocity field prediction (rectified flow, flow matching loss)
    "pi0":            "flow_matching",    # Flow matching on PaLI backbone
    "pi0.5":          "flow_matching",    # Extended pi0, open-world
    "pi*0.6":         "flow_matching",    # pi0 + RL (RECAP); generation = flow
    "FLOWER":         "flow_matching",    # Rectified flow, 4-8 steps
    "SmolVLA":        "flow_matching",    # Compact flow matching
    "InstructVLA":    "flow_matching",    # Flow matching + MoE (MoE = routing, not generation)
    "AtomicVLA":      "flow_matching",    # Flow matching + SG-MoE (MoE = routing)
    "Fast-WAM":       "flow_matching",    # Flow matching for both video & action
    "GST-VLA":        "flow_matching",    # Flow matching action expert
    "FLARE":          "flow_matching",    # DiT + flow matching + latent prediction
    "UniVLA":         "flow_matching",    # Flow matching, unified tokenization
    "LingBot-VLA":    "flow_matching",    # MoT + flow matching (generation = flow)
    "Motus":          "flow_matching",    # Transformer action expert, flow matching
    "Vlaser":         "flow_matching",    # Flow matching action expert
    "GigaWorld-Policy": "flow_matching",  # Flow matching action decoder
    "GR-Dexter":      "flow_matching",    # Flow matching DiT for dexterous hands

    # === discrete_diffusion ===
    # Masking-based diffusion on discrete tokens (iterative unmasking)
    "DD-VLA":         "discrete_diffusion",  # Masked CE, adaptive unmasking, 12 steps
    "E0":             "discrete_diffusion",  # Continuized discrete diffusion
    "dVLA":           "discrete_diffusion",  # Discrete diffusion + multimodal CoT
    "UD-VLA":         "discrete_diffusion",  # Joint discrete denoising (JD3P)

    # === regression ===
    # Single-pass direct prediction (L1/L2, no iterative refinement)
    "OpenVLA-OFT":    "regression",       # Parallel decoding + L1 loss, single pass
    "HPT":            "regression",       # Shared trunk + linear heads
    "CrossFormer":    "regression",       # L1 regression, variable-dim heads
    "RoboMamba":      "regression",       # Lightweight MLP head (3.7M)
    "NanoVLA":        "regression",       # "Efficient action head" (direct)
    "DeeR-VLA":       "regression",       # LSTM+MLP head (early exit = efficiency, not generation)
    "FALCON":         "regression",       # Spatial-grounded action head (direct)

    # === inverse_dynamics ===
    # Action inferred from predicted state transitions
    "UniPi":          "inverse_dynamics", # Video generation → inverse dynamics
    "AVDC":           "inverse_dynamics", # Optical flow → rigid body transform → action
    "DreamVLA":       "inverse_dynamics", # World knowledge → DiT inverse dynamics

    # === hybrid ===
    # Explicitly combines two different generation mechanisms
    "HybridVLA":      "hybrid",           # AR reasoning + diffusion action (two mechanisms)
    "Diffusion-VLA":  "hybrid",           # AR reasoning (NTP) + diffusion action (FiLM)
    "DeepThinkVLA":   "hybrid",           # Causal CoT (AR) + bidirectional parallel action
    "RoboDual":       "hybrid",           # AR VLA (System 2) + DiT specialist (System 1)
    "InternVLA-A1":   "hybrid",           # MoT: understanding (AR) + generation + action experts
    "InternVLA-M1":   "hybrid",           # 2-stage: spatial reasoning → action generation

    # === other ===
    # Specialized architectures that don't fit above categories
    "Gemini Robotics": "other",           # Proprietary, architecture undisclosed
    "Humanoid-VLA":    "other",           # Language-motion pre-alignment (specialized)
    "AcceRL":          "other",           # RL framework (system, not action head)
    "PLD":             "other",           # Residual RL framework (system, not action head)
    "DepthCache":      "other",           # Token merging (efficiency, not action head)
    "SAMoE-VLA":       "other",           # Autonomous driving BEV-based (different domain)
    "GR00T-N2":        "other",           # DreamZero-based (undisclosed)
    "UP-VLA":          "other",           # Understanding + prediction (auxiliary focused)
    "Mobility VLA":    "other",           # Hierarchical navigation (topo graph)
    "NaVILA":          "other",           # Hierarchical VLA + locomotion RL
    "TwinBrainVLA":    "other",           # Asymmetric MoT (architectural innovation, not generation)
    "HiMoE-VLA":       "regression",       # MoE routing + MLP experts (MoE = routing, base = regression)

    # === FAST tokenizer models ===
    # FAST is a tokenization scheme, not a generation mechanism
    # Classification depends on the underlying generation: AR with FAST tokens
    "pi0-FAST":       "autoregressive",   # FAST tokenization + AR generation
    "NORA":           "autoregressive",   # FAST+ tokenization + AR generation
}


def main():
    # Verify all models are classified
    yaml_files = sorted(MODELS_DIR.glob("*.yaml"))
    models = {}
    for yf in yaml_files:
        with open(yf) as f:
            data = yaml.safe_load(f)
        models[data["name"]] = (yf, data)

    # Check coverage
    unclassified = set(models.keys()) - set(CLASSIFICATION.keys())
    if unclassified:
        print(f"ERROR: {len(unclassified)} models not classified:")
        for name in sorted(unclassified):
            ah = models[name][1].get("architecture", {}).get("action_head", "?")
            print(f"  {name}: {ah}")
        return 1

    extra = set(CLASSIFICATION.keys()) - set(models.keys())
    if extra:
        print(f"WARNING: {len(extra)} classifications for non-existent models: {extra}")

    # Apply categories
    updated = 0
    for name, (yf, data) in models.items():
        category = CLASSIFICATION.get(name)
        if not category:
            continue

        arch = data.get("architecture", {})
        if arch.get("action_head_category") != category:
            arch["action_head_category"] = category
            data["architecture"] = arch

            with open(yf, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True,
                          sort_keys=False, width=120)
            updated += 1

    # Summary
    from collections import Counter
    counts = Counter(CLASSIFICATION.values())
    print(f"\nAction Head Category Distribution:")
    for cat, count in counts.most_common():
        print(f"  {cat:22s}: {count}")
    print(f"\nTotal: {sum(counts.values())} models, {updated} files updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
