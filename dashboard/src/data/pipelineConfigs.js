// ── Detailed pipeline configurations for VLA model architecture diagrams ─────
//
// Schema per model:
//   inputs:  [{ label, sub?, color }]        – input nodes (top)
//   stages:  [Stage]                          – processing blocks (middle)
//   output:  { label, sub?, color }           – output node (bottom)
//   meta?:   { loss?, loop?, notes?: [] }     – footer metadata
//
// Stage variants:
//   Simple : { label, sub?, color }
//   Group  : { group, sub?, color, children: [{ label, sub?, color }], bottom?: { label, sub?, color } }
//
// Colors: b=blue, p=purple, g=green, o=orange, r=red, t=teal,
//         a=amber, x=gray, i=indigo, k=pink, e=emerald, c=cyan, y=yellow

export const PIPELINE_CONFIGS = {

  // ═══════════════════════════════════════════════════════════════════════════
  // #1  PLD — NVIDIA GEAR Lab
  // ═══════════════════════════════════════════════════════════════════════════
  'PLD': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
    ],
    stages: [
      { group: 'Base VLA (any generalist)', sub: 'e.g. OpenVLA / pi0', color: 'p', children: [
        { label: 'VLM backbone', sub: 'frozen', color: 'p' },
        { label: 'Action head', sub: 'frozen', color: 'i' },
      ]},
      { group: 'PLD: Probe → Learn → Distill', sub: 'residual RL framework', color: 'o', children: [
        { label: 'Probe', sub: 'identify weak skills', color: 'a' },
        { label: 'Learn', sub: 'RL on weak skills', color: 'r' },
        { label: 'Distill', sub: 'back to generalist', color: 'g' },
      ]},
    ],
    output: { label: 'Improved actions', sub: 'plug-and-play on any VLA', color: 'e' },
    meta: { loss: 'RL (outcome-based) + distillation', notes: ['Residual RL wrapper', 'Model-agnostic'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #2  SimpleVLA-RL
  // ═══════════════════════════════════════════════════════════════════════════
  'SimpleVLA-RL': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'PrismaticVLM', sub: '7B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'DinoV2', sub: 'vision encoder', color: 'k' },
      ], bottom: { label: 'Llama-2 7B', sub: 'language model', color: 'i' } },
      { label: 'Projector', sub: 'vision → LLM token space', color: 'x' },
      { group: 'Action Head + GRPO RL', sub: 'reinforcement learning fine-tuning', color: 'o', children: [
        { label: 'AR tokenizer', sub: 'OpenVLA-OFT base', color: 'a' },
        { label: 'GRPO', sub: 'outcome-based reward', color: 'r' },
      ]},
    ],
    output: { label: 'Tokenized actions', sub: '7-DoF, RL-improved', color: 'e' },
    meta: { loss: 'GRPO reinforcement learning', notes: ['Simple outcome rewards', 'On OpenVLA-OFT base'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #3  X-VLA
  // ═══════════════════════════════════════════════════════════════════════════
  'X-VLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Embodiment ID', color: 't' },
    ],
    stages: [
      { group: 'Florence-Large VLM', sub: '0.9B', color: 'p', children: [
        { label: 'Florence encoder', sub: 'vision-language', color: 'p' },
        { label: 'Soft prompts', sub: 'per-embodiment', color: 't' },
      ]},
      { label: 'Autoregressive decoder', sub: 'with embodiment-specific adaptation', color: 'i' },
    ],
    output: { label: 'Action tokens', sub: 'embodiment-adapted', color: 'e' },
    meta: { loss: 'Cross-entropy (next-token)', notes: ['25 Hz inference', 'Embodiment soft prompts'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #4  Fast-WAM
  // ═══════════════════════════════════════════════════════════════════════════
  'Fast-WAM': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Wan2.2 Video DiT', sub: '5B world model', color: 'p', children: [
        { label: 'Video tokenizer', sub: '3D causal VAE', color: 'p' },
        { label: 'DiT blocks', sub: 'video generation', color: 'i' },
      ]},
      { label: 'Action-conditioned world model', sub: 'joint video + action prediction', color: 'r' },
      { label: 'Flow matching head', sub: 'velocity field prediction', color: 'o' },
    ],
    output: { label: 'Action chunks', sub: 'continuous, state-relative', color: 'e' },
    meta: { loss: 'Flow matching + world modeling', notes: ['5 Hz inference', '6B total parameters'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #5  VLA-Thinker
  // ═══════════════════════════════════════════════════════════════════════════
  'VLA-Thinker': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'PrismaticVLM', sub: '7B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'DinoV2', sub: 'vision encoder', color: 'k' },
      ], bottom: { label: 'Llama-2 7B', sub: 'language model', color: 'i' } },
      { group: 'Chain-of-Thought Reasoning', sub: 'think before acting', color: 'a', children: [
        { label: 'Thought generation', sub: 'CoT reasoning', color: 'y' },
        { label: 'Reinforced self-improvement', sub: 'GRPO', color: 'r' },
      ]},
      { label: 'AR action decoder', sub: 'tokenized output', color: 'o' },
    ],
    output: { label: 'Tokenized actions', sub: '7-DoF with reasoning trace', color: 'e' },
    meta: { loss: 'Cross-entropy + GRPO RL', notes: ['Reasoning-augmented VLA'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #6  pi*0.6  (pi-star 0.6)
  // ═══════════════════════════════════════════════════════════════════════════
  'pi*0.6': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Gemma 3 VLM', sub: '4B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'Gemma 3 4B', sub: 'language model', color: 'i' },
      ]},
      { label: 'VLM → Action expert connector', sub: 'cross-attention projection', color: 'x' },
      { group: 'Flow Matching Action Expert', sub: '~860M', color: 'r', children: [
        { label: 'Action DiT', sub: 'denoising transformer', color: 'r' },
        { label: 'RECAP RL', sub: 'experience + corrections', color: 'o' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'RL-refined, multi-embodiment', color: 'e' },
    meta: { loss: 'Flow matching + RECAP RL', notes: ['~5B total', 'Demos + rollouts + corrections'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #7  DexVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'DexVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Qwen2-VL VLM', sub: '2B', color: 'p', children: [
        { label: 'ViT encoder', sub: 'Qwen2-VL vision', color: 'p' },
        { label: 'Qwen2 1.5B', sub: 'language model', color: 'i' },
      ]},
      { label: 'Coarse language plan', sub: 'hierarchical planning via VLM', color: 'a', arrowLabel: 'plan tokens' },
      { group: 'Diffusion Action Expert', sub: '~1B', color: 'r', children: [
        { label: 'Plan encoder', sub: 'from VLM plan tokens', color: 'a' },
        { label: 'Diffusion DiT', sub: 'fine-grained actions', color: 'r' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'dexterous manipulation', color: 'e' },
    meta: { loss: 'Cross-entropy (plan) + diffusion (actions)', loop: 'K-step denoising', notes: ['~3B total', 'Hierarchical: plan → action'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #8  HybridVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'HybridVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'PrismaticVLM', sub: '7B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'DinoV2', sub: 'vision encoder', color: 'k' },
      ], bottom: { label: 'Llama-2 7B', sub: 'language model', color: 'i' } },
      { group: 'Hybrid action head', sub: 'best of both worlds', color: 'o', children: [
        { label: 'AR branch', sub: 'discrete tokens', color: 'a' },
        { label: 'Diffusion branch', sub: 'continuous flow', color: 'r' },
      ]},
    ],
    output: { label: 'Actions', sub: 'hybrid AR + diffusion', color: 'e' },
    meta: { loss: 'Cross-entropy + diffusion', notes: ['Combines AR precision + diffusion smoothness'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #9  GR00T-N1.6
  // ═══════════════════════════════════════════════════════════════════════════
  'GR00T-N1.6': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Proprioception', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'System 2: Cosmos-Reason-2B VLM', sub: 'flexible resolution, no padding', color: 'p', children: [
        { label: 'SigLIP-2', sub: 'vision encoder', color: 'k' },
        { label: 'T5', sub: 'text encoder', color: 'c' },
      ], bottom: { label: 'LLM layers', sub: 'top-4 layers unfrozen', color: 'i' } },
      { label: 'Modified MLP connector', sub: 'VL token embeddings φ_t', color: 'x', arrowLabel: 'VL tokens' },
      { group: 'System 1: Diffusion Transformer', sub: '32 layers', color: 'r', children: [
        { label: 'Cross-attention', sub: 'K/V from VL tokens', color: 'r' },
        { label: 'Self-attention', sub: 'state ⊕ action tokens', color: 't' },
        { label: 'AdaLN', sub: 'timestep cond.', color: 'a' },
      ], bottom: { label: 'Action decoder MLP', sub: 'per-embodiment, H=16 tokens', color: 'g' } },
    ],
    output: { label: 'State-relative action chunks', sub: 'Δa_t, Δa_{t+1}, …, Δa_{t+15}', color: 'e' },
    meta: { loss: 'Flow matching + world modeling', loop: 'K=4 denoising steps', notes: ['3B params', 'Dual-system architecture'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #10  OpenVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'OpenVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'PrismaticVLM', sub: '7B', color: 'p', children: [
        { label: 'SigLIP', sub: '400M vision encoder', color: 'k' },
        { label: 'DinoV2', sub: 'vision encoder', color: 'k' },
      ], bottom: { label: 'Llama-2 7B', sub: 'language backbone', color: 'i' } },
      { label: 'Action tokenizer', sub: '256 bins per dimension', color: 'a' },
      { label: 'Autoregressive decoder', sub: 'next-token prediction, 7 tokens', color: 'o' },
    ],
    output: { label: 'Tokenized actions', sub: '7-DoF continuous (de-tokenized)', color: 'e' },
    meta: { loss: 'Cross-entropy (next-token)', notes: ['970K trajectories (OXE)', '64 A100s, ~14 days'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #11  OpenVLA-OFT
  // ═══════════════════════════════════════════════════════════════════════════
  'OpenVLA-OFT': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'PrismaticVLM', sub: '7B, LoRA fine-tuned', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'DinoV2', sub: 'vision encoder', color: 'k' },
      ], bottom: { label: 'Llama-2 7B', sub: 'LoRA adapters', color: 'i' } },
      { label: 'Parallel action decoder', sub: 'removes autoregressive bottleneck', color: 'o' },
      { label: 'Continuous action head', sub: 'direct regression, action chunking', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: '7-DoF, chunked', color: 'e' },
    meta: { loss: 'MSE (continuous)', notes: ['8 A100s', 'LoRA fine-tuning'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #12  CogACT
  // ═══════════════════════════════════════════════════════════════════════════
  'CogACT': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'PrismaticVLM', sub: '7B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'DinoV2', sub: 'vision encoder', color: 'k' },
      ], bottom: { label: 'Llama-2 7B', sub: 'language model', color: 'i' } },
      { label: 'VLM feature projection', sub: 'condition tokens for diffusion', color: 'x' },
      { group: 'Diffusion action head', sub: 'continuous generation', color: 'r', children: [
        { label: 'Denoising network', sub: 'conditioned on VLM features', color: 'r' },
        { label: 'Action chunking', sub: 'multi-step prediction', color: 'o' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: '7-DoF, denoised', color: 'e' },
    meta: { loss: 'Diffusion (DDPM)', loop: 'K-step denoising', notes: ['32 A100s', 'Replaces AR with diffusion'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #13  CoT-VLA
  // ═══════════════════════════════════════════════════════════════════════════
  'CoT-VLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'VILA-U VLM', sub: '7B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'LLaMA-2 7B', sub: 'language model', color: 'i' },
      ]},
      { label: 'Chain-of-thought reasoning', sub: 'intermediate language rationale', color: 'a' },
      { label: 'Diffusion policy head', sub: 'conditioned on CoT + VLM features', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'reasoning-guided', color: 'e' },
    meta: { loss: 'Cross-entropy (CoT) + diffusion (action)', notes: ['8 A100s', 'Two-stage training'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #14  DD-VLA
  // ═══════════════════════════════════════════════════════════════════════════
  'DD-VLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'PrismaticVLM', sub: '7B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'DinoV2', sub: 'vision encoder', color: 'k' },
      ], bottom: { label: 'Llama-2 7B', sub: 'language model', color: 'i' } },
      { label: 'Action tokenizer', sub: 'continuous → discrete tokens', color: 'a' },
      { group: 'Discrete diffusion head', sub: 'adaptive decoding order', color: 'r', children: [
        { label: 'Mask-predict', sub: 'parallel denoising', color: 'r' },
        { label: 'Adaptive order', sub: 'learned decode sequence', color: 'o' },
      ]},
    ],
    output: { label: 'Action tokens', sub: 'discrete diffusion → continuous', color: 'e' },
    meta: { loss: 'Discrete diffusion (masked)', notes: ['Adaptive decoding order', 'Parallel generation'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #15  ECoT
  // ═══════════════════════════════════════════════════════════════════════════
  'ECoT': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'PrismaticVLM', sub: '7B (OpenVLA base)', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'DinoV2', sub: 'vision encoder', color: 'k' },
      ], bottom: { label: 'Llama-2 7B', sub: 'language model', color: 'i' } },
      { group: 'Embodied CoT reasoning', sub: 'structured chain', color: 'a', children: [
        { label: 'TASK', sub: 'description', color: 'y' },
        { label: 'PLAN', sub: 'step sequence', color: 'a' },
        { label: 'SUBTASK', sub: 'current step', color: 't' },
        { label: 'MOVE', sub: 'motion plan', color: 'g' },
      ]},
      { label: 'AR action decoder', sub: 'tokenized 7-DoF', color: 'o' },
    ],
    output: { label: 'Tokenized actions', sub: '7-DoF with reasoning trace', color: 'e' },
    meta: { loss: 'Cross-entropy (reasoning + action tokens)', notes: ['TASK→PLAN→SUBTASK→MOVE chain'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #16  TraceVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'TraceVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Visual trace', sub: '2D trajectory overlay', color: 't' },
    ],
    stages: [
      { group: 'PrismaticVLM', sub: '7B (OpenVLA base)', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'DinoV2', sub: 'vision encoder', color: 'k' },
      ], bottom: { label: 'Llama-2 7B', sub: 'language model', color: 'i' } },
      { label: 'Trace-conditioned reasoning', sub: 'visual trace → spatial grounding', color: 't' },
      { label: 'AR action decoder', sub: 'tokenized output', color: 'o' },
    ],
    output: { label: 'Tokenized actions', sub: '7-DoF, trace-guided', color: 'e' },
    meta: { loss: 'Cross-entropy', notes: ['16 A100s', 'Visual trace augmentation'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #17  UniVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'UniVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'PrismaticVLM', sub: '7B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'DinoV2', sub: 'vision encoder', color: 'k' },
      ], bottom: { label: 'Llama-2 7B', sub: 'language model', color: 'i' } },
      { label: 'Unified tokenization', sub: 'vision + language + action in one space', color: 'a' },
      { label: 'Flow matching head', sub: 'continuous denoising', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'flow-matched', color: 'e' },
    meta: { loss: 'Flow matching', notes: ['16 A100s', 'Unified token space'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #18  DreamVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'DreamVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'PrismaticVLM', sub: '7B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'DinoV2', sub: 'vision encoder', color: 'k' },
      ], bottom: { label: 'Llama-2 7B', sub: 'language model', color: 'i' } },
      { group: 'World knowledge forecasts', sub: 'predict future representations', color: 'c', children: [
        { label: 'Depth pred.', sub: 'future depth', color: 'c' },
        { label: 'DINOv2 pred.', sub: 'future features', color: 'k' },
        { label: 'SAM pred.', sub: 'future segments', color: 't' },
      ]},
      { label: 'Inverse dynamics', sub: 'forecast → actions', color: 'o' },
    ],
    output: { label: 'Actions', sub: 'from world-knowledge inverse dynamics', color: 'e' },
    meta: { loss: 'Forecast loss + inverse dynamics', notes: ['Dream-based world model', 'No explicit action labels needed'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #19  MemoryVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'MemoryVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'PrismaticVLM', sub: '7B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'DinoV2', sub: 'vision encoder', color: 'k' },
      ], bottom: { label: 'Llama-2 7B', sub: 'language model', color: 'i' } },
      { label: 'Memory bank', sub: 'history-augmented tokens', color: 'c' },
      { label: 'Diffusion transformer', sub: 'conditioned on memory + VLM features', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'memory-augmented', color: 'e' },
    meta: { loss: 'Diffusion', loop: 'K-step denoising', notes: ['Episodic memory tokens'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #20  pi0
  // ═══════════════════════════════════════════════════════════════════════════
  'pi0': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'PaliGemma VLM', sub: '3B pre-trained', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'Gemma 2B', sub: 'language model', color: 'i' },
      ]},
      { label: 'VLM → action expert projection', sub: 'cross-attention bridge', color: 'x' },
      { group: 'Flow matching action expert', sub: 'denoising DiT', color: 'r', children: [
        { label: 'Action DiT', sub: 'transformer denoiser', color: 'r' },
        { label: 'State encoder', sub: 'proprioception', color: 'g' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'multi-embodiment, flow-matched', color: 'e' },
    meta: { loss: 'Flow matching', loop: 'K-step denoising', notes: ['50 Hz inference', '3B params'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #21  pi0.5
  // ═══════════════════════════════════════════════════════════════════════════
  'pi0.5': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'PaliGemma VLM', sub: '3B, web-scale pre-training', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'Gemma 2B', sub: 'language model', color: 'i' },
      ]},
      { label: 'Long-horizon planner', sub: 'improved reasoning over pi0', color: 'a' },
      { group: 'Flow matching action expert', sub: 'denoising DiT', color: 'r', children: [
        { label: 'Action DiT', sub: 'transformer denoiser', color: 'r' },
        { label: 'State encoder', sub: 'proprioception', color: 'g' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'long-horizon, flow-matched', color: 'e' },
    meta: { loss: 'Flow matching', loop: 'K-step denoising', notes: ['45 Hz inference', '3B+ params'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #22  pi0-FAST
  // ═══════════════════════════════════════════════════════════════════════════
  'pi0-FAST': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
    ],
    stages: [
      { group: 'PaliGemma VLM', sub: '3B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'Gemma 2B', sub: 'language model', color: 'i' },
      ]},
      { group: 'FAST tokenizer', sub: 'DCT → discrete tokens', color: 'a', children: [
        { label: 'DCT transform', sub: 'continuous → frequency domain', color: 'a' },
        { label: 'Quantization', sub: 'frequency → discrete tokens', color: 'o' },
      ]},
      { label: 'AR token decoder', sub: 'next-token prediction', color: 'o' },
    ],
    output: { label: 'Action tokens', sub: 'iDCT → continuous, 230 Hz', color: 'e' },
    meta: { loss: 'Cross-entropy (next-token)', notes: ['230 Hz inference', 'FAST tokenization (DCT)'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #23  GR00T-N1
  // ═══════════════════════════════════════════════════════════════════════════
  'GR00T-N1': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'System 2 — Reasoning (7-9 Hz)', sub: 'Eagle-2 VLM', color: 'p', children: [
        { label: 'SigLIP-2', sub: 'vision encoder', color: 'k' },
        { label: 'SmolLM2', sub: 'language model', color: 'i' },
      ]},
      { label: 'S2 → S1 conditioning', sub: 'latent goal tokens', color: 'x' },
      { group: 'System 1 — Motor (120 Hz)', sub: 'diffusion transformer', color: 'r', children: [
        { label: 'DiT denoiser', sub: 'fast motor policy', color: 'r' },
        { label: 'State encoder', sub: 'proprioception', color: 'g' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'dual-system, 120 Hz motor', color: 'e' },
    meta: { loss: 'Diffusion (S1) + VLM (S2)', loop: 'K-step denoising', notes: ['2.2B params', 'Humanoid-optimized'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #24  GR00T-N1.5
  // ═══════════════════════════════════════════════════════════════════════════
  'GR00T-N1.5': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Eagle 2.5 VLM', sub: 'reasoning backbone', color: 'p', children: [
        { label: 'Vision encoder', sub: 'Eagle 2.5', color: 'k' },
        { label: 'Language model', sub: 'Eagle 2.5', color: 'i' },
      ]},
      { label: 'FLARE objective', sub: 'future latent representation alignment', color: 'c' },
      { group: 'Diffusion transformer', sub: 'motor policy', color: 'r', children: [
        { label: 'DiT denoiser', sub: 'FLARE-conditioned', color: 'r' },
        { label: 'State encoder', sub: 'proprioception', color: 'g' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'FLARE-guided', color: 'e' },
    meta: { loss: 'Diffusion + FLARE', notes: ['3B params', '1K H100s, 36h training', 'Learns from human video'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #25  GR00T-N1.7
  // ═══════════════════════════════════════════════════════════════════════════
  'GR00T-N1.7': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Reasoning VLM', sub: '~3B, production-grade', color: 'p', children: [
        { label: 'Vision encoder', sub: 'reasoning-capable', color: 'k' },
        { label: 'Language model', sub: 'reasoning VLM', color: 'i' },
      ]},
      { label: 'Reasoning module', sub: 'first reasoning VLA for deployment', color: 'a' },
      { label: 'Diffusion transformer', sub: 'motor policy', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'reasoning-guided', color: 'e' },
    meta: { loss: 'Diffusion + reasoning', notes: ['~3B params', 'Commercial license', 'NIM microservices'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #26  GR00T-N2
  // ═══════════════════════════════════════════════════════════════════════════
  'GR00T-N2': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
    ],
    stages: [
      { group: 'World Action Model', sub: 'non-VLM architecture', color: 'c', children: [
        { label: 'World model encoder', sub: 'state + dynamics', color: 'c' },
        { label: 'Action planner', sub: 'DreamZero-based', color: 'r' },
      ]},
      { label: 'Imagination rollouts', sub: 'plan via simulated futures', color: 't' },
      { label: 'Action decoder', sub: 'planned trajectory → motor', color: 'o' },
    ],
    output: { label: 'Continuous actions', sub: 'world-model planned', color: 'e' },
    meta: { loss: 'World model + planning', notes: ['Non-VLM architecture', '2x+ success vs leading VLAs', 'Preview (late 2026)'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #27  Octo
  // ═══════════════════════════════════════════════════════════════════════════
  'Octo': {
    inputs: [
      { label: 'RGB frames', sub: 'variable cameras', color: 'b' },
      { label: 'Language instr.', sub: 'optional', color: 'b' },
      { label: 'Robot state', sub: 'variable dims', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Modular encoder', sub: 'variable obs/action spaces', color: 'p', children: [
        { label: 'Image tokenizer', sub: 'ViT patches', color: 'k' },
        { label: 'Language tokenizer', sub: 'T5 embeddings', color: 'i' },
        { label: 'State tokenizer', sub: 'proprioception MLP', color: 'g' },
      ]},
      { label: 'Transformer backbone', sub: '93M params, encoder-decoder', color: 'x' },
      { label: 'Diffusion action head', sub: 'continuous denoising', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'variable dims per embodiment', color: 'e' },
    meta: { loss: 'Diffusion (DDPM)', loop: 'K-step denoising', notes: ['93M params', '800K+ OXE trajectories'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #28  RDT-1B
  // ═══════════════════════════════════════════════════════════════════════════
  'RDT-1B': {
    inputs: [
      { label: 'RGB frames', sub: 'multi-view', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { label: 'SigLIP-So400m', sub: 'vision encoder (frozen)', color: 'k' },
      { label: 'T5-XXL', sub: 'language encoder (frozen)', color: 'i' },
      { group: 'Scalable diffusion transformer', sub: '1.2B params', color: 'r', children: [
        { label: 'DiT backbone', sub: 'largest robotic diffusion model', color: 'r' },
        { label: 'State encoder', sub: 'bimanual proprioception', color: 'g' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'bimanual, multi-embodiment', color: 'e' },
    meta: { loss: 'Diffusion', loop: 'K-step denoising', notes: ['1.2B params', '64 A100s', 'Bimanual support'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #29  Diffusion Policy
  // ═══════════════════════════════════════════════════════════════════════════
  'Diffusion Policy': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { label: 'ResNet-18', sub: 'visual feature extractor', color: 'k' },
      { group: 'Diffusion denoiser', sub: 'DDPM, ~25M params', color: 'r', children: [
        { label: 'U-Net / Transformer', sub: 'noise prediction network', color: 'r' },
        { label: 'FiLM conditioning', sub: 'obs features → denoiser', color: 'x' },
      ]},
      { label: 'Action chunking', sub: 'multi-step prediction', color: 'o' },
    ],
    output: { label: 'Continuous actions', sub: 'multi-step, multimodal dist.', color: 'e' },
    meta: { loss: 'Diffusion (DDPM)', loop: 'K-step denoising', notes: ['~25M params', 'Task-specific', 'Foundational work (2023)'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #30  CrossFormer
  // ═══════════════════════════════════════════════════════════════════════════
  'CrossFormer': {
    inputs: [
      { label: 'RGB frames', sub: 'multi-view', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { label: 'ResNet encoder', sub: 'per-view image features', color: 'k' },
      { label: 'Language encoder', sub: 'instruction embeddings', color: 'i' },
      { group: 'Cross-embodiment transformer', sub: 'shared backbone', color: 'p', children: [
        { label: 'Cross-attention layers', sub: 'multi-view + language fusion', color: 'x' },
        { label: 'Embodiment tokens', sub: 'robot-specific adapters', color: 'g' },
      ]},
      { label: 'Diffusion action head', sub: 'continuous denoising', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'cross-embodiment', color: 'e' },
    meta: { loss: 'Diffusion', loop: 'K-step denoising', notes: ['Cross-embodiment generalization', 'Multi-view fusion'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #31  SpatialVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'SpatialVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Depth maps', sub: 'spatial priors', color: 't' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'PaliGemma2 VLM', sub: '4B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'Gemma 2', sub: 'language model', color: 'i' },
      ]},
      { label: '3D spatial tokenizer', sub: 'depth → adaptive spatial tokens', color: 't' },
      { label: 'AR action decoder', sub: 'tokenized 7-DoF', color: 'o' },
    ],
    output: { label: 'Tokenized actions', sub: '7-DoF, spatially-grounded', color: 'e' },
    meta: { loss: 'Cross-entropy (next-token)', notes: ['4B params', '3D spatial tokenization'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #32  SmolVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'SmolVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'SmolVLM2', sub: '450M total', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'SmolLM2', sub: 'language model', color: 'i' },
      ]},
      { label: 'Flow matching head', sub: 'continuous denoising', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'flow-matched', color: 'e' },
    meta: { loss: 'Flow matching', notes: ['450M params', 'Single GPU trainable', 'Competitive with 10x larger'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #33  TinyVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'TinyVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Compact VLM', sub: '422M–1.3B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'Phi-2', sub: '2.7B language model', color: 'i' },
      ]},
      { label: 'LoRA adapters', sub: 'parameter-efficient fine-tuning', color: 'x' },
      { label: 'Diffusion action head', sub: 'continuous denoising', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'lightweight diffusion', color: 'e' },
    meta: { loss: 'Diffusion', notes: ['422M–1.3B params', 'LoRA-based training'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #34  NanoVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'NanoVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { label: 'ResNet-18', sub: 'lightweight vision encoder', color: 'k' },
      { label: 'BERT-base / Qwen2.5-0.5B', sub: 'language encoder', color: 'i' },
      { label: 'Routing-decoupled fusion', sub: 'separate vision-language pathways', color: 'x' },
      { label: 'Efficient action head', sub: 'edge-optimized', color: 'o' },
    ],
    output: { label: 'Continuous actions', sub: 'edge-device, ~50M params', color: 'e' },
    meta: { loss: 'Action prediction', notes: ['~50M params', '52x faster on edge', '98% fewer params vs SOTA'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #35  SAM2Act
  // ═══════════════════════════════════════════════════════════════════════════
  'SAM2Act': {
    inputs: [
      { label: 'Multi-view RGB', sub: 'multiple cameras', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { label: 'SAM2 encoder', sub: 'Segment Anything 2', color: 'k' },
      { label: 'SAM2 memory bank', sub: 'temporal tracking across frames', color: 'c' },
      { group: 'RVT-2 multi-view transformer', sub: 'coarse-to-fine', color: 'p', children: [
        { label: 'Coarse heatmap', sub: 'approximate location', color: 'r' },
        { label: 'Fine heatmap', sub: 'precise action point', color: 'r' },
      ]},
    ],
    output: { label: 'Action keypoints', sub: 'coarse-to-fine heatmap', color: 'e' },
    meta: { loss: 'Heatmap + action regression', notes: ['SAM2 memory for temporal reasoning', 'SOTA RLBench 86.8%'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #36  RoboDual
  // ═══════════════════════════════════════════════════════════════════════════
  'RoboDual': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Generalist: OpenVLA 7B', sub: 'high-level reasoning', color: 'p', children: [
        { label: 'SigLIP + DinoV2', sub: 'vision encoders', color: 'k' },
        { label: 'Llama-2 7B', sub: 'language backbone', color: 'i' },
      ]},
      { label: 'Cross-attention fusion', sub: 'generalist → specialist bridge', color: 'x' },
      { label: 'Specialist: DiT 20M', sub: 'diffusion transformer, low-level motor', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'dual-system fused', color: 'e' },
    meta: { loss: 'Diffusion (specialist) + CE (generalist)', notes: ['7B + 20M dual system', 'Cross-attention fusion'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #37  RT-2-X
  // ═══════════════════════════════════════════════════════════════════════════
  'RT-2-X': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'PaLI-X VLM', sub: '55B', color: 'p', children: [
        { label: 'ViT-22B', sub: 'vision encoder', color: 'k' },
        { label: 'PaLM-E', sub: 'language model', color: 'i' },
      ]},
      { label: 'Web-scale co-training', sub: 'internet + robot data', color: 'c' },
      { label: 'AR action decoder', sub: 'tokenized actions', color: 'o' },
    ],
    output: { label: 'Tokenized actions', sub: 'discretized 7-DoF', color: 'e' },
    meta: { loss: 'Cross-entropy (next-token)', notes: ['55B params', 'First large-scale cross-embodiment VLA'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #38  HPT
  // ═══════════════════════════════════════════════════════════════════════════
  'HPT': {
    inputs: [
      { label: 'RGB frames', sub: 'per-embodiment view', color: 'b' },
      { label: 'Robot state', color: 'g' },
    ],
    stages: [
      { label: 'Heterogeneous stem encoders', sub: 'per-embodiment specialized', color: 'k' },
      { group: 'Shared transformer trunk', sub: '300M params', color: 'p', children: [
        { label: 'Universal representation', sub: 'cross-embodiment features', color: 'x' },
      ]},
      { label: 'Task-specific heads', sub: 'per-embodiment action decoders', color: 'o' },
    ],
    output: { label: 'Continuous actions', sub: 'per-embodiment output', color: 'e' },
    meta: { loss: 'Action prediction', notes: ['300M params', '52 diverse datasets', 'Heterogeneous pre-training'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #39  FALCON
  // ═══════════════════════════════════════════════════════════════════════════
  'FALCON': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Spatial annotations', sub: 'grounding priors', color: 't' },
    ],
    stages: [
      { group: 'Kosmos-2 VLM', sub: '1.6B', color: 'p', children: [
        { label: 'Vision encoder', sub: 'Kosmos-2', color: 'k' },
        { label: 'Language model', sub: 'Kosmos-2', color: 'i' },
      ]},
      { label: 'Embodied Spatial Model', sub: '1.0B, spatial grounding', color: 't' },
      { label: 'Spatial-grounded action head', sub: 'location-aware actions', color: 'o' },
    ],
    output: { label: 'Continuous actions', sub: 'spatially-grounded', color: 'e' },
    meta: { loss: 'Spatial grounding + action', notes: ['2.9B total (1.6B + 1.0B)', 'Decoupled spatial understanding'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #40  FLARE
  // ═══════════════════════════════════════════════════════════════════════════
  'FLARE': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { label: 'SigLIP2-Large', sub: 'vision encoder (frozen)', color: 'k' },
      { label: 'Q-Former', sub: 'visual feature compression', color: 'x' },
      { label: 'Future latent alignment', sub: 'implicit world modeling', color: 'c' },
      { group: 'Diffusion Transformer', sub: '3B params', color: 'r', children: [
        { label: 'DiT backbone', sub: 'FLARE-conditioned denoiser', color: 'r' },
        { label: 'State encoder', sub: 'proprioception', color: 'g' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'future-aligned', color: 'e' },
    meta: { loss: 'Diffusion + future latent alignment', loop: 'K-step denoising', notes: ['3B params', 'Implicit world model'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #41  3D-VLA
  // ═══════════════════════════════════════════════════════════════════════════
  '3D-VLA': {
    inputs: [
      { label: '3D point clouds', sub: 'multi-view', color: 't' },
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { label: 'Q-Former', sub: '3D multi-view features', color: 'k' },
      { group: 'BLIP2 + FlanT5', sub: '3B–11B', color: 'p', children: [
        { label: '3D-LLM backbone', sub: 'point-cloud aligned', color: 'i' },
      ]},
      { label: '3D world model', sub: 'goal generation', color: 'c' },
      { label: 'Embodied diffusion', sub: 'action denoising', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: '3D-grounded', color: 'e' },
    meta: { loss: 'Diffusion + 3D alignment', loop: 'K-step denoising', notes: ['3B–11B params', '3D point-cloud world model'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #42  3D Diffuser Actor
  // ═══════════════════════════════════════════════════════════════════════════
  '3D Diffuser Actor': {
    inputs: [
      { label: '3D scene point cloud', color: 't' },
      { label: 'Language instr.', sub: 'CLIP embeddings', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { label: '3D scene encoder', sub: 'point cloud features', color: 'k' },
      { label: 'CLIP language encoder', sub: 'task conditioning', color: 'i' },
      { label: '3D diffusion denoiser', sub: 'SE(3) pose denoising', color: 'r' },
    ],
    output: { label: 'SE(3) poses', sub: '3D action trajectories', color: 'e' },
    meta: { loss: 'Diffusion (3D)', loop: 'K-step denoising', notes: ['~50M params', 'Diffusion over SE(3)'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #43  AtomicVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'AtomicVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'PaliGemma VLM', sub: '4.17B total', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'Gemma 2B', sub: 'language model', color: 'i' },
      ]},
      { label: 'Flexible routing encoder', sub: 'atomic skill decomposition', color: 'x' },
      { label: 'SG-MoE action head', sub: 'Skill-Guided Mixture of Experts', color: 'o' },
    ],
    output: { label: 'Continuous actions', sub: 'skill-routed', color: 'e' },
    meta: { loss: 'Skill-guided MoE', notes: ['4.17B params', 'Continual skill acquisition'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #44  AVDC
  // ═══════════════════════════════════════════════════════════════════════════
  'AVDC': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { label: 'Video diffusion model', sub: 'future frame generation', color: 'r' },
      { label: 'Optical flow extraction', sub: 'motion fields', color: 'c' },
      { label: 'Inverse dynamics', sub: 'flow → actions', color: 'o' },
    ],
    output: { label: 'Continuous actions', sub: 'flow-derived', color: 'e' },
    meta: { loss: 'Video diffusion + inverse dynamics', notes: ['No explicit LLM', 'Video as intermediate representation'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #45  DeeR-VLA
  // ═══════════════════════════════════════════════════════════════════════════
  'DeeR-VLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'OpenFlamingo VLM', sub: '3B', color: 'p', children: [
        { label: 'Vision encoder', sub: 'OpenFlamingo', color: 'k' },
        { label: 'MPT-1B', sub: 'language model', color: 'i' },
      ]},
      { label: 'Multi-exit layers', sub: 'early stopping via action consistency', color: 'x' },
      { label: 'Action head', sub: 'adaptive compute depth', color: 'o' },
    ],
    output: { label: 'Continuous actions', sub: 'early-exit optimized', color: 'e' },
    meta: { loss: 'Action prediction + consistency', notes: ['3B params', '5.2–6.5x compute reduction'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #46  Diffusion-VLA
  // ═══════════════════════════════════════════════════════════════════════════
  'Diffusion-VLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Qwen2 VLM', sub: '2B–72B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'Qwen2', sub: 'language + reasoning', color: 'i' },
      ]},
      { label: 'AR reasoning tokens', sub: 'chain-of-thought injection', color: 'c' },
      { label: 'Diffusion action head', sub: 'reasoning-injected denoising', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: '82 Hz inference', color: 'e' },
    meta: { loss: 'CE (reasoning) + diffusion (actions)', loop: 'K-step denoising', notes: ['2B–72B variants', '82 Hz fast inference'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #47  dVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'dVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { label: 'MAGViT-v2', sub: 'vision tokenizer', color: 'k' },
      { group: 'LLaDA-8B-Instruct', sub: 'discrete diffusion LM', color: 'p', children: [
        { label: 'Multimodal CoT', sub: 'chain-of-thought reasoning', color: 'c' },
      ]},
      { label: 'Discrete diffusion head', sub: 'token-level denoising', color: 'r' },
    ],
    output: { label: 'Action tokens', sub: 'discrete diffusion', color: 'e' },
    meta: { loss: 'Discrete diffusion + CoT', notes: ['8B params', 'Multimodal chain-of-thought'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #48  E0
  // ═══════════════════════════════════════════════════════════════════════════
  'E0': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'PaliGemma VLM', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'Gemma', sub: 'language model', color: 'i' },
      ]},
      { label: 'Action quantizer', sub: 'continuous → discrete tokens', color: 'x' },
      { label: 'Continuized discrete diffusion', sub: 'hybrid denoising', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'de-quantized', color: 'e' },
    meta: { loss: 'Continuized discrete diffusion', notes: ['Bridges discrete & continuous diffusion'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #49  FLOWER
  // ═══════════════════════════════════════════════════════════════════════════
  'FLOWER': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { label: 'Florence-2-Large', sub: 'vision encoder (encoder only)', color: 'k' },
      { label: 'Flow matching head', sub: 'continuous action denoising', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'flow-matched', color: 'e' },
    meta: { loss: 'Flow matching', notes: ['950M params', 'Sub-1B generalist', 'No LLM backbone'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #50  Gemini Robotics
  // ═══════════════════════════════════════════════════════════════════════════
  'Gemini Robotics': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
    ],
    stages: [
      { group: 'Gemini 2.0 VLM', sub: 'foundation model', color: 'p', children: [
        { label: 'Vision encoder', sub: 'Gemini multimodal', color: 'k' },
        { label: 'Language model', sub: 'Gemini 2.0', color: 'i' },
      ]},
      { label: 'Embodied reasoning', sub: 'spatial + temporal understanding', color: 'c' },
      { label: 'End-to-end action head', sub: 'VLA output', color: 'o' },
    ],
    output: { label: 'Continuous actions', sub: 'end-to-end VLA', color: 'e' },
    meta: { loss: 'End-to-end VLA', notes: ['Gemini 2.0 scale', 'Proprietary', 'On-device variants'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #51  GR-Dexter
  // ═══════════════════════════════════════════════════════════════════════════
  'GR-Dexter': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Qwen2.5-VL-3B', sub: 'VLM backbone', color: 'p', children: [
        { label: 'Vision encoder', sub: 'Qwen2.5-VL', color: 'k' },
        { label: 'Qwen2.5 3B', sub: 'language model', color: 'i' },
      ]},
      { label: 'Flow-matching DiT', sub: '56-DoF bimanual dexterous', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: '56-DoF bimanual hands', color: 'e' },
    meta: { loss: 'Flow matching', notes: ['56-DoF dexterous hands', 'ByteDexter V2'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #52  GR-1
  // ═══════════════════════════════════════════════════════════════════════════
  'GR-1': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'GPT-style transformer', sub: '130M', color: 'p', children: [
        { label: 'Vision tokenizer', sub: 'frame encoding', color: 'k' },
        { label: 'Language tokens', sub: 'task conditioning', color: 'i' },
      ]},
      { label: 'Video prediction', sub: 'auxiliary future frames', color: 'c' },
      { label: 'AR action decoder', sub: 'next-token prediction', color: 'o' },
    ],
    output: { label: 'Actions + video', sub: 'unified generation', color: 'e' },
    meta: { loss: 'Cross-entropy (video + action)', notes: ['130M params', 'Video as auxiliary objective'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #53  GR-2
  // ═══════════════════════════════════════════════════════════════════════════
  'GR-2': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'Video GPT transformer', sub: '1.5B', color: 'p', children: [
        { label: 'Vision tokenizer', sub: 'video encoding', color: 'k' },
        { label: 'Language tokens', sub: 'task conditioning', color: 'i' },
      ]},
      { label: 'World model', sub: '38M video pre-training', color: 'c' },
      { label: 'AR action decoder', sub: 'video prediction + action', color: 'o' },
    ],
    output: { label: 'Actions + video', sub: 'world-model grounded', color: 'e' },
    meta: { loss: 'Cross-entropy (video + action)', notes: ['1.5B params', '38M internet videos pre-training'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #54  GST-VLA
  // ═══════════════════════════════════════════════════════════════════════════
  'GST-VLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Depth maps', color: 't' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'PrismaticVLM backbone', sub: 'OpenVLA-based', color: 'p', children: [
        { label: 'SigLIP + DinoV2', sub: 'vision encoders', color: 'k' },
        { label: 'Llama-2 7B', sub: 'language model', color: 'i' },
      ]},
      { label: 'Gaussian Spatial Tokenizer', sub: 'depth + semantic → 3D Gaussian primitives', color: 't' },
      { label: '3D Depth-Aware CoT', sub: 'spatial reasoning', color: 'c' },
      { label: 'Flow matching action expert', sub: '300M denoiser', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: '3D-aware flow-matched', color: 'e' },
    meta: { loss: 'Flow matching + CoT', notes: ['300M action expert', '3D Gaussian spatial tokens'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #55  HiMoE-VLA
  // ═══════════════════════════════════════════════════════════════════════════
  'HiMoE-VLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'PaliGemma VLM', sub: '3B+', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'Gemma 2B', sub: 'language model', color: 'i' },
      ]},
      { group: 'Hierarchical MoE', sub: 'multi-level expert routing', color: 'x', children: [
        { label: 'Task-level experts', sub: 'high-level routing', color: 'x' },
        { label: 'Action-Space MoE', sub: 'boundary layer experts', color: 'o' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'MoE-routed', color: 'e' },
    meta: { loss: 'MoE routing + action prediction', notes: ['3B+ params', 'Hierarchical MoE', 'pi0-based'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #56  Humanoid-VLA
  // ═══════════════════════════════════════════════════════════════════════════
  'Humanoid-VLA': {
    inputs: [
      { label: 'Egocentric video', sub: 'first-person view', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { label: 'CLIP ViT', sub: 'egocentric video encoder', color: 'k' },
      { label: 'Language-motion pre-alignment', sub: 'LLM + motion tokens', color: 'i' },
      { label: 'Egocentric fine-tuning', sub: 'video-conditioned adaptation', color: 'x' },
      { label: 'Whole-body action head', sub: 'humanoid control', color: 'o' },
    ],
    output: { label: 'Whole-body actions', sub: 'humanoid joints', color: 'e' },
    meta: { loss: 'Language-motion alignment', notes: ['Humanoid embodiment', 'Egocentric video'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #57  InstructVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'InstructVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'InternVL2 VLM', sub: '4B', color: 'p', children: [
        { label: 'Vision encoder', sub: 'InternVL2', color: 'k' },
        { label: 'InternLM2', sub: 'language model', color: 'i' },
      ]},
      { label: 'Instruction-following formulation', sub: 'preserves VLM knowledge', color: 'c' },
      { label: 'Flow matching head', sub: 'continuous actions', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'instruction-aligned', color: 'e' },
    meta: { loss: 'Flow matching + instruction', notes: ['4B params', '650K VLA-IT annotations'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #58  InternVLA-A1
  // ═══════════════════════════════════════════════════════════════════════════
  'InternVLA-A1': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'InternVL3 / Qwen3-VL', sub: '2B–3B', color: 'p', children: [
        { label: 'Vision encoder', sub: 'InternVL3 / Qwen3-VL', color: 'k' },
        { label: 'Qwen2.5 / Qwen3', sub: 'language model', color: 'i' },
      ]},
      { group: 'Mixture-of-Transformers', sub: 'UGA experts', color: 'x', children: [
        { label: 'Understanding expert', sub: 'perception', color: 'x' },
        { label: 'Generation expert', sub: 'planning', color: 'c' },
        { label: 'Action expert', sub: 'motor control', color: 'o' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'MoT-unified', color: 'e' },
    meta: { loss: 'MoT multi-expert', notes: ['2B–3B params', 'Unified understanding-generation-action'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #59  InternVLA-M1
  // ═══════════════════════════════════════════════════════════════════════════
  'InternVLA-M1': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Qwen2.5-VL', sub: 'VLM backbone', color: 'p', children: [
        { label: 'Vision encoder', sub: 'Qwen2.5-VL', color: 'k' },
        { label: 'Qwen2.5', sub: 'language + spatial planner', color: 'i' },
      ]},
      { label: 'Spatial grounding', sub: 'dual-system planning', color: 'c' },
      { label: 'DiT action head', sub: 'diffusion transformer', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'spatially grounded', color: 'e' },
    meta: { loss: 'Diffusion + spatial planning', notes: ['Qwen2.5-VL based', 'SOTA SimplerEnv'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #60  LingBot-VLA
  // ═══════════════════════════════════════════════════════════════════════════
  'LingBot-VLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Qwen2.5-VL-3B', sub: 'VLM backbone', color: 'p', children: [
        { label: 'Vision encoder', sub: 'Qwen2.5-VL', color: 'k' },
        { label: 'Qwen2.5 3B', sub: 'language model', color: 'i' },
      ]},
      { label: 'LingBot-Depth distillation', sub: 'depth feature injection', color: 't' },
      { label: 'MoT + flow matching', sub: 'Mixture of Transformers action expert', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'multi-embodiment', color: 'e' },
    meta: { loss: 'Flow matching + depth distillation', notes: ['~4B params', '20K hours real-world data', '9 robot platforms'] },
  },

}
