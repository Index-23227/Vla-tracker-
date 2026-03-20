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

}
