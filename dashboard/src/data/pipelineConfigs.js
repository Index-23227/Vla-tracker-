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
      { label: 'Robot state', color: 'g' },
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
      { label: 'Robot state', color: 'g' },
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
      { label: 'Robot state', color: 'g' },
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
    ],
    stages: [
      { group: 'VILA-U VLM', sub: '7B', color: 'p', children: [
        { label: 'VILA-U vision tower', sub: 'discrete visual tokens', color: 'k' },
        { label: 'LLaMA-2 7B', sub: 'language model', color: 'i' },
      ]},
      { label: 'Visual CoT reasoning', sub: 'generate subgoal image (causal attention)', color: 'a' },
      { label: 'Action chunking', sub: 'full-attention parallel decode', color: 'o' },
    ],
    output: { label: 'Continuous actions', sub: 'subgoal-guided, chunked', color: 'e' },
    meta: { loss: 'Cross-entropy (next-token)', notes: ['8 A100s', 'Hybrid attention: causal (CoT) + full (action)'] },
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
    ],
    stages: [
      { group: 'PrismaticVLM', sub: '7B', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'DinoV2', sub: 'vision encoder', color: 'k' },
      ], bottom: { label: 'Llama-2 7B', sub: 'language model', color: 'i' } },
      { label: 'VQ-VAE latent action tokenizer', sub: 'task-centric latent actions from video', color: 'a' },
      { label: 'AR latent action decoder', sub: 'next-token prediction over ACT tokens', color: 'o' },
    ],
    output: { label: 'Latent action tokens', sub: 'decoded via embodiment-specific head (12M)', color: 'e' },
    meta: { loss: 'Cross-entropy (next-token)', notes: ['1/20 compute vs OpenVLA', 'Task-centric latent actions from video'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #18  DreamVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'DreamVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Modality-specific encoders', color: 'p', children: [
        { label: 'MAE ViT-B', sub: 'vision encoder (spatiotemporal)', color: 'k' },
        { label: 'CLIP text', sub: 'language encoder', color: 'i' },
        { label: 'Conv + FC', sub: 'proprioceptive encoder', color: 'g' },
      ], bottom: { label: 'GPT-2', sub: 'central LLM backbone', color: 'i' } },
      { group: 'World knowledge forecasts', sub: '<dream> queries', color: 'c', children: [
        { label: 'Dynamic pred.', sub: 'future dynamics', color: 'c' },
        { label: 'Depth pred.', sub: 'future depth', color: 't' },
        { label: 'Semantic pred.', sub: 'future segments', color: 'k' },
      ]},
      { label: 'Diffusion action head', sub: 'disentangled action transformer', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'world-knowledge guided', color: 'e' },
    meta: { loss: 'World knowledge forecast + diffusion', notes: ['CLIP + MAE + GPT-2 backbone', 'Block-wise structured attention'] },
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
      { label: 'Robot state', color: 'g' },
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
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'SmolVLM2', sub: '~350M (frozen, first 16 layers)', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder (64 tokens/frame)', color: 'k' },
        { label: 'SmolLM2', sub: 'language decoder', color: 'i' },
      ]},
      { group: 'Flow Matching Action Expert', sub: '~100M, interleaved attention', color: 'r', children: [
        { label: 'Cross-attention', sub: 'attend to VLM features', color: 'r' },
        { label: 'Causal self-attention', sub: 'temporal action coherence', color: 'o' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'flow-matched action chunks', color: 'e' },
    meta: { loss: 'Flow matching', notes: ['450M total', 'Single GPU trainable', 'Interleaved CA + SA'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #33  TinyVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'TinyVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
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
      { label: 'Robot state', color: 'g' },
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
      { label: 'Robot state', color: 'g' },
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
      { group: 'Florence-2-Large (encoder only)', sub: 'decoder removed for efficiency', color: 'p', children: [
        { label: 'DaViT', sub: 'vision encoder', color: 'k' },
        { label: 'Encoder LLM layers', sub: 'multimodal features', color: 'i' },
      ]},
      { group: 'Flow Transformer (DiT)', sub: '18 layers, 1024 dim', color: 'r', children: [
        { label: 'Cross-attention', sub: 'from VLM intermediate features', color: 'r' },
        { label: 'AdaLN-Zero', sub: 'timestep + embodiment cond.', color: 'a' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'rectified flow, 4 denoise steps', color: 'e' },
    meta: { loss: 'Rectified flow matching', notes: ['947M params', '200 GPU-hours pretrain', '<8GB VRAM inference'] },
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
      { label: 'Robot state', color: 'g' },
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
      { label: 'Robot state', color: 'g' },
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
      { label: 'Robot state', color: 'g' },
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

  // ═══════════════════════════════════════════════════════════════════════════
  // #61  LLARVA
  // ═══════════════════════════════════════════════════════════════════════════
  'LLARVA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'LLaVA backbone', sub: '7B', color: 'p', children: [
        { label: 'CLIP ViT-L', sub: 'vision encoder', color: 'k' },
        { label: 'Llama-2 7B', sub: 'language model', color: 'i' },
      ]},
      { label: '2D visual trace prediction', sub: 'trajectory waypoints', color: 'c' },
      { label: 'AR action decoder', sub: 'text-based actions', color: 'o' },
    ],
    output: { label: 'Actions + traces', sub: 'text + 2D trajectories', color: 'e' },
    meta: { loss: 'Cross-entropy', notes: ['7B params', 'Visual trace prediction'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #62  Mobility VLA
  // ═══════════════════════════════════════════════════════════════════════════
  'Mobility VLA': {
    inputs: [
      { label: 'RGB frames', sub: 'navigation view', color: 'b' },
      { label: 'Language goal', color: 'b' },
    ],
    stages: [
      { group: 'Gemini 1.5 Pro', sub: 'long-context VLM', color: 'p', children: [
        { label: 'Vision encoder', sub: 'Gemini multimodal', color: 'k' },
        { label: 'Language model', sub: 'Gemini 1.5 Pro', color: 'i' },
      ]},
      { label: 'Goal identification', sub: 'long-context reasoning', color: 'c' },
      { label: 'Topological graph nav', sub: 'path planning', color: 'o' },
    ],
    output: { label: 'Navigation actions', sub: 'hierarchical waypoints', color: 'e' },
    meta: { loss: 'Hierarchical planning', notes: ['Gemini 1.5 Pro', 'Navigation specialized'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #63  NaVILA
  // ═══════════════════════════════════════════════════════════════════════════
  'NaVILA': {
    inputs: [
      { label: 'RGB frames', sub: 'legged robot view', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'NVILA-8B VLM', sub: 'high-level VLA', color: 'p', children: [
        { label: 'SigLIP', sub: 'vision encoder', color: 'k' },
        { label: 'LLaMA 3 8B', sub: 'language model', color: 'i' },
      ]},
      { label: 'High-level policy', sub: 'VLA goal commands', color: 'c' },
      { label: 'Low-level RL controller', sub: 'visual locomotion', color: 'o' },
    ],
    output: { label: 'Locomotion actions', sub: 'hierarchical VLA + RL', color: 'e' },
    meta: { loss: 'VLA + RL', notes: ['8B params', 'Legged robot navigation'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #64  NORA
  // ═══════════════════════════════════════════════════════════════════════════
  'NORA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'Qwen2.5-VL', sub: '3B', color: 'p', children: [
        { label: 'Vision encoder', sub: 'Qwen2.5-VL', color: 'k' },
        { label: 'Qwen2.5 3B', sub: 'language model', color: 'i' },
      ]},
      { label: 'FAST+ tokenizer', sub: 'action tokenization', color: 'o' },
    ],
    output: { label: 'Tokenized actions', sub: 'FAST+ encoded', color: 'e' },
    meta: { loss: 'Cross-entropy (next-token)', notes: ['3B params', '970K demos', 'FAST+ action tokens'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #65  RoboMamba
  // ═══════════════════════════════════════════════════════════════════════════
  'RoboMamba': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { label: 'CLIP ViT-L/14', sub: 'vision encoder', color: 'k' },
      { label: 'Mamba 2.7B', sub: 'state-space model (not transformer)', color: 'i' },
      { label: 'MLP policy head', sub: '3.7M, SE(3) pose', color: 'o' },
    ],
    output: { label: 'SE(3) poses', sub: 'lightweight MLP output', color: 'e' },
    meta: { loss: 'Action regression', notes: ['3.2B total', 'SSM instead of transformer', '3.7M action head'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #66  RoboVLM
  // ═══════════════════════════════════════════════════════════════════════════
  'RoboVLM': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'KosMos VLM', sub: '2B', color: 'p', children: [
        { label: 'Vision encoder', sub: 'KosMos', color: 'k' },
        { label: 'KosMos 2B', sub: 'language model', color: 'i' },
      ]},
      { label: 'VQ-VAE latent tokenizer', sub: 'actions → visual tokens', color: 'x' },
      { label: 'AR action decoder', sub: 'latent token prediction', color: 'o' },
    ],
    output: { label: 'Continuous actions', sub: 'VQ-VAE decoded', color: 'e' },
    meta: { loss: 'Cross-entropy + VQ-VAE', notes: ['2B params', 'Latent action tokenization'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #67  SuSIE
  // ═══════════════════════════════════════════════════════════════════════════
  'SuSIE': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { label: 'InstructPix2Pix', sub: 'high-level subgoal image generator', color: 'c' },
      { label: 'Subgoal image', sub: 'visual plan', color: 'b' },
      { label: 'Low-level diffusion policy', sub: 'reach subgoal', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'subgoal-conditioned', color: 'e' },
    meta: { loss: 'Diffusion (two-level)', loop: 'K-step denoising', notes: ['No explicit LLM', 'Hierarchical image subgoals'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #68  TwinBrainVLA
  // ═══════════════════════════════════════════════════════════════════════════
  'TwinBrainVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Qwen2.5-VL / Qwen3-VL', sub: '3B–4B', color: 'p', children: [
        { label: 'Vision encoder', sub: 'Qwen VL', color: 'k' },
        { label: 'Qwen LM', sub: 'language backbone', color: 'i' },
      ]},
      { group: 'Asymmetric MoT', sub: 'dual-brain', color: 'x', children: [
        { label: 'Left Brain (frozen)', sub: 'generalist knowledge', color: 'c' },
        { label: 'Right Brain (trainable)', sub: 'specialist motor skills', color: 'o' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'dual-brain fused', color: 'e' },
    meta: { loss: 'MoT + anti-forgetting', notes: ['3B–4B params', 'Frozen generalist + trainable specialist'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #69  UD-VLA
  // ═══════════════════════════════════════════════════════════════════════════
  'UD-VLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'Emu3 VLM', sub: '3B', color: 'p', children: [
        { label: 'MOVQ tokenizer', sub: 'visual tokenization', color: 'k' },
        { label: 'Emu3 LM', sub: 'language model', color: 'i' },
      ]},
      { label: 'JD3P', sub: 'joint discrete denoising diffusion', color: 'r' },
    ],
    output: { label: 'Action tokens', sub: 'unified discrete diffusion', color: 'e' },
    meta: { loss: 'Discrete denoising diffusion', notes: ['3B params', '4x faster than AR', 'Joint vision-action diffusion'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #70  UniPi
  // ═══════════════════════════════════════════════════════════════════════════
  'UniPi': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { label: 'Video diffusion model', sub: 'text-conditioned future generation', color: 'r' },
      { label: 'Generated video plan', sub: 'visual trajectory', color: 'c' },
      { label: 'Inverse dynamics model', sub: 'video → actions', color: 'o' },
    ],
    output: { label: 'Continuous actions', sub: 'video-plan derived', color: 'e' },
    meta: { loss: 'Video diffusion + inverse dynamics', notes: ['No explicit LLM', 'Video as universal interface'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #71  UP-VLA
  // ═══════════════════════════════════════════════════════════════════════════
  'UP-VLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'Show-o backbone', sub: '1.3B', color: 'p', children: [
        { label: 'CLIP ViT', sub: 'vision encoder', color: 'k' },
        { label: 'Phi-1.5', sub: 'language model', color: 'i' },
      ]},
      { label: 'Future image prediction', sub: 'spatial awareness', color: 'c' },
      { label: 'Unified action head', sub: 'understanding + prediction', color: 'o' },
    ],
    output: { label: 'Continuous actions', sub: 'future-aware', color: 'e' },
    meta: { loss: 'Unified understanding + prediction', notes: ['1.3B params', 'Lightweight'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #72  VLA-RL
  // ═══════════════════════════════════════════════════════════════════════════
  'VLA-RL': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
    ],
    stages: [
      { group: 'OpenVLA backbone', sub: '7B', color: 'p', children: [
        { label: 'SigLIP + DinoV2', sub: 'vision encoders', color: 'k' },
        { label: 'Llama-2 7B', sub: 'language model', color: 'i' },
      ]},
      { label: 'VLM process reward model', sub: 'trajectory-level feedback', color: 'c' },
      { label: 'RL fine-tuning', sub: 'trajectory-level optimization', color: 'x' },
    ],
    output: { label: 'Continuous actions', sub: 'RL-refined', color: 'e' },
    meta: { loss: 'RL (trajectory-level) + VLM reward', notes: ['7B params', 'RL on top of VLA'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #73  Vlaser
  // ═══════════════════════════════════════════════════════════════════════════
  'Vlaser': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'InternVL3 VLM', sub: '2B–8B', color: 'p', children: [
        { label: 'Vision encoder', sub: 'InternVL3', color: 'k' },
        { label: 'Qwen2.5', sub: '1.5B or 7B', color: 'i' },
      ]},
      { label: 'Embodied reasoning', sub: 'integrated CoT', color: 'c' },
      { label: 'Flow matching action expert', sub: 'continuous denoising', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'reasoning-guided', color: 'e' },
    meta: { loss: 'Flow matching + reasoning', notes: ['2B–8B params', 'Vlaser-6M dataset'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #74  OmniVLA-RL — Country Garden / Omni AI / VBot / ECNU
  // ═══════════════════════════════════════════════════════════════════════════
  'OmniVLA-RL': {
    inputs: [
      { label: 'Multi-view RGB', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Encoders', sub: 'tri-modal', color: 'k', children: [
        { label: 'SigLIP', sub: 'semantic vision', color: 'k' },
        { label: 'VGGT', sub: '3D spatial encoder', color: 'c' },
        { label: 'Text encoder', sub: 'instructions', color: 'b' },
      ]},
      { group: 'Mix-of-Transformers (MoT) backbone', sub: 'PaLiGemma VLM init, 3 experts share layers', color: 'p', children: [
        { label: 'Reasoning expert', sub: 'semantics + language', color: 'i' },
        { label: 'Spatial expert', sub: '3D geometry', color: 'c' },
        { label: 'Action expert', sub: 'flow matching head', color: 'r' },
      ]},
      { label: 'Block-wise Causal Attention', sub: 'omni-visible prefix + causal action suffix', color: 'x' },
      { group: 'Flow-GSPO online RL', sub: 'SDE reformulation + Group Segmented PO', color: 'a', children: [
        { label: 'Conditional Flow Matching', sub: 'H=16, K=10 denoising', color: 'r' },
        { label: 'SDE sampling', sub: 'Fokker–Planck', color: 'y' },
        { label: 'GSPO update', sub: 'G=8, ε=0.2, β=0.01', color: 'a' },
      ]},
    ],
    output: { label: 'Continuous action chunks', sub: 'spatially grounded, RL-refined', color: 'e' },
    meta: { loss: 'CFM + Flow-GSPO', loop: '10 denoising / 200 RL steps', notes: ['LIBERO avg 97.6%', 'LIBERO-Plus 80.3% (+39.1pp over SFT)', 'No real-world eval'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #75  HEX — BICHR / XJTU / Nankai / PKU
  // ═══════════════════════════════════════════════════════════════════════════
  'HEX': {
    inputs: [
      { label: 'Multi-view RGB', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Humanoid proprio', sub: 'upper+lower body, IMU, tactile', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Qwen3-VL-2B-Instruct', sub: 'VLM backbone, 2B', color: 'p', children: [
        { label: 'Vision encoder', sub: 'Qwen3-VL', color: 'k' },
        { label: 'Qwen3 LLM', sub: '2B language', color: 'i' },
        { label: 'History query tokens', sub: 'lightweight temporal cache', color: 'c' },
      ]},
      { group: 'Unified Proprioceptive Predictor (UPP)', sub: '4-layer, hidden 768, morphology-aware MoE', color: 'o', children: [
        { label: 'Humanoid-aligned state repr.', sub: 'body-part encoders', color: 'g' },
        { label: 'MoE: 16 routed + 2 shared', sub: 'top-1 softmax, LB=0.01', color: 'a' },
        { label: 'Future state predictor', sub: '50-step proprio horizon', color: 't' },
      ]},
      { label: 'Residual-gated fusion', sub: 'blends VL features + predicted future states', color: 'x' },
      { group: 'DiT-B flow-matching action head', sub: '16-layer, hidden 1024, 100-step chunks', color: 'r', children: [
        { label: 'Flow-matching objective', sub: 'velocity field V_θ', color: 'r' },
      ]},
    ],
    output: { label: 'Whole-body action chunks', sub: '100-step, cross-embodiment', color: 'e' },
    meta: { loss: 'L_action + α·L_state (flow + proprio pred)', loop: 'Real-time @ 73.34ms on RTX 4090 (≈13.6 Hz)', notes: ['2.4B params', 'Seen-scenario avg 79.8%', 'Generalization avg 61.8%', '~1K A100 GPU hrs'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #76  Goal2Skill — BUPT / InspireOmni AI / Tsinghua
  // ═══════════════════════════════════════════════════════════════════════════
  'Goal2Skill': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Long-horizon goal G', color: 'b' },
      { label: 'Proprio s_t', sub: '14-dim', color: 'g' },
    ],
    stages: [
      { group: 'High-level VLM planner', sub: 'agentic (VLM identity not disclosed)', color: 'p', children: [
        { label: 'Φ_plan', sub: 'goal decomposition', color: 'i' },
        { label: 'Φ_verify', sub: 'VLM outcome check', color: 'c' },
        { label: 'Φ_reflect', sub: 'failure diagnosis + replan', color: 'y' },
        { label: 'Φ_mem', sub: 'memory updater', color: 'a' },
      ]},
      { group: 'Structured task memory M_t', sub: 'natural-language', color: 't', children: [
        { label: 'Episodic H_t', sub: 'sliding-window summaries', color: 't' },
        { label: 'Working W_t', sub: 'current state summary', color: 't' },
        { label: 'Error register E_t', sub: 'diagnosed failures', color: 'r' },
      ]},
      { label: 'Geometry-preserving filter', sub: 'Î_t = I_t ⊙ (1−Q_t), zero-shot mask + temporal update', color: 'x' },
      { group: 'Low-level skill library', sub: 'discrete diffusion primitives {π_1..π_J}', color: 'r', children: [
        { label: 'Skill selector j_k', sub: 'chosen by planner', color: 'a' },
        { label: 'Reverse diffusion (Eq.19)', sub: 'µ_θ + σ_m·ε, conditioned on Î, s, ℓ', color: 'r' },
        { label: 'Receding-horizon chunks', sub: 'with VLM checkpoint verify', color: 'x' },
      ]},
    ],
    output: { label: 'Continuous action chunks', sub: 'sub-task level, memory-aware', color: 'e' },
    meta: { loss: 'Per-skill diffusion (50 demos/task, 30k steps)', loop: 'Closed-loop plan → exec → verify → (retry | replan)', notes: ['RMBench total 32.4% (vs 9.8% best baseline)', 'M(n) avg 38.7% vs 9.0%', 'VLM / executor identities NOT disclosed'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #77  UniT — XPeng / Tsinghua / HKU
  // ═══════════════════════════════════════════════════════════════════════════
  'UniT': {
    inputs: [
      { label: 'Egocentric human video', color: 'b' },
      { label: 'Humanoid robot RGB', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'State + action (human/robot)', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'UniT tokenizer (pretrain)', sub: 'tri-branch cross-reconstruction + shared RQ-VAE codebook', color: 'c', children: [
        { label: 'Visual branch E_v', sub: 'frozen DINOv2 → IDM', color: 'k' },
        { label: 'Action branch E_a', sub: 'state + action chunk', color: 'g' },
        { label: 'Fusion branch E_m', sub: 'synergizes E_v + E_a features', color: 't' },
      ]},
      { group: 'VLA-UniT (policy)', sub: 'GR00T n1.5 framework + Qwen2.5-VL', color: 'p', children: [
        { label: 'Qwen2.5-VL', sub: 'VL backbone', color: 'i' },
        { label: 'Learnable UniT queries', sub: 'parallel CE classification (Eq.4)', color: 'a' },
        { label: 'Flow-matching velocity head V_θ', sub: 'continuous actions (Eq.5)', color: 'r' },
      ]},
      { group: 'WM-UniT (world model, optional)', sub: 'Cosmos Predict 2.5', color: 'y', children: [
        { label: 'UniT action features z_a', sub: 'control interface', color: 'g' },
        { label: 'Video generation head', sub: 'flow matching on latent frames', color: 'y' },
      ]},
    ],
    output: { label: 'Continuous actions + future frames', sub: 'embodiment-agnostic', color: 'e' },
    meta: { loss: 'L_VLA = L_token (CE) + λ_fm·L_fm (flow)', loop: 'WM-UniT autoregressive rollout', notes: ['RoboCasa GR1 overall 66.7% (vs FLARE 55.0)', 'IRON real-world 78% / 75% (vs GR00T 30% / 5%)', '10% data UniT ≈ full-data GR00T baseline'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #78  AnySlot — Stony Brook / CMU / MBZUAI / PKU
  // ═══════════════════════════════════════════════════════════════════════════
  'AnySlot': {
    inputs: [
      { label: 'Multi-view RGB I^{1:m}', color: 'b' },
      { label: 'Compositional language instr.', color: 'b' },
      { label: 'Robot state q_t', color: 'g' },
    ],
    stages: [
      { group: 'High-level: Scene-marker grounding', sub: 'Nano-Banana 2 (11s latency on H200)', color: 'y', children: [
        { label: 'Guidance image Ĩ', sub: 'blue sphere overlay at target slot', color: 'a' },
        { label: 'Marker center (u,v)', sub: 'pixel-aligned goal', color: 'y' },
      ]},
      { group: 'Low-level: π0.5 VLA (fully fine-tuned)', sub: 'PaliGemma-3B + flow-matching action expert', color: 'p', children: [
        { label: 'PaliGemma-3B', sub: 'VL backbone', color: 'i' },
        { label: 'Goal-conditioned prefix', sub: 'Ĩ injected into context', color: 'c' },
        { label: 'Flow-matching action expert', sub: 'continuous denoising', color: 'r' },
      ]},
    ],
    output: { label: 'Precise slot placement actions', sub: 'sub-centimeter accuracy', color: 'e' },
    meta: { loss: 'Flow matching fine-tune (D_syn, 20k steps, H200 bs=64)', notes: ['3B params', 'SlotBench avg SR 89.6% / IA 92.7%', 'Flat π0.5 scores 18% on Ord only, 0% elsewhere', 'Oracle ~96.7% → bottleneck is high-level grounding'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #79  StableIDM — GigaAI / CASIA / BIT
  // ═══════════════════════════════════════════════════════════════════════════
  'StableIDM': {
    inputs: [
      { label: 'RGB frames I_{t-K:t}', sub: 'causal history window', color: 'b' },
      { label: 'Robot mask M_t', sub: 'from SAM', color: 'g' },
    ],
    stages: [
      { group: 'Preprocessing + visual encoding', sub: 'fixed-length sliding window', color: 'k', children: [
        { label: 'SAM', sub: 'robot-centric masking (suppresses bg)', color: 'c' },
        { label: 'DINOv2', sub: 'frozen encoder → Embeddings + CLS tokens', color: 'k' },
      ]},
      { label: 'TDR step 1: Temporal Fusion (pre-DFA)', sub: 'repairs current visual features using adjacent frames', color: 't' },
      { group: 'Directional Feature Aggregation (DFA)', sub: 'anisotropic spatial aggregation', color: 'o', children: [
        { label: 'Directional weights from CLS tokens', sub: 'MLP over CLS → axis-aware coefficients', color: 'i' },
        { label: 'Weighted global average pooling', sub: 'produces direction-aware feature', color: 'a' },
      ]},
      { label: 'TDR step 2: Temporal Regressor (post-DFA)', sub: 'MLP + causal TCN, dilations {1,2,4,8}', color: 't' },
      { label: 'Residual action prediction', sub: 'denormalize via μ_train, σ_train per-dim', color: 'x' },
    ],
    output: { label: 'Current action a_t', sub: 'dual-arm joint angles + gripper', color: 'e' },
    meta: { loss: 'Supervised IDM regression', loop: 'Single-step; fixed-length causal window', notes: ['AgiBot heavy trunc: acc 0.307, L1 0.493 (best)', 'Real-robot replay avg 47.4%', 'As π0.5 annotator: 35.3% → 52.9%'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #80  DA-PTQ — Tongji / UTS
  // ═══════════════════════════════════════════════════════════════════════════
  'DA-PTQ': {
    inputs: [
      { label: 'Pretrained CogACT (FP)', sub: 'base VLA', color: 'p' },
      { label: 'Calibration: 512 trajs', sub: 'BridgeData V2, 6 spatial bins', color: 'b' },
    ],
    stages: [
      { group: 'Cross-Space Representation Compensation (CSRC)', sub: 'orthogonalize VL↔action interface', color: 'c', children: [
        { label: 'Block-wise SVD rotation', sub: 'block size 16×16', color: 'c' },
        { label: 'Smoothing + shrinkage', sub: 'λ_smooth=0.15, shrink=0.55', color: 't' },
        { label: 'Fold compensation into weights', sub: 'zero inference overhead', color: 'x' },
      ]},
      { group: 'Drift-Aware Mixed-Precision Allocation (DA-MPA)', sub: 'trajectory-level drift optimization', color: 'o', children: [
        { label: 'Structural Jacobian profiling', sub: 'Tikhonov damping λ=3e-4', color: 'a' },
        { label: 'Per-dim motion-error scoring', sub: 'w_trans=1.8, w_rot=0.15', color: 'y' },
        { label: 'Top-30% layers → BF16', sub: 'rest 70% → W4', color: 'r' },
      ]},
      { label: 'Final 2 DiT blocks preserved', sub: 'proximity to continuous output', color: 'x' },
    ],
    output: { label: 'Quantized CogACT (W4A8)', sub: '7-DoF diffusion actions, drift-stabilized', color: 'e' },
    meta: { loss: 'Training-free PTQ (calibration only)', notes: ['42.5% memory reduction', '54.8% inference speedup', 'WidowX VM 48.9 (vs FP 51.3)', 'Google VM 68.5 (vs FP 74.8)', 'Google VA 51.7 (vs FP 61.3)'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #81  VLA-IAP — HKUST / CUHK / SCNU / NUDT / USTB
  // ═══════════════════════════════════════════════════════════════════════════
  'VLA-IAP': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Pretrained VLA (any)', sub: 'OpenVLA-OFT / π0 / π0.5 / DreamVLA', color: 'p' },
    ],
    stages: [
      { group: 'Interaction-First signal extraction (training-free)', sub: 'per-frame geometric + motion priors', color: 'c', children: [
        { label: 'Sobel edge map', sub: 'structural geometric anchors', color: 'c' },
        { label: 'Motion denoise', sub: 'history smoothing, IoU tracking', color: 't' },
        { label: 'Semantic-motion alignment', sub: 'low-IoU → pruned, high-IoU → preserved', color: 'a' },
      ]},
      { group: 'Dynamic pruning schedule', sub: 'conservative → aggressive', color: 'o', children: [
        { label: 'Early phase', sub: 'minimal pruning (stability)', color: 'g' },
        { label: 'Interaction phase', sub: 'aggressive pruning (speedup)', color: 'r' },
      ]},
      { label: 'Inject pruned tokens into backbone', sub: 'unchanged action head', color: 'x' },
    ],
    output: { label: 'Accelerated actions', sub: 'unchanged quality, 1.25–1.54× speedup', color: 'e' },
    meta: { loss: 'None (training-free)', notes: ['LIBERO 97.8% with 1.25× speedup', 'Up to 1.54× speedup with comparable performance', 'Works across OpenVLA-OFT, π0, π0.5, DreamVLA', 'Validated on real robot + 3 sim envs'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #82  DeepThinkVLA — HUST / Tsinghua / RUC
  // ═══════════════════════════════════════════════════════════════════════════
  'DeepThinkVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
    ],
    stages: [
      { group: 'PaliGemma 2 backbone (π0-FAST base)', sub: '2.9B params', color: 'p', children: [
        { label: 'Vision encoder', sub: 'PaliGemma 2 ViT', color: 'k' },
        { label: 'Gemma-2 2B', sub: 'language model', color: 'i' },
      ]},
      { group: 'Hybrid-Attention Decoder', sub: 'condition 1 of CoT effectiveness', color: 'o', children: [
        { label: 'Causal attention for language CoT', sub: 'sequential reasoning tokens', color: 'a' },
        { label: 'Bidirectional attention for actions', sub: 'parallel action decoding', color: 'y' },
      ]},
      { group: 'SFT-then-RL pipeline', sub: 'condition 2: outcome-aligned causal CoT', color: 'r', children: [
        { label: 'Stage 1: SFT with CoT annotations', sub: 'reasoning-action chain', color: 't' },
        { label: 'Stage 2: outcome-based RL', sub: 'sparse task-success reward', color: 'r' },
      ]},
    ],
    output: { label: 'Action tokens (FAST-style)', sub: 'reasoning-grounded', color: 'e' },
    meta: { loss: 'CE on FAST action tokens + RL task reward', notes: ['LIBERO 97.0%', 'LIBERO-Plus 79.0% (vs π0-FAST 61.6%)', 'RoboTwin 2.0 59.3% (+21.7pp over best)', '2.9B params'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #83  StructVLA — USTC / MBZUAI
  // ═══════════════════════════════════════════════════════════════════════════
  'StructVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
    ],
    stages: [
      { group: 'Structured World-Model Planner', sub: 'sparse physically-meaningful future frames', color: 'c', children: [
        { label: 'Kinematic milestone detector', sub: 'gripper transitions + turning points', color: 'k' },
        { label: 'Structured frame predictor', sub: 'unified discrete token vocabulary', color: 't' },
      ]},
      { label: 'Unified discrete token vocabulary', sub: 'shared across visual + action tokens', color: 'p' },
      { group: 'Two-stage training', sub: 'decoupled plan vs execute', color: 'o', children: [
        { label: 'Stage 1: WM predicts structured frames', sub: 'sparse milestones (not dense rollout)', color: 'a' },
        { label: 'Stage 2: map structured foresight → actions', sub: 'low-level motor tokens', color: 'r' },
      ]},
    ],
    output: { label: 'Actions', sub: 'milestone-aligned', color: 'e' },
    meta: { loss: 'Token prediction on unified vocab', notes: ['LIBERO 94.8%', 'SimplerEnv-WidowX 75.0%', 'Sparse structured frames avoid dense-rollout error accumulation', 'Real-world deployments validated'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #84  SmoothVLA — CQUPT / CIGIT CAS / USTC
  // ═══════════════════════════════════════════════════════════════════════════
  'SmoothVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
    ],
    stages: [
      { group: 'Prismatic-7B (OpenVLA base)', sub: 'autoregressive VLA', color: 'p', children: [
        { label: 'SigLIP + DinoV2', sub: 'vision encoders', color: 'k' },
        { label: 'Llama-2 7B', sub: 'language model', color: 'i' },
        { label: 'AR action head + chunking', sub: 'autoregressive tokens', color: 'a' },
      ]},
      { group: 'GRPO RL fine-tuning', sub: 'intrinsic smoothness optimization', color: 'o', children: [
        { label: 'Binary sparse task reward', sub: 'success/fail signal', color: 'r' },
        { label: 'Continuous jerk penalty', sub: 'trajectory jerk from policy rollouts', color: 't' },
        { label: 'Hybrid reward combination', sub: 'task + physics-informed smoothness', color: 'y' },
      ]},
    ],
    output: { label: 'Smooth continuous actions', sub: 'physically feasible trajectories', color: 'e' },
    meta: { loss: 'GRPO with task + jerk hybrid reward', notes: ['LIBERO 80.5% avg', '+13.8% smoothness over standard RL', 'Outperforms SFT generalization', 'Intrinsic reward (no extrinsic env feedback)'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #85  DMW (Drive My Way) — UC Riverside / U Michigan  (autonomous driving VLA)
  // ═══════════════════════════════════════════════════════════════════════════
  'DMW': {
    inputs: [
      { label: 'Front-view RGB', color: 'b' },
      { label: 'Natural language instr.', sub: 'short-term style guidance', color: 'b' },
      { label: 'User embedding', sub: 'long-term preference', color: 'k' },
    ],
    stages: [
      { group: 'SimLingo VLA backbone', sub: 'built on InternVL2-1B', color: 'p', children: [
        { label: 'InternVL2 vision encoder', color: 'k' },
        { label: 'InternVL2-1B LLM', sub: '1B', color: 'i' },
      ]},
      { group: 'Preference-alignment fine-tuning', sub: 'per-driver policy adaptation', color: 'o', children: [
        { label: 'LLM-generated reward weights', sub: 'per-style safety/efficiency/comfort', color: 'y' },
        { label: 'Policy FT with reward shaping', sub: 'AdamW lr=1e-4, wd=1e-3', color: 'r' },
      ]},
    ],
    output: { label: 'Continuous driving control', sub: 'throttle / brake / steer, personalized', color: 'e' },
    meta: { loss: 'RL-style reward shaping with LLM-generated weights', notes: ['1B params', 'Bench2Drive closed-loop', '25 drivers for preference alignment', 'CVPR 2026, open-source at dmw-cvpr.github.io'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #86  KineVLA — Gaoge Han et al.
  // ═══════════════════════════════════════════════════════════════════════════
  'KineVLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
    ],
    stages: [
      { group: 'Prismatic-7B VLM (OpenVLA base)', sub: '7B', color: 'p', children: [
        { label: 'SigLIP + DinoV2', sub: 'vision encoders', color: 'k' },
        { label: 'Llama-2 7B', sub: 'language model', color: 'i' },
      ]},
      { group: 'Bi-level action tokenization (RVQ-VAE)', sub: 'goal vs kinematics decomposition', color: 'o', children: [
        { label: 'Goal codebook', sub: 'coarse, goal-invariant intent', color: 'a' },
        { label: 'Kinematics codebook', sub: 'fine, kinematics-variant execution', color: 't' },
        { label: 'Mutual-information regularizer', sub: 'disentangle the two levels', color: 'y' },
      ]},
      { label: 'Autoregressive action-token decoder', sub: 'bi-level tokens', color: 'r' },
    ],
    output: { label: 'Discrete action tokens', sub: 'two-level structure', color: 'e' },
    meta: { loss: 'CE on bi-level tokens + MI regularizer', notes: ['~7B params', 'Decouples task intent from motor execution', 'OpenVLA-style base'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #87  LLaVA-VLA — HKUST (GZ) / HUST / Westlake
  // ═══════════════════════════════════════════════════════════════════════════
  'LLaVA-VLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
    ],
    stages: [
      { group: 'LLaVA-OneVision-0.5B', sub: 'compact VLM', color: 'p', children: [
        { label: 'LLaVA-OneVision vision tower', color: 'k' },
        { label: 'Qwen-2 0.5B', sub: 'language model', color: 'i' },
      ]},
      { label: 'Discrete action tokenizer', sub: 'chunk size 5', color: 'o' },
      { label: 'Autoregressive decoder', sub: 'action tokens', color: 'r' },
    ],
    output: { label: 'Action chunks', sub: '5-step discrete tokens', color: 'e' },
    meta: { loss: 'CE on action tokens', notes: ['0.5B params (ultra-compact)', 'CALVIN 3.68 avg length', 'Action chunking on small VLM'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #88  Motus — Tsinghua / PKU / Horizon Robotics
  // ═══════════════════════════════════════════════════════════════════════════
  'Motus': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Mixture-of-Transformers backbone', sub: '~8B total', color: 'p', children: [
        { label: 'Wan 2.2 VGM', sub: 'video generative model (5.0B)', color: 'y' },
        { label: 'Qwen3-VL-2B', sub: 'VLM (2.13B)', color: 'i' },
        { label: 'Understanding module', sub: '253.5M', color: 'c' },
      ]},
      { group: 'Action expert', sub: '30-layer transformer, 641.5M', color: 'o', children: [
        { label: 'Flow-matching objective', sub: 'velocity prediction', color: 'r' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'video-grounded', color: 'e' },
    meta: { loss: 'Flow matching + video generation', notes: ['~8B total params', 'RoboTwin v2 top-1 (87.8)', 'MoT combines video generation with VLA'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #89  GigaWorld-Policy — GigaAI
  // ═══════════════════════════════════════════════════════════════════════════
  'GigaWorld-Policy': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Wan 2.2 Diffusion Transformer', sub: '5B (video-generative backbone)', color: 'p', children: [
        { label: 'Multi-modal encoder', color: 'k' },
        { label: 'DiT backbone', sub: 'video generation', color: 'y' },
      ]},
      { label: 'Flow-matching action decoder', sub: 'velocity field head', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'world-model-grounded', color: 'e' },
    meta: { loss: 'Flow matching', notes: ['5B params', 'RoboTwin v2 #2 (86.0)', 'Shared video-generation prior'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #90  MMaDA-VLA — (organization TBD from paper)
  // ═══════════════════════════════════════════════════════════════════════════
  'MMaDA-VLA': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', color: 'g' },
      { label: 'Masked action tokens', color: 'o' },
    ],
    stages: [
      { label: 'Multi-modal encoder', sub: 'unified VL tokens', color: 'p' },
      { group: 'Discrete-diffusion action decoder', sub: 'masked token denoising', color: 'o', children: [
        { label: 'Iterative order-free refinement', sub: 'multi-step unmask', color: 'a' },
        { label: 'Parallel token recovery', sub: 'non-causal', color: 'y' },
      ]},
    ],
    output: { label: 'Discrete action tokens', sub: 'iteratively denoised', color: 'e' },
    meta: { loss: 'Masked denoising CE over action codebook', notes: ['Discrete diffusion over action tokens', 'Order-free iterative refinement'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // #91  UniDex — Tsinghua / Shanghai Qizhi / SYSU
  // ═══════════════════════════════════════════════════════════════════════════
  'UniDex': {
    inputs: [
      { label: 'RGB frames', color: 'b' },
      { label: 'Point cloud', sub: '3D geometry', color: 'c' },
      { label: 'Language instr.', color: 'b' },
      { label: 'Robot state', sub: 'dexterous hand 6-24 DoF', color: 'g' },
      { label: 'Noise ε', color: 'o' },
    ],
    stages: [
      { group: 'Multimodal encoding', sub: 'vision + 3D + language', color: 'p', children: [
        { label: 'Uni3D (ViT)', sub: 'point-cloud encoder (2D ViT init)', color: 'k' },
        { label: 'PaliGemma-style vision-language', sub: 'RGB + text', color: 'i' },
      ]},
      { label: 'FAAS (Function-Actuator-Aligned Space)', sub: '82-dim unified action space across 8 hands', color: 'c' },
      { group: 'Flow-matching action head', sub: 'conditional flow matching', color: 'o', children: [
        { label: 'Velocity field prediction', sub: 'Euler integration', color: 'r' },
        { label: 'Step δ=0.1', sub: '10-step denoising', color: 'a' },
      ]},
    ],
    output: { label: 'Dexterous hand actions', sub: '82-dim FAAS → embodiment-specific', color: 'e' },
    meta: { loss: 'Conditional flow matching', notes: ['8 dexterous hands supported', '6-24 DoF range', 'FAAS enables cross-hand transfer'] },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Round 30 — 22 pipeline configs from PDF-verified Round 25-27 YAMLs
  // ═══════════════════════════════════════════════════════════════════════════

  'DualCoT-VLA': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language instr.', color: 'b' }, { label: 'Robot state', color: 'g' }, { label: 'Noise ε', color: 'o' }],
    stages: [
      { group: 'Qwen3-VL-4B + dual frozen teachers', sub: 'Depth-Anything-3 (visual) + Qwen3-0.6B (linguistic)', color: 'p', children: [
        { label: 'Qwen3-VL ViT', color: 'k' },
        { label: 'Qwen3-VL-4B LLM', color: 'i' },
        { label: '16 visual + 16 linguistic CoT query tokens', sub: 'parallel reasoning', color: 'c' },
      ]},
      { label: 'Diffusion Transformer (DiT) action expert', sub: 'Flow-Matching, conditioned on VLM hidden states', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'parallel CoT reasoning', color: 'e' },
    meta: { loss: 'Flow matching + CoT distillation', notes: ['LIBERO 98.8 (#3)', 'RoboCasa GR1 55.1', 'Latency 3156→58.1 ms'] },
  },

  'SnapFlow': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }, { label: 'Noise ε', color: 'o' }],
    stages: [
      { group: 'Frozen base flow VLA', sub: 'pi0.5 (3B) or SmolVLA (500M)', color: 'p', children: [
        { label: 'PaliGemma / SmolVLM', color: 'k' },
        { label: 'Original flow-matching expert', color: 'r' },
      ]},
      { group: 'Two-Phase Distillation', sub: 'progressive FM/consistency mix', color: 'o', children: [
        { label: 'Stage 1: progressive FM training', sub: 'reduce step count', color: 'a' },
        { label: 'Stage 2: consistency distillation', sub: 'collapse to 1 step', color: 'y' },
      ]},
      { label: '1-NFE flow head', sub: 'single forward pass action', color: 'r' },
    ],
    output: { label: 'Action chunk in 1 step', sub: 'speedup w/o quality loss', color: 'e' },
    meta: { loss: 'FM + consistency distillation', notes: ['LIBERO 98.75 (#4)', 'pi0.5 1-NFE', 'Pareto frontier vs concurrent methods'] },
  },

  'LaMP': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }, { label: 'Noise ε', color: 'o' }],
    stages: [
      { group: 'Qwen3-VL-4B-Instruct (frozen)', color: 'p', children: [
        { label: 'Qwen3-VL ViT', color: 'k' },
        { label: 'Qwen3-VL-4B LLM', color: 'i' },
      ]},
      { label: '3D Scene Flow Motion Expert', sub: 'predicts latent motion primitives', color: 'c' },
      { group: 'Action Expert', sub: 'flow matching + gated cross-attention', color: 'o', children: [
        { label: 'Single-layer gated CA', sub: 'motion guidance injection', color: 'a' },
        { label: 'Flow-matching head', color: 'r' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'motion-guided', color: 'e' },
    meta: { loss: 'Flow matching + motion regression', notes: ['LIBERO 98.3 (#6)', 'SimplerEnv WidowX 79.2 (#2)', 'LIBERO-Plus 7-axis robust'] },
  },

  'TAG': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }],
    stages: [
      { label: 'Frozen base VLA (pi0/pi0.5)', sub: 'Gemma + flow-matching head', color: 'p' },
      { group: 'Test-time guidance (CFG-style)', sub: 'no parameter changes', color: 'y', children: [
        { label: 'Target-Agnostic background prompt', sub: 'guidance signal', color: 'a' },
        { label: 'Classifier-free composition', sub: 'shifts denoising trajectory', color: 'r' },
      ]},
    ],
    output: { label: 'More stable actions', sub: 'object-centric', color: 'e' },
    meta: { loss: 'None (test-time only)', notes: ['LIBERO 97.9 (with pi0.5)', 'LIBERO-Plus +9.3pp', 'VLABench +26.01pp'] },
  },

  'DepthCache': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Depth map', color: 'c' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }],
    stages: [
      { label: 'Host VLA backbone (frozen)', sub: 'pi0.5 / OpenVLA / GR00T', color: 'p' },
      { group: 'Depth-guided token merging (training-free)', sub: 'inference-time wrapper', color: 'o', children: [
        { label: 'Depth-region segmentation', color: 'c' },
        { label: 'Progressive cross-frame merging', color: 't' },
        { label: 'Dual protection mechanism', sub: 'preserves critical tokens', color: 'a' },
      ]},
      { label: 'Inject merged tokens into base', sub: 'unchanged action head', color: 'x' },
    ],
    output: { label: 'Accelerated actions', sub: '1.07-1.33× speedup', color: 'e' },
    meta: { loss: 'None (training-free)', notes: ['LIBERO 97.6 (pi0.5+DepthCache)', '0.2-1.0pp drop', 'Real 191→143ms (1.33×)'] },
  },

  'AnchorRefine': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }],
    stages: [
      { label: 'Base policy (frozen GR-1 / X-VLA)', color: 'p' },
      { group: 'Hierarchical Two-Branch Decoder', sub: 'anchor + residual + decision', color: 'o', children: [
        { label: 'Trajectory Anchor Planner', sub: 'coarse arm scaffold (0.22B)', color: 'a' },
        { label: 'Residual Refiner', sub: 'fine-grained correction (1.1B)', color: 'r' },
        { label: 'Decision-aware gripper branch', color: 'y' },
      ]},
    ],
    output: { label: 'Refined actions', sub: 'anchor + residual', color: 'e' },
    meta: { loss: 'Hybrid: anchor + residual + gripper', notes: ['LIBERO-Long 97.4 (X-VLA base)', 'CALVIN ABC→D 4.40', 'Real-world Avg SR 59%'] },
  },

  'OFlow': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }, { label: 'Noise ε', color: 'o' }],
    stages: [
      { group: 'GR00T-N1.5 base', sub: 'Eagle-2.5 VLM + DINOv2 foresight', color: 'p', children: [
        { label: 'Eagle-2.5 (SigLIP + Qwen2.5-VL)', color: 'k' },
        { label: 'DINOv2-Base foresight model', color: 'c' },
      ]},
      { group: 'Object-Aware Flow Matching DiT', sub: 'ControlNet-style injection', color: 'o', children: [
        { label: 'Zero-init cross-attention', sub: 'object-aware features', color: 'a' },
        { label: 'Flow-matching velocity head', color: 'r' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'object-anchored', color: 'e' },
    meta: { loss: 'Flow matching + object-aware reg.', notes: ['LIBERO 96.6', 'SimplerEnv 67.1', 'LIBERO-Plus avg 72.3'] },
  },

  'A1': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }, { label: 'Noise ε', color: 'o' }],
    stages: [
      { label: 'Molmo VLM backbone', sub: 'open-source VLM', color: 'p' },
      { group: 'Flow-matching action expert', sub: '~400M Qwen3 with KV self-attention', color: 'o', children: [
        { label: 'KV-conditioned self-attention', color: 'a' },
        { label: 'Flow-matching head', color: 'r' },
      ]},
    ],
    output: { label: 'Continuous actions', color: 'e' },
    meta: { loss: 'Flow matching', notes: ['LIBERO 96.6 avg', 'VLABench avg 53.5', 'RoboChallenge 29.0', 'Open-source: github.com/ATeam-Research/A1'] },
  },

  'ProbeFlow': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }],
    stages: [
      { label: 'Evo-1 + frozen InternVL3-1B', sub: 'host flow-matching VLA', color: 'p' },
      { group: 'Adaptive ODE Solver (training-free)', sub: 'wrapper on existing flow head', color: 'o', children: [
        { label: 'Probe step', sub: 'estimate local trajectory curvature', color: 'a' },
        { label: 'Adaptive step size', sub: 'denser steps where needed', color: 'y' },
      ]},
      { label: 'Existing 8-layer DiT, 1024 dim', sub: 'unmodified', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'no quality loss, faster', color: 'e' },
    meta: { loss: 'None (training-free)', notes: ['LIBERO 88.7', 'MetaWorld 83.2', 'Zero training cost'] },
  },

  'RoboAlign': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }, { label: 'Noise ε', color: 'o' }],
    stages: [
      { group: 'Qwen2.5-VL-7B MLLM (frozen)', sub: 'reasoning aligned via GRPO', color: 'p', children: [
        { label: 'Qwen2.5-VL ViT', color: 'k' },
        { label: 'Qwen2.5 LLM', color: 'i' },
      ]},
      { group: 'Test-time reasoning + RL alignment', sub: 'GRPO objective', color: 'o', children: [
        { label: 'CoT prompt template', color: 'a' },
        { label: 'GRPO RL fine-tune', color: 'y' },
      ]},
      { label: 'Diffusion action expert (newly initialized)', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'reasoning-aligned', color: 'e' },
    meta: { loss: 'Diffusion + GRPO reward', notes: ['LIBERO 86.8', 'CALVIN 2.57', 'Real-robot 66.7%'] },
  },

  'DIAL': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }, { label: 'Noise ε', color: 'o' }],
    stages: [
      { group: 'Qwen2.5-VL-3B base', color: 'p', children: [
        { label: 'Qwen2.5-VL ViT', color: 'k' },
        { label: 'Qwen2.5-VL-3B LLM', color: 'i' },
      ]},
      { label: 'Latent World Model', sub: 'predicts latent foresight', color: 'c' },
      { group: 'Action decoder', sub: '4-layer fuser + 16-layer DiT', color: 'o', children: [
        { label: 'Self-attention fuser', sub: 'visual + foresight fusion', color: 'a' },
        { label: '16-layer Diffusion Transformer', color: 'r' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'foresight-conditioned', color: 'e' },
    meta: { loss: 'Diffusion + latent WM loss', notes: ['RoboCasa GR1 70.2 (#1!)', 'XPeng project page', 'Decoupled intent + action'] },
  },

  'E-VLA': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Event camera frames', color: 'c' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }],
    stages: [
      { group: 'SmolVLA backbone', sub: 'frozen base VLA', color: 'p', children: [
        { label: 'SigLIP visual encoder', color: 'k' },
        { label: 'SmolVLM LLM', color: 'i' },
      ]},
      { group: '4-block Event Adapter (~13M)', sub: 'hierarchical event encoding', color: 'o', children: [
        { label: 'Pre-encoding overlay', sub: 'optional alternative', color: 'a' },
        { label: 'Weight-sharing ViT blocks', color: 'y' },
      ]},
      { label: 'SmolVLA action expert', sub: 'interleaved self+cross-attn', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'robust to dark/blur', color: 'e' },
    meta: { loss: 'Flow matching (inherited)', notes: ['Real-world: 0→90% in 20-lux dark', 'OOD blur robustness', 'Event-augmented'] },
  },

  'HELM': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }, { label: 'Episodic memory', color: 't' }],
    stages: [
      { label: 'Frozen base VLA (OpenVLA / Octo)', color: 'p' },
      { group: 'Harness Wrapper', sub: 'long-horizon memory + recovery', color: 'o', children: [
        { label: 'Episodic Memory Module (EMM)', sub: 'CLIP ViT-B/32 keyframe RAG', color: 't' },
        { label: 'State Verifier (MLP)', sub: 'progress detection', color: 'c' },
        { label: 'Harness Controller', sub: 'rollback / replan', color: 'r' },
      ]},
    ],
    output: { label: 'Long-horizon actions', sub: 'with memory + recovery', color: 'e' },
    meta: { loss: 'Wrapper-only (frozen base)', notes: ['LIBERO-Long 81.5 (only Long suite reported)', 'CALVIN ABC→D 3.58', 'Code on acceptance'] },
  },

  'HiVLA': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }, { label: 'Noise ε', color: 'o' }],
    stages: [
      { group: 'Qwen3-VL 8B planner', sub: 'fine-tuned high-level', color: 'p', children: [
        { label: 'DINOv2 + SigLIP vision', color: 'k' },
        { label: 'Qwen3-VL-8B LLM', color: 'i' },
      ]},
      { group: 'Cascaded cross-attention DiT', sub: 'CFM, init from H-RDT', color: 'o', children: [
        { label: 'LLaMA-style RoPE backbone', color: 'a' },
        { label: 'Conditional Flow Matching', color: 'r' },
      ]},
    ],
    output: { label: 'Continuous actions', sub: 'hierarchical planner+executor', color: 'e' },
    meta: { loss: 'Conditional flow matching', notes: ['RoboTwin v2 9-task avg 83.3 (#3)', 'Easy 96.0 / Hard 73.2', 'Click Bell + Pick&Place real-world'] },
  },

  'HY-Embodied-0.5': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }, { label: 'Noise ε', color: 'o' }],
    stages: [
      { group: 'HY-Embodied-0.5 MoT', sub: '2B activated / 4B total', color: 'p', children: [
        { label: 'HY-ViT 2.0 (400M, native-res)', color: 'k' },
        { label: 'Mixture-of-Transformers', sub: '32B MoE-A32B variant available', color: 'i' },
        { label: 'Visual latent tokens', color: 'c' },
      ]},
      { label: 'pi0/pi0.5-style Action Expert', sub: '5K-hour UMI pretrain', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'real-world humanoid', color: 'e' },
    meta: { loss: 'Flow matching', notes: ['Tencent open-source: github.com/Tencent-Hunyuan/HY-Embodied', 'Mug Hanging 75% vs π0 45%', 'Packing 85% / Stacking 80%'] },
  },

  'ProGAL-VLA': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }],
    stages: [
      { group: 'Dual-system architecture', sub: 'fast pi0 + slow planner', color: 'p', children: [
        { label: 'OpenVLA-7B fast policy', color: 'i' },
        { label: 'Qwen-2.5-VL-Instruct-7B slow planner', color: 'a' },
      ]},
      { group: 'Goal Alignment + Contrast (GAC)', sub: 'verified-goal-embedding bottleneck', color: 'o', children: [
        { label: 'State-grounded goal prediction', color: 'c' },
        { label: 'GAC contrastive training', color: 't' },
      ]},
      { label: 'Imitation regression head', sub: 'OpenVLA-7B controller', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'goal-aligned', color: 'e' },
    meta: { loss: 'Regression + GAC contrastive', notes: ['LIBERO-Plus 85.5 avg', 'Hybrid: planner + verified bottleneck', '7B controller'] },
  },

  'ReFineVLA': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }],
    stages: [
      { group: 'PaliGemma 2 + SpatialVLA', sub: '3.5B base', color: 'p', children: [
        { label: 'PaliGemma 2 ViT', color: 'k' },
        { label: 'Gemma LLM', color: 'i' },
      ]},
      { group: 'Multi-objective Hybrid Head', sub: 'action + reasoning generation', color: 'o', children: [
        { label: 'SpatialVLA action head', color: 'r' },
        { label: 'Multimodal reasoning generation', sub: 'natural-language CoT', color: 'a' },
      ]},
    ],
    output: { label: 'Action + reasoning', sub: 'teacher-guided', color: 'e' },
    meta: { loss: 'Action + reasoning multi-task', notes: ['SimplerEnv WidowX avg 47.7', 'Google VM 76.6 / VA 68.8', '3.5B params'] },
  },

  'SAMoE-VLA': {
    inputs: [{ label: 'Multi-camera RGB', color: 'b' }, { label: 'BEV features', color: 'c' }, { label: 'Language', color: 'b' }, { label: 'Ego state', color: 'g' }, { label: 'Noise ε', color: 'o' }],
    stages: [
      { group: 'Multi-modal encoder', sub: 'autonomous driving stack', color: 'p', children: [
        { label: 'OpenCLIP-ConvNext + CPFPN', color: 'k' },
        { label: 'BEVFormer BEV encoder', color: 't' },
        { label: 'InternVL2-2B planner', color: 'i' },
      ]},
      { group: 'SA-MoE Flow Matching', sub: 'every 4th transformer FFN replaced', color: 'o', children: [
        { label: 'Scene-Adaptive routing', color: 'a' },
        { label: 'Conditional Cross-Modal Causal Attention', color: 'y' },
      ]},
    ],
    output: { label: 'Driving trajectory', sub: 'nuScenes / LangAuto', color: 'e' },
    meta: { loss: 'Flow matching', notes: ['Autonomous driving (not manipulation)', 'nuScenes + LangAuto SOTA', '3.6B'] },
  },

  'TacVLA': {
    inputs: [{ label: 'Front RGB', color: 'b' }, { label: 'Wrist RGB', color: 'b' }, { label: 'Tactile (15 sensors)', color: 'c' }, { label: 'Language', color: 'b' }, { label: 'State', color: 'g' }, { label: 'Noise ε', color: 'o' }],
    stages: [
      { group: 'Pi0.5 backbone (frozen partially)', color: 'p', children: [
        { label: 'SigLIP visual encoder', color: 'k' },
        { label: 'PaliGemma / Gemma 2.6B', color: 'i' },
      ]},
      { label: 'MLP tactile encoder', sub: '15-sensor input', color: 'c' },
      { label: 'Contact-aware gating', sub: 'fuses tactile + visual', color: 'a' },
      { label: 'Pi0.5-style flow-matching action expert', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'tactile-aware', color: 'e' },
    meta: { loss: 'Flow matching', notes: ['Real-world only (Franka)', '4 disassembly + in-box picking', 'Vision+tactile fusion'] },
  },

  'UniDriveVLA': {
    inputs: [{ label: 'Front-view RGB', color: 'b' }, { label: 'Driving instr.', color: 'b' }, { label: 'Ego state', color: 'g' }, { label: 'Noise ε', color: 'o' }],
    stages: [
      { group: 'Qwen3-VL backbone', sub: '2B Base / 8B Large', color: 'p', children: [
        { label: 'SigLIP-2 vision', color: 'k' },
        { label: 'Qwen3 LM', color: 'i' },
      ]},
      { group: 'Mixture-of-Transformers', sub: 'autonomous driving experts', color: 'o', children: [
        { label: 'Understanding expert', color: 'c' },
        { label: 'Perception expert', color: 't' },
        { label: 'Action expert', sub: 'flow matching', color: 'r' },
      ]},
    ],
    output: { label: 'Driving trajectory', sub: 'Bench2Drive / nuScenes', color: 'e' },
    meta: { loss: 'Flow matching', notes: ['Bench2Drive Driving Score 78.37', 'nuScenes L2 0.90', 'Xiaomi open-source'] },
  },

  'VP-VLA': {
    inputs: [{ label: 'RGB frames', color: 'b' }, { label: 'Language + visual prompt', color: 'b' }, { label: 'State', color: 'g' }],
    stages: [
      { group: 'Dual-system planner + controller', sub: 'visual-prompt interface', color: 'p', children: [
        { label: 'Qwen3-VL-4B Planner', sub: 'parses visual prompts', color: 'a' },
        { label: 'QwenOFT Controller', sub: 'OpenVLA-OFT base + Qwen3-VL-4B', color: 'i' },
      ]},
      { label: 'Imitation regression head', sub: 'OFT-style action chunks', color: 'r' },
    ],
    output: { label: 'Continuous actions', sub: 'visual-prompt grounded', color: 'e' },
    meta: { loss: 'Regression', notes: ['SimplerEnv WidowX 58.3', 'RoboCasa-GR1 53.8', 'Visual prompting interface'] },
  },

}
