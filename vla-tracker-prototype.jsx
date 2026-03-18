import { useState, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from "recharts";

const MODELS = [
  {
    name: "π₀.5", org: "Physical Intelligence", type: "Flow Matching",
    libero: { spatial: 96.2, object: 97.1, goal: 89.4, long: 72.8 },
    color: "#7F77DD", history: [
      { month: "Oct 24", avg: 0 }, { month: "Jan 25", avg: 0 },
      { month: "Apr 25", avg: 85.2 }, { month: "Jul 25", avg: 86.8 },
      { month: "Oct 25", avg: 88.1 }, { month: "Jan 26", avg: 88.9 }
    ]
  },
  {
    name: "OpenVLA-OFT", org: "Stanford / UC Berkeley", type: "Autoregressive + LoRA",
    libero: { spatial: 94.8, object: 95.2, goal: 91.0, long: 68.4 },
    color: "#1D9E75", history: [
      { month: "Oct 24", avg: 74.5 }, { month: "Jan 25", avg: 78.2 },
      { month: "Apr 25", avg: 82.4 }, { month: "Jul 25", avg: 85.1 },
      { month: "Oct 25", avg: 86.8 }, { month: "Jan 26", avg: 87.4 }
    ]
  },
  {
    name: "CoT-VLA", org: "UC Berkeley", type: "Chain-of-Thought + Diffusion",
    libero: { spatial: 92.1, object: 93.8, goal: 88.2, long: 71.2 },
    color: "#D85A30", history: [
      { month: "Oct 24", avg: 0 }, { month: "Jan 25", avg: 0 },
      { month: "Apr 25", avg: 0 }, { month: "Jul 25", avg: 82.3 },
      { month: "Oct 25", avg: 84.7 }, { month: "Jan 26", avg: 86.3 }
    ]
  },
  {
    name: "HybridVLA", org: "ByteDance", type: "Diffusion + Autoregressive",
    libero: { spatial: 91.4, object: 94.2, goal: 87.6, long: 69.8 },
    color: "#D4537E", history: [
      { month: "Oct 24", avg: 0 }, { month: "Jan 25", avg: 0 },
      { month: "Apr 25", avg: 0 }, { month: "Jul 25", avg: 0 },
      { month: "Oct 25", avg: 83.2 }, { month: "Jan 26", avg: 85.8 }
    ]
  },
  {
    name: "OpenVLA", org: "Stanford / UC Berkeley", type: "Autoregressive",
    libero: { spatial: 84.2, object: 88.4, goal: 72.0, long: 53.4 },
    color: "#888780", history: [
      { month: "Oct 24", avg: 74.5 }, { month: "Jan 25", avg: 74.5 },
      { month: "Apr 25", avg: 74.5 }, { month: "Jul 25", avg: 74.5 },
      { month: "Oct 25", avg: 74.5 }, { month: "Jan 26", avg: 74.5 }
    ]
  },
  {
    name: "Octo", org: "UC Berkeley", type: "Diffusion",
    libero: { spatial: 78.9, object: 82.3, goal: 65.4, long: 42.1 },
    color: "#639922", history: [
      { month: "Oct 24", avg: 67.2 }, { month: "Jan 25", avg: 67.2 },
      { month: "Apr 25", avg: 67.2 }, { month: "Jul 25", avg: 67.2 },
      { month: "Oct 25", avg: 67.2 }, { month: "Jan 26", avg: 67.2 }
    ]
  }
];

const WEEKLY_DIGEST = {
  week: "Week 12, March 2026",
  headline: "Flow matching VLAs now dominate 4/5 LIBERO suites",
  insights: [
    {
      title: "Action head paradigm shift",
      body: "π₀.5 and HybridVLA both use flow matching action heads. Among the top 5 models on LIBERO, 3 now use flow matching or diffusion, up from 1/5 six months ago. Autoregressive tokenization (OpenVLA's approach) is losing ground except in long-horizon tasks.",
      delta: "+12.3%p",
      direction: "up"
    },
    {
      title: "Long-horizon remains unsolved",
      body: "LIBERO-Long scores still lag behind other suites by 15-25%p across all models. Interestingly, CoT-VLA's chain-of-thought reasoning shows the smallest gap (17.1%p) vs π₀.5 (23.4%p gap), suggesting explicit reasoning may help in sequential decision-making.",
      delta: "17.1%p gap",
      direction: "neutral"
    },
    {
      title: "Open-source catching up",
      body: "OpenVLA-OFT (open-source) now trails π₀.5 (closed weights) by only 1.5%p on average LIBERO score. Six months ago, the gap was 6.8%p. LoRA fine-tuning on domain-specific data is proving highly effective.",
      delta: "1.5%p gap",
      direction: "down"
    }
  ]
};

const Tab = ({ active, label, onClick }) => (
  <button
    onClick={onClick}
    style={{
      padding: "8px 18px",
      fontSize: 13,
      fontWeight: active ? 600 : 400,
      color: active ? "var(--color-text-primary)" : "var(--color-text-secondary)",
      background: active ? "var(--color-background-secondary)" : "transparent",
      border: "1px solid",
      borderColor: active ? "var(--color-border-secondary)" : "transparent",
      borderRadius: 8,
      cursor: "pointer",
      transition: "all 0.2s"
    }}
  >{label}</button>
);

const Badge = ({ text, color }) => (
  <span style={{
    display: "inline-block",
    padding: "2px 8px",
    fontSize: 11,
    fontWeight: 500,
    borderRadius: 6,
    background: color + "18",
    color: color,
    marginLeft: 6
  }}>{text}</span>
);

export default function VLATracker() {
  const [activeTab, setActiveTab] = useState("leaderboard");
  const [benchmark, setBenchmark] = useState("avg");
  const [selectedModels, setSelectedModels] = useState(["π₀.5", "OpenVLA-OFT", "CoT-VLA"]);

  const sorted = useMemo(() => {
    return [...MODELS].sort((a, b) => {
      if (benchmark === "avg") {
        const avgA = (a.libero.spatial + a.libero.object + a.libero.goal + a.libero.long) / 4;
        const avgB = (b.libero.spatial + b.libero.object + b.libero.goal + b.libero.long) / 4;
        return avgB - avgA;
      }
      return b.libero[benchmark] - a.libero[benchmark];
    });
  }, [benchmark]);

  const chartData = useMemo(() => {
    const months = ["Oct 24", "Jan 25", "Apr 25", "Jul 25", "Oct 25", "Jan 26"];
    return months.map(m => {
      const point = { month: m };
      MODELS.filter(mod => selectedModels.includes(mod.name)).forEach(mod => {
        const h = mod.history.find(h => h.month === m);
        if (h && h.avg > 0) point[mod.name] = h.avg;
      });
      return point;
    });
  }, [selectedModels]);

  const barData = useMemo(() => {
    const suites = ["spatial", "object", "goal", "long"];
    return suites.map(s => {
      const point = { suite: s.charAt(0).toUpperCase() + s.slice(1) };
      MODELS.slice(0, 4).forEach(m => { point[m.name] = m.libero[s]; });
      return point;
    });
  }, []);

  return (
    <div style={{ fontFamily: '"SF Pro Display", -apple-system, sans-serif', maxWidth: 720, margin: "0 auto" }}>
      <div style={{ marginBottom: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <span style={{ fontSize: 20 }}>🔬</span>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: "var(--color-text-primary)", margin: 0 }}>
            VLA-Tracker
          </h1>
          <Badge text="LIVE" color="#1D9E75" />
        </div>
        <p style={{ fontSize: 13, color: "var(--color-text-secondary)", margin: 0 }}>
          AI-powered benchmark tracking for Vision-Language-Action models · Updated weekly
        </p>
      </div>

      <div style={{
        background: "var(--color-background-info)",
        border: "1px solid var(--color-border-info)",
        borderRadius: 10,
        padding: "12px 16px",
        marginBottom: 20,
        fontSize: 13
      }}>
        <div style={{ fontWeight: 600, color: "var(--color-text-info)", marginBottom: 4 }}>
          📡 {WEEKLY_DIGEST.week}
        </div>
        <div style={{ color: "var(--color-text-primary)" }}>
          {WEEKLY_DIGEST.headline}
        </div>
      </div>

      <div style={{ display: "flex", gap: 6, marginBottom: 20, flexWrap: "wrap" }}>
        <Tab active={activeTab === "leaderboard"} label="Leaderboard" onClick={() => setActiveTab("leaderboard")} />
        <Tab active={activeTab === "trends"} label="Trends" onClick={() => setActiveTab("trends")} />
        <Tab active={activeTab === "compare"} label="Compare" onClick={() => setActiveTab("compare")} />
        <Tab active={activeTab === "digest"} label="AI Analysis" onClick={() => setActiveTab("digest")} />
      </div>

      {activeTab === "leaderboard" && (
        <div>
          <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
            {["avg", "spatial", "object", "goal", "long"].map(b => (
              <button key={b} onClick={() => setBenchmark(b)} style={{
                padding: "4px 12px", fontSize: 12, borderRadius: 6,
                border: benchmark === b ? "1px solid var(--color-border-primary)" : "1px solid var(--color-border-tertiary)",
                background: benchmark === b ? "var(--color-background-secondary)" : "transparent",
                color: "var(--color-text-primary)",
                cursor: "pointer", fontWeight: benchmark === b ? 600 : 400,
                textTransform: "capitalize"
              }}>{b === "avg" ? "Average" : `LIBERO-${b.charAt(0).toUpperCase() + b.slice(1)}`}</button>
            ))}
          </div>

          <div style={{ border: "1px solid var(--color-border-tertiary)", borderRadius: 10, overflow: "hidden" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ background: "var(--color-background-secondary)" }}>
                  <th style={{ padding: "10px 12px", textAlign: "left", fontWeight: 600, color: "var(--color-text-secondary)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.5px" }}>#</th>
                  <th style={{ padding: "10px 12px", textAlign: "left", fontWeight: 600, color: "var(--color-text-secondary)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.5px" }}>Model</th>
                  <th style={{ padding: "10px 12px", textAlign: "left", fontWeight: 600, color: "var(--color-text-secondary)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.5px" }}>Type</th>
                  <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 600, color: "var(--color-text-secondary)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.5px" }}>Score</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((m, i) => {
                  const score = benchmark === "avg"
                    ? ((m.libero.spatial + m.libero.object + m.libero.goal + m.libero.long) / 4).toFixed(1)
                    : m.libero[benchmark].toFixed(1);
                  const medals = ["🥇", "🥈", "🥉"];
                  return (
                    <tr key={m.name} style={{ borderTop: "1px solid var(--color-border-tertiary)" }}>
                      <td style={{ padding: "10px 12px", fontWeight: 500 }}>
                        {i < 3 ? medals[i] : i + 1}
                      </td>
                      <td style={{ padding: "10px 12px" }}>
                        <div style={{ fontWeight: 600, color: "var(--color-text-primary)" }}>{m.name}</div>
                        <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>{m.org}</div>
                      </td>
                      <td style={{ padding: "10px 12px" }}>
                        <Badge text={m.type} color={m.color} />
                      </td>
                      <td style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700, fontSize: 15, fontVariantNumeric: "tabular-nums", color: i === 0 ? "#1D9E75" : "var(--color-text-primary)" }}>
                        {score}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 8 }}>
            Benchmark: LIBERO · Metric: Success Rate (%) · Higher is better · Last updated: 2026-03-15
          </p>
        </div>
      )}

      {activeTab === "trends" && (
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 6, color: "var(--color-text-primary)" }}>
            LIBERO average score over time
          </div>
          <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
            {MODELS.map(m => (
              <label key={m.name} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12, cursor: "pointer", color: "var(--color-text-secondary)" }}>
                <input type="checkbox" checked={selectedModels.includes(m.name)}
                  onChange={() => setSelectedModels(prev =>
                    prev.includes(m.name) ? prev.filter(n => n !== m.name) : [...prev, m.name]
                  )}
                />
                <span style={{ width: 10, height: 10, borderRadius: 3, background: m.color, display: "inline-block" }} />
                {m.name}
              </label>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
              <XAxis dataKey="month" tick={{ fontSize: 11, fill: "var(--color-text-tertiary)" }} />
              <YAxis domain={[60, 100]} tick={{ fontSize: 11, fill: "var(--color-text-tertiary)" }} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid var(--color-border-secondary)", background: "var(--color-background-primary)" }} />
              {MODELS.filter(m => selectedModels.includes(m.name)).map(m => (
                <Line key={m.name} type="monotone" dataKey={m.name} stroke={m.color} strokeWidth={2} dot={{ r: 3 }} connectNulls={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
          <p style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 4 }}>
            Models appear when first published. Gap = not yet released.
          </p>
        </div>
      )}

      {activeTab === "compare" && (
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: "var(--color-text-primary)" }}>
            Top 4 models across LIBERO suites
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
              <XAxis dataKey="suite" tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }} />
              <YAxis domain={[40, 100]} tick={{ fontSize: 11, fill: "var(--color-text-tertiary)" }} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid var(--color-border-secondary)", background: "var(--color-background-primary)" }} />
              {MODELS.slice(0, 4).map(m => (
                <Bar key={m.name} dataKey={m.name} fill={m.color} radius={[4, 4, 0, 0]} barSize={18} />
              ))}
            </BarChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", gap: 12, justifyContent: "center", marginTop: 8, flexWrap: "wrap" }}>
            {MODELS.slice(0, 4).map(m => (
              <span key={m.name} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--color-text-secondary)" }}>
                <span style={{ width: 10, height: 10, borderRadius: 3, background: m.color, display: "inline-block" }} />
                {m.name}
              </span>
            ))}
          </div>
          <div style={{
            marginTop: 16,
            padding: "12px 16px",
            background: "var(--color-background-warning)",
            border: "1px solid var(--color-border-warning)",
            borderRadius: 10,
            fontSize: 12,
            color: "var(--color-text-warning)"
          }}>
            <strong>AI Insight:</strong> LIBERO-Long remains the most challenging suite. Even the top model (π₀.5) scores 72.8% here vs 96.2% on Spatial — a 23.4%p gap. CoT-VLA's chain-of-thought approach shows the smallest gap at 17.1%p, suggesting explicit reasoning helps sequential tasks.
          </div>
        </div>
      )}

      {activeTab === "digest" && (
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4, color: "var(--color-text-primary)" }}>
            Weekly AI Analysis
          </div>
          <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 16 }}>
            Auto-generated by Claude · {WEEKLY_DIGEST.week}
          </div>
          {WEEKLY_DIGEST.insights.map((ins, i) => (
            <div key={i} style={{
              border: "1px solid var(--color-border-tertiary)",
              borderRadius: 10,
              padding: "14px 16px",
              marginBottom: 10
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                <span style={{ fontWeight: 600, fontSize: 13, color: "var(--color-text-primary)" }}>
                  {ins.title}
                </span>
                <span style={{
                  fontSize: 12, fontWeight: 600, fontVariantNumeric: "tabular-nums",
                  color: ins.direction === "up" ? "#1D9E75" : ins.direction === "down" ? "#D85A30" : "var(--color-text-secondary)"
                }}>
                  {ins.direction === "up" ? "▲" : ins.direction === "down" ? "▼" : "●"} {ins.delta}
                </span>
              </div>
              <p style={{ fontSize: 12, lineHeight: 1.6, color: "var(--color-text-secondary)", margin: 0 }}>
                {ins.body}
              </p>
            </div>
          ))}
          <div style={{
            marginTop: 16,
            padding: 14,
            background: "var(--color-background-secondary)",
            borderRadius: 10,
            fontSize: 12,
            color: "var(--color-text-secondary)",
            textAlign: "center"
          }}>
            📚 Past analyses: <a href="#" style={{ color: "var(--color-text-info)" }}>W11</a> · <a href="#" style={{ color: "var(--color-text-info)" }}>W10</a> · <a href="#" style={{ color: "var(--color-text-info)" }}>W09</a> · <a href="#" style={{ color: "var(--color-text-info)" }}>2025 Q4 Report</a>
          </div>
        </div>
      )}

      <div style={{
        marginTop: 24,
        padding: "10px 16px",
        background: "var(--color-background-secondary)",
        borderRadius: 10,
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        fontSize: 11,
        color: "var(--color-text-tertiary)"
      }}>
        <span>6 models · 5 benchmarks · Updated 2026-03-15</span>
        <span>⭐ Star on GitHub</span>
      </div>
    </div>
  );
}
