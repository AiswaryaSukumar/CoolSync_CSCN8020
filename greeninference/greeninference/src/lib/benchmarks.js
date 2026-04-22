export const BENCHMARK_PROFILES = [
  {
    id: "short_qa",
    label: "Short QA",
    prompt: "Answer this question in two concise sentences and cite one key risk.",
    qosLabel: "Interactive < 1.0s first token",
    tokenTarget: 64,
    qualityTier: "Interactive",
  },
  {
    id: "long_context",
    label: "Long Context Summary",
    prompt:
      "Summarize this 20-page report into 8 bullets, highlight operational risks, and identify follow-up actions for the sustainability team.",
    qosLabel: "Interactive < 2.5s first token",
    tokenTarget: 320,
    qualityTier: "Analyst",
  },
  {
    id: "batch_brief",
    label: "Batch Briefing",
    prompt:
      "Generate a structured briefing for regional operations, including carbon signal interpretation, routing advice, and a final recommendation paragraph.",
    qosLabel: "Batch < 8s completion",
    tokenTarget: 520,
    qualityTier: "Batch",
  },
];

export function getBenchmarkProfile(profileId) {
  return BENCHMARK_PROFILES.find((profile) => profile.id === profileId) ?? BENCHMARK_PROFILES[0];
}

export function buildBenchmarkResult({
  profile,
  activeRun,
  activeMetrics,
  traceSource = "simulation",
  strategy = "reactive",
}) {
  if (!profile || !activeRun || !activeMetrics) return null;

  return {
    id: profile.id,
    label: profile.label,
    qosLabel: profile.qosLabel,
    qualityTier: profile.qualityTier,
    traceSource,
    strategy,
    whPerRequest: Number(activeRun.whPerRequest ?? 0),
    jPerToken: Number(activeRun.jPerToken ?? 0),
    co2ePerRequest: Number(activeRun.co2ePerRequest ?? 0),
    stabilityScore: Number(activeRun.stabilityScore ?? 0),
    overheadRatio: Number(activeMetrics.overheadCurrentMeasured ?? activeMetrics.overheadCurrent ?? 0),
    estimatedTag: activeRun.isEstimated ? "Estimated" : "Observed",
    comparabilityNote: activeRun.isEstimated
      ? "Comparable estimate under the selected benchmark profile."
      : "Measured trace aligned to the selected benchmark profile.",
  };
}

export function createBenchmarkSnapshot({
  benchmarkResult,
  profile,
  traceSource = "simulation",
  strategy = "reactive",
} = {}) {
  if (!benchmarkResult || !profile) return null;

  const now = new Date();
  return {
    id: `${profile.id}_${now.getTime()}`,
    capturedAt: now.toISOString(),
    label: profile.label,
    traceSource,
    strategy,
    whPerRequest: Number(benchmarkResult.whPerRequest ?? 0),
    jPerToken: Number(benchmarkResult.jPerToken ?? 0),
    co2ePerRequest: Number(benchmarkResult.co2ePerRequest ?? 0),
    stabilityScore: Number(benchmarkResult.stabilityScore ?? 0),
    overheadRatio: Number(benchmarkResult.overheadRatio ?? 0),
    estimatedTag: benchmarkResult.estimatedTag,
  };
}
