export const DEFAULT_CAPACITY_CONTROLS = {
  headroomPct: 18,
  oversubscriptionMode: "guarded",
  batchingMode: "adaptive",
  powerCapMode: "balanced",
};

const OVERSUBSCRIPTION_PROFILES = {
  guarded: {
    label: "Guarded",
    throughputMultiplier: 0.98,
    computeMultiplier: 0.98,
    lagMultiplier: 0.92,
    queueDelayMs: 28,
    utilizationBoost: -4,
    stabilityAdjustment: 6,
    alertBias: -1,
    pressureBias: -1,
  },
  balanced: {
    label: "Balanced",
    throughputMultiplier: 1.06,
    computeMultiplier: 1.02,
    lagMultiplier: 1.03,
    queueDelayMs: 70,
    utilizationBoost: 4,
    stabilityAdjustment: 0,
    alertBias: 0,
    pressureBias: 0,
  },
  aggressive: {
    label: "Aggressive",
    throughputMultiplier: 1.16,
    computeMultiplier: 1.08,
    lagMultiplier: 1.16,
    queueDelayMs: 145,
    utilizationBoost: 11,
    stabilityAdjustment: -7,
    alertBias: 1,
    pressureBias: 1,
  },
};

const BATCHING_PROFILES = {
  low: {
    label: "Low batching",
    throughputMultiplier: 0.94,
    computeMultiplier: 1.04,
    overheadDelta: 0.01,
    queueDelayMs: 10,
    stabilityAdjustment: 2,
  },
  adaptive: {
    label: "Adaptive batching",
    throughputMultiplier: 1.04,
    computeMultiplier: 0.97,
    overheadDelta: -0.01,
    queueDelayMs: 35,
    stabilityAdjustment: 3,
  },
  high: {
    label: "High batching",
    throughputMultiplier: 1.12,
    computeMultiplier: 0.93,
    overheadDelta: -0.015,
    queueDelayMs: 85,
    stabilityAdjustment: -2,
  },
};

const POWER_CAP_PROFILES = {
  disabled: {
    label: "Cap disabled",
    powerMultiplier: 1.08,
    throughputMultiplier: 1.04,
    lagMultiplier: 1.08,
    overheadDelta: 0.012,
    stabilityAdjustment: -4,
    capW: 0,
  },
  balanced: {
    label: "Balanced cap",
    powerMultiplier: 1.0,
    throughputMultiplier: 1.0,
    lagMultiplier: 1.0,
    overheadDelta: 0,
    stabilityAdjustment: 1,
    capW: 420,
  },
  strict: {
    label: "Strict cap",
    powerMultiplier: 0.91,
    throughputMultiplier: 0.93,
    lagMultiplier: 0.94,
    overheadDelta: -0.01,
    stabilityAdjustment: 5,
    capW: 360,
  },
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function classifyPressure(score) {
  if (score >= 2) return "HIGH";
  if (score >= 1) return "MEDIUM";
  return "LOW";
}

function getBaseServiceTime(tokens = 0, strategy = "reactive") {
  const base = 520 + tokens * 5.4;
  return strategy === "coordinated" ? base * 0.96 : base;
}

export function buildCapacityPlan({
  controls = DEFAULT_CAPACITY_CONTROLS,
  tokens = 0,
  strategy = "reactive",
} = {}) {
  const headroomPct = clamp(Number(controls.headroomPct ?? 18), 8, 35);
  const oversubscription =
    OVERSUBSCRIPTION_PROFILES[controls.oversubscriptionMode] ??
    OVERSUBSCRIPTION_PROFILES.guarded;
  const batching =
    BATCHING_PROFILES[controls.batchingMode] ?? BATCHING_PROFILES.adaptive;
  const powerCap =
    POWER_CAP_PROFILES[controls.powerCapMode] ?? POWER_CAP_PROFILES.balanced;

  const headroomFactor = (20 - headroomPct) / 100;
  const throughputMultiplier = clamp(
    oversubscription.throughputMultiplier *
      batching.throughputMultiplier *
      powerCap.throughputMultiplier *
      (1 + headroomFactor * 0.4),
    0.78,
    1.35
  );
  const computeMultiplier = clamp(
    oversubscription.computeMultiplier * batching.computeMultiplier,
    0.88,
    1.12
  );
  const powerMultiplier = clamp(
    powerCap.powerMultiplier * (1 + headroomFactor * 0.25),
    0.84,
    1.12
  );
  const lagMultiplier = clamp(
    oversubscription.lagMultiplier *
      powerCap.lagMultiplier *
      (1 + Math.max(0, 18 - headroomPct) * 0.012),
    0.85,
    1.28
  );
  const overheadDelta = clamp(
    batching.overheadDelta +
      powerCap.overheadDelta +
      Math.max(0, 16 - headroomPct) * 0.0025,
    -0.03,
    0.05
  );

  const queueDelayMs = Math.round(
    oversubscription.queueDelayMs +
      batching.queueDelayMs +
      Math.max(0, 16 - headroomPct) * 7
  );
  const baseServiceMs = getBaseServiceTime(tokens, strategy);
  const effectiveServiceMs = Math.max(
    300,
    Math.round(baseServiceMs / throughputMultiplier + queueDelayMs)
  );
  const throughputRps = Number((1000 / effectiveServiceMs).toFixed(3));

  const utilizationPct = clamp(
    100 - headroomPct + oversubscription.utilizationBoost,
    52,
    98
  );
  const oversubscriptionRatio = Number(
    (utilizationPct / Math.max(10, 100 - headroomPct)).toFixed(2)
  );
  const stabilityAdjustment =
    oversubscription.stabilityAdjustment +
    batching.stabilityAdjustment +
    powerCap.stabilityAdjustment +
    (headroomPct >= 22 ? 3 : headroomPct <= 12 ? -4 : 0);

  const pressureScore =
    (utilizationPct >= 90 ? 1 : 0) +
    (queueDelayMs >= 120 ? 1 : 0) +
    oversubscription.pressureBias;
  const capacityPressure = classifyPressure(pressureScore);

  const recommendations = [
    headroomPct < 14
      ? "Increase reserved headroom before admission spikes hit the rack."
      : "Current headroom reserve is adequate for interactive inference bursts.",
    oversubscriptionRatio > 1.08
      ? "Oversubscription is pushing queue delay upward; use guarded mode for latency-sensitive traffic."
      : "Oversubscription remains within a manageable operating band.",
    powerCap.capW > 0
      ? `Power cap holds the node near ${powerCap.capW} W to smooth thermal excursions.`
      : "Power cap is disabled, so peak throughput comes with higher thermal volatility.",
  ];

  return {
    controls: {
      headroomPct,
      oversubscriptionMode: controls.oversubscriptionMode,
      batchingMode: controls.batchingMode,
      powerCapMode: controls.powerCapMode,
    },
    labels: {
      oversubscription: oversubscription.label,
      batching: batching.label,
      powerCap: powerCap.label,
    },
    throughputMultiplier,
    computeMultiplier,
    powerMultiplier,
    lagMultiplier,
    overheadDelta,
    queueDelayMs,
    throughputRps,
    utilizationPct,
    oversubscriptionRatio,
    stabilityAdjustment,
    alertBias: oversubscription.alertBias,
    capacityPressure,
    powerCapW: powerCap.capW,
    recommendations,
  };
}

export function buildCapacityAssessment({
  capacityPlan,
  activeRun,
  activeMetrics,
  traceSource = "simulation",
} = {}) {
  if (!capacityPlan) return null;

  const throughputRps = Number(
    activeMetrics?.throughputRps ?? capacityPlan.throughputRps ?? 0
  );
  const queueDelayMs = Number(
    activeMetrics?.queueDelayMs ?? capacityPlan.queueDelayMs ?? 0
  );
  const utilizationPct = Number(
    activeMetrics?.utilizationPct ?? capacityPlan.utilizationPct ?? 0
  );
  const projectedStability = clamp(
    Number(activeRun?.stabilityScore ?? 72) + Number(capacityPlan.stabilityAdjustment ?? 0),
    0,
    100
  );
  const energyDeltaPct = Number(
    (((capacityPlan.computeMultiplier ?? 1) - 1 + (capacityPlan.overheadDelta ?? 0)) * 100).toFixed(1)
  );

  return {
    mode: traceSource === "telemetry" ? "Projected from telemetry context" : "Applied to simulation",
    throughputRps,
    queueDelayMs,
    utilizationPct,
    projectedStability,
    capacityPressure: capacityPlan.capacityPressure,
    oversubscriptionRatio: capacityPlan.oversubscriptionRatio,
    powerCapW: capacityPlan.powerCapW,
    energyDeltaPct,
    recommendations: capacityPlan.recommendations,
  };
}
