import { estimateFacilityOverhead } from "./overheadModel";
import {
  buildPhysicsRun,
  buildConservativeRun,
  PHYS,
} from "./physics";

const USE_PHYSICS_ENGINE = true;

export function estimateTokensFromText(text) {
  const words = (text || "").trim().split(/\s+/).filter(Boolean).length;
  return Math.max(0, Math.round(words * 1.3));
}

export function complexityClass(tokens) {
  if (tokens < 64) return "LOW";
  if (tokens < 256) return "MEDIUM";
  if (tokens < 800) return "HIGH";
  return "EXTREME";
}

export function classifyPromptRisk(tokens = 0, model = "LLM-70B (Dense)") {
  const modelRiskMap = {
    "Small Model": 0.85,
    "Medium Model": 1.0,
    "LLM-70B (Dense)": 1.2,
    "MoE Model": 1.1,
  };

  const weighted = tokens * (modelRiskMap[model] ?? 1);

  if (weighted < 120) return "LOW";
  if (weighted < 320) return "MEDIUM";
  return "HIGH";
}

export function estimateWhPerRequest(
  tokens = 0,
  model = "LLM-70B (Dense)",
  overhead = 0.37,
  calibration = {}
) {
  const whPerTokenMap = {
    "Small Model": 0.0002,
    "Medium Model": 0.0006,
    "LLM-70B (Dense)": 0.0012,
    "MoE Model": 0.0008,
  };

  const whPerToken = (whPerTokenMap[model] ?? 0.001) * (calibration.modelWhMultiplier ?? 1);
  const baseComputeWh = tokens * whPerToken;
  const totalWh = baseComputeWh * (1 + overhead * (calibration.siteOverheadMultiplier ?? 1));

  return {
    whPerToken,
    baseComputeWh: Number(baseComputeWh.toFixed(3)),
    totalWh: Number(totalWh.toFixed(3)),
  };
}

export function estimateJPerToken(totalWh = 0, tokens = 0) {
  if (!tokens) return 0;
  const joules = totalWh * 3600;
  return Number((joules / tokens).toFixed(3));
}

function average(values = []) {
  if (!values.length) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function formatAlertTime(baseTime, offsetMs) {
  return new Date(baseTime.getTime() + offsetMs).toTimeString().slice(0, 8);
}

function safePctReduction(baseline = 0, next = 0) {
  if (!baseline) return 0;
  return Number((((baseline - next) / baseline) * 100).toFixed(1));
}

function safePctIncrease(baseline = 0, next = 0) {
  if (!baseline) return 0;
  return Number((((next - baseline) / baseline) * 100).toFixed(1));
}

function classifyLevel(value, { medium = 0, high = 0 }) {
  if (value >= high) return "HIGH";
  if (value >= medium) return "MEDIUM";
  return "LOW";
}

export function detectCoordinationSignals({
  tokens = 0,
  model = "LLM-70B (Dense)",
  points = [],
  lagMs = 0,
}) {
  const promptRisk = classifyPromptRisk(tokens, model);

  const peakGpuW = Math.max(...points.map((p) => Number(p.gpuW) || 0), 0);
  const peakInlet = Math.max(...points.map((p) => Number(p.inlet) || 0), 0);

  const earlyWindow = points.filter((p) => Number(p.t) <= 350);
  const inletStart = Number(points[0]?.inlet ?? 24.2);
  const inletEarlyPeak = Math.max(...earlyWindow.map((p) => Number(p.inlet) || 0), inletStart);
  const inletRise = Math.max(0, inletEarlyPeak - inletStart);

  // Thresholds account for physics-based values (rack W) or heuristic (GPU W)
  const isPhysicsScale = peakGpuW > 1000;
  const spikeRisk = classifyLevel(peakGpuW, isPhysicsScale
    ? { medium: 15_000, high: 25_000 }
    : { medium: 320, high: 400 }
  );

  const inletRisk = classifyLevel(inletRise, {
    medium: 0.35,
    high: 0.7,
  });

  const lagSeverity = classifyLevel(lagMs, {
    medium: 60_000,
    high: 120_000,
  });

  const coordinationGap =
    (spikeRisk === "HIGH" || promptRisk === "HIGH") &&
    (lagSeverity === "HIGH" || inletRisk !== "LOW");

  return {
    promptRisk,
    peakGpuW: Number(peakGpuW.toFixed(1)),
    peakInlet: Number(peakInlet.toFixed(2)),
    inletRise: Number(inletRise.toFixed(2)),
    spikeRisk,
    inletRisk,
    lagSeverity,
    coordinationGap,
  };
}

export function buildDecisionSupport({
  strategy = "reactive",
  tokens = 0,
  model = "LLM-70B (Dense)",
  points = [],
  lagMs = 0,
}) {
  const signals = detectCoordinationSignals({
    tokens,
    model,
    points,
    lagMs,
  });

  const rationale = [];
  const detectedSignals = [];

  if (signals.spikeRisk !== "LOW") {
    detectedSignals.push(`Fast start risk: ${signals.spikeRisk}`);
    rationale.push(
      "Power rises quickly at the start, so the task begins with a strong burst of work."
    );
  }

  if (signals.inletRisk !== "LOW") {
    detectedSignals.push(`Temperature rise risk: ${signals.inletRisk}`);
    rationale.push(
      "Temperature rises early, which suggests heat is building before cooling fully catches up."
    );
  }

  if (signals.lagSeverity !== "LOW") {
    detectedSignals.push(`Cooling delay: ${signals.lagSeverity}`);
    rationale.push(
      `Cooling responds ${signals.lagSeverity.toLowerCase()} slowly, which increases the chance of a late response.`
    );
  }

  if (signals.coordinationGap) {
    detectedSignals.push("Earlier cooling may help");
    rationale.push(
      "The work rises quickly and cooling reacts late, so earlier cooling would likely help."
    );
  }

  let recommendation = "Keep the current setting";
  let recommendationLevel = "LOW";

  if (signals.coordinationGap && strategy === "reactive") {
    recommendation = "Switch to coordinated mode now";
    recommendationLevel = "HIGH";
    rationale.push(
      "The current mode is likely to react after heat has already started rising, so earlier cooling is recommended."
    );
  } else if (
    strategy === "coordinated" &&
    (signals.spikeRisk !== "LOW" || signals.lagSeverity !== "LOW")
  ) {
    recommendation = "Coordinated mode is a good fit";
    recommendationLevel = "MEDIUM";
    rationale.push(
      "This mode matches the current workload better and helps reduce late cooling and temperature overshoot."
    );
  } else if (signals.lagSeverity === "MEDIUM") {
    recommendation = "Watch the cooling response";
    recommendationLevel = "MEDIUM";
    rationale.push(
      "The current conditions are manageable, but cooling timing should still be watched closely."
    );
  }

  if (!detectedSignals.length) {
    detectedSignals.push("No major warning signs");
    rationale.push("This task stays in a lower-risk range under the current assumptions.");
  }

  return {
    ...signals,
    detectedSignals,
    recommendation,
    recommendationLevel,
    rationale,
  };
}

export function buildRunSummary({
  strategy = "reactive",
  source = "simulation",
  tokens = 0,
  model = "LLM-70B (Dense)",
  points = [],
  metrics = {},
  lagMs = 0,
  alerts = [],
}) {
  const complexity = complexityClass(tokens);
  const promptRisk = classifyPromptRisk(tokens, model);

  const peakGpuW = Math.max(...points.map((p) => Number(p.gpuW) || 0), 0);
  const peakInlet = Math.max(...points.map((p) => Number(p.inlet) || 0), 0);
  const avgCoolingKw = average(points.map((p) => Number(p.coolingKw) || 0));
  const minInlet = Math.min(...points.map((p) => Number(p.inlet) || 0), peakInlet || 0);

  const thresholdBreaches = points.filter((p) => Number(p.inlet) >= 25.0).length;
  const tempSpread = Math.max(0, peakInlet - minInlet);

  const stabilityScoreRaw =
    100 -
    lagMs / 14 -
    thresholdBreaches * 2.8 -
    tempSpread * 8 +
    (strategy === "coordinated" ? 6 : 0);

  const stabilityScore = Math.max(0, Math.min(100, Number(stabilityScoreRaw.toFixed(1))));
  const alertCount = alerts.length;

  const whPerRequest =
    strategy === "coordinated"
      ? Number(metrics.totalWhTarget ?? 0)
      : Number(metrics.totalWhCurrent ?? 0);

  const co2ePerRequest =
    strategy === "coordinated"
      ? Number(metrics.co2e_g_target ?? 0)
      : Number(metrics.co2e_g_current ?? 0);

  const jPerToken = estimateJPerToken(whPerRequest, tokens);

  return {
    source,
    strategy,
    tokens,
    complexity,
    promptRisk,
    peakGpuW: Number(peakGpuW.toFixed(1)),
    peakInlet: Number(peakInlet.toFixed(2)),
    avgCoolingKw: Number(avgCoolingKw.toFixed(2)),
    alertCount,
    lagMs: Number(lagMs.toFixed(0)),
    thresholdBreaches,
    stabilityScore,
    whPerRequest,
    jPerToken,
    co2ePerRequest: Number(co2ePerRequest.toFixed(2)),
    isEstimated: source === "simulation",
  };
}

export function buildObservedRunSummary({
  strategy = "reactive",
  source = "telemetry",
  tokens = 0,
  model = "LLM-70B (Dense)",
  points = [],
  metrics = {},
  lagMs = 0,
  alerts = [],
}) {
  return buildRunSummary({
    strategy,
    source,
    tokens,
    model,
    points,
    metrics,
    lagMs,
    alerts,
  });
}

export function estimateLagFromPoints(points = []) {
  const normalizedPoints = [...points]
    .filter((p) => Number.isFinite(Number(p?.t)))
    .sort((a, b) => Number(a.t) - Number(b.t));

  if (normalizedPoints.length < 2) return 0;

  const baselineCooling = average(
    normalizedPoints
      .filter((p) => Number(p.t) <= 200)
      .map((p) => Number(p.coolingKw) || 0)
  );

  const peakGpuPoint = normalizedPoints.reduce(
    (peak, point) => ((Number(point.gpuW) || 0) > (Number(peak.gpuW) || 0) ? point : peak),
    normalizedPoints[0]
  );

  const adjustmentPoint = normalizedPoints.find(
    (point) =>
      Number(point.t) >= Number(peakGpuPoint.t) &&
      (Number(point.coolingKw) || 0) >= baselineCooling + 0.4
  );

  if (!adjustmentPoint) return 0;
  return Math.max(0, Number(adjustmentPoint.t) - Number(peakGpuPoint.t));
}

export function buildTelemetryAlerts({
  strategy = "reactive",
  points = [],
  lagMs = 0,
  traceLabel = "Observed telemetry trace",
}) {
  if (!points.length) return [];

  const baseTime = new Date();
  const firstTime = Number(points[0]?.t ?? 0);
  const maxGpu = Math.max(...points.map((p) => Number(p.gpuW) || 0), 0);
  const peakGpuPoint = points.find((p) => Number(p.gpuW) === maxGpu) ?? points[0];
  const warningPoint = points.find((p) => Number(p.inlet) >= 25.6);
  const coolingChangePoint = points.find(
    (p) => Number(p.t) >= Number(peakGpuPoint.t) && (Number(p.coolingKw) || 0) > 18.3
  );
  const lastPoint = points[points.length - 1];

  const alerts = [
    `[${formatAlertTime(baseTime, 0)}] ${traceLabel} loaded for ${strategy} review.`,
    `[${formatAlertTime(baseTime, Number(peakGpuPoint.t) - firstTime)}] Highest compute power seen: ${Math.round(Number(peakGpuPoint.gpuW) || 0)} W.`,
  ];

  if (coolingChangePoint) {
    alerts.push(
      `[${formatAlertTime(baseTime, Number(coolingChangePoint.t) - firstTime)}] Cooling response observed ${Math.round(lagMs)} ms after the spike window.`
    );
  }

  if (warningPoint) {
    alerts.push(
      `[${formatAlertTime(baseTime, Number(warningPoint.t) - firstTime)}] Inlet crossed the warning band at ${Number(warningPoint.inlet).toFixed(2)} C.`
    );
  }

  alerts.push(
    `[${formatAlertTime(baseTime, Number(lastPoint?.t ?? 0) - firstTime)}] Uploaded data is ready for review and comparison.`
  );

  return alerts;
}

export function compareRuns(baselineRun, coordinatedRun) {
  if (!baselineRun || !coordinatedRun) {
    return {
      peakInletReductionPct: 0,
      lagReductionPct: 0,
      alertReductionPct: 0,
      coolingReductionPct: 0,
      stabilityImprovementPct: 0,
      thresholdReductionPct: 0,
      summary: [],
    };
  }

  const peakInletReductionPct = safePctReduction(
    baselineRun.peakInlet,
    coordinatedRun.peakInlet
  );
  const lagReductionPct = safePctReduction(baselineRun.lagMs, coordinatedRun.lagMs);
  const alertReductionPct = safePctReduction(
    baselineRun.alertCount,
    coordinatedRun.alertCount
  );
  const coolingReductionPct = safePctReduction(
    baselineRun.avgCoolingKw,
    coordinatedRun.avgCoolingKw
  );
  const thresholdReductionPct = safePctReduction(
    baselineRun.thresholdBreaches,
    coordinatedRun.thresholdBreaches
  );
  const stabilityImprovementPct = safePctIncrease(
    baselineRun.stabilityScore,
    coordinatedRun.stabilityScore
  );

  const summary = [
    `Peak inlet temperature dropped by ${peakInletReductionPct.toFixed(1)}% because cooling started earlier.`,
    `Cooling delay fell by ${lagReductionPct.toFixed(1)}%, so the system reacted sooner during the early spike.`,
    `Temperature stability improved by ${stabilityImprovementPct.toFixed(1)}% with smoother cooling timing.`,
  ];

  if (alertReductionPct > 0) {
    summary.push(`Warnings dropped by ${alertReductionPct.toFixed(1)}% in coordinated mode.`);
  }

  if (thresholdReductionPct > 0) {
    summary.push(
      `Warning-limit crossings fell by ${thresholdReductionPct.toFixed(1)}%, which lowers heat stress.`
    );
  }

  return {
    peakInletReductionPct,
    lagReductionPct,
    alertReductionPct,
    coolingReductionPct,
    stabilityImprovementPct,
    thresholdReductionPct,
    summary,
  };
}

export function buildComparisonNarrative(comparison = {}) {
  const summary = [];

  if (Number(comparison.peakInletReductionPct) > 0) {
    summary.push(
      `Peak inlet temperature dropped by ${Number(comparison.peakInletReductionPct).toFixed(1)}%, keeping the system farther from the warning band.`
    );
  }

  if (Number(comparison.lagReductionPct) > 0) {
    summary.push(
      `Cooling delay improved by ${Number(comparison.lagReductionPct).toFixed(1)}%, so cooling starts earlier during the fast rise at the start.`
    );
  }

  if (Number(comparison.alertReductionPct) > 0) {
    summary.push(
      `Warnings fell by ${Number(comparison.alertReductionPct).toFixed(1)}%, showing a more stable heat response.`
    );
  }

  if (Number(comparison.coolingReductionPct) > 0) {
    summary.push(
      `Average cooling demand dropped by ${Number(comparison.coolingReductionPct).toFixed(1)}%, which suggests less extra facility load.`
    );
  }

  if (Number(comparison.stabilityImprovementPct) > 0) {
    summary.push(
      `Overall stability improved by ${Number(comparison.stabilityImprovementPct).toFixed(1)}% with coordinated cooling.`
    );
  }

  return summary.slice(0, 5);
}

export function runSimulation({
  tokens = 0,
  model = "LLM-70B (Dense)",
  grid_g_per_kWh = 110,
  overheadCurrent = 0.37,
  overheadTarget = 0.27,
  strategy = "reactive",
  calibration = null,
  capacityPlan = null,
}) {
  const activeCalibration  = calibration?.factors ?? {};
  const activeCapacityPlan = capacityPlan ?? {};

  // ── Physics-based simulation ────────────────────────────────────────────────
  const physRun = USE_PHYSICS_ENGINE
    ? buildPhysicsRun({ tokens, strategy })
    : null;

  const points = physRun?.points ?? [];
  const lagMs  = physRun?.lagMs  ?? (strategy === "coordinated" ? 0 : PHYS.DEAD_STEPS * 60_000);

  // ── Energy & carbon bookkeeping (token-level model, kept for overhead panel) ─
  const effectiveOverheadCurrent = Math.max(
    0.12,
    overheadCurrent + (activeCapacityPlan.overheadDelta ?? 0)
  );
  const effectiveOverheadTarget = Math.max(
    0.1,
    overheadTarget + (activeCapacityPlan.overheadDelta ?? 0) * 0.7
  );
  const effectiveCalibration = {
    ...activeCalibration,
    modelWhMultiplier:
      (activeCalibration.modelWhMultiplier ?? 1) *
      (activeCapacityPlan.computeMultiplier ?? 1),
  };

  const whEstCurrent = estimateWhPerRequest(tokens, model, effectiveOverheadCurrent, effectiveCalibration);
  const whEstTarget  = estimateWhPerRequest(tokens, model, effectiveOverheadTarget,  effectiveCalibration);

  const whPerToken      = whEstCurrent.whPerToken;
  const baseComputeWh   = whEstCurrent.baseComputeWh;
  const totalWhCurrent  = USE_PHYSICS_ENGINE
    ? Number((physRun.energyWh).toFixed(3))
    : whEstCurrent.totalWh;
  const totalWhTarget   = whEstTarget.totalWh;
  const savedWh         = Math.max(0, totalWhCurrent - totalWhTarget);

  const heuristicOverheadCurrent = estimateFacilityOverhead({
    computeWh: baseComputeWh,
    overheadRatio: effectiveOverheadCurrent,
    strategy,
    capacityPlan: activeCapacityPlan,
  });
  const heuristicOverheadTarget = estimateFacilityOverhead({
    computeWh: whEstTarget.baseComputeWh,
    overheadRatio: effectiveOverheadTarget,
    strategy: "coordinated",
    capacityPlan: activeCapacityPlan,
  });

  const co2e_g_current = (totalWhCurrent / 1000) * grid_g_per_kWh;
  const co2e_g_target  = (totalWhTarget  / 1000) * grid_g_per_kWh;
  const savedCo2e_g    = Math.max(0, co2e_g_current - co2e_g_target);

  // ── Alerts ──────────────────────────────────────────────────────────────────
  const isCoordinated = strategy === "coordinated";
  const alerts = [];
  const now    = new Date();
  const stamp  = (ms) => new Date(now.getTime() + ms).toTimeString().slice(0, 8);
  const peakT  = physRun?.peakT ?? 0;

  alerts.push(`[${stamp(0)}] Strategy selected → ${isCoordinated ? "Coordinated (LSTM pre-cooling)" : "Reactive PID"}.`);
  alerts.push(`[${stamp(20)}] System pressure: ${activeCapacityPlan.capacityPressure ?? "LOW"}, ${Math.round(activeCapacityPlan.utilizationPct ?? 0)}% compute use.`);
  if (peakT > PHYS.T_DANGER) {
    alerts.push(`[${stamp(PHYS.DEAD_STEPS * 60_000)}] Peak rack temperature ${peakT.toFixed(1)}°C exceeded ${PHYS.T_DANGER}°C threshold.`);
  }
  alerts.push(`[${stamp(8 * 60_000)}] Burst workload begins — heat rises above ${(PHYS.Q_BASE_W / 1000).toFixed(0)} kW baseline.`);
  if (isCoordinated) {
    alerts.push(`[${stamp(8 * 60_000 - PHYS.DEAD_STEPS * 60_000)}] LSTM predicted burst → pre-cooling activated ${PHYS.DEAD_STEPS} min early.`);
  } else {
    alerts.push(`[${stamp((8 + PHYS.DEAD_STEPS) * 60_000)}] Reactive PID responds after ${PHYS.DEAD_STEPS}-min dead-time lag.`);
  }
  if (Number(activeCapacityPlan.queueDelayMs ?? 0) > 0) {
    alerts.push(`[${stamp(2 * 60_000)}] Queue + batching adds ~${Math.round(activeCapacityPlan.queueDelayMs)} ms latency.`);
  }
  alerts.push(`[${stamp((points.length - 1) * 60_000)}] Simulation complete — ${(totalWhCurrent).toFixed(1)} Wh total cooling energy.`);

  // ── Metrics object ──────────────────────────────────────────────────────────
  const metrics = {
    tokens,
    whPerToken,
    baseComputeWh:          Number(baseComputeWh.toFixed(3)),
    totalWhCurrent:         Number(totalWhCurrent.toFixed(3)),
    totalWhTarget:          Number(totalWhTarget.toFixed(3)),
    savedWh:                Number(savedWh.toFixed(3)),
    itEnergyWh:             Number(heuristicOverheadCurrent.itEnergyWh.toFixed(3)),
    coolingEnergyWh:        Number(heuristicOverheadCurrent.coolingEnergyWh.toFixed(3)),
    powerDeliveryLossWh:    Number(heuristicOverheadCurrent.powerDeliveryLossWh.toFixed(3)),
    idleReserveWh:          Number(heuristicOverheadCurrent.idleReserveWh.toFixed(3)),
    totalFacilityEnergyWh:  Number(heuristicOverheadCurrent.totalFacilityEnergyWh.toFixed(3)),
    co2e_g_current:         Number(co2e_g_current.toFixed(2)),
    co2e_g_target:          Number(co2e_g_target.toFixed(2)),
    savedCo2e_g:            Number(savedCo2e_g.toFixed(2)),
    overheadCurrent:        Number(effectiveOverheadCurrent.toFixed(3)),
    overheadTarget:         Number(effectiveOverheadTarget.toFixed(3)),
    grid_g_per_kWh,
    throughputRps:          Number((activeCapacityPlan.throughputRps ?? 0).toFixed(3)),
    queueDelayMs:           Math.round(activeCapacityPlan.queueDelayMs ?? 0),
    utilizationPct:         Number((activeCapacityPlan.utilizationPct ?? 0).toFixed(1)),
    capacityPressure:       activeCapacityPlan.capacityPressure ?? "LOW",
    oversubscriptionRatio:  Number((activeCapacityPlan.oversubscriptionRatio ?? 1).toFixed(2)),
    powerCapW:              Number(activeCapacityPlan.powerCapW ?? 0),
    headroomPct:            Number(activeCapacityPlan.controls?.headroomPct ?? 0),
    oversubscriptionMode:   activeCapacityPlan.controls?.oversubscriptionMode ?? "guarded",
    batchingMode:           activeCapacityPlan.controls?.batchingMode ?? "adaptive",
    powerCapMode:           activeCapacityPlan.controls?.powerCapMode ?? "balanced",
    heuristicOverheadAttribution: heuristicOverheadCurrent,
    targetOverheadAttribution:    heuristicOverheadTarget,
    phaseEnergyWh: { prefillComputeWh: 0, decodeComputeWh: 0, baselineComputeWh: 0 },
    calibrationApplied: calibration?.status === "CALIBRATED",
    // Physics-specific extras
    peakTRack:    physRun?.peakT    ?? 0,
    breachSteps:  physRun?.breachSteps ?? 0,
  };

  const runSummary = buildRunSummary({
    strategy, source: "simulation", tokens, model, points, metrics, lagMs, alerts,
  });

  const decisionSupport = buildDecisionSupport({ strategy, tokens, model, points, lagMs });

  const pump     = isCoordinated ? 68 : 80;
  const valve    = isCoordinated ? "COORDINATED" : "DYNAMIC";

  return {
    strategy,
    points,
    lagMs,
    dclc: {
      liquidIn:   PHYS.T_SUPPLY,
      liquidOut:  isCoordinated ? 26.2 : 26.8,
      pumpSpeed:  pump,
      valveState: valve,
    },
    capacityPlan: activeCapacityPlan,
    metrics,
    alerts,
    runSummary,
    decisionSupport,
  };
}

// ─── Conservative simulation (PUE 1.56 baseline) ─────────────────────────────
export function runConservativeSimulation({
  tokens = 0,
  model = "LLM-70B (Dense)",
  grid_g_per_kWh = 110,
  overheadCurrent = 0.56,
  capacityPlan = null,
}) {
  const activeCapacityPlan = capacityPlan ?? {};
  const consRun = buildConservativeRun({ tokens });
  const points  = consRun.points;
  const lagMs   = consRun.lagMs;

  const effectiveOverhead = Math.max(0.12, overheadCurrent + (activeCapacityPlan.overheadDelta ?? 0));
  const whEst = estimateWhPerRequest(tokens, model, effectiveOverhead, {});
  const totalWhCurrent = Number(consRun.energyWh.toFixed(3));
  const co2e_g = (totalWhCurrent / 1000) * grid_g_per_kWh;

  const heuristicOverhead = estimateFacilityOverhead({
    computeWh: whEst.baseComputeWh,
    overheadRatio: effectiveOverhead,
    strategy: "reactive",
    capacityPlan: activeCapacityPlan,
  });

  const alerts = [`[${new Date().toTimeString().slice(0, 8)}] Conservative PID (T_sp=20°C, T_sup=16°C, PUE ~1.56).`];

  const metrics = {
    tokens,
    whPerToken: whEst.whPerToken,
    baseComputeWh:       Number(whEst.baseComputeWh.toFixed(3)),
    totalWhCurrent,
    totalWhTarget:       totalWhCurrent,
    savedWh:             0,
    itEnergyWh:          Number(heuristicOverhead.itEnergyWh.toFixed(3)),
    coolingEnergyWh:     Number(heuristicOverhead.coolingEnergyWh.toFixed(3)),
    powerDeliveryLossWh: Number(heuristicOverhead.powerDeliveryLossWh.toFixed(3)),
    idleReserveWh:       Number(heuristicOverhead.idleReserveWh.toFixed(3)),
    totalFacilityEnergyWh: Number(heuristicOverhead.totalFacilityEnergyWh.toFixed(3)),
    co2e_g_current:      Number(co2e_g.toFixed(2)),
    co2e_g_target:       Number(co2e_g.toFixed(2)),
    savedCo2e_g:         0,
    overheadCurrent:     Number(effectiveOverhead.toFixed(3)),
    overheadTarget:      Number(effectiveOverhead.toFixed(3)),
    grid_g_per_kWh,
    throughputRps: 0, queueDelayMs: 0, utilizationPct: 0,
    capacityPressure: "LOW", oversubscriptionRatio: 1, powerCapW: 0,
    headroomPct: 0, oversubscriptionMode: "guarded", batchingMode: "adaptive", powerCapMode: "balanced",
    heuristicOverheadAttribution: heuristicOverhead,
    targetOverheadAttribution:    heuristicOverhead,
    phaseEnergyWh: { prefillComputeWh: 0, decodeComputeWh: 0, baselineComputeWh: 0 },
    calibrationApplied: false,
    peakTRack:   consRun.peakT ?? 0,
    breachSteps: consRun.breachSteps ?? 0,
  };

  const runSummary    = buildRunSummary({ strategy: "reactive", source: "simulation", tokens, model, points, metrics, lagMs, alerts });
  const decisionSupport = buildDecisionSupport({ strategy: "reactive", tokens, model, points, lagMs });

  return {
    strategy: "conservative",
    points,
    lagMs,
    dclc: { liquidIn: 16.0, liquidOut: 25.0, pumpSpeed: 88, valveState: "CONSERVATIVE" },
    capacityPlan: activeCapacityPlan,
    metrics,
    alerts,
    runSummary,
    decisionSupport,
  };
}