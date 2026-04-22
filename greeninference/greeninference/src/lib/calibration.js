function safeDivide(numerator, denominator, fallback = 1) {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) {
    return fallback;
  }

  return numerator / denominator;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function pctError(estimated = 0, observed = 0) {
  if (!Number.isFinite(observed) || observed === 0) return 0;
  return Number((((estimated - observed) / observed) * 100).toFixed(1));
}

function absPctError(estimated = 0, observed = 0) {
  return Math.abs(pctError(estimated, observed));
}

function getCalibrationConfidence({
  telemetryRows = 0,
  hasFacilityMetering = false,
  hasComponentBreakdown = false,
  beforeError = 0,
  afterError = 0,
}) {
  const improvement = Math.max(0, beforeError - afterError);
  let score = 0;

  if (telemetryRows >= 20) score += 2;
  else if (telemetryRows >= 10) score += 1;

  if (hasFacilityMetering) score += 1;
  if (hasComponentBreakdown) score += 1;
  if (improvement >= 12) score += 2;
  else if (improvement >= 5) score += 1;

  if (score >= 4) return "HIGH";
  if (score >= 2) return "MEDIUM";
  return "LOW";
}

export function getDefaultCalibrationState(model = "LLM-70B (Dense)") {
  const seededFactorsByModel = {
    "Small Model": {
      modelWhMultiplier: 0.96,
      prefillMultiplier: 1.04,
      decodeMultiplier: 0.93,
      siteOverheadMultiplier: 1.05,
    },
    "Medium Model": {
      modelWhMultiplier: 1.03,
      prefillMultiplier: 1.08,
      decodeMultiplier: 0.95,
      siteOverheadMultiplier: 1.07,
    },
    "LLM-70B (Dense)": {
      modelWhMultiplier: 1.08,
      prefillMultiplier: 1.12,
      decodeMultiplier: 0.92,
      siteOverheadMultiplier: 1.1,
    },
    "MoE Model": {
      modelWhMultiplier: 1.02,
      prefillMultiplier: 1.09,
      decodeMultiplier: 0.9,
      siteOverheadMultiplier: 1.06,
    },
  };
  const seededFactors =
    seededFactorsByModel[model] ?? seededFactorsByModel["LLM-70B (Dense)"];

  return {
    status: "SEEDED",
    confidence: "STARTING VALUES",
    model,
    factors: seededFactors,
    baseline: {
      estimatedWhPerRequest: 0,
      observedWhPerRequest: 0,
      estimatedJPerToken: 0,
      observedJPerToken: 0,
      estimatedOverhead: 0,
      observedOverhead: 0,
    },
    errors: {
      beforePct: 0,
      afterPct: 0,
      beforeJPerTokenPct: 0,
      afterJPerTokenPct: 0,
      beforeOverheadPct: 0,
      afterOverheadPct: 0,
    },
    residuals: {
      whPerRequest: 0,
      jPerToken: 0,
      overhead: 0,
    },
    notes: [
      `These starting values are shown for ${model} before any uploaded data is used.`,
      "Upload data to replace these starting values with site-specific adjustments.",
    ],
  };
}

export function calibrateRunAgainstTelemetry({
  model = "LLM-70B (Dense)",
  estimatedRun,
  observedRun,
  estimatedMetrics,
  observedMetrics,
  telemetryMeta = {},
}) {
  if (!estimatedRun || !observedRun || !estimatedMetrics || !observedMetrics) {
    return getDefaultCalibrationState(model);
  }

  const estimatedPhases = estimatedMetrics.phaseEnergyWh ?? {};
  const observedPhases = observedMetrics.phaseEnergyWh ?? {};

  const modelWhMultiplier = clamp(
    safeDivide(observedMetrics.baseComputeWh, estimatedMetrics.baseComputeWh, 1),
    0.65,
    1.7
  );
  const prefillMultiplier = clamp(
    safeDivide(observedPhases.prefillComputeWh, estimatedPhases.prefillComputeWh, modelWhMultiplier),
    0.7,
    1.8
  );
  const decodeMultiplier = clamp(
    safeDivide(observedPhases.decodeComputeWh, estimatedPhases.decodeComputeWh, modelWhMultiplier),
    0.7,
    1.8
  );
  const siteOverheadMultiplier = clamp(
    safeDivide(observedMetrics.overheadCurrentMeasured, estimatedMetrics.overheadCurrent, 1),
    0.6,
    1.9
  );

  const calibratedWhPerRequest =
    estimatedMetrics.baseComputeWh * modelWhMultiplier * (1 + estimatedMetrics.overheadCurrent * siteOverheadMultiplier);
  const calibratedJPerToken =
    observedRun.tokens > 0 ? Number(((calibratedWhPerRequest * 3600) / observedRun.tokens).toFixed(3)) : 0;
  const calibratedOverhead = estimatedMetrics.overheadCurrent * siteOverheadMultiplier;

  const beforeError = absPctError(estimatedRun.whPerRequest, observedRun.whPerRequest);
  const afterError = absPctError(calibratedWhPerRequest, observedRun.whPerRequest);
  const beforeJError = absPctError(estimatedRun.jPerToken, observedRun.jPerToken);
  const afterJError = absPctError(calibratedJPerToken, observedRun.jPerToken);
  const beforeOverheadError = absPctError(
    estimatedMetrics.overheadCurrent,
    observedMetrics.overheadCurrentMeasured
  );
  const afterOverheadError = absPctError(calibratedOverhead, observedMetrics.overheadCurrentMeasured);

  const confidence = getCalibrationConfidence({
    telemetryRows: telemetryMeta.rows ?? 0,
    hasFacilityMetering: telemetryMeta.hasFacilityMetering ?? false,
    hasComponentBreakdown: telemetryMeta.hasComponentBreakdown ?? false,
    beforeError,
    afterError,
  });

  return {
    status: "CALIBRATED",
    confidence,
    model,
    factors: {
      modelWhMultiplier: Number(modelWhMultiplier.toFixed(3)),
      prefillMultiplier: Number(prefillMultiplier.toFixed(3)),
      decodeMultiplier: Number(decodeMultiplier.toFixed(3)),
      siteOverheadMultiplier: Number(siteOverheadMultiplier.toFixed(3)),
    },
    baseline: {
      estimatedWhPerRequest: Number(estimatedRun.whPerRequest.toFixed(3)),
      observedWhPerRequest: Number(observedRun.whPerRequest.toFixed(3)),
      estimatedJPerToken: Number(estimatedRun.jPerToken.toFixed(3)),
      observedJPerToken: Number(observedRun.jPerToken.toFixed(3)),
      estimatedOverhead: Number((estimatedMetrics.overheadCurrent ?? 0).toFixed(3)),
      observedOverhead: Number((observedMetrics.overheadCurrentMeasured ?? 0).toFixed(3)),
    },
    errors: {
      beforePct: beforeError,
      afterPct: afterError,
      beforeJPerTokenPct: beforeJError,
      afterJPerTokenPct: afterJError,
      beforeOverheadPct: beforeOverheadError,
      afterOverheadPct: afterOverheadError,
    },
    residuals: {
      whPerRequest: Number((observedRun.whPerRequest - calibratedWhPerRequest).toFixed(3)),
      jPerToken: Number((observedRun.jPerToken - calibratedJPerToken).toFixed(3)),
      overhead: Number((observedMetrics.overheadCurrentMeasured - calibratedOverhead).toFixed(3)),
    },
    notes: [
      `The model energy value changed to ${modelWhMultiplier.toFixed(2)}x for ${model}.`,
      `The start phase changed to ${prefillMultiplier.toFixed(2)}x and the output phase changed to ${decodeMultiplier.toFixed(2)}x to better match the uploaded data.`,
      `The building power value changed to ${siteOverheadMultiplier.toFixed(2)}x to better match the uploaded data.`,
    ],
  };
}
