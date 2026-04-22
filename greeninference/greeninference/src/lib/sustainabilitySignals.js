export const GRID_SIGNAL_PRESETS = [
  {
    id: "qc",
    label: "QC (low)",
    region: "Quebec",
    g: 50,
    forecast_g: 46,
    deferred_g: 42,
    waterIntensity: "LOW",
    waterIndex: 0.3,
    waterCost: "LOW",
    latencyClass: "LOW",
  },
  {
    id: "on",
    label: "ON (mid)",
    region: "Ontario",
    g: 110,
    forecast_g: 104,
    deferred_g: 92,
    waterIntensity: "MEDIUM",
    waterIndex: 0.5,
    waterCost: "MEDIUM",
    latencyClass: "LOW",
  },
  {
    id: "us_avg",
    label: "US Avg",
    region: "US Average",
    g: 380,
    forecast_g: 360,
    deferred_g: 310,
    waterIntensity: "MEDIUM",
    waterIndex: 0.58,
    waterCost: "MEDIUM",
    latencyClass: "MEDIUM",
  },
  {
    id: "ab",
    label: "AB (high)",
    region: "Alberta",
    g: 650,
    forecast_g: 610,
    deferred_g: 520,
    waterIntensity: "HIGH",
    waterIndex: 0.82,
    waterCost: "HIGH",
    latencyClass: "MEDIUM",
  },
];

function scoreBand(value, low, high) {
  if (value >= high) return "HIGH";
  if (value >= low) return "MEDIUM";
  return "LOW";
}

export function getGridSignalPreset(gridValue) {
  return GRID_SIGNAL_PRESETS.find((preset) => preset.g === gridValue) ?? GRID_SIGNAL_PRESETS[1];
}

export function buildSustainabilitySignals({ gridPreset, activeRun, workloadFlexibility = "urgent" }) {
  const currentCarbon = Number(gridPreset?.g ?? 0);
  const forecastCarbon = Number(gridPreset?.forecast_g ?? currentCarbon);
  const deferredCarbon = Number(gridPreset?.deferred_g ?? forecastCarbon);
  const whPerRequest = Number(activeRun?.whPerRequest ?? 0);
  const currentCo2e = (whPerRequest / 1000) * currentCarbon;
  const forecastCo2e = (whPerRequest / 1000) * forecastCarbon;
  const deferredCo2e = (whPerRequest / 1000) * deferredCarbon;

  return {
    region: gridPreset?.region ?? "Unknown",
    currentCarbon,
    forecastCarbon,
    deferredCarbon,
    currentCo2e: Number(currentCo2e.toFixed(2)),
    forecastCo2e: Number(forecastCo2e.toFixed(2)),
    deferredCo2e: Number(deferredCo2e.toFixed(2)),
    carbonRisk: scoreBand(currentCarbon, 150, 400),
    waterRisk: scoreBand(Number(gridPreset?.waterIndex ?? 0), 0.45, 0.75),
    waterIntensity: gridPreset?.waterIntensity ?? "MEDIUM",
    waterCost: gridPreset?.waterCost ?? "MEDIUM",
    latencyClass: gridPreset?.latencyClass ?? "LOW",
    workloadFlexibility,
    deferBenefitPct:
      currentCo2e > 0
        ? Number((((currentCo2e - deferredCo2e) / currentCo2e) * 100).toFixed(1))
        : 0,
  };
}
