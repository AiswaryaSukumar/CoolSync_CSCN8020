import { GRID_SIGNAL_PRESETS } from "./sustainabilitySignals";

function bestAlternative(activeId) {
  return GRID_SIGNAL_PRESETS.filter((preset) => preset.id !== activeId).sort((a, b) => a.g - b.g)[0];
}

export function buildRoutingRecommendation({
  sustainabilitySignals,
  activeRun,
  workloadFlexibility = "urgent",
  strategy = "reactive",
}) {
  const activePreset =
    GRID_SIGNAL_PRESETS.find((preset) => preset.region === sustainabilitySignals.region) ??
    GRID_SIGNAL_PRESETS[1];
  const alternative = bestAlternative(activePreset.id);
  const currentWh = Number(activeRun?.whPerRequest ?? 0);
  const alternativeCo2e = Number(((currentWh / 1000) * Number(alternative?.g ?? sustainabilitySignals.currentCarbon)).toFixed(2));
  const currentCo2e = Number(sustainabilitySignals.currentCo2e ?? 0);
  const routeSavings = Math.max(0, Number((currentCo2e - alternativeCo2e).toFixed(2)));
  const deferSavings = Math.max(
    0,
    Number((currentCo2e - Number(sustainabilitySignals.deferredCo2e ?? currentCo2e)).toFixed(2))
  );

  let recommendation = "Run in current region";
  let recommendationLevel = "LOW";
  let action = "Proceed now";

  if (workloadFlexibility === "flexible" && sustainabilitySignals.carbonRisk === "HIGH") {
    recommendation = `Delay workload for cleaner window in ${sustainabilitySignals.region}`;
    recommendationLevel = "HIGH";
    action = "Defer";
  } else if (routeSavings > 0.2 && sustainabilitySignals.carbonRisk !== "LOW") {
    recommendation = `Route workload to ${alternative?.region ?? "lower-carbon region"}`;
    recommendationLevel = "MEDIUM";
    action = "Route";
  } else if (sustainabilitySignals.waterRisk === "HIGH" && workloadFlexibility === "flexible") {
    recommendation = "Avoid water-stressed site for flexible traffic";
    recommendationLevel = "MEDIUM";
    action = "Rebalance";
  }

  if (strategy === "reactive" && recommendationLevel !== "LOW") {
    recommendation = `${recommendation} and prefer coordinated cooling`;
  }

  const rationale = [
    `Current region intensity is ${sustainabilitySignals.currentCarbon} gCO2/kWh with ${sustainabilitySignals.waterIntensity.toLowerCase()} water stress.`,
    `Deferring this request could reduce request emissions by ${deferSavings.toFixed(2)} gCO2e.`,
    `Routing to ${alternative?.region ?? "the best available region"} would save approximately ${routeSavings.toFixed(2)} gCO2e per request.`,
  ];

  if (workloadFlexibility === "urgent") {
    rationale.push("Urgent workload classification limits the option to defer execution.");
  } else {
    rationale.push("Flexible workload classification enables routing and time-shifting recommendations.");
  }

  return {
    action,
    recommendation,
    recommendationLevel,
    currentRegion: sustainabilitySignals.region,
    alternativeRegion: alternative?.region ?? sustainabilitySignals.region,
    currentCo2e,
    alternativeCo2e,
    deferredCo2e: Number(sustainabilitySignals.deferredCo2e ?? currentCo2e),
    routeSavings,
    deferSavings,
    waterRisk: sustainabilitySignals.waterRisk,
    carbonRisk: sustainabilitySignals.carbonRisk,
    rationale,
  };
}
