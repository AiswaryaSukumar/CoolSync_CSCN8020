function safeRound(value, digits = 3) {
  if (!Number.isFinite(value)) return 0;
  return Number(value.toFixed(digits));
}

function safeRatio(part = 0, whole = 0) {
  if (!Number.isFinite(part) || !Number.isFinite(whole) || whole <= 0) return 0;
  return part / whole;
}

export function estimateFacilityOverhead({
  computeWh = 0,
  overheadRatio = 0.37,
  strategy = "reactive",
  capacityPlan = null,
} = {}) {
  const ratio = Math.max(0, Number(overheadRatio) || 0);
  const totalOverheadWh = Math.max(0, computeWh * ratio);
  const coordinatedBias = strategy === "coordinated" ? -0.02 : 0;
  const capacityBias = Number(capacityPlan?.overheadDelta ?? 0);

  const coolingShare = Math.min(0.72, Math.max(0.42, 0.56 + coordinatedBias + capacityBias * 0.8));
  const deliveryShare = Math.min(0.26, Math.max(0.12, 0.18 - coordinatedBias * 0.4));
  const idleShare = Math.max(0.08, 1 - coolingShare - deliveryShare);

  const coolingEnergyWh = totalOverheadWh * coolingShare;
  const powerDeliveryLossWh = totalOverheadWh * deliveryShare;
  const idleReserveWh = totalOverheadWh * idleShare;
  const totalFacilityEnergyWh = computeWh + totalOverheadWh;

  return {
    mode: "heuristic",
    itEnergyWh: safeRound(computeWh),
    coolingEnergyWh: safeRound(coolingEnergyWh),
    powerDeliveryLossWh: safeRound(powerDeliveryLossWh),
    idleReserveWh: safeRound(idleReserveWh),
    totalOverheadWh: safeRound(totalOverheadWh),
    totalFacilityEnergyWh: safeRound(totalFacilityEnergyWh),
    overheadRatio: safeRound(safeRatio(totalOverheadWh, computeWh)),
  };
}

export function buildObservedOverheadAttribution({
  computeWh = 0,
  coolingWh = 0,
  pumpWh = 0,
  otherOverheadWh = 0,
  totalWhObserved = 0,
} = {}) {
  const totalFacilityEnergyWh = Math.max(
    0,
    Number(totalWhObserved) || computeWh + coolingWh + pumpWh + otherOverheadWh
  );
  const knownCoolingWh = Math.max(0, (Number(coolingWh) || 0) + (Number(pumpWh) || 0));
  const residualWh = Math.max(
    0,
    totalFacilityEnergyWh - (Number(computeWh) || 0) - knownCoolingWh - (Number(otherOverheadWh) || 0)
  );
  const powerDeliveryLossWh = residualWh * 0.58;
  const idleReserveWh = residualWh * 0.42 + Math.max(0, Number(otherOverheadWh) || 0);
  const totalOverheadWh = knownCoolingWh + powerDeliveryLossWh + idleReserveWh;

  return {
    mode: "observed",
    itEnergyWh: safeRound(computeWh),
    coolingEnergyWh: safeRound(knownCoolingWh),
    powerDeliveryLossWh: safeRound(powerDeliveryLossWh),
    idleReserveWh: safeRound(idleReserveWh),
    totalOverheadWh: safeRound(totalOverheadWh),
    totalFacilityEnergyWh: safeRound(totalFacilityEnergyWh),
    overheadRatio: safeRound(safeRatio(totalOverheadWh, computeWh)),
  };
}

export function buildCalibratedOverheadAttribution({
  heuristicAttribution,
  calibrationState,
} = {}) {
  if (!heuristicAttribution) return null;

  const siteMultiplier = Number(calibrationState?.factors?.siteOverheadMultiplier ?? 1);
  const coolingEnergyWh = heuristicAttribution.coolingEnergyWh * siteMultiplier;
  const powerDeliveryLossWh = heuristicAttribution.powerDeliveryLossWh * siteMultiplier;
  const idleReserveWh = heuristicAttribution.idleReserveWh * siteMultiplier;
  const totalOverheadWh = coolingEnergyWh + powerDeliveryLossWh + idleReserveWh;
  const totalFacilityEnergyWh = heuristicAttribution.itEnergyWh + totalOverheadWh;

  return {
    mode: "calibrated",
    itEnergyWh: safeRound(heuristicAttribution.itEnergyWh),
    coolingEnergyWh: safeRound(coolingEnergyWh),
    powerDeliveryLossWh: safeRound(powerDeliveryLossWh),
    idleReserveWh: safeRound(idleReserveWh),
    totalOverheadWh: safeRound(totalOverheadWh),
    totalFacilityEnergyWh: safeRound(totalFacilityEnergyWh),
    overheadRatio: safeRound(safeRatio(totalOverheadWh, heuristicAttribution.itEnergyWh)),
  };
}
