/**
 * Calls the CoolSync FastAPI backend for DQN+LSTM simulation.
 * Falls back to JS physics engine (runCoordinated) if backend is unreachable.
 *
 * Returns the same shape as buildScenarioResults() so all downstream
 * UI components (KpiCard, charts, comparison table) work unchanged.
 */
import { runPID, runCoordinated, runPIDConservative, PHYS } from "./physics";

const API_BASE = "/api";

async function fetchBatch(heatTrace, strategies = ["pid", "pid_conservative", "dqn"]) {
  const res = await fetch(`${API_BASE}/simulate/batch`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ heat_trace: Array.from(heatTrace), strategies }),
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

function makeRunResult(points, strategy, grid_g_per_kWh) {
  const energyWh = points[points.length - 1]?.energy_wh ?? 0;
  const peakT    = Math.max(...points.map(p => p.T_rack ?? p.inlet ?? 0));
  const breaches = points.filter(p => (p.T_rack ?? p.inlet ?? 0) > PHYS.T_DANGER).length;
  const lagMs    = strategy === "coordinated" || strategy === "dqn" ? 0 : PHYS.DEAD_STEPS * 60_000;
  const avgCool  = points.reduce((s, p) => s + (p.coolingKw ?? 0), 0) / Math.max(points.length, 1);

  return {
    strategy,
    points,
    lagMs,
    dclc: {
      liquidIn:  PHYS.T_SUPPLY,
      liquidOut: 26.5,
      pumpSpeed: 70,
      valveState: strategy.toUpperCase(),
    },
    capacityPlan: {},
    metrics: {
      totalWhCurrent:       Number(energyWh.toFixed(3)),
      totalWhTarget:        Number(energyWh.toFixed(3)),
      savedWh:              0,
      co2e_g_current:       Number((energyWh / 1000 * grid_g_per_kWh).toFixed(2)),
      co2e_g_target:        0,
      savedCo2e_g:          0,
      grid_g_per_kWh,
      peakTRack:            peakT,
      breachSteps:          breaches,
      itEnergyWh:           0,
      coolingEnergyWh:      Number(energyWh.toFixed(3)),
      powerDeliveryLossWh:  0,
      idleReserveWh:        0,
      totalFacilityEnergyWh: Number(energyWh.toFixed(3)),
      overheadCurrent: 0.37, overheadTarget: 0.27,
      throughputRps: 0, queueDelayMs: 0, utilizationPct: 0,
      capacityPressure: "LOW", oversubscriptionRatio: 1, powerCapW: 0,
      headroomPct: 0, oversubscriptionMode: "guarded",
      batchingMode: "adaptive", powerCapMode: "balanced",
      heuristicOverheadAttribution: { coolingEnergyWh: energyWh * 0.5, itEnergyWh: energyWh * 0.32, powerDeliveryLossWh: energyWh * 0.1, idleReserveWh: energyWh * 0.08, totalFacilityEnergyWh: energyWh },
      targetOverheadAttribution:    { coolingEnergyWh: energyWh * 0.4, itEnergyWh: energyWh * 0.42, powerDeliveryLossWh: energyWh * 0.1, idleReserveWh: energyWh * 0.08, totalFacilityEnergyWh: energyWh },
      phaseEnergyWh: { prefillComputeWh: 0, decodeComputeWh: 0, baselineComputeWh: 0 },
      calibrationApplied: false, tokens: 0, whPerToken: 0, baseComputeWh: 0,
    },
    alerts: [`[DQN+LSTM] ${points.length}-step simulation. Energy: ${energyWh.toFixed(1)} Wh. Peak T: ${peakT.toFixed(1)}°C.`],
    runSummary: {
      strategy, source: "simulation",
      peakGpuW:         Math.max(...points.map(p => p.gpuW ?? p.heatW ?? 0)),
      peakInlet:        Number(peakT.toFixed(2)),
      avgCoolingKw:     Number(avgCool.toFixed(2)),
      alertCount:       breaches > 0 ? 1 : 0,
      lagMs,
      thresholdBreaches: breaches,
      stabilityScore:   Math.max(0, 100 - breaches * 3),
      whPerRequest:     Number(energyWh.toFixed(3)),
      jPerToken:        0,
      co2ePerRequest:   Number((energyWh / 1000 * grid_g_per_kWh).toFixed(2)),
      complexity: "MEDIUM", promptRisk: "MEDIUM", isEstimated: true,
    },
    decisionSupport: {
      detectedSignals: breaches > 0 ? [`T > ${PHYS.T_DANGER}°C for ${breaches} steps`] : [],
      recommendation:  strategy === "dqn" ? "DQN+LSTM active — physics-based" : "Physics simulation",
      recommendationLevel: breaches > 0 ? "HIGH" : "LOW",
      rationale: [],
      coordinationGap: breaches > 0,
    },
  };
}

/**
 * Fetch DQN+LSTM results from backend, replacing the "coordinated" run.
 * If backend is unavailable, returns null (caller falls back to JS physics).
 */
export async function fetchDqnResults(heatTrace, grid_g_per_kWh = 110) {
  try {
    const data = await fetchBatch(heatTrace, ["pid", "pid_conservative", "dqn"]);

    const reactive     = data.pid            ? makeRunResult(data.pid.points,            "reactive",     grid_g_per_kWh) : null;
    const coordinated  = data.dqn            ? makeRunResult(data.dqn.points,            "coordinated",  grid_g_per_kWh) : null;
    const conservative = data.pid_conservative ? makeRunResult(data.pid_conservative.points, "conservative", grid_g_per_kWh) : null;

    return { reactive, coordinated, conservative, source: "dqn" };
  } catch {
    return null;
  }
}

/** Check if backend is reachable. */
export async function checkBackendHealth() {
  try {
    const res  = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    const data = await res.json();
    return data?.status === "ok" && data?.models_loaded === true;
  } catch {
    return false;
  }
}
