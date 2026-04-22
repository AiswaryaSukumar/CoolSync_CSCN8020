// src/sections/PrototypeDemoSection.jsx
import { useEffect, useMemo, useRef, useState } from "react";
import Section from "../components/Section";
import Card from "../components/Card";
import KpiCard from "../components/KpiCard";
import AnimatedNumber from "../components/AnimatedNumber";
import CalibrationPanel from "../components/CalibrationPanel";
import {
  runSimulation,
  runConservativeSimulation,
  estimateTokensFromText,
  complexityClass,
  classifyPromptRisk,
  compareRuns,
  buildObservedRunSummary,
  buildComparisonNarrative,
  buildDecisionSupport,
  estimateLagFromPoints,
  buildTelemetryAlerts,
} from "../lib/simulate";
import { parseTelemetryCsv, computeTelemetryMetrics } from "../lib/telemetry";
import { COOLSYNC_SCENARIOS, loadScenario, buildHeatTrace } from "../lib/scenarioLoader";
import { parseScheduleCsv, scheduleEventsToScenario, estimateClass as estimatePromptClass, BURST_CLASS_LABEL, BURST_CLASS_COLOR } from "../lib/scheduleParser";
import { runPID, runCoordinated, runPIDConservative, PHYS, generateWorkload, tokensToBurstClass, BURST_PARAMS } from "../lib/physics";
import { fetchDqnResults, checkBackendHealth } from "../lib/simulationAdapter";
import { getTelemetryAcceptedColumnsText } from "../lib/telemetrySchema";
import {
  calibrateRunAgainstTelemetry,
  getDefaultCalibrationState,
} from "../lib/calibration";
import {
  buildCapacityAssessment,
  buildCapacityPlan,
  DEFAULT_CAPACITY_CONTROLS,
} from "../lib/capacityManager";
import { GRID_SIGNAL_PRESETS } from "../lib/sustainabilitySignals";
import {
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceArea,
  ReferenceLine,
  LineChart,
  Line,
  ComposedChart,
  Bar,
  CartesianGrid,
} from "recharts";
import {
  Zap,
  Play,
  SlidersHorizontal,
  Upload,
  FileDown,
  ToggleLeft,
  ToggleRight,
} from "lucide-react";

const MODELS = ["Small Model", "Medium Model", "LLM-70B (Dense)", "MoE Model"];

const BURST_PARAMS_REF = {
  0: { dur: 3 }, 1: { dur: 7 }, 2: { dur: 13 }, 3: { dur: 20 },
};

const GRID_PRESETS = GRID_SIGNAL_PRESETS;

const TRACE_SOURCE_LABEL = {
  simulation: "Simulation",
  telemetry: "Telemetry",
};

const STRATEGY_LABEL = {
  reactive: "Reactive",
  coordinated: "Coordinated",
};

const TELEMETRY_ACCEPTED_COLUMNS = getTelemetryAcceptedColumnsText();

// Example prompts per burst class (used for burst-class preset buttons and Add-row auto-fill)
const EXAMPLE_PROMPTS = [
  "What is 2 + 2?",
  "Translate this sentence to French: Good morning, how can I help you today?",
  "Write a Python REST API with JWT authentication and rate limiting.",
  "Perform a thorough architectural review of a microservices system handling 10M daily users: analyze service decomposition, inter-service communication, data consistency, observability, disaster recovery, capacity planning, and provide a phased migration roadmap.",
];

const BURST_CLASS_PRESETS = [
  { id: "burst_short",    label: "Short",     burstClass: 0 },
  { id: "burst_medium",   label: "Medium",    burstClass: 1 },
  { id: "burst_long",     label: "Long",      burstClass: 2 },
  { id: "burst_verylong", label: "Very Long", burstClass: 3 },
];

// Only 3 CoolSync scenarios shown in UI
const CS_UI_IDS = ["cs_multi_user", "cs_peak_hour", "cs_chaos_load"];
const COOLSYNC_SCENARIO_PRESETS = COOLSYNC_SCENARIOS
  .filter(s => CS_UI_IDS.includes(s.id))
  .map(s => ({
    id:    s.id,
    label: `[CoolSync] ${s.label}`,
    prompt: s.description,
    model:  s.model ?? "LLM-70B (Dense)",
    workloadFlexibility: s.workloadFlexibility ?? "urgent",
    coolsyncFile: s.file,
  }));

// Scenario presets (CoolSync only — burst buttons are handled separately)
const SCENARIO_PRESETS = [...COOLSYNC_SCENARIO_PRESETS];

function ModuleTag({ label }) {
  return (
    <span className="demo-module-tag inline-flex items-center rounded-full border border-[#D3D9D0] bg-[#ECEFE8] px-3 py-1 text-[11px] uppercase tracking-[0.14em] text-[#5C645D]">
      Module: {label}
    </span>
  );
}

function StatusPill({ tone = "slate", children }) {
  const toneMap = {
    emerald: "border-[#C9DCCB] bg-[#EEF5EF] text-[#8BC34A]",
    amber: "border-[#DDC8B2] bg-[#F6EFE8] text-[#B4691F]",
    sky: "border-[#D3D9D0] bg-[#ECEFE8] text-[#5F6B67]",
    slate: "border-[#D3D9D0] bg-[#DDE4DA] text-[#3A3A3A]",
    red: "border-[#D8B8B2] bg-[#F2E5E2] text-[#7C5A53]",
  };

  return (
    <span
      className={`demo-status-pill inline-flex items-center rounded-full border px-3 py-1 text-xs ${toneMap[tone]}`}
    >
      {children}
    </span>
  );
}

function formatCarbonImpact(value) {
  const amount = Number(value) || 0;
  if (amount === 0) return "below 1 mgCO2e";
  if (Math.abs(amount) < 0.01) return `${(amount * 1000).toFixed(1)} mgCO2e`;
  return `${amount.toFixed(2)} gCO2e`;
}

function getExportTimestamp() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");

  return [
    now.getFullYear(),
    pad(now.getMonth() + 1),
    pad(now.getDate()),
    "_",
    pad(now.getHours()),
    pad(now.getMinutes()),
    pad(now.getSeconds()),
  ].join("");
}

function formatChartTime(t) {
  const n = Number(t);
  if (!Number.isFinite(n)) return "";
  if (n >= 60_000) return `${(n / 60_000).toFixed(0)}min`;
  return `${(n / 1000).toFixed(2)}s`;
}

function formatInteger(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "0";
  return Math.round(n).toLocaleString();
}

function CustomLegend({ payload }) {
  if (!payload?.length) return null;

  const legendLabelMap = {
    "Reactive Inlet Temperature (°C)": "Reactive temperature",
    "Coordinated Inlet Temperature (°C)": "Coordinated temperature",
    "Reactive Cooling Demand (kW)": "Reactive cooling",
    "Coordinated Cooling Demand (kW)": "Coordinated cooling",
    "Concurrent Requests": "Concurrent requests",
    "Total Power Demand (W)": "Total power demand",
  };

  return (
    <div className="mt-2 flex flex-wrap gap-2">
      {payload.map((entry, index) => (
        <span
          key={index}
          className="demo-chart-legend-item inline-flex items-center gap-2 rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1 text-xs text-[#3A3A3A]"
        >
          <span className="h-2 w-2 rounded-full" style={{ background: entry.color }} />
          {legendLabelMap[entry.value] || entry.value}
        </span>
      ))}
    </div>
  );
}

function PhaseMarkers({ phaseWindows }) {
  const { prefillEnd, decodeEnd } = phaseWindows;
  if (!Number.isFinite(prefillEnd) || !Number.isFinite(decodeEnd)) return null;

  return (
    <>
      <ReferenceArea x1={0} x2={prefillEnd} fillOpacity={0.08} fill="#C67A2B" />
      <ReferenceLine x={decodeEnd} stroke="#8D9691" strokeDasharray="5 4" />
    </>
  );
}

function buildScenarioResults({ tokens, model, grid, calibrationState, capacityPlan, heatTrace }) {
  if (heatTrace) {
    // CoolSync physics scenario: run directly from heat trace
    const pidPoints  = runPID(heatTrace);
    const dqnPoints  = runCoordinated(heatTrace);
    const consPoints = runPIDConservative(heatTrace);

    const makeResult = (points, strategy) => {
      const energyWh = points[points.length - 1]?.energy_wh ?? 0;
      const peakT    = Math.max(...points.map(p => p.T_rack));
      const breaches = points.filter(p => p.T_rack > PHYS.T_DANGER).length;
      const lagMs    = strategy === "coordinated" ? 0 : PHYS.DEAD_STEPS * 60_000;
      const metrics  = {
        totalWhCurrent: Number(energyWh.toFixed(3)), totalWhTarget: Number(energyWh.toFixed(3)),
        savedWh: 0, co2e_g_current: Number((energyWh / 1000 * grid).toFixed(2)),
        co2e_g_target: 0, savedCo2e_g: 0, grid_g_per_kWh: grid,
        peakTRack: peakT, breachSteps: breaches,
        itEnergyWh: 0, coolingEnergyWh: Number(energyWh.toFixed(3)),
        powerDeliveryLossWh: 0, idleReserveWh: 0,
        totalFacilityEnergyWh: Number(energyWh.toFixed(3)),
        overheadCurrent: 0.37, overheadTarget: 0.27,
        throughputRps: 0, queueDelayMs: 0, utilizationPct: 0,
        capacityPressure: "LOW", oversubscriptionRatio: 1, powerCapW: 0,
        headroomPct: 0, oversubscriptionMode: "guarded",
        batchingMode: "adaptive", powerCapMode: "balanced",
        heuristicOverheadAttribution: { coolingEnergyWh: energyWh * 0.5, powerDeliveryLossWh: energyWh * 0.1, idleReserveWh: energyWh * 0.08, itEnergyWh: energyWh * 0.32, totalFacilityEnergyWh: energyWh },
        targetOverheadAttribution: { coolingEnergyWh: energyWh * 0.4, powerDeliveryLossWh: energyWh * 0.1, idleReserveWh: energyWh * 0.08, itEnergyWh: energyWh * 0.42, totalFacilityEnergyWh: energyWh },
        phaseEnergyWh: { prefillComputeWh: 0, decodeComputeWh: 0, baselineComputeWh: 0 },
        calibrationApplied: false, tokens, whPerToken: 0, baseComputeWh: 0,
      };
      const alerts = [`Scenario simulation complete — ${points.length} steps, ${energyWh.toFixed(1)} Wh.`];
      const { buildRunSummary: brs, buildDecisionSupport: bds } = (window._simExports ?? {});
      return {
        strategy, points, lagMs,
        dclc: { liquidIn: PHYS.T_SUPPLY, liquidOut: 26.5, pumpSpeed: 70, valveState: strategy.toUpperCase() },
        capacityPlan: capacityPlan ?? {}, metrics, alerts,
        runSummary: { strategy, source: "simulation", tokens, model,
          peakGpuW: Math.max(...points.map(p => p.gpuW ?? 0)), peakInlet: peakT,
          avgCoolingKw: points.reduce((s, p) => s + (p.coolingKw ?? 0), 0) / points.length,
          alertCount: 0, lagMs, thresholdBreaches: breaches,
          stabilityScore: Math.max(0, 100 - breaches * 3 - lagMs / 60000 * 5),
          whPerRequest: energyWh, jPerToken: 0,
          co2ePerRequest: Number((energyWh / 1000 * grid).toFixed(2)),
          complexity: "MEDIUM", promptRisk: "MEDIUM", isEstimated: true,
        },
        decisionSupport: { detectedSignals: [], recommendation: "Physics-based scenario", recommendationLevel: "LOW", rationale: [], coordinationGap: false },
      };
    };

    const reactive    = makeResult(pidPoints,  "reactive");
    const coordinated = makeResult(dqnPoints,  "coordinated");
    const conservative = makeResult(consPoints, "conservative");
    // Adjust conservative energy label
    conservative.strategy = "conservative";
    return { reactive, coordinated, conservative };
  }

  const reactive = runSimulation({
    tokens, model, grid_g_per_kWh: grid,
    overheadCurrent: 0.37, overheadTarget: 0.27,
    strategy: "reactive", calibration: calibrationState, capacityPlan,
  });

  const coordinated = runSimulation({
    tokens, model, grid_g_per_kWh: grid,
    overheadCurrent: 0.37, overheadTarget: 0.27,
    strategy: "coordinated", calibration: calibrationState, capacityPlan,
  });

  const conservative = runConservativeSimulation({
    tokens, model, grid_g_per_kWh: grid,
    overheadCurrent: 0.56, capacityPlan,
  });

  return { reactive, coordinated, conservative };
}

export default function PrototypeDemoSection() {
  const panelBRef = useRef(null);
  const guidedTimersRef = useRef([]);
  const [isGuidedDemoRunning, setIsGuidedDemoRunning] = useState(false);

  const [prompt, setPrompt] = useState(
    "Summarize this report into 5 bullets and list key risks and mitigations."
  );
  const [scenarioPresetId, setScenarioPresetId] = useState(null);
  const [model, setModel] = useState("LLM-70B (Dense)");
  const [grid, setGrid] = useState(GRID_PRESETS[1]?.g ?? 110);
  const [traceSource, setTraceSource] = useState("simulation");
  const [strategy, setStrategy] = useState("coordinated");
  const [workloadFlexibility, setWorkloadFlexibility] = useState("urgent");
  const [capacityControls, setCapacityControls] = useState(DEFAULT_CAPACITY_CONTROLS);
  const [calibrationState, setCalibrationState] = useState(() =>
    getDefaultCalibrationState("LLM-70B (Dense)")
  );

  const [telemetry, setTelemetry] = useState({
    points: [],
    meta: { format: "none" },
  });
  const [fileInputKey, setFileInputKey] = useState(0);
  const [uploadState, setUploadState] = useState("none");

  // CoolSync scenario heat trace (null = use token-based generate_workload)
  const [csHeatTrace, setCsHeatTrace] = useState(null);
  const [csScenarioLoading, setCsScenarioLoading] = useState(false);

  // Prompt schedule (events[] from scenario JSON or uploaded CSV)
  const [scheduleEvents, setScheduleEvents]     = useState([]);
  const [scheduleSource, setScheduleSource]     = useState(null); // "scenario" | "csv"
  const [scheduleInputKey, setScheduleInputKey] = useState(0);

  // Add-row form state
  const [addRowOpen,    setAddRowOpen]    = useState(false);
  const [newRowTime,    setNewRowTime]    = useState("");
  const [newRowPrompt,  setNewRowPrompt]  = useState("");
  const [newRowUsers,   setNewRowUsers]   = useState("1");
  const [newRowClass,   setNewRowClass]   = useState(null); // 0-3 or null (free text)

  // DQN backend: override scenarioResults when real model is available
  const [dqnResults, setDqnResults]       = useState(null);
  const [backendOnline, setBackendOnline] = useState(false);

  const tokens = useMemo(() => estimateTokensFromText(prompt), [prompt]);
  const complexity = useMemo(() => complexityClass(tokens), [tokens]);
  const promptRisk = useMemo(() => classifyPromptRisk(tokens, model), [tokens, model]);
  const defaultCalibration = useMemo(() => getDefaultCalibrationState(model), [model]);

  const hasTelemetry = telemetry?.points?.length > 0;
  const useTelemetry = traceSource === "telemetry" && hasTelemetry;
  const valueTypeLabel = useTelemetry ? "Observed" : "Estimated";
  const valueTypeMeta = useTelemetry
    ? "Observed telemetry view"
    : "Published benchmark + model estimate";
  const topConfidenceLabel = useTelemetry
    ? "Model confidence: Higher (telemetry-adjusted)"
    : "Model confidence: Medium (simulation-based)";
  const outputDataSourceLabel = useTelemetry
    ? "Data source: Observed telemetry"
    : "Data source: Simulation model";

  const capacityPlan = useMemo(
    () =>
      buildCapacityPlan({
        controls: capacityControls,
        tokens,
        strategy,
      }),
    [capacityControls, tokens, strategy]
  );

  const uncalibratedScenarioResults = useMemo(
    () =>
      buildScenarioResults({
        tokens, model, grid,
        calibrationState: defaultCalibration,
        capacityPlan,
        heatTrace: csHeatTrace,
      }),
    [tokens, model, grid, defaultCalibration, capacityPlan, csHeatTrace]
  );

  const scenarioResults = useMemo(
    () =>
      buildScenarioResults({
        tokens, model, grid,
        calibrationState,
        capacityPlan,
        heatTrace: csHeatTrace,
      }),
    [tokens, model, grid, calibrationState, capacityPlan, csHeatTrace]
  );

  // Merge DQN backend results when available (replaces coordinated + conservative)
  const mergedScenarioResults = dqnResults
    ? {
        reactive:     dqnResults.reactive    ?? scenarioResults.reactive,
        coordinated:  dqnResults.coordinated ?? scenarioResults.coordinated,
        conservative: dqnResults.conservative ?? scenarioResults.conservative,
      }
    : scenarioResults;

  const activeResult =
    strategy === "coordinated"
      ? mergedScenarioResults.coordinated
      : mergedScenarioResults.reactive;

  const [alertsLog, setAlertsLog] = useState(() => activeResult?.alerts || []);

  const clearTelemetry = () => {
    setTelemetry({ points: [], meta: { format: "none", rows: 0, error: null } });
    setTraceSource("simulation");
    setFileInputKey((k) => k + 1);
    setUploadState("none");
    setCalibrationState(getDefaultCalibrationState(model));
  };

  function clearGuidedTimers() {
    guidedTimersRef.current.forEach((t) => clearTimeout(t));
    guidedTimersRef.current = [];
  }

  useEffect(() => {
    return () => clearGuidedTimers();
  }, []);

  // Load CoolSync JSON heat trace when a CoolSync scenario is selected
  useEffect(() => {
    if (!scenarioPresetId) return;
    const preset = SCENARIO_PRESETS.find(s => s.id === scenarioPresetId);
    if (!preset?.coolsyncFile) return;
    setCsScenarioLoading(true);
    loadScenario(preset.coolsyncFile)
      .then(({ heatTrace, scenario }) => {
        setCsHeatTrace(heatTrace);
        const enriched = (scenario?.events ?? []).map(ev => ({
          ...ev,
          force_class: ev.force_class !== undefined
            ? Number(ev.force_class)
            : estimatePromptClass(ev.prompt ?? ""),
        }));
        setScheduleEvents(enriched);
        setScheduleSource("scenario");
        setCsScenarioLoading(false);
      })
      .catch(() => {
        setCsHeatTrace(null);
        setScheduleEvents([]);
        setScheduleSource(null);
        setCsScenarioLoading(false);
      });
  }, [scenarioPresetId]);

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth().then(setBackendOnline);
  }, []);

  // Fetch DQN+LSTM results from backend when heat trace or grid changes
  useEffect(() => {
    if (!backendOnline) { setDqnResults(null); return; }
    setDqnResults(null);
    let cancelled = false;
    const heatTrace = csHeatTrace
      ?? generateWorkload(
           Math.max(BURST_PARAMS_REF[tokensToBurstClass(tokens)]?.dur + 20, 30),
           8, tokensToBurstClass(tokens), 1,
         );
    fetchDqnResults(heatTrace, grid).then(res => {
      if (!cancelled && res) setDqnResults(res);
    });
    return () => { cancelled = true; };
  }, [backendOnline, csHeatTrace, tokens, grid]);

  const phaseWindows = useMemo(() => {
    const pts =
      (useTelemetry && hasTelemetry ? telemetry.points : activeResult?.points) || [];
    const maxT = pts.length ? pts[pts.length - 1].t : 0;
    const prefillEnd = maxT * 0.17;
    const decodeEnd = maxT * 0.78;
    return { prefillEnd, decodeEnd, maxT };
  }, [useTelemetry, hasTelemetry, telemetry, activeResult]);

  const telemetryLagMs = useMemo(
    () => estimateLagFromPoints(telemetry.points),
    [telemetry.points]
  );

  const telemetryMetrics = useMemo(() => {
    if (!hasTelemetry) return null;

    return computeTelemetryMetrics(telemetry.points, {
      grid_g_per_kWh: grid,
      overheadTarget: 0.27,
      tokens,
    });
  }, [hasTelemetry, telemetry.points, grid, tokens]);

  const telemetryWorkloadSummary = useMemo(() => {
    return telemetry?.meta?.workloadSummary || telemetryMetrics?.workload || null;
  }, [telemetry, telemetryMetrics]);

  const telemetryBurstNarrative = useMemo(() => {
    if (!telemetryWorkloadSummary?.hasBurstSignals) return null;

    const peakRequests = Number(telemetryWorkloadSummary.peakRequests || 0);
    const peakWorkloadTokens = Number(telemetryWorkloadSummary.peakWorkloadTokens || 0);
    const dominantPromptType = telemetryWorkloadSummary.dominantPromptType || "mixed";
    const burstRisk = telemetryWorkloadSummary.burstRisk || "LOW";

    if (burstRisk === "HIGH") {
      return `High burst load detected → ${peakRequests} concurrent requests and ${formatInteger(
        peakWorkloadTokens
      )} workload tokens at peak. ${dominantPromptType} dominated the trace.`;
    }

    if (burstRisk === "MEDIUM") {
      return `Moderate burst load detected → ${peakRequests} concurrent requests and ${formatInteger(
        peakWorkloadTokens
      )} workload tokens at peak. ${dominantPromptType} was the most frequent prompt type.`;
    }

    return `Light burst activity detected → peak of ${peakRequests} concurrent requests with ${formatInteger(
      peakWorkloadTokens
    )} workload tokens.`;
  }, [telemetryWorkloadSummary]);

  const telemetryAlerts = useMemo(() => {
    if (!hasTelemetry) return [];

    const baseAlerts = buildTelemetryAlerts({
      strategy,
      points: telemetry.points,
      lagMs: telemetryLagMs,
      traceLabel: "Observed telemetry trace",
    });

    const workloadAlerts = [];

    if (telemetryWorkloadSummary?.hasPromptMix && telemetryWorkloadSummary?.dominantPromptType) {
      workloadAlerts.push(
        `[Telemetry] Prompt mix detected → dominant workload: ${telemetryWorkloadSummary.dominantPromptType}.`
      );
    }

    if (telemetryWorkloadSummary?.hasBurstSignals) {
      workloadAlerts.push(
        `[Telemetry] Burst load detected → peak concurrent requests: ${formatInteger(
          telemetryWorkloadSummary.peakRequests
        )}.`
      );
    }

    if (telemetryWorkloadSummary?.hasTokenSignals) {
      workloadAlerts.push(
        `[Telemetry] Aggregate demand detected → peak workload tokens: ${formatInteger(
          telemetryWorkloadSummary.peakWorkloadTokens
        )}.`
      );
    }

    return [...workloadAlerts, ...baseAlerts];
  }, [
    hasTelemetry,
    strategy,
    telemetry.points,
    telemetryLagMs,
    telemetryWorkloadSummary,
  ]);

  const telemetryDecisionSupport = useMemo(() => {
    if (!hasTelemetry) return null;

    return buildDecisionSupport({
      strategy,
      tokens,
      model,
      points: telemetry.points,
      lagMs: telemetryLagMs,
    });
  }, [hasTelemetry, strategy, tokens, model, telemetry.points, telemetryLagMs]);

  const telemetryRunSummary = useMemo(() => {
    if (!hasTelemetry || !telemetryMetrics) return null;

    return buildObservedRunSummary({
      strategy,
      source: "telemetry",
      tokens,
      model,
      points: telemetry.points,
      metrics: telemetryMetrics,
      lagMs: telemetryLagMs,
      alerts: telemetryAlerts,
    });
  }, [
    hasTelemetry,
    telemetryMetrics,
    strategy,
    tokens,
    model,
    telemetry.points,
    telemetryLagMs,
    telemetryAlerts,
  ]);

  const uncalibratedEstimatedRun = useMemo(() => {
    if (strategy === "coordinated") return uncalibratedScenarioResults.coordinated?.runSummary;
    return uncalibratedScenarioResults.reactive?.runSummary;
  }, [strategy, uncalibratedScenarioResults]);

  const derivedCalibrationState = useMemo(() => {
    if (!telemetryRunSummary || !telemetryMetrics || !uncalibratedEstimatedRun) {
      return getDefaultCalibrationState(model);
    }

    const estimatedMetrics =
      strategy === "coordinated"
        ? uncalibratedScenarioResults.coordinated?.metrics
        : uncalibratedScenarioResults.reactive?.metrics;

    return calibrateRunAgainstTelemetry({
      model,
      estimatedRun: uncalibratedEstimatedRun,
      observedRun: telemetryRunSummary,
      estimatedMetrics,
      observedMetrics: telemetryMetrics,
      telemetryMeta: {
        rows: telemetry.meta?.rows || 0,
        hasFacilityMetering: telemetryMetrics.meta?.hasFacilityMetering || false,
        hasComponentBreakdown: telemetryMetrics.meta?.hasComponentBreakdown || false,
      },
    });
  }, [
    telemetryRunSummary,
    telemetryMetrics,
    uncalibratedEstimatedRun,
    strategy,
    uncalibratedScenarioResults,
    telemetry.meta,
    model,
  ]);

  const displayCalibrationState = useTelemetry ? derivedCalibrationState : calibrationState;

  const simulationBaselineRun    = mergedScenarioResults.reactive?.runSummary;
  const simulationCoordinatedRun = mergedScenarioResults.coordinated?.runSummary;
  const simulationConservativeRun = mergedScenarioResults.conservative?.runSummary;
  const baselineRun    = simulationBaselineRun;
  const coordinatedRun = simulationCoordinatedRun;
  const conservativeRun = simulationConservativeRun;
  const activeRunSummary = useTelemetry ? telemetryRunSummary : activeResult?.runSummary;

  const comparison = useMemo(() => {
    return compareRuns(simulationBaselineRun, simulationCoordinatedRun);
  }, [simulationBaselineRun, simulationCoordinatedRun]);

  const comparisonNarrative = useMemo(() => {
    return buildComparisonNarrative(comparison);
  }, [comparison]);

  const derivedAlertsLog = useMemo(() => {
    if (useTelemetry) return telemetryAlerts;
    return activeResult?.alerts || [];
  }, [useTelemetry, telemetryAlerts, activeResult]);

  const visibleAlertsLog = alertsLog?.length ? alertsLog : derivedAlertsLog;

  const execute = () => {
    const nextActive =
      strategy === "coordinated" ? mergedScenarioResults.coordinated : mergedScenarioResults.reactive;
    setAlertsLog(nextActive?.alerts || []);
  };

  const onUploadCsv = async (file) => {
    if (!file) return;

    const text = await file.text();
    const parsed = parseTelemetryCsv(text);
    setTelemetry(parsed);

    const now = new Date().toTimeString().slice(0, 8);

    if (parsed.points.length > 0) {
      const observedMetrics = computeTelemetryMetrics(parsed.points, {
        grid_g_per_kWh: grid,
        overheadTarget: 0.27,
        tokens,
      });

      const observedLagMs = estimateLagFromPoints(parsed.points);
      const observedWorkload = parsed.meta?.workloadSummary || observedMetrics?.workload || null;
      const observedAlerts = buildTelemetryAlerts({
        strategy,
        points: parsed.points,
        lagMs: observedLagMs,
        traceLabel: "Observed telemetry trace",
      });

      if (observedWorkload?.hasPromptMix && observedWorkload?.dominantPromptType) {
        observedAlerts.unshift(
          `Prompt mix detected → dominant workload: ${observedWorkload.dominantPromptType}.`
        );
      }

      if (observedWorkload?.hasBurstSignals) {
        observedAlerts.unshift(
          `Burst load detected → peak concurrent requests: ${formatInteger(
            observedWorkload.peakRequests
          )}.`
        );
      }

      if (observedWorkload?.hasTokenSignals) {
        observedAlerts.unshift(
          `Aggregate workload detected → peak workload tokens: ${formatInteger(
            observedWorkload.peakWorkloadTokens
          )}.`
        );
      }

      const observedRun = buildObservedRunSummary({
        strategy,
        source: "telemetry",
        tokens,
        model,
        points: parsed.points,
        metrics: observedMetrics,
        lagMs: observedLagMs,
        alerts: observedAlerts,
      });

      const estimatedRun =
        strategy === "coordinated"
          ? uncalibratedScenarioResults.coordinated?.runSummary
          : uncalibratedScenarioResults.reactive?.runSummary;

      const estimatedMetrics =
        strategy === "coordinated"
          ? uncalibratedScenarioResults.coordinated?.metrics
          : uncalibratedScenarioResults.reactive?.metrics;

      setTraceSource("telemetry");
      setUploadState("valid");
      setCalibrationState(
        calibrateRunAgainstTelemetry({
          model,
          estimatedRun,
          observedRun,
          estimatedMetrics,
          observedMetrics,
          telemetryMeta: {
            rows: parsed.meta?.rows || 0,
            hasFacilityMetering: observedMetrics.meta?.hasFacilityMetering || false,
            hasComponentBreakdown: observedMetrics.meta?.hasComponentBreakdown || false,
          },
        })
      );

      const uploadMessages = [
        `[${now}] Telemetry validated → observed metrics and comparison labels updated.`,
        `[${now}] Phase markers inferred from uploaded telemetry.`,
        `[${now}] Strategy view remains ${
          strategy === "coordinated" ? "coordinated" : "reactive"
        } for comparison reporting.`,
      ];

      if (observedWorkload?.hasPromptMix && observedWorkload?.dominantPromptType) {
        uploadMessages.unshift(
          `[${now}] Prompt mix detected → dominant workload: ${observedWorkload.dominantPromptType}.`
        );
      }

      if (observedWorkload?.hasBurstSignals) {
        uploadMessages.unshift(
          `[${now}] Burst load detected → peak concurrent requests: ${formatInteger(
            observedWorkload.peakRequests
          )}.`
        );
      }

      if (observedWorkload?.hasTokenSignals) {
        uploadMessages.unshift(
          `[${now}] Aggregate demand detected → peak workload tokens: ${formatInteger(
            observedWorkload.peakWorkloadTokens
          )}.`
        );
      }

      setAlertsLog(uploadMessages);
    } else {
      setTraceSource("simulation");
      setUploadState("invalid");
      setCalibrationState(getDefaultCalibrationState(model));
      setAlertsLog([
        `[${now}] Telemetry upload failed → invalid columns, simulation estimate remains active.`,
      ]);
    }
  };

  const runGuidedDemo = () => {
    if (isGuidedDemoRunning) return;

    setIsGuidedDemoRunning(true);
    clearGuidedTimers();

    clearTelemetry();
    setTraceSource("simulation");
    setStrategy("reactive");
    setAlertsLog([]);

    guidedTimersRef.current.push(
      setTimeout(() => {
        panelBRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 250)
    );

    const base = new Date();
    const ts = (ms) => new Date(base.getTime() + ms).toTimeString().slice(0, 8);

    const steps = [
      {
        t: 550,
        action: () => setStrategy("reactive"),
        msg: "Step 1/4: Reactive baseline loaded to show the initial cooling delay.",
      },
      {
        t: 1200,
        action: () => setStrategy("coordinated"),
        msg: "Step 2/4: Coordinated mode enabled to reduce the modeled lag window.",
      },
      {
        t: 2200,
        msg: "Step 3/4: Upload telemetry to compare observed data with the estimate.",
      },
      {
        t: 3200,
        msg: "Step 4/4: Export the summary with labels for research, estimated, and observed values.",
      },
    ];

    steps.forEach((s) => {
      guidedTimersRef.current.push(
        setTimeout(() => {
          if (s.action) s.action();
          setAlertsLog((prev) => [`[${ts(s.t)}] ${s.msg}`, ...prev]);
        }, s.t)
      );
    });

    guidedTimersRef.current.push(
      setTimeout(() => {
        setIsGuidedDemoRunning(false);
      }, 4000)
    );
  };

  const applyScenarioPreset = (preset) => {
    if (!preset) return;
    setDqnResults(null);
    setScenarioPresetId(preset.id);
    setPrompt(preset.prompt);
    setModel(preset.model);
    setWorkloadFlexibility(preset.workloadFlexibility);
    setCalibrationState(getDefaultCalibrationState(preset.model));
    setTraceSource("simulation");
    setUploadState("none");
    setAlertsLog([]);
  };

  // Set schedule to a single burst event at time_min=0 (replaces existing schedule)
  const onAddBurstEvent = (burstClass) => {
    const cls   = burstClass;
    const { dur } = BURST_PARAMS[cls];
    const event = {
      time_min:    0,
      prompt:      EXAMPLE_PROMPTS[cls],
      users:       1,
      force_class: cls,
      note:        BURST_CLASS_LABEL[cls],
    };
    const totalMinutes = dur + 20;
    const scenario     = scheduleEventsToScenario([event], totalMinutes);
    const heatTrace    = buildHeatTrace(scenario);
    setScheduleEvents([event]);
    setScheduleSource("scenario");
    setScenarioPresetId(null);
    setCsHeatTrace(heatTrace);
    setDqnResults(null);
  };

  const onDeleteScheduleRow = (idx) => {
    const updated = scheduleEvents.filter((_, i) => i !== idx);
    setScheduleEvents(updated);
    if (updated.length === 0) {
      setCsHeatTrace(null);
      setScheduleSource(null);
      setDqnResults(null);
    } else {
      const totalMinutes = Math.ceil(updated[updated.length - 1].time_min) + 30;
      const scenario     = scheduleEventsToScenario(updated, totalMinutes);
      const heatTrace    = buildHeatTrace(scenario);
      setCsHeatTrace(heatTrace);
      setDqnResults(null);
    }
  };

  const onAddScheduleRow = () => {
    const time_min = Number(newRowTime);
    const prompt   = newRowPrompt.trim();
    const users    = Math.max(1, Number(newRowUsers) || 1);
    if (!isFinite(time_min) || time_min < 0 || !prompt) return;
    const force_class   = newRowClass !== null ? newRowClass : estimatePromptClass(prompt);
    const newEvent      = { time_min, prompt, users, force_class, note: `${BURST_CLASS_LABEL[force_class]} ×${users}` };
    const updated       = [...scheduleEvents, newEvent].sort((a, b) => a.time_min - b.time_min);
    const totalMinutes  = Math.ceil(updated[updated.length - 1].time_min) + 30;
    const scenario      = scheduleEventsToScenario(updated, totalMinutes);
    const heatTrace     = buildHeatTrace(scenario);
    setScheduleEvents(updated);
    setScheduleSource(scheduleSource ?? "csv");
    setCsHeatTrace(heatTrace);
    setDqnResults(null);
    setAddRowOpen(false);
    setNewRowTime(""); setNewRowPrompt(""); setNewRowUsers("1"); setNewRowClass(null);
  };

  const onScheduleCsvUpload = async (file) => {
    if (!file) return;
    const text = await file.text();
    const { events, totalMinutes, error } = parseScheduleCsv(text);
    if (error || events.length === 0) {
      setAlertsLog([`[System] 스케줄 CSV 오류: ${error ?? "이벤트 없음"}`]);
      return;
    }
    const scenario  = scheduleEventsToScenario(events, totalMinutes);
    const heatTrace = buildHeatTrace(scenario);
    setCsHeatTrace(heatTrace);
    setScheduleEvents(events);
    setScheduleSource("csv");
    setDqnResults(null);
    setAlertsLog([`[System] 스케줄 CSV 로드 완료 — ${events.length}개 이벤트, ${totalMinutes}분`]);
  };

  const activeMetrics =
    useTelemetry && telemetryMetrics ? telemetryMetrics : activeResult.metrics;

  const activeRun = activeRunSummary;

  const activeDecisionSupport = useTelemetry
    ? telemetryDecisionSupport
    : activeResult?.decisionSupport;

  const capacityAssessment = buildCapacityAssessment({
    capacityPlan,
    activeRun: activeRunSummary,
    activeMetrics,
    traceSource,
  });

  const strategyWhRequest = Number(activeRunSummary?.whPerRequest || 0);

const observedCo2eRequest = Number(telemetryRunSummary?.co2ePerRequest || 0);
const modeledStrategyCo2eRequest = Number(
  strategy === "coordinated"
    ? mergedScenarioResults.coordinated?.runSummary?.co2ePerRequest || 0
    : mergedScenarioResults.reactive?.runSummary?.co2ePerRequest || 0
);

const strategyCo2eRequest = useTelemetry
  ? observedCo2eRequest
  : modeledStrategyCo2eRequest;

const carbonCardLabel = useTelemetry
  ? "Observed carbon impact"
  : "Estimated carbon impact";

const carbonCardHint = useTelemetry
  ? "Observed telemetry trace; strategy savings are shown in the comparison below"
  : "Modeled per-request carbon under the selected strategy";

  const activePreset =
    SCENARIO_PRESETS.find((preset) => preset.id === scenarioPresetId) ?? null;

  const comparisonWhSaved = Math.max(
    0,
    Number(mergedScenarioResults.reactive?.metrics?.totalWhCurrent || 0) -
      Number(mergedScenarioResults.coordinated?.metrics?.totalWhTarget || 0)
  );

  const comparisonCo2eSaved = Math.max(
    0,
    Number(mergedScenarioResults.reactive?.metrics?.co2e_g_current || 0) -
      Number(mergedScenarioResults.coordinated?.metrics?.co2e_g_target || 0)
  );

  const comparisonFacilityEnergySaved = Math.max(
    0,
    Number(mergedScenarioResults.reactive?.metrics?.totalFacilityEnergyWh || 0) -
      Number(mergedScenarioResults.coordinated?.metrics?.totalFacilityEnergyWh || 0)
  );

  const facilityEnergyComparisonLabel =
    comparisonFacilityEnergySaved < 0.001
      ? "Modeled difference → near-flat"
      : `-${comparisonFacilityEnergySaved.toFixed(3)} Wh`;

  const predictedOverheadDelta = Math.max(
    0,
    Number(activeMetrics?.overheadCurrent || 0) - Number(activeMetrics?.overheadTarget || 0)
  );

  const consoleSummaryItems = [
    { label: "Estimated tokens", value: `${tokens}`, meta: "Modeled workload estimate" },
    { label: "Complexity", value: complexity, meta: "Modeled prompt class" },
    { label: "Strategy", value: STRATEGY_LABEL[strategy], meta: "Decision policy view" },
    {
      label: "Trace source",
      value: TRACE_SOURCE_LABEL[traceSource],
      meta: useTelemetry ? "Observed telemetry active" : "Simulation active",
    },
    {
      label: "Estimated cooling gap",
      value: `${(predictedOverheadDelta * 100).toFixed(0)} pp`,
      meta: "Published benchmark + model estimate",
    },
  ];

  const comparisonChartData = useMemo(() => {
    const reactivePoints    = mergedScenarioResults.reactive?.points    || [];
    const coordinatedPoints = mergedScenarioResults.coordinated?.points || [];

    const maxLen = Math.max(reactivePoints.length, coordinatedPoints.length);

    return Array.from({ length: maxLen }).map((_, i) => ({
      t: reactivePoints[i]?.t || coordinatedPoints[i]?.t || i * 50,
      reactiveInlet: reactivePoints[i]?.inlet || null,
      coordinatedInlet: coordinatedPoints[i]?.inlet || null,
      reactiveCoolingKw: reactivePoints[i]?.coolingKw || null,
      coordinatedCoolingKw: coordinatedPoints[i]?.coolingKw || null,
    }));
  }, [scenarioResults, dqnResults]);

  // Temperature Y-axis: fixed [18, 30] but expands if data exceeds the range
  const tempYDomain = useMemo(() => {
    const vals = comparisonChartData.flatMap(d =>
      [d.reactiveInlet, d.coordinatedInlet].filter(v => v != null)
    );
    if (vals.length === 0) return [18, 30];
    const dataMin = Math.min(...vals);
    const dataMax = Math.max(...vals);
    return [
      Math.min(18, Math.floor(dataMin - 0.5)),
      Math.max(30, Math.ceil(dataMax + 0.5)),
    ];
  }, [comparisonChartData]);

  const workloadEnergyData = useMemo(() => {
    if (!telemetry?.points?.length) return [];

    return telemetry.points.map((p) => ({
      t: p.t,
      requestCount: Number(p.requestCount || 0),
      workloadTokens: Number(p.workloadTokens || 0),
      totalPowerW: Number(p.totalPowerW || 0),
    }));
  }, [telemetry]);

  const eventLogItems = visibleAlertsLog.map((entry) => {
    const match = String(entry).match(/^\[(.*?)\]\s*(.*)$/);

    if (match) {
      return { time: match[1], text: match[2] };
    }

    return { time: "System", text: String(entry) };
  });

  const controlTraceItems = [
    {
      label: "Selected mode",
      value: STRATEGY_LABEL[strategy],
      tone:
        strategy === "coordinated"
          ? "border-[#C9DCCB] bg-[#EEF5EF] text-[#1F7A3A]"
          : "border-[#DDC8B2] bg-[#F6EFE8] text-[#B4691F]",
    },
    {
      label: "Policy target",
      value:
        strategy === "coordinated"
          ? "Pre-cool window opened"
          : "Reactive cooling maintained",
      tone: "border-[#D3D9D0] bg-[#F3F5F0] text-[#3A3A3A]",
    },
    {
      label: "Expected effect",
      value:
        strategy === "coordinated"
          ? "Modeled lag window reduced"
          : "Modeled lag window remains exposed",
      tone: "border-[#D3D9D0] bg-[#F3F5F0] text-[#3A3A3A]",
    },
  ];

  const downloadSampleTelemetry = () => {
    const rows = [
      [
        "time_ms",
        "facility_kw",
        "rack_kw",
        "server_kw",
        "gpu_power_w",
        "cpu_power_w",
        "dram_power_w",
        "nic_power_w",
        "inlet_temp_c",
        "cooling_kw",
        "pump_kw",
        "other_overhead_kw",
        "water_lpm",
        "prompt_type",
        "request_count",
        "tokens_per_request",
      ],
    ];

    for (let i = 0; i <= 36; i++) {
      const t = i * 50;
      const prefill = t <= 300;
      const decode = t > 300 && t <= 1400;

      let gpuW = 110;
      if (prefill) gpuW = 140 + 320 * Math.sin((Math.PI * t) / 300);
      else if (decode) gpuW = 190 + 30 * Math.sin((Math.PI * (t - 300)) / 500);

      const cpuW = prefill ? 76 : decode ? 58 : 42;
      const dramW = prefill ? 34 : decode ? 28 : 18;
      const nicW = decode ? 16 : 10;
      const inlet = 24.2 + (gpuW - 120) / 400;
      const coolingKw = t >= 550 ? 18 + ((gpuW - 120) / 400) * 10 : 18;
      const pumpKw = t >= 550 ? 1.2 + ((gpuW - 120) / 400) * 1.4 : 1.1;
      const otherOverheadKw = 0.9;
      const serverKw = (gpuW + cpuW + dramW + nicW) / 1000;
      const rackKw = serverKw * 1.08;
      const facilityKw = rackKw + coolingKw + pumpKw + otherOverheadKw;
      const waterLpm = 22 + Math.max(0, (gpuW - 120) / 40);

      let promptType = "short_qa";
      let requestCount = 2;
      let tokensPerRequest = 180;

      if (t >= 350 && t < 800) {
        promptType = "long_summary";
        requestCount = 6;
        tokensPerRequest = 2200;
      } else if (t >= 800 && t < 1200) {
        promptType = "code_gen";
        requestCount = 8;
        tokensPerRequest = 3200;
      } else if (t >= 1200) {
        promptType = "mixed_burst";
        requestCount = 10;
        tokensPerRequest = 1500;
      }

      rows.push([
        t,
        facilityKw.toFixed(3),
        rackKw.toFixed(3),
        serverKw.toFixed(3),
        Math.round(gpuW),
        cpuW.toFixed(0),
        dramW.toFixed(0),
        nicW.toFixed(0),
        inlet.toFixed(2),
        coolingKw.toFixed(2),
        pumpKw.toFixed(2),
        otherOverheadKw.toFixed(2),
        waterLpm.toFixed(1),
        promptType,
        requestCount,
        tokensPerRequest,
      ]);
    }

    const csv = rows.map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "greeninference_valid_telemetry.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const downloadInvalidTelemetry = () => {
    const csv = [
      "time_ms,facility_kw,gpu_power_w,inlet_temp_c",
      "0,20.1,120,24.1",
      "50,20.6,broken,24.3",
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "greeninference_invalid_telemetry.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const exportCsv = () => {
    const m = activeMetrics;
    const exportStamp = getExportTimestamp();

    const rows = [
      ["timestamp", new Date().toISOString()],
      ["prompt", prompt.replace(/\s+/g, " ").trim()],
      ["model", model],
      ["trace_source", traceSource],
      ["strategy", strategy],
      ["grid_gCO2_per_kWh", String(grid)],
      ["workload_flexibility", String(workloadFlexibility)],
      ["value_type", useTelemetry ? "observed" : "estimated"],
      ["data_source", useTelemetry ? "telemetry" : "simulation"],
      ["model_basis", "published research benchmarks + modeled estimates"],
      ["capacity_headroom_pct", String(capacityControls.headroomPct)],
      ["capacity_oversubscription_mode", String(capacityControls.oversubscriptionMode)],
      ["capacity_batching_mode", String(capacityControls.batchingMode)],
      ["capacity_power_cap_mode", String(capacityControls.powerCapMode)],
      ["tokens_est", String(m.tokens)],
      ["complexity", complexity],
      ["prompt_risk", promptRisk],
      ["wh_per_token", String(m.whPerToken)],
      ["base_compute_Wh", String(m.baseComputeWh)],
      ["total_Wh_today", String(m.totalWhCurrent)],
      ["total_Wh_target", String(m.totalWhTarget)],
      ["saved_Wh", String(m.savedWh)],
      ["co2e_g_today", String(m.co2e_g_current)],
      ["co2e_g_target", String(m.co2e_g_target)],
      ["saved_co2e_g", String(m.savedCo2e_g)],
      ["cooling_overhead_today", String(m.overheadCurrent)],
      ["cooling_overhead_target", String(m.overheadTarget)],
      ["capacity_throughput_rps", String(capacityAssessment?.throughputRps ?? "")],
      ["capacity_queue_delay_ms", String(capacityAssessment?.queueDelayMs ?? "")],
      ["capacity_utilization_pct", String(capacityAssessment?.utilizationPct ?? "")],
      ["capacity_pressure", String(capacityAssessment?.capacityPressure ?? "")],
      ["reactive_peak_inlet", String(simulationBaselineRun?.peakInlet || "")],
      ["coordinated_peak_inlet", String(simulationCoordinatedRun?.peakInlet || "")],
      ["reactive_lag_ms", String(simulationBaselineRun?.lagMs || "")],
      ["coordinated_lag_ms", String(simulationCoordinatedRun?.lagMs || "")],
      ["telemetry_enabled", String(useTelemetry)],
      ["telemetry_format", String(telemetry?.meta?.format || "none")],
      ["telemetry_rows", String(telemetry?.meta?.rows || 0)],
      ["telemetry_confidence", String(telemetry?.meta?.confidence || "none")],
      [
        "telemetry_peak_requests",
        String(
          telemetryWorkloadSummary?.peakRequests ??
            telemetryMetrics?.workload?.peakRequests ??
            ""
        ),
      ],
      [
        "telemetry_peak_workload_tokens",
        String(
          telemetryWorkloadSummary?.peakWorkloadTokens ??
            telemetryMetrics?.workload?.peakWorkloadTokens ??
            ""
        ),
      ],
      [
        "telemetry_dominant_prompt_type",
        String(
          telemetryWorkloadSummary?.dominantPromptType ??
            telemetryMetrics?.workload?.dominantPromptType ??
            ""
        ),
      ],
      [
        "telemetry_burst_risk",
        String(
          telemetryWorkloadSummary?.burstRisk ?? telemetryMetrics?.workload?.burstRisk ?? ""
        ),
      ],
      ["comparison_peakInletReductionPct", String(comparison.peakInletReductionPct)],
      ["comparison_lagReductionPct", String(comparison.lagReductionPct)],
      ["comparison_coolingReductionPct", String(comparison.coolingReductionPct)],
      ["comparison_stabilityImprovementPct", String(comparison.stabilityImprovementPct)],
      ["policy_recommendation", String(activeDecisionSupport?.recommendation || "")],
      ["policy_recommendationLevel", String(activeDecisionSupport?.recommendationLevel || "")],
      [
        "policy_detectedSignals",
        String((activeDecisionSupport?.detectedSignals || []).join("; ")),
      ],
      ["policy_rationale", String((activeDecisionSupport?.rationale || []).join("; "))],
      ["calibration_status", String(displayCalibrationState?.status || "")],
      ["calibration_confidence", String(displayCalibrationState?.confidence || "")],
    ];

    const csv = rows
      .map((r) => r.map((x) => `"${String(x).replaceAll('"', '""')}"`).join(","))
      .join("\n");

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `greeninference_transparency_${exportStamp}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <Section
      id="demo"
      eyebrow="Simulation"
      title="Simulation"
      subtitle="How cooling timing changes estimated energy and carbon impact"
    >
      <div className="simulation-disclaimer mb-4 rounded-[18px] border border-[rgba(120,140,120,0.2)] bg-[rgba(240,244,240,0.8)] px-4 py-3 text-[13px] text-[#2F3B2F] shadow-[0_8px_18px_rgba(56,96,68,0.04)]">
        <p>
          This simulation combines:
          <span className="ml-2 inline-flex items-center rounded-full border border-[#C9DCCB] bg-[#EEF5EF] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#1E7D3A]">
            Published research benchmarks
          </span>
          <span className="ml-2 inline-flex items-center rounded-full border border-[#E2C89E] bg-[#FFF4E5] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#A15C00]">
            Modeled system estimates
          </span>
          <span className="ml-2 inline-flex items-center rounded-full border border-[#BDD0FF] bg-[#E8F0FF] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#1F4FD8]">
            Optional telemetry data
          </span>
        </p>
        <p className="mt-2 text-[12px] text-[#5E6A5F]">
          Estimated values are derived from published data center energy, cooling,
          and workload studies. Observed values appear when telemetry is uploaded.
        </p>
      </div>

      <div className="demo-summary-grid mb-4 grid grid-cols-1 gap-3 md:grid-cols-5">
        {consoleSummaryItems.map((item) => (
          <div
            key={item.label}
            className="demo-summary-card rounded-[18px] border border-[rgba(95,104,96,0.1)] bg-[rgba(250,252,249,0.9)] px-4 py-3 shadow-[0_8px_18px_rgba(56,96,68,0.04)]"
          >
            <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#6A776F]">
              {item.label}
            </div>
            <div className="mt-2 text-sm font-semibold text-[#18212F]">{item.value}</div>
            <div className="mt-1 text-xs text-[#6A776F]">{item.meta}</div>
          </div>
        ))}
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <StatusPill tone="sky">{topConfidenceLabel}</StatusPill>
      </div>

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-2">
        <Card className="demo-input-shell ds-card--secondary h-fit lg:sticky lg:top-24">
          <div className="demo-left-shell">
            <div>
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-2">
                  <Zap className="text-[#7A7F54]" size={18} />
                  <div className="font-semibold">Console Input</div>
                </div>

                <div className="flex flex-wrap items-center gap-2 sm:justify-end">
                  <button
                    onClick={runGuidedDemo}
                    disabled={isGuidedDemoRunning}
                    className={[
                      "inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm transition",
                      isGuidedDemoRunning
                        ? "border-[#D3D9D0] bg-[#DDE4DA] text-[#3A3A3A] opacity-70 cursor-not-allowed"
                        : "border-[#D3D9D0] bg-[#DDE4DA] text-[#262626] hover:bg-[#BFD98A]/45",
                    ].join(" ")}
                    type="button"
                  >
                    <Play size={16} className="text-[#6B7B48]" />
                    {isGuidedDemoRunning ? "Running Demo..." : "Run Guided Demo"}
                  </button>

                  <button
                    onClick={execute}
                    className="inline-flex items-center gap-2 rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-4 py-2 text-sm text-[#262626] transition-colors hover:bg-[#BFD98A]/45"
                    type="button"
                  >
                    <Play size={16} className="text-[#6B7B48]" />
                    Run
                  </button>

                  <button
                    onClick={exportCsv}
                    className="inline-flex items-center gap-2 rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-4 py-2 text-sm text-[#262626] transition-colors hover:bg-[#F3F5F0]"
                    type="button"
                  >
                    <FileDown size={16} className="text-[#5E7766]" />
                    Export CSV
                  </button>
                </div>
              </div>

              <div className="mt-4 flex flex-wrap items-center gap-2">
                <ModuleTag label="Prompt input" />
                <ModuleTag label="Simulation view" />
                <span className="demo-signal-chip inline-flex items-center rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1 text-xs text-[#3A3A3A]">
                  Data:
                  <span className="ml-1 text-[#6B7B48]">{TRACE_SOURCE_LABEL[traceSource]}</span>
                </span>
                <span className="demo-signal-chip inline-flex items-center rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1 text-xs text-[#3A3A3A]">
                  Mode:
                  <span className="ml-1 text-[#7A7F54]">{STRATEGY_LABEL[strategy]}</span>
                </span>
              </div>

              <div className="demo-policy-trace-card demo-support-surface mt-4 rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-[#262626]">Decision trace</div>
                    <div className="mt-1 text-xs text-[#6F756E]">
                      Request → Estimate → Policy → Cooling → Results
                    </div>
                  </div>
                  <StatusPill tone="sky">Modeled view</StatusPill>
                </div>

                <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
                  {controlTraceItems.map((item) => (
                    <div
                      key={item.label}
                      className={`demo-policy-trace-item rounded-xl border px-3 py-3 ${item.tone}`}
                    >
                      <div className="text-[11px] font-semibold uppercase tracking-[0.12em] opacity-80">
                        {item.label}
                      </div>
                      <div className="mt-2 text-sm font-semibold">{item.value}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-4">
                <label className="text-xs text-[#6F756E]">Prompt</label>

                {/* Row 1: Quick-add burst event buttons */}
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <span className="text-[11px] text-[#9AA09A]">+ Add event:</span>
                  {BURST_CLASS_PRESETS.map((bp) => (
                    <button
                      key={bp.id}
                      type="button"
                      onClick={() => onAddBurstEvent(bp.burstClass)}
                      className="rounded-full border px-3 py-1.5 text-xs font-medium transition border-[#D3D9D0] bg-[#F3F5F0] text-[#3A3A3A] hover:bg-[#DDE4DA] active:scale-95"
                      style={{ borderColor: BURST_CLASS_COLOR[bp.burstClass] + "88" }}
                    >
                      <span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: BURST_CLASS_COLOR[bp.burstClass] }} />
                      {bp.label}
                    </button>
                  ))}
                </div>

                {/* Row 2: CoolSync scenario buttons */}
                <div className="mt-2 flex flex-wrap gap-2">
                  {COOLSYNC_SCENARIO_PRESETS.map((preset) => (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() => applyScenarioPreset(preset)}
                      className={[
                        "rounded-full border px-3 py-1.5 text-xs transition",
                        scenarioPresetId === preset.id
                          ? "border-[#C7CDC5] bg-[#E8ECE6] text-[#525B55]"
                          : "border-[#D3D9D0] bg-[#F3F5F0] text-[#3A3A3A] hover:bg-[#DDE4DA]",
                      ].join(" ")}
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>

                <div className="mt-2 text-xs text-[#6F756E]">
                  Active preset: <span className="text-[#3A3A3A]">{activePreset?.label ?? "—"}</span>
                </div>

                {/* ── Prompt Schedule Panel ── */}
                <div className="mt-3 rounded-2xl border border-[#C7CDC5] bg-[#F3F5F0] p-3">
                  <div className="flex items-center justify-between gap-2">
                    <div className="text-xs font-semibold uppercase tracking-[0.1em] text-[#5F6B67]">
                      Prompt Schedule
                      {scheduleSource && (
                        <span className="ml-2 rounded-full border border-[#D3D9D0] bg-[#E8ECE6] px-2 py-0.5 text-[10px] text-[#6A776F]">
                          {scheduleSource === "csv" ? "CSV" : "Scenario"}
                        </span>
                      )}
                      <span className="ml-2 text-[#9AA09A] normal-case font-normal">
                        {scheduleEvents.length} event{scheduleEvents.length !== 1 ? "s" : ""}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setAddRowOpen(v => !v)}
                        className="rounded-full border border-[#C7CDC5] bg-[#E8ECE6] px-2.5 py-1 text-[10px] font-semibold text-[#525B55] hover:bg-[#DDE4DA] transition"
                      >
                        + Add row
                      </button>
                      {scheduleEvents.length > 0 && (
                        <button
                          type="button"
                          onClick={() => {
                            setScheduleEvents([]);
                            setScheduleSource(null);
                            if (scheduleSource === "csv") { setCsHeatTrace(null); setDqnResults(null); }
                            setScheduleInputKey(k => k + 1);
                            setAddRowOpen(false);
                          }}
                          className="text-[10px] text-[#9AA09A] hover:text-[#5F6B67]"
                        >
                          Clear
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Add-row inline form */}
                  {addRowOpen && (
                    <div className="mt-2 rounded-xl border border-[#D3D9D0] bg-[#FAFBF9] p-2.5 text-[11px]">
                      {/* Type quick-fill buttons */}
                      <div className="mb-2">
                        <span className="text-[10px] text-[#6A776F]">Type — click to auto-fill prompt</span>
                        <div className="mt-1 flex gap-1.5 flex-wrap">
                          {BURST_CLASS_PRESETS.map((bp) => (
                            <button
                              key={bp.id}
                              type="button"
                              onClick={() => {
                                setNewRowClass(bp.burstClass);
                                setNewRowPrompt(EXAMPLE_PROMPTS[bp.burstClass]);
                              }}
                              className={[
                                "rounded-full border px-2.5 py-0.5 text-[10px] font-medium transition",
                                newRowClass === bp.burstClass
                                  ? "border-[#8B9B5A] text-white"
                                  : "border-[#D3D9D0] bg-[#F3F5F0] text-[#525B55] hover:bg-[#E8ECE6]",
                              ].join(" ")}
                              style={newRowClass === bp.burstClass ? { background: BURST_CLASS_COLOR[bp.burstClass] } : {}}
                            >
                              {bp.label}
                            </button>
                          ))}
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <div className="flex flex-col gap-1">
                          <span className="text-[10px] text-[#6A776F]">Time (min)</span>
                          <input
                            type="number"
                            min="0"
                            value={newRowTime}
                            onChange={e => setNewRowTime(e.target.value)}
                            placeholder="e.g. 10"
                            className="w-20 rounded-lg border border-[#D3D9D0] bg-white px-2 py-1 text-[11px] text-[#262626] outline-none focus:border-[#B8C3B5]"
                          />
                        </div>
                        <div className="flex flex-1 flex-col gap-1">
                          <span className="text-[10px] text-[#6A776F]">Prompt</span>
                          <input
                            type="text"
                            value={newRowPrompt}
                            onChange={e => { setNewRowPrompt(e.target.value); setNewRowClass(null); }}
                            placeholder="Enter prompt text or click a type button above..."
                            className="w-full rounded-lg border border-[#D3D9D0] bg-white px-2 py-1 text-[11px] text-[#262626] outline-none focus:border-[#B8C3B5]"
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-[10px] text-[#6A776F]">Users</span>
                          <input
                            type="number"
                            min="1"
                            max="10"
                            value={newRowUsers}
                            onChange={e => setNewRowUsers(e.target.value)}
                            className="w-14 rounded-lg border border-[#D3D9D0] bg-white px-2 py-1 text-[11px] text-[#262626] outline-none focus:border-[#B8C3B5]"
                          />
                        </div>
                      </div>
                      <div className="mt-2 flex gap-2">
                        <button
                          type="button"
                          onClick={onAddScheduleRow}
                          disabled={!newRowTime || !newRowPrompt}
                          className="rounded-full border border-[#7A9B5A] bg-[#EEF5EF] px-3 py-1 text-[10px] font-semibold text-[#3A6B28] disabled:opacity-40 hover:bg-[#DDF0DD] transition"
                        >
                          Add
                        </button>
                        <button
                          type="button"
                          onClick={() => { setAddRowOpen(false); setNewRowTime(""); setNewRowPrompt(""); setNewRowUsers("1"); setNewRowClass(null); }}
                          className="rounded-full border border-[#D3D9D0] bg-[#F3F5F0] px-3 py-1 text-[10px] text-[#6A776F] hover:bg-[#E8ECE6] transition"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}

                  {scheduleEvents.length > 0 ? (
                    <div className="mt-2 max-h-52 overflow-y-auto">
                      <table className="w-full text-[11px]">
                        <thead>
                          <tr className="border-b border-[#D3D9D0] text-[#6A776F]">
                            <th className="pb-1 pr-2 text-left font-semibold">Time</th>
                            <th className="pb-1 pr-2 text-left font-semibold">Prompt</th>
                            <th className="pb-1 pr-2 text-center font-semibold">Users</th>
                            <th className="pb-1 pr-2 text-center font-semibold">Type</th>
                            <th className="pb-1 text-center font-semibold"></th>
                          </tr>
                        </thead>
                        <tbody>
                          {scheduleEvents.map((ev, idx) => (
                            <tr key={idx} className="border-b border-[#EAEDE8] last:border-0">
                              <td className="py-1.5 pr-2 font-mono text-[#3A3A3A] whitespace-nowrap">
                                {ev.time_min}min
                              </td>
                              <td className="py-1.5 pr-2 text-[#3A3A3A]" style={{ maxWidth: "180px" }}>
                                <span title={ev.prompt} className="block overflow-hidden text-ellipsis whitespace-nowrap">
                                  {ev.prompt}
                                </span>
                              </td>
                              <td className="py-1.5 pr-2 text-center text-[#3A3A3A]">{ev.users ?? 1}</td>
                              <td className="py-1.5 pr-2 text-center">
                                {(() => { const cls = ev.force_class ?? estimatePromptClass(ev.prompt ?? ""); return (
                                  <span className="rounded-full px-2 py-0.5 text-white text-[10px]" style={{ background: BURST_CLASS_COLOR[cls] }}>
                                    {BURST_CLASS_LABEL[cls]}
                                  </span>
                                ); })()}
                              </td>
                              <td className="py-1.5 text-center">
                                <button
                                  type="button"
                                  onClick={() => onDeleteScheduleRow(idx)}
                                  className="text-[#9AA09A] hover:text-[#C0392B] transition-colors leading-none"
                                  title="Remove event"
                                >
                                  ×
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="mt-2 text-[11px] text-[#9AA09A]">
                      No events yet. Select a [CoolSync] scenario, upload a CSV, or add rows manually.
                    </p>
                  )}
                </div>

                {/* ── Schedule CSV Upload ── */}
                <div className="mt-2">
                  <label className="flex cursor-pointer items-center gap-2 rounded-2xl border border-dashed border-[#C7CDC5] bg-[#F8FAF8] px-3 py-2 text-xs text-[#6A776F] transition hover:border-[#A8B4A5] hover:bg-[#F3F5F0]">
                    <Upload size={13} className="shrink-0 text-[#7A7F54]" />
                    <span>
                      Upload Schedule CSV
                      <span className="ml-1 text-[#9AA09A]">(time_min, prompt, users)</span>
                    </span>
                    <input
                      key={scheduleInputKey}
                      type="file"
                      accept=".csv"
                      className="hidden"
                      onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (f) onScheduleCsvUpload(f);
                        setScheduleInputKey(k => k + 1);
                      }}
                    />
                  </label>
                </div>

                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={6}
                  className="mt-2 w-full rounded-2xl border border-[#D3D9D0] bg-[#F3F5F0] p-3 text-sm text-[#262626] outline-none transition-colors focus:border-[#B8C3B5]"
                />
              </div>

              <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
                <div className="demo-controls-card demo-support-surface rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
                  <div className="flex items-center gap-2 text-sm font-semibold">
                    <SlidersHorizontal size={16} className="text-[#5E7766]" />
                    Controls
                  </div>

                  <div className="mt-3 space-y-3">
                    <div>
                      <div className="text-xs text-[#6F756E]">Model</div>
                      <select
                        value={model}
                        onChange={(e) => {
                          const nextModel = e.target.value;
                          setModel(nextModel);
                          setCalibrationState(getDefaultCalibrationState(nextModel));
                        }}
                        className="mt-1 w-full rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] p-2 text-sm text-[#262626]"
                      >
                        {MODELS.map((m) => (
                          <option key={m} value={m}>
                            {m}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <div className="text-xs text-[#6F756E]">Grid Intensity (gCO2/kWh)</div>
                      <select
                        value={grid}
                        onChange={(e) => setGrid(Number(e.target.value))}
                        className="mt-1 w-full rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] p-2 text-sm text-[#262626]"
                      >
                        {GRID_PRESETS.map((p) => (
                          <option key={`${p.label}-${p.g}`} value={p.g}>
                            {p.label} - {p.g}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <div className="text-xs text-[#6F756E]">Strategy</div>
                      <div className="mt-2 grid grid-cols-2 gap-2">
                        <button
                          type="button"
                          onClick={() => setStrategy("reactive")}
                          className={[
                            "rounded-xl border px-3 py-2 text-sm transition",
                            strategy === "reactive"
                              ? "border-[#DDC8B2] bg-[#F6EFE8] text-[#B4691F]"
                              : "border-[#D3D9D0] bg-[#F6F7F3] text-[#3A3A3A] hover:bg-[#DDE4DA]",
                          ].join(" ")}
                        >
                          Reactive
                        </button>
                        <button
                          type="button"
                          onClick={() => setStrategy("coordinated")}
                          className={[
                            "rounded-xl border px-3 py-2 text-sm transition",
                            strategy === "coordinated"
                              ? "border-[#C9DCCB] bg-[#EEF5EF] text-[#1F7A3A]"
                              : "border-[#D3D9D0] bg-[#F6F7F3] text-[#3A3A3A] hover:bg-[#DDE4DA]",
                          ].join(" ")}
                        >
                          Coordinated
                        </button>
                      </div>
                    </div>

                    <div>
                      <div className="text-xs text-[#6F756E]">Workload flexibility</div>
                      <select
                        value={workloadFlexibility}
                        onChange={(e) => setWorkloadFlexibility(e.target.value)}
                        className="mt-1 w-full rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] p-2 text-sm text-[#262626]"
                      >
                        <option value="urgent">Urgent / needed now</option>
                        <option value="flexible">Flexible / can wait</option>
                      </select>
                    </div>

                    <div className="rounded-2xl border border-[#D3D9D0] bg-[#F3F5F0] p-3">
                      <div className="text-xs text-[#6F756E]">System pressure</div>
                      <div className="mt-1 text-sm font-semibold text-[#262626]">
                        {capacityAssessment?.capacityPressure || "LOW"}
                      </div>
                      <div className="mt-1 text-xs text-[#6F756E]">
                        {capacityAssessment?.throughputRps?.toFixed(3) || "0.000"} req/s,{" "}
                        {capacityAssessment?.queueDelayMs || 0} ms waiting time
                      </div>
                    </div>
                  </div>
                </div>

                <div className="demo-support-grid grid grid-cols-2 gap-3">
                  <div className="demo-stat-card rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
                    <div className="text-xs text-[#6F756E]">Estimated tokens</div>
                    <div className="mt-1 text-xl font-extrabold text-[#6B7B48]">{tokens}</div>
                  </div>

                  <div className="demo-stat-card rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
                    <div className="text-xs text-[#6F756E]">Complexity</div>
                    <div className="mt-1 text-xl font-extrabold text-[#7B8F4B]">{complexity}</div>
                  </div>

                  <div className="demo-stat-card rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
                    <div className="text-xs text-[#6F756E]">Load risk</div>
                    <div className="mt-1 text-xl font-extrabold text-[#7A7F54]">{promptRisk}</div>
                  </div>

                  <div className="demo-stat-card rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
                    <div className="text-xs text-[#6F756E]">{valueTypeLabel} energy</div>
                    <div className="mt-1 text-lg font-bold text-[#262626]">
                      {activeRunSummary?.whPerRequest?.toFixed(3) || "0.000"} Wh
                    </div>
                    <div className="mt-1 text-xs text-[#6F756E]">
                      Based on GPU power models and literature ranges
                    </div>
                  </div>

                  <div className="demo-stat-card col-span-2 rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
                    <div className="text-xs text-[#6F756E]">Estimated cooling gap</div>
                    <div className="mt-1 text-lg font-bold text-[#262626]">
                      {(predictedOverheadDelta * 100).toFixed(0)} percentage points
                    </div>
                    <div className="mt-1 text-xs text-[#6F756E]">
                      {useTelemetry
                        ? "Observed telemetry trace active"
                        : "Published benchmark + simulation trace active"}
                    </div>
                  </div>
                </div>
              </div>

              <div className="demo-telemetry-card demo-support-surface mt-4 rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 text-sm font-semibold">
                    <Upload size={16} className="text-[#7A7F54]" />
                    Telemetry input (CSV)
                  </div>

                  <button
                    onClick={() => {
                      if (uploadState === "invalid") return;
                      if (!hasTelemetry) return;
                      setTraceSource((current) =>
                        current === "telemetry" ? "simulation" : "telemetry"
                      );
                    }}
                    className={[
                      "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs transition",
                      uploadState === "invalid"
                        ? "border-[#D8B8B2] bg-[#F2E5E2] text-[#7C5A53]"
                        : "border-[#D3D9D0] text-[#3A3A3A]",
                      hasTelemetry && uploadState !== "invalid"
                        ? "bg-[#DDE4DA] hover:bg-[#BFD98A]/45"
                        : "bg-[#DDE4DA] opacity-50 cursor-not-allowed",
                    ].join(" ")}
                    type="button"
                  >
                    {uploadState === "invalid" ? (
                      <>
                        <Upload size={16} className="text-[#7C5A53]" />
                        Bad CSV
                      </>
                    ) : useTelemetry ? (
                      <>
                        <ToggleRight size={16} className="text-[#6B7B48]" />
                        Telemetry active
                      </>
                    ) : (
                      <>
                        <ToggleLeft size={16} className="text-[#3A3A3A]" />
                        Using simulation
                      </>
                    )}
                  </button>
                </div>

                <div className="mt-3 flex items-center justify-between gap-3">
                  <input
                    type="file"
                    key={fileInputKey}
                    accept=".csv,text/csv"
                    onChange={(e) => onUploadCsv(e.target.files?.[0])}
                    className="block w-full text-xs text-[#3A3A3A] file:mr-3 file:rounded-full file:border file:border-[#D3D9D0] file:bg-[#DDE4DA] file:px-4 file:py-2 file:text-xs file:text-[#3A3A3A] hover:file:bg-[#DDE4DA]"
                  />

                  <div className="whitespace-nowrap text-xs text-[#6F756E]">
                    {telemetry?.meta?.format === "unknown" ? (
                      <span className="text-[#B4691F]">Wrong columns</span>
                    ) : hasTelemetry ? (
                      <span className="text-[#5F6B67]">
                        {telemetry.meta.format} | {telemetry.meta.rows} rows
                      </span>
                    ) : (
                      <span>No file yet</span>
                    )}
                  </div>
                </div>

                <div className="mt-3 flex items-center justify-end gap-2">
                  <button
                    onClick={clearTelemetry}
                    className="inline-flex items-center gap-2 rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1.5 text-xs text-[#3A3A3A] hover:bg-[#DDE4DA]"
                    type="button"
                  >
                    Clear data
                  </button>

                  <button
                    onClick={downloadSampleTelemetry}
                    className="inline-flex items-center gap-2 rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1.5 text-xs text-[#3A3A3A] hover:bg-[#DDE4DA]"
                    type="button"
                  >
                    <FileDown size={14} className="text-[#3A3A3A]" />
                    Download sample CSV
                  </button>

                  <button
                    onClick={downloadInvalidTelemetry}
                    className="inline-flex items-center gap-2 rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1.5 text-xs text-[#3A3A3A] hover:bg-[#DDE4DA]"
                    type="button"
                  >
                    <FileDown size={14} className="text-[#3A3A3A]" />
                    Download bad CSV
                  </button>
                </div>

                {telemetry?.meta?.error && (
                  <div className="mt-2 text-xs text-[#7A7F54]">{telemetry.meta.error}</div>
                )}

                <div className="mt-2 text-xs text-[#6F756E]">
                  Accepted columns:{" "}
                  {TELEMETRY_ACCEPTED_COLUMNS.map((columns, index) => (
                    <span key={columns}>
                      <span className="text-[#3A3A3A]">{columns}</span>
                      {index < TELEMETRY_ACCEPTED_COLUMNS.length - 1 ? " OR " : ""}
                    </span>
                  ))}
                </div>

                {hasTelemetry && telemetryWorkloadSummary?.hasBurstSignals && (
                  <div className="mt-3 rounded-xl border border-[#C9DCCB] bg-[#EEF5EF] px-3 py-3 text-xs text-[#35563D]">
                    <div className="font-semibold text-[#1F7A3A]">
                      Concurrent request signal detected
                    </div>
                    <div className="mt-1">
                      {telemetryBurstNarrative ||
                        "Telemetry includes request burst fields for multi-user workload analysis."}
                    </div>
                  </div>
                )}
              </div>

              {uploadState === "invalid" && (
                <div className="mt-3 rounded-lg border border-[#D8B8B2] bg-[#F2E5E2] px-3 py-2 text-xs text-[#7C5A53]">
                  CSV invalid → simulation estimate remains active
                </div>
              )}

              <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                <KpiCard
                  label={`${valueTypeLabel} energy`}
                  value={
                    <span>
                      <AnimatedNumber value={strategyWhRequest} decimals={3} /> Wh
                    </span>
                  }
                  hint={valueTypeMeta}
                  badge={useTelemetry ? "Observed" : "Estimated"}
                  accent={strategy === "coordinated" ? "emerald" : "amber"}
                />

                <KpiCard
                  label={carbonCardLabel}
                  value={<span>{formatCarbonImpact(strategyCo2eRequest)}</span>}
                  hint={carbonCardHint}
                  badge={useTelemetry ? "Observed" : "Estimated"}
                  accent="teal"
                />

                <KpiCard
                  label={useTelemetry ? "Observed cooling demand" : "Estimated cooling demand"}
                  value={
                    <span>
                      <AnimatedNumber value={activeRun?.avgCoolingKw || 0} decimals={2} /> kW
                    </span>
                  }
                  hint={
                    useTelemetry
                      ? "Derived from uploaded telemetry"
                      : "Cooling share benchmark typically falls in the 30–50% range"
                  }
                  badge={useTelemetry ? "Observed" : "Estimated"}
                  accent="teal"
                />
              </div>

              {hasTelemetry && telemetryWorkloadSummary?.hasBurstSignals && (
                <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div className="demo-result-strip demo-saving-card rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
                    <div className="text-xs text-[#6F756E]">Peak concurrent requests</div>
                    <div className="mt-1 text-lg font-bold text-[#8BC34A]">
                      {formatInteger(telemetryWorkloadSummary.peakRequests)}
                    </div>
                    <div className="mt-1 text-xs text-[#6F756E]">
                      Detected from uploaded telemetry request_count
                    </div>
                  </div>

                  <div className="demo-result-strip demo-saving-card rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
                    <div className="text-xs text-[#6F756E]">Peak workload tokens</div>
                    <div className="mt-1 text-lg font-bold text-[#5E7766]">
                      {formatInteger(telemetryWorkloadSummary.peakWorkloadTokens)}
                    </div>
                    <div className="mt-1 text-xs text-[#6F756E]">
                      Concurrent demand = request count × tokens per request
                    </div>
                  </div>
                </div>
              )}

              {hasTelemetry && telemetryWorkloadSummary?.hasPromptMix && (
                <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div className="demo-result-strip demo-saving-card rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
                    <div className="text-xs text-[#6F756E]">Dominant prompt type</div>
                    <div className="mt-1 text-lg font-bold text-[#262626]">
                      {telemetryWorkloadSummary.dominantPromptType || "Mixed"}
                    </div>
                    <div className="mt-1 text-xs text-[#6F756E]">
                      Most frequent request type in the uploaded telemetry
                    </div>
                  </div>

                  <div className="demo-result-strip demo-saving-card rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
                    <div className="text-xs text-[#6F756E]">Burst risk</div>
                    <div className="mt-1 text-lg font-bold text-[#262626]">
                      {telemetryWorkloadSummary.burstRisk || "LOW"}
                    </div>
                    <div className="mt-1 text-xs text-[#6F756E]">
                      Multi-user load classification from telemetry signals
                    </div>
                  </div>
                </div>
              )}

              <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="demo-result-strip demo-saving-card rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
                  <div className="text-xs text-[#6F756E]">Modeled energy difference</div>
                  <div className="mt-1 text-lg font-bold text-[#8BC34A]">
                    {comparisonWhSaved.toFixed(3)} Wh
                  </div>
                  <div className="mt-1 text-xs text-[#6F756E]">
                    Reactive estimate compared with coordinated estimate
                  </div>
                </div>

                <div className="demo-result-strip demo-saving-card rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
                  <div className="text-xs text-[#6F756E]">Modeled carbon difference</div>
                  <div className="mt-1 text-lg font-bold text-[#5E7766]">
                    {formatCarbonImpact(comparisonCo2eSaved)}
                  </div>
                  <div className="mt-1 text-xs text-[#6F756E]">
                    Better timing can reduce estimated carbon impact
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>

        <div ref={panelBRef}>
          <Card className="demo-output-shell ds-card--primary">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <Zap className="text-[#5F6B67]" size={18} />
                <div className="font-semibold">Console Output</div>

                <div className="ml-0 text-xs text-[#6F756E] sm:ml-2">
                  Trace: <span className="text-[#3A3A3A]">{TRACE_SOURCE_LABEL[traceSource]}</span>
                </div>

                <div className="text-xs text-[#6F756E]">
                  Strategy:{" "}
                  <span
                    className={
                      strategy === "coordinated" ? "text-[#8BC34A]" : "text-[#B4691F]"
                    }
                  >
                    {strategy === "coordinated" ? "Coordinated" : "Reactive"}
                  </span>
                </div>

                <StatusPill tone="slate">{outputDataSourceLabel}</StatusPill>
              </div>

              <div className="mt-4 rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-[#262626]">
                      Thermal vs cooling over time
                    </div>
                    <div className="mt-1 text-xs text-[#6F756E]">
                      Problem graph: heat rises first and cooling responds after.
                    </div>
                  </div>
                  <StatusPill tone="sky">Comparison view</StatusPill>
                </div>

                <p className="mt-3 rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] px-3 py-2 text-xs text-[#5E6A5F]">
                  Heat rises before cooling catches up. This is the coordination gap the system is trying to solve.
                </p>

                <div className="demo-proof-banner mt-3">
                  Coordinated cooling reduces delay and stabilizes temperature under load.
                </div>

                <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div className="demo-proof-card rounded-2xl border border-[#C9DCCB] bg-[#F7FAF5] p-3 shadow-[0_10px_24px_rgba(34,197,94,0.10)]">
                    <div className="card-heading text-xs">Modeled cooling delay reduction</div>
                    <div className="mt-1 text-2xl font-black text-[#166534]">
                      {comparison.lagReductionPct.toFixed(1)}%
                    </div>
                    <div className="kpi-delta mt-1">
                      {baselineRun?.lagMs} ms → {coordinatedRun?.lagMs} ms
                    </div>
                    <div className="card-subtext mt-1 text-xs">
                      Based on simulated workload conditions
                    </div>
                  </div>

                  <div className="demo-support-card rounded-2xl border border-[#D3D9D0] bg-[#F3F5F0] p-3">
                    <div className="text-xs text-[#6F756E]">Modeled peak temperature reduction</div>
                    <div className="mt-1 text-xl font-extrabold text-[#8BC34A]">
                      {comparison.peakInletReductionPct.toFixed(1)}%
                    </div>
                    <div className="mt-1 text-xs text-[#6F756E]">
                      {baselineRun?.peakInlet?.toFixed(2)}°C → {coordinatedRun?.peakInlet?.toFixed(2)}°C
                    </div>
                  </div>
                </div>

                {/* ── Chart 1: Inlet Temperature ── */}
                <div className="chart-shell chart-shell--demo mt-4">
                  <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.1em] text-[#6A776F]">
                    Inlet Temperature (°C)
                  </div>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart
                      data={comparisonChartData}
                      syncId="compChart"
                      margin={{ top: 4, right: 16, left: 8, bottom: 4 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#E1E5DD" />
                      <XAxis
                        dataKey="t"
                        tick={{ fill: "#4B5563", fontSize: 11 }}
                        tickFormatter={formatChartTime}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        domain={tempYDomain}
                        tick={{ fill: "#4B5563", fontSize: 11 }}
                        width={38}
                        label={{
                          value: "°C",
                          angle: -90,
                          position: "insideLeft",
                          offset: 8,
                          style: { fill: "#4B5563", fontSize: 11 },
                        }}
                      />
                      <Tooltip
                        formatter={(v, name) => [Number(v).toFixed(2) + " °C", name]}
                        labelFormatter={(v) => `t = ${formatChartTime(v)}`}
                        contentStyle={{ background: "#fff", border: "1px solid #D7DED8", borderRadius: 12, fontSize: 11 }}
                      />
                      <ReferenceLine y={25} stroke="#C67A2B" strokeDasharray="4 3" strokeOpacity={0.6} />
                      <ReferenceLine y={27} stroke="#8B2020" strokeDasharray="4 3" strokeOpacity={0.5} />
                      <PhaseMarkers phaseWindows={phaseWindows} />
                      <Line type="monotone" dataKey="reactiveInlet" name="Reactive" dot={false} stroke="#9CA3AF" strokeWidth={2.5} />
                      <Line type="monotone" dataKey="coordinatedInlet" name="Coordinated" dot={false} stroke="#8BC34A" strokeWidth={2.5} />
                    </LineChart>
                  </ResponsiveContainer>
                  <div className="mt-1 flex gap-3 text-[10px] text-[#6A776F]">
                    <span className="flex items-center gap-1"><span className="inline-block h-0.5 w-4 rounded" style={{ background: "#9CA3AF" }} />Reactive</span>
                    <span className="flex items-center gap-1"><span className="inline-block h-0.5 w-4 rounded" style={{ background: "#8BC34A" }} />Coordinated</span>
                    <span className="flex items-center gap-1 text-[#C67A2B]"><span className="inline-block h-0.5 w-4" style={{ borderTop: "2px dashed #C67A2B" }} />Safety</span>
                    <span className="flex items-center gap-1 text-[#8B2020]"><span className="inline-block h-0.5 w-4" style={{ borderTop: "2px dashed #8B2020" }} />Warning</span>
                  </div>
                </div>

                {/* ── Chart 2: Cooling Demand ── */}
                <div className="chart-shell chart-shell--demo mt-3">
                  <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.1em] text-[#6A776F]">
                    Cooling Demand (kW)
                  </div>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart
                      data={comparisonChartData}
                      syncId="compChart"
                      margin={{ top: 4, right: 16, left: 8, bottom: 20 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#E1E5DD" />
                      <XAxis
                        dataKey="t"
                        tick={{ fill: "#4B5563", fontSize: 11 }}
                        tickFormatter={formatChartTime}
                        label={{ value: "Time", position: "insideBottom", offset: -10, style: { fill: "#4B5563", fontSize: 11 } }}
                      />
                      <YAxis
                        tick={{ fill: "#4B5563", fontSize: 11 }}
                        width={38}
                        label={{
                          value: "kW",
                          angle: -90,
                          position: "insideLeft",
                          offset: 8,
                          style: { fill: "#4B5563", fontSize: 11 },
                        }}
                      />
                      <Tooltip
                        formatter={(v, name) => [Number(v).toFixed(2) + " kW", name]}
                        labelFormatter={(v) => `t = ${formatChartTime(v)}`}
                        contentStyle={{ background: "#fff", border: "1px solid #D7DED8", borderRadius: 12, fontSize: 11 }}
                      />
                      <PhaseMarkers phaseWindows={phaseWindows} />
                      <Line type="monotone" dataKey="reactiveCoolingKw" name="Reactive" dot={false} stroke="#9CA3AF" strokeDasharray="6 4" strokeWidth={2} />
                      <Line type="monotone" dataKey="coordinatedCoolingKw" name="Coordinated" dot={false} stroke="#1F7A3A" strokeDasharray="6 4" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                  <div className="mt-1 flex gap-3 text-[10px] text-[#6A776F]">
                    <span className="flex items-center gap-1"><span className="inline-block h-0.5 w-4 rounded border-t-2 border-dashed" style={{ borderColor: "#9CA3AF" }} />Reactive</span>
                    <span className="flex items-center gap-1"><span className="inline-block h-0.5 w-4 rounded border-t-2 border-dashed" style={{ borderColor: "#1F7A3A" }} />Coordinated</span>
                  </div>
                </div>

                <p className="chart-interpretation mt-3 rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] px-3 py-2 text-xs text-[#5E6A5F]">
                  Reactive mode lets temperature rise before cooling catches up. Coordinated mode shifts cooling earlier, reduces peak temperature, and narrows the lag window.
                </p>

                <div className="demo-comparison-table mt-4 rounded-2xl border border-[#D3D9D0] bg-[#F3F5F0] p-3">
                  <div className="text-sm font-semibold text-[#262626]">Comparison snapshot</div>
                  <p className="comparison-note mt-1 text-xs text-[#6F756E]">
                    Modeled comparison between reactive and coordinated cooling strategies.
                  </p>

                  <div className="mt-3 overflow-x-auto">
                    <table className="w-full text-left text-xs text-[#3A3A3A]">
                      <thead>
                        <tr className="border-b border-[#D3D9D0] text-[#6F756E]">
                          <th className="pb-2 pr-3 font-medium">Metric</th>
                          <th className="pb-2 pr-3 font-medium">Reactive</th>
                          <th className="pb-2 pr-3 font-medium">Coordinated</th>
                          <th className="pb-2 pr-3 font-medium">Change</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b border-[#E1E5DD]">
                          <td className="py-2 pr-3">Modeled cooling lag</td>
                          <td className="py-2 pr-3">{(baselineRun?.lagMs ?? 0) >= 60000 ? `${((baselineRun.lagMs)/60000).toFixed(0)} min` : `${baselineRun?.lagMs ?? 0} ms`}</td>
                          <td className="py-2 pr-3">{(coordinatedRun?.lagMs ?? 0) === 0 ? "0 (pre-cooled)" : `${((coordinatedRun.lagMs)/60000).toFixed(0)} min`}</td>
                          <td className="py-2 pr-3">-{comparison.lagReductionPct.toFixed(1)}%</td>
                        </tr>
                        <tr className="border-b border-[#E1E5DD]">
                          <td className="py-2 pr-3">Modeled building energy</td>
                          <td className="py-2 pr-3">
                            {Number(
                              mergedScenarioResults.reactive?.metrics?.totalFacilityEnergyWh || 0
                            ).toFixed(3)}{" "}
                            Wh
                          </td>
                          <td className="py-2 pr-3">
                            {Number(
                              mergedScenarioResults.coordinated?.metrics?.totalFacilityEnergyWh || 0
                            ).toFixed(3)}{" "}
                            Wh
                          </td>
                          <td className="py-2 pr-3">{facilityEnergyComparisonLabel}</td>
                        </tr>
                        <tr className="border-b border-[#E1E5DD]">
                          <td className="py-2 pr-3">Modeled carbon impact</td>
                          <td className="py-2 pr-3">
                            {formatCarbonImpact(
                              mergedScenarioResults.reactive?.runSummary?.co2ePerRequest
                            )}
                          </td>
                          <td className="py-2 pr-3">
                            {formatCarbonImpact(
                              mergedScenarioResults.coordinated?.runSummary?.co2ePerRequest
                            )}
                          </td>
                          <td className="py-2 pr-3">{formatCarbonImpact(comparisonCo2eSaved)}</td>
                        </tr>
                        {conservativeRun && (() => {
                          const consE  = mergedScenarioResults.conservative?.metrics?.totalWhCurrent ?? 0;
                          const dqnE   = mergedScenarioResults.coordinated?.metrics?.totalWhCurrent  ?? 0;
                          const pidE   = mergedScenarioResults.reactive?.metrics?.totalWhCurrent     ?? 0;
                          const savVsC = consE > 0 ? ((consE - dqnE) / consE * 100).toFixed(1) : "—";
                          const savPid = consE > 0 ? ((consE - pidE) / consE * 100).toFixed(1) : "—";
                          const peakC  = mergedScenarioResults.conservative?.metrics?.peakTRack?.toFixed(1) ?? "—";
                          const peakP  = mergedScenarioResults.reactive?.metrics?.peakTRack?.toFixed(1) ?? "—";
                          const peakD  = mergedScenarioResults.coordinated?.metrics?.peakTRack?.toFixed(1) ?? "—";
                          return (
                            <>
                              <tr className="border-b border-[#E1E5DD] bg-[#F6EFE8]">
                                <td className="py-2 pr-3 font-medium text-[#B4691F]">PUE 1.56 baseline (Conservative PID)</td>
                                <td className="py-2 pr-3 text-[#B4691F]">{consE.toFixed(1)} Wh</td>
                                <td className="py-2 pr-3 text-[#1F7A3A]">Coordinated saves <strong>-{savVsC}%</strong> vs baseline</td>
                                <td className="py-2 pr-3 text-[#5C645D]">Reactive saves -{savPid}%</td>
                              </tr>
                              <tr>
                                <td className="py-2 pr-3">Peak rack temperature</td>
                                <td className="py-2 pr-3">{peakP}°C (Reactive)</td>
                                <td className="py-2 pr-3">{peakD}°C (Coordinated)</td>
                                <td className="py-2 pr-3 text-[#B4691F]">{peakC}°C (Conservative)</td>
                              </tr>
                            </>
                          );
                        })()}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {hasTelemetry && telemetryWorkloadSummary?.hasBurstSignals && (
                <div className="mt-4 rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-[#262626]">
                        Workload vs energy over time
                      </div>
                      <div className="mt-1 text-xs text-[#6F756E]">
                        Cause graph: prompt bursts drive power demand before thermal stress appears.
                      </div>
                    </div>
                    <StatusPill tone="sky">Telemetry view</StatusPill>
                  </div>

                  <p className="mt-3 rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] px-3 py-2 text-xs text-[#5E6A5F]">
                    This graph shows the cause. When prompt traffic increases, power demand rises, which then drives heat and cooling stress.
                  </p>

                  <div className="chart-shell chart-shell--demo mt-4">
                    <ResponsiveContainer width="100%" height={320}>
                      <ComposedChart data={workloadEnergyData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E1E5DD" />
                        <XAxis
                          dataKey="t"
                          tick={{ fill: "#4B5563", fontSize: 11 }}
                          tickFormatter={formatChartTime}
                          label={{
                            value: "Time (s)",
                            position: "insideBottom",
                            offset: -6,
                            style: { fill: "#4B5563", fontSize: 12 },
                          }}
                        />
                        <YAxis
                          yAxisId="requests"
                          tick={{ fill: "#4B5563", fontSize: 11 }}
                          label={{
                            value: "Concurrent Requests",
                            angle: -90,
                            position: "insideLeft",
                            style: { fill: "#4B5563", fontSize: 12 },
                          }}
                        />
                        <YAxis
                          yAxisId="power"
                          orientation="right"
                          tick={{ fill: "#4B5563", fontSize: 11 }}
                          label={{
                            value: "Total Power Demand (W)",
                            angle: 90,
                            position: "insideRight",
                            style: { fill: "#4B5563", fontSize: 12 },
                          }}
                        />
                        <Tooltip
                          formatter={(value, name, item) => {
                            const payload = item?.payload || {};
                            if (name === "Concurrent Requests") {
                              return [
                                `${formatInteger(value)} requests`,
                                `${name} · ${formatInteger(payload.workloadTokens)} workload tokens`,
                              ];
                            }

                            return [
                              `${formatInteger(value)} W`,
                              name,
                            ];
                          }}
                          labelFormatter={(value) => `t = ${formatChartTime(value)}`}
                        />
                        <Legend content={CustomLegend} />
                        <Bar
                          yAxisId="requests"
                          dataKey="requestCount"
                          name="Concurrent Requests"
                          fill="#8BC34A"
                          radius={[6, 6, 0, 0]}
                        />
                        <Line
                          yAxisId="power"
                          type="monotone"
                          dataKey="totalPowerW"
                          name="Total Power Demand (W)"
                          dot={false}
                          stroke="#5E7766"
                          strokeWidth={2.5}
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>

                  <p className="chart-interpretation mt-3 rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] px-3 py-2 text-xs text-[#5E6A5F]">
                    Prompt bursts increase workload demand, which raises power consumption and contributes to heat buildup.
                  </p>
                </div>
              )}

              {hasTelemetry && telemetryWorkloadSummary?.hasBurstSignals && (
                <div className="mt-4 rounded-2xl border border-[#C9DCCB] bg-[#F7FAF5] p-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-[#1F2D22]">
                        Concurrent workload insight
                      </div>
                      <div className="mt-1 text-xs text-[#5E6A5F]">
                        Telemetry-derived view of what happens when many users submit prompts at once.
                      </div>
                    </div>
                    <StatusPill
                      tone={
                        telemetryWorkloadSummary?.burstRisk === "HIGH"
                          ? "amber"
                          : "emerald"
                      }
                    >
                      {telemetryWorkloadSummary?.burstRisk || "LOW"} burst risk
                    </StatusPill>
                  </div>

                  <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
                    <div className="rounded-xl border border-[#D3D9D0] bg-white/70 px-3 py-3">
                      <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#6A776F]">
                        Peak concurrent requests
                      </div>
                      <div className="mt-2 text-lg font-bold text-[#262626]">
                        {formatInteger(telemetryWorkloadSummary.peakRequests)}
                      </div>
                    </div>

                    <div className="rounded-xl border border-[#D3D9D0] bg-white/70 px-3 py-3">
                      <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#6A776F]">
                        Peak workload tokens
                      </div>
                      <div className="mt-2 text-lg font-bold text-[#262626]">
                        {formatInteger(telemetryWorkloadSummary.peakWorkloadTokens)}
                      </div>
                    </div>

                    <div className="rounded-xl border border-[#D3D9D0] bg-white/70 px-3 py-3">
                      <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#6A776F]">
                        Dominant prompt type
                      </div>
                      <div className="mt-2 text-lg font-bold text-[#262626]">
                        {telemetryWorkloadSummary.dominantPromptType || "Mixed"}
                      </div>
                    </div>
                  </div>

                  <div className="mt-3 rounded-xl border border-[#D3D9D0] bg-white/70 px-3 py-3 text-sm text-[#2D3A2D]">
                    {telemetryBurstNarrative ||
                      "Uploaded telemetry contains prompt mix and concurrent request signals."}
                  </div>
                </div>
              )}

              <div className="mt-4 rounded-2xl border border-[#C9DCCB] bg-[#F7FAF5] p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-[#1F2D22]">Research context</div>
                    <div className="mt-1 text-xs text-[#5E6A5F]">
                      Context used to ground the simulation and the decision-support narrative.
                    </div>
                  </div>
                  <StatusPill tone="emerald">Research-backed</StatusPill>
                </div>

                <ul className="mt-3 space-y-2 text-sm text-[#2D3A2D]">
                  <li className="rounded-xl border border-[#D3D9D0] bg-white/70 px-3 py-2">
                    Cooling accounts for ~30–50% of total data center energy use.
                  </li>
                  <li className="rounded-xl border border-[#D3D9D0] bg-white/70 px-3 py-2">
                    Coordinated cooling can reduce thermal lag and improve efficiency.
                  </li>
                  <li className="rounded-xl border border-[#D3D9D0] bg-white/70 px-3 py-2">
                    Data centers are among the most energy-intensive digital infrastructures.
                  </li>
                </ul>
              </div>

              <div className="demo-insight-card demo-recommendation-card mt-4 rounded-2xl border border-[rgba(180,105,31,0.22)] bg-[#FFF7EE] p-4 shadow-[0_10px_24px_rgba(180,105,31,0.08)]">
                <div className="text-sm font-bold text-[#8A5923]">&#9888; Recommended</div>
                <div className="mt-2 text-xs text-[#8A5923]">Recommended action (modeled)</div>

                <div className="mt-3 space-y-2 text-sm text-[#3A3A3A]">
                  {comparisonNarrative.slice(0, 3).map((item, idx) => (
                    <div
                      key={idx}
                      className="demo-recommendation-item rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] px-3 py-2"
                    >
                      {item}
                    </div>
                  ))}
                  <div className="demo-recommendation-item rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] px-3 py-2">
                    Coordinated mode is recommended for this workload profile.
                  </div>
                  {useTelemetry && telemetryWorkloadSummary?.hasBurstSignals && (
                    <div className="demo-recommendation-item rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] px-3 py-2">
                      Concurrent prompt bursts are visible in telemetry, so workload shaping or
                      pre-cooling should be considered before peak demand windows.
                    </div>
                  )}
                </div>
              </div>

              <div className="mt-4 rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-[#262626]">Telemetry and calibration</div>
                    <div className="mt-1 text-xs text-[#6F756E]">
                      Uploaded data can adjust the estimate and increase confidence.
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <StatusPill tone={useTelemetry ? "emerald" : "amber"}>
                      {useTelemetry ? "Observed + calibrated" : "Estimated view"}
                    </StatusPill>
                    <StatusPill tone="slate">
                      {displayCalibrationState?.status || "UNCALIBRATED"}
                    </StatusPill>
                  </div>
                </div>

                {useTelemetry && (
                  <div className="mt-3 rounded-xl border border-[#BDD0FF] bg-[#E8F0FF] px-3 py-2 text-xs text-[#1F4FD8]">
                    Observed values are derived from uploaded telemetry data.
                  </div>
                )}

                <div className="mt-4">
                  <CalibrationPanel calibrationState={displayCalibrationState} />
                </div>
              </div>

              <div className="limit-box demo-limit-card mt-4 rounded-2xl p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div className="limit-title text-sm">Console limits</div>
                  <StatusPill tone="amber">Important note</StatusPill>
                </div>
                <div className="grid grid-cols-1 gap-2 text-sm">
                  <div className="demo-limit-item rounded-xl border border-[rgba(198,122,43,0.18)] bg-[#FFF9F2] px-3 py-2">
                    Building control and cooling system are not connected.
                  </div>
                  <div className="demo-limit-item rounded-xl border border-[rgba(198,122,43,0.18)] bg-[#FFF9F2] px-3 py-2">
                    Telemetry comes from CSV upload only.
                  </div>
                  <div className="demo-limit-item rounded-xl border border-[rgba(198,122,43,0.18)] bg-[#FFF9F2] px-3 py-2">
                    Recommendations are rule-based for this demo.
                  </div>
                </div>
              </div>

              <div className="demo-event-log-card mt-4 rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-4">
                <div className="card-heading text-sm">Event log</div>
                <div className="card-muted mt-1 text-xs">
                  Recent system activity from the current run or uploaded telemetry.
                </div>
                <div className="mt-3 space-y-2 text-sm">
                  {eventLogItems.slice(0, 6).map((item, idx) => (
                    <div
                      key={idx}
                      className="event-log-item demo-event-log-item rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] px-3 py-2"
                    >
                      <div className="event-time">{item.time}</div>
                      <div className="event-text mt-1">{item.text}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </Section>
  );
}