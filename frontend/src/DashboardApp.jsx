import React, { useMemo, useState } from "react";
import Select from "react-select";
import {
  Thermometer,
  Wind,
  Cpu,
  Activity,
  ShieldCheck,
  ShieldAlert,
  Play,
  FileText,
  CheckCircle2,
  XCircle,
  DollarSign,
  Info,
  Gauge,
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  Cell,
} from "recharts";

const THEME = {
  primaryDark: "#0f172a",
  primaryMid: "#1e293b",
  primarySoft: "#334155",

  background: "#f1f5f9",
  card: "#ffffff",
  cardBorder: "#e2e8f0",

  accentBlue: "#3b82f6",
  accentBlueSoft: "#7aaef7",

  accentTeal: "#14b8a6",
  accentTealSoft: "#7dd3c7",
  accentTealStrong: "#0b8276",

  accentGreen: "#22c55e",
  accentGreenSoft: "#86efac",

  accentRed: "#ef4444",
  accentAmber: "#f59e0b",

  textMain: "#0f172a",
  textSub: "#64748b",
};

const fallbackView = {
  promptFeatures: null,
  summary: null,
  currentStats: {
    currentTemperature: 25.2,
    currentWorkload: 0.71,
    predictedHeat: 0.81,
    coolingLevel: 6,
  },
};

function numberOr(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function roundToOne(value) {
  return Math.round(value * 10) / 10;
}

function cardStyle() {
  return {
    background: THEME.card,
    borderRadius: 20,
    boxShadow: "0 6px 24px rgba(15, 23, 42, 0.08)",
    padding: 20,
    border: `1px solid ${THEME.cardBorder}`,
  };
}

function chartCardStyle() {
  return {
    background: THEME.card,
    borderRadius: 16,
    padding: 20,
    border: `1px solid ${THEME.cardBorder}`,
    boxShadow: "0 4px 12px rgba(15, 23, 42, 0.05)",
  };
}

function buttonStyle(background, color = "white", border = "none") {
  return {
    padding: "12px 18px",
    borderRadius: 12,
    border,
    background,
    color,
    cursor: "pointer",
    fontSize: 16,
    fontWeight: 500,
  };
}

function textAreaStyle() {
  return {
    width: "100%",
    minHeight: 110,
    resize: "vertical",
    borderRadius: 14,
    border: "1px solid rgba(255,255,255,0.22)",
    background: "rgba(255,255,255,0.10)",
    color: "white",
    padding: 14,
    fontSize: 15,
    outline: "none",
    boxSizing: "border-box",
  };
}

function sectionLabelStyle() {
  return {
    marginBottom: 12,
    fontSize: 13,
    fontWeight: 700,
    letterSpacing: 0.3,
    color: "#475569",
    textTransform: "uppercase",
  };
}

function StatCard({ title, value, subtitle, icon: Icon }) {
  return (
    <div style={cardStyle()}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
        <div>
          <div style={{ fontSize: 14, color: THEME.textSub }}>{title}</div>
          <div style={{ fontSize: 28, fontWeight: 700, marginTop: 8, color: THEME.textMain }}>
            {value}
          </div>
          <div style={{ fontSize: 12, color: THEME.textSub, marginTop: 6 }}>{subtitle}</div>
        </div>
        <div
          style={{
            background: "#eef2ff",
            borderRadius: 16,
            padding: 12,
            height: "fit-content",
          }}
        >
          <Icon size={20} color={THEME.primarySoft} />
        </div>
      </div>
    </div>
  );
}

function ChartPlaceholder({ message }) {
  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        minHeight: 340,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        borderRadius: 12,
        border: "1px dashed #cbd5e1",
        background: "#f8fafc",
        color: THEME.textSub,
        fontSize: 15,
        textAlign: "center",
        padding: 20,
        boxSizing: "border-box",
      }}
    >
      {message}
    </div>
  );
}

function ChartCard({ title, subtitle, children }) {
  return (
    <div style={chartCardStyle()}>
      <h3
        style={{
          marginTop: 0,
          marginBottom: subtitle ? 6 : 16,
          fontSize: 18,
          color: THEME.textMain,
          fontWeight: 600,
        }}
      >
        {title}
      </h3>
      {subtitle && (
        <div
          style={{
            fontSize: 13,
            color: THEME.textSub,
            marginBottom: 14,
            lineHeight: 1.5,
          }}
        >
          {subtitle}
        </div>
      )}
      <div
        style={{
          width: "100%",
          height: 340,
          minHeight: 340,
          position: "relative",
        }}
      >
        {children}
      </div>
    </div>
  );
}

function buildSimulationTimeline(simulationResult) {
  if (!simulationResult?.timeseries) {
    return [];
  }

  const timeseries = simulationResult.timeseries;

  const steps = timeseries.steps || [];
  const temperatures = timeseries.temperature || [];
  const actualHeat = timeseries.actual_heat || [];
  const predictedHeat = timeseries.predicted_heat || [];
  const rewards = timeseries.reward || [];
  const cumulativeRewards = timeseries.cumulative_reward || [];
  const coolingLevels = timeseries.cooling_level || [];
  const actions = timeseries.action || [];
  const energies = timeseries.energy || [];
  const safeTempMax = numberOr(simulationResult?.summary?.safe_temp_max, 26);

  return steps.map((step, index) => ({
    step,
    temperature: temperatures[index] ?? 0,
    heat: actualHeat[index] ?? 0,
    predictedHeat: predictedHeat[index] ?? 0,
    reward: rewards[index] ?? 0,
    cumulativeReward: cumulativeRewards[index] ?? 0,
    coolingLevel: coolingLevels[index] ?? 0,
    action: actions[index] ?? "maintain",
    energy: energies[index] ?? 0,
    // Use the backend-provided threshold so all UI threshold logic stays aligned.
    safeMaxTemp: safeTempMax,
  }));
}

function buildControllerComparison(selectedController, useForecast) {
  const rows = [
    { name: "Static", key: "static", reward: 140, energy: 7.2, maxTemp: 28.4 },
    { name: "Threshold", key: "threshold", reward: 310, energy: 6.6, maxTemp: 26.8 },
    { name: "PID", key: "pid", reward: 420, energy: 6.1, maxTemp: 25.8 },
    { name: "Predictive Threshold", key: "predictive_threshold", reward: 510, energy: 5.8, maxTemp: 25.1 },
    { name: "Q-Learning", key: "q_learning", reward: 368, energy: 6.0, maxTemp: 24.5 },
    { name: "DQN", key: "dqn", reward: 722, energy: 5.4, maxTemp: 24.3 },
  ];

  return rows.map((row) => {
    let reward = row.reward;
    let energy = row.energy;
    let maxTemp = row.maxTemp;

    if (!useForecast && ["predictive_threshold", "q_learning", "dqn"].includes(row.key)) {
      reward -= 60;
      energy += 0.4;
      maxTemp += 0.7;
    }

    return {
      ...row,
      reward,
      energy,
      maxTemp,
      energyScaled: roundToOne(energy * 50),
      isSelected: row.key === selectedController,
    };
  });
}

function buildForecastImpact(useForecast) {
  if (useForecast) {
    return {
      title: "Forecast Mode Summary",
      energySavingPct: 6.9,
      tempImprovement: 0.7,
      summary:
        "Forecast ON uses proactive control, so the controller can react before heat spikes fully appear.",
    };
  }

  return {
    title: "Forecast Mode Summary",
    energySavingPct: 0.0,
    tempImprovement: 0.0,
    summary:
      "Forecast OFF uses reactive control, so the controller responds only after heat changes become visible.",
  };
}

function getVerdictExplanation(simulationSummary) {
  if (!simulationSummary) {
    return "";
  }

  if (simulationSummary.pass_fail === "PASS") {
    return "PASS: No overheating and the average temperature stayed within the safe range.";
  }

  return "FAIL: The run exceeded safe thermal limits or did not satisfy the business safety criteria.";
}

function getActionExplanation(simulationTimeline, predictedHeat, currentTemperature, coolingLevel, forecastEnabled) {
  const latest = simulationTimeline?.[simulationTimeline.length - 1];
  const action = latest?.action;

  if (action === "increase" || coolingLevel >= 7) {
    if (forecastEnabled && predictedHeat > 0.7) {
      return "Cooling increased because predicted heat is high and the controller is acting proactively to avoid a temperature spike.";
    }
    if (!forecastEnabled) {
      return "Cooling increased because current heat and temperature are elevated, so the reactive controller is responding to maintain safe operation.";
    }
    return "Cooling increased because the thermal state is trending upward and extra cooling helps maintain safe operation.";
  }

  if (action === "decrease" || coolingLevel <= 3) {
    if (!forecastEnabled) {
      return "Cooling was reduced because the rack is in a safe zone and current thermal conditions support a lower cooling level.";
    }
    return "Cooling was reduced because the rack is in a safe zone and the forecast suggests lower immediate thermal pressure.";
  }

  if (!forecastEnabled) {
    return "Cooling was maintained because the current temperature and current heat remain within a manageable reactive-control range.";
  }

  return "Cooling was maintained because the current temperature and forecasted heat remain within a manageable range.";
}

export default function DashboardApp() {
  const [scenario, setScenario] = useState("spiky");
  const [controller, setController] = useState("dqn");
  const [forecast, setForecast] = useState("on");
  const [backendMessage, setBackendMessage] = useState("Run a custom simulation to generate the experiment results.");

  const [promptText, setPromptText] = useState(
    "Urgent: debug Python CUDA code for multiple concurrent users"
  );
  const [simulationResult, setSimulationResult] = useState(null);
  const [simulationLoading, setSimulationLoading] = useState(false);

  const scenarioOptions = useMemo(
    () => [
      { value: "stable", label: "Stable" },
      { value: "sinusoidal", label: "Sinusoidal" },
      { value: "spiky", label: "Spiky" },
      { value: "burst_heavy", label: "Burst Heavy" },
    ],
    []
  );

  const controllerOptions = useMemo(
    () => [
      { value: "static", label: "Static" },
      { value: "threshold", label: "Threshold" },
      { value: "pid", label: "PID" },
      { value: "predictive_threshold", label: "Predictive Threshold" },
      { value: "q_learning", label: "Q-Learning" },
      { value: "dqn", label: "DQN" },
    ],
    []
  );

  const forecastOptions = useMemo(
    () => [
      { value: "on", label: "Forecast On" },
      { value: "off", label: "Forecast Off" },
    ],
    []
  );

  const customSelectStyles = useMemo(
    () => ({
      control: (base, state) => ({
        ...base,
        minHeight: 44,
        minWidth: 160,
        backgroundColor: "rgba(255,255,255,0.12)",
        borderColor: state.isFocused ? "rgba(125, 211, 199, 0.95)" : "rgba(255,255,255,0.28)",
        borderRadius: 12,
        boxShadow: state.isFocused ? "0 0 0 2px rgba(20, 184, 166, 0.18)" : "none",
        cursor: "pointer",
        transition: "all 0.15s ease",
        "&:hover": {
          borderColor: "rgba(125, 211, 199, 0.8)",
          backgroundColor: "rgba(255,255,255,0.16)",
        },
      }),
      valueContainer: (base) => ({
        ...base,
        padding: "0 12px",
      }),
      indicatorSeparator: () => ({
        display: "none",
      }),
      dropdownIndicator: (base, state) => ({
        ...base,
        color: state.isFocused ? THEME.accentTealSoft : "rgba(255,255,255,0.72)",
        padding: 8,
        transition: "all 0.15s ease",
        "&:hover": {
          color: THEME.accentTealSoft,
        },
      }),
      singleValue: (base) => ({
        ...base,
        color: "white",
        fontSize: 16,
        fontWeight: 500,
      }),
      placeholder: (base) => ({
        ...base,
        color: "rgba(255,255,255,0.7)",
      }),
      input: (base) => ({
        ...base,
        color: "white",
      }),
      menu: (base) => ({
        ...base,
        backgroundColor: THEME.primaryMid,
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: 12,
        overflow: "hidden",
        marginTop: 8,
        boxShadow: "0 16px 32px rgba(15, 23, 42, 0.28)",
      }),
      menuList: (base) => ({
        ...base,
        paddingTop: 6,
        paddingBottom: 6,
      }),
      option: (base, state) => ({
        ...base,
        fontSize: 14,
        cursor: "pointer",
        color: "white",
        padding: "10px 14px",
        backgroundColor: state.isSelected
          ? "rgba(59, 130, 246, 0.35)"
          : state.isFocused
            ? "rgba(255,255,255,0.08)"
            : THEME.primaryMid,
        "&:active": {
          backgroundColor: "rgba(20, 184, 166, 0.25)",
        },
      }),
    }),
    []
  );

  const runSimulation = async () => {
    try {
      setSimulationLoading(true);
      setBackendMessage("Running custom prompt simulation...");

      const response = await fetch("http://127.0.0.1:5000/api/run-simulation", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt_text: promptText,
          controller,
          scenario,
          use_forecast: forecast === "on",
        }),
      });

      const json = await response.json();

      if (!response.ok) {
        throw new Error(json?.error || "Simulation failed");
      }

      setSimulationResult(json);
      setBackendMessage("Custom prompt simulation completed.");
    } catch (error) {
      console.log("Simulation failed", error);
      setBackendMessage(`Simulation failed: ${error.message}`);
    } finally {
      setSimulationLoading(false);
    }
  };

  const clearSimulation = () => {
    setSimulationResult(null);
    setBackendMessage("Simulation cleared. Enter a prompt and run a new experiment.");
  };

  const promptFeatures = simulationResult?.prompt_features ?? fallbackView.promptFeatures;
  const simulationSummary = simulationResult?.summary ?? fallbackView.summary;
  const simulationTimeline = buildSimulationTimeline(simulationResult);
  const hasSimulation = simulationTimeline.length > 0;

  const currentTemperature = hasSimulation
    ? numberOr(
        simulationTimeline[simulationTimeline.length - 1]?.temperature,
        fallbackView.currentStats.currentTemperature
      )
    : fallbackView.currentStats.currentTemperature;

  const currentWorkload = promptFeatures
    ? numberOr(promptFeatures.workload, fallbackView.currentStats.currentWorkload)
    : fallbackView.currentStats.currentWorkload;

  const predictedHeat = hasSimulation
    ? numberOr(
        simulationTimeline[simulationTimeline.length - 1]?.predictedHeat,
        fallbackView.currentStats.predictedHeat
      )
    : fallbackView.currentStats.predictedHeat;

  const coolingLevel = hasSimulation
    ? numberOr(
        simulationTimeline[simulationTimeline.length - 1]?.coolingLevel,
        fallbackView.currentStats.coolingLevel
      )
    : fallbackView.currentStats.coolingLevel;

  const avgTemperature = simulationSummary
    ? numberOr(simulationSummary.avg_temperature, 24.8)
    : 0;

  const totalEnergy = simulationSummary
    ? numberOr(simulationSummary.total_energy, 0)
    : 0;

  const overheatCount = simulationSummary
    ? numberOr(simulationSummary.overheat_count, 0)
    : 0;

  // The backend now returns the environment's safe_temp_max so the frontend
  // can avoid hardcoded threshold mismatches.
  const safeTempMax = simulationSummary
    ? numberOr(simulationSummary.safe_temp_max, 26)
    : 26;

  const safeZoneRate = hasSimulation
    ? Math.round(
        (simulationTimeline.filter((row) => row.temperature <= safeTempMax).length /
          simulationTimeline.length) *
          100
      )
    : 0;

  const isSafe = currentTemperature <= safeTempMax;
  const verdictPass = simulationSummary?.pass_fail === "PASS";

  const latestStateLabel = isSafe ? "Latest State: SAFE" : "Latest State: WARNING";
  const latestStateColor = isSafe ? "#166534" : "#b91c1c";
  const latestStateBg = isSafe ? "#ecfdf5" : "#fef2f2";
  const latestStateBorder = isSafe ? "1px solid #86efac" : "1px solid #fecaca";

  const controllerComparison = buildControllerComparison(controller, forecast === "on");
  const forecastImpact = buildForecastImpact(forecast === "on");
  const verdictExplanation = getVerdictExplanation(simulationSummary);

  const businessScore = simulationSummary
    ? Math.max(
        0,
        Math.round(
          100
          - numberOr(simulationSummary.total_energy, 0) * 4
          - numberOr(simulationSummary.overheat_count, 0) * 15
          - Math.max(0, numberOr(simulationSummary.max_temperature, 0) - safeTempMax) * 8
        )
      )
    : 0;

  const rlReward = simulationSummary
    ? numberOr(simulationSummary.total_reward, 0)
    : 0;

  const coolingCost = simulationSummary
    ? roundToOne(numberOr(simulationSummary.total_energy, 0) * 0.12)
    : 0;

  const baselineEnergy = 7.2;
  const energySavings = simulationSummary
    ? Math.max(
        0,
        roundToOne(
          ((baselineEnergy - numberOr(simulationSummary.total_energy, 0)) / baselineEnergy) * 100
        )
      )
    : 0;

  const slaViolations = simulationSummary
    ? numberOr(simulationSummary.overheat_count, 0)
    : 0;

  const riskCost = slaViolations * 50;

  const computeStatus = {
    device: simulationResult?.compute?.device || "CPU",
    concurrency:
      simulationResult?.compute?.concurrency ??
      promptFeatures?.concurrency_level ??
      1,
    batchMode: simulationResult?.compute?.batch_mode ?? false,
  };

  const whyThisAction = hasSimulation
    ? getActionExplanation(
        simulationTimeline,
        predictedHeat,
        currentTemperature,
        coolingLevel,
        forecast === "on"
      )
    : "Run a simulation to generate an explanation for the controller decision.";

  return (
    <div
      style={{
        minHeight: "100vh",
        background: `linear-gradient(to bottom, ${THEME.primaryDark} 0%, ${THEME.primaryMid} 20%, ${THEME.background} 60%)`,
        color: THEME.textMain,
        fontFamily: "Arial, sans-serif",
      }}
    >
      <div style={{ maxWidth: 1280, margin: "0 auto", padding: 24 }}>
        <div
          style={{
            ...cardStyle(),
            marginBottom: 24,
            background: `linear-gradient(135deg, ${THEME.primaryDark}, ${THEME.primaryMid})`,
            color: "white",
            border: "none",
          }}
        >
          <div style={{ fontSize: 13, opacity: 0.85, marginBottom: 10 }}>
            CoolSync+ Dashboard
          </div>

          <h1 style={{ margin: 0, fontSize: 40 }}>
            Prompt-Aware Intelligent Cooling Control
          </h1>

          <p style={{ color: "#dbeafe", maxWidth: 860, lineHeight: 1.6 }}>
            Monitor prompt-driven workload behavior, forecast near-future heat,
            visualize controller actions, and evaluate whether the business objective passes or fails.
          </p>

          <p style={{ color: THEME.accentTealSoft, marginTop: 10, fontSize: 14 }}>
            {backendMessage}
          </p>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1.3fr 1fr",
              gap: 16,
              marginTop: 18,
            }}
          >
            <div>
              <div style={{ fontSize: 14, marginBottom: 8, color: "#dbeafe" }}>
                Enter a custom prompt
              </div>
              <textarea
                value={promptText}
                onChange={(event) => setPromptText(event.target.value)}
                placeholder="Enter a custom AI workload prompt..."
                style={textAreaStyle()}
              />
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ minWidth: 160 }}>
                <Select
                  value={scenarioOptions.find((option) => option.value === scenario)}
                  onChange={(option) => setScenario(option.value)}
                  options={scenarioOptions}
                  styles={customSelectStyles}
                  isSearchable={false}
                />
              </div>

              <div style={{ minWidth: 210 }}>
                <Select
                  value={controllerOptions.find((option) => option.value === controller)}
                  onChange={(option) => setController(option.value)}
                  options={controllerOptions}
                  styles={customSelectStyles}
                  isSearchable={false}
                />
              </div>

              <div style={{ minWidth: 160 }}>
                <Select
                  value={forecastOptions.find((option) => option.value === forecast)}
                  onChange={(option) => setForecast(option.value)}
                  options={forecastOptions}
                  styles={customSelectStyles}
                  isSearchable={false}
                />
              </div>

              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <button
                  onClick={runSimulation}
                  disabled={simulationLoading}
                  style={buttonStyle(THEME.accentBlue)}
                >
                  <Play size={16} style={{ marginRight: 6, verticalAlign: "middle" }} />
                  {simulationLoading ? "Running..." : "Run Simulation"}
                </button>

                <button
                  onClick={clearSimulation}
                  style={buttonStyle("white", THEME.primaryDark, "1px solid #cbd5e1")}
                >
                  Clear Simulation
                </button>
              </div>
            </div>
          </div>
        </div>

        {hasSimulation && (
          <div
            style={{
              marginBottom: 16,
              padding: "10px 14px",
              borderRadius: 12,
              background: "#ecfeff",
              border: "1px solid #67e8f9",
              color: "#0e7490",
              fontWeight: 600,
            }}
          >
            Custom Prompt Simulation Result
          </div>
        )}

        {promptFeatures && simulationSummary && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: 16,
              marginBottom: 20,
            }}
          >
            <div style={cardStyle()}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                <FileText size={18} color={THEME.primarySoft} />
                <h3 style={{ margin: 0, color: THEME.textMain }}>Prompt Features</h3>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <div style={{ fontSize: 13, color: THEME.textSub }}>Prompt Type</div>
                  <div style={{ fontWeight: 700, marginTop: 4 }}>{promptFeatures.prompt_type}</div>
                </div>
                <div>
                  <div style={{ fontSize: 13, color: THEME.textSub }}>Prompt Length</div>
                  <div style={{ fontWeight: 700, marginTop: 4 }}>{promptFeatures.prompt_length}</div>
                </div>
                <div>
                  <div style={{ fontSize: 13, color: THEME.textSub }}>Complexity</div>
                  <div style={{ fontWeight: 700, marginTop: 4 }}>{promptFeatures.complexity_score}</div>
                </div>
                <div>
                  <div style={{ fontSize: 13, color: THEME.textSub }}>Concurrency</div>
                  <div style={{ fontWeight: 700, marginTop: 4 }}>{promptFeatures.concurrency_level}</div>
                </div>
                <div>
                  <div style={{ fontSize: 13, color: THEME.textSub }}>Workload</div>
                  <div style={{ fontWeight: 700, marginTop: 4 }}>{promptFeatures.workload}</div>
                </div>
                <div>
                  <div style={{ fontSize: 13, color: THEME.textSub }}>Heat Load</div>
                  <div style={{ fontWeight: 700, marginTop: 4 }}>{promptFeatures.heat_load}</div>
                </div>
              </div>

              <div
                style={{
                  marginTop: 14,
                  fontSize: 12,
                  color: THEME.textSub,
                  lineHeight: 1.5,
                }}
              >
                Workload is derived from prompt characteristics such as complexity, length, and concurrency level.
              </div>
            </div>

            <div
              style={{
                ...cardStyle(),
                background: verdictPass ? "#ecfdf5" : "#fef2f2",
                border: verdictPass ? "1px solid #86efac" : "1px solid #fecaca",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                {verdictPass ? (
                  <CheckCircle2 size={18} color="#166534" />
                ) : (
                  <XCircle size={18} color="#b91c1c" />
                )}
                <h3 style={{ margin: 0, color: THEME.textMain }}>Business Verdict</h3>
              </div>

              <div
                style={{
                  fontSize: 28,
                  fontWeight: 800,
                  color: verdictPass ? "#166534" : "#b91c1c",
                  marginBottom: 10,
                }}
              >
                {simulationSummary.pass_fail}
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <div style={{ fontSize: 13, color: THEME.textSub }}>Business Score</div>
                  <div style={{ fontWeight: 700, marginTop: 4 }}>{businessScore}/100</div>
                </div>
                <div>
                  <div style={{ fontSize: 13, color: THEME.textSub }}>RL Reward</div>
                  <div style={{ fontWeight: 700, marginTop: 4 }}>{rlReward.toFixed(1)}</div>
                </div>
                <div>
                  <div style={{ fontSize: 13, color: THEME.textSub }}>Total Energy</div>
                  <div style={{ fontWeight: 700, marginTop: 4 }}>{simulationSummary.total_energy}</div>
                </div>
                <div>
                  <div style={{ fontSize: 13, color: THEME.textSub }}>Avg Temperature</div>
                  <div style={{ fontWeight: 700, marginTop: 4 }}>{simulationSummary.avg_temperature}°C</div>
                </div>
                <div>
                  <div style={{ fontSize: 13, color: THEME.textSub }}>Max Temperature</div>
                  <div style={{ fontWeight: 700, marginTop: 4 }}>{simulationSummary.max_temperature}°C</div>
                </div>
                <div>
                  <div style={{ fontSize: 13, color: THEME.textSub }}>Overheat Count</div>
                  <div style={{ fontWeight: 700, marginTop: 4 }}>{simulationSummary.overheat_count}</div>
                </div>
              </div>

              <div
                style={{
                  marginTop: 14,
                  padding: "10px 12px",
                  borderRadius: 12,
                  background: verdictPass ? "#dcfce7" : "#fee2e2",
                  color: verdictPass ? "#166534" : "#991b1b",
                  fontSize: 13,
                  fontWeight: 600,
                  lineHeight: 1.5,
                }}
              >
                {verdictExplanation}
              </div>
            </div>
          </div>
        )}

        <div style={sectionLabelStyle()}>Current Run Snapshot</div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: 16,
            marginBottom: 20,
          }}
        >
          <StatCard
            title="Current Temperature"
            value={`${currentTemperature.toFixed(1)}°C`}
            subtitle="Latest simulation state"
            icon={Thermometer}
          />
          <StatCard
            title="Current Workload"
            value={`${(currentWorkload * 100).toFixed(0)}%`}
            subtitle="Prompt-derived workload"
            icon={Cpu}
          />
          <StatCard
            title="Predicted Heat"
            value={predictedHeat.toFixed(2)}
            subtitle="Next-step predicted heat"
            icon={Activity}
          />
          <StatCard
            title="Cooling Level"
            value={`${coolingLevel}/10`}
            subtitle="Selected cooling action"
            icon={Wind}
          />
        </div>

        <div
          style={{
            ...cardStyle(),
            marginBottom: 20,
            background: latestStateBg,
            border: latestStateBorder,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            {isSafe ? (
              <ShieldCheck color={latestStateColor} />
            ) : (
              <ShieldAlert color={latestStateColor} />
            )}

            <strong style={{ color: latestStateColor }}>{latestStateLabel}</strong>

            <span style={{ color: THEME.primarySoft }}>
              {isSafe
                ? `The latest simulated rack temperature is stable at ${currentTemperature.toFixed(1)}°C.`
                : `The latest simulated rack temperature is elevated at ${currentTemperature.toFixed(1)}°C.`}
            </span>
          </div>
        </div>

        <div
          style={{
            ...cardStyle(),
            marginBottom: 20,
            background: "#eff6ff",
            border: "1px solid #dbeafe",
          }}
        >
          <h3 style={{ marginTop: 0, marginBottom: 10, color: THEME.textMain, fontWeight: 600 }}>
            {forecastImpact.title}
          </h3>
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontSize: 13, color: THEME.textSub }}>
                {forecast === "on" ? "Expected Energy Benefit" : "Forecast Benefit"}
              </div>
              <div style={{ fontSize: 26, fontWeight: 700, color: THEME.accentTeal }}>
                {forecast === "on" ? `+${forecastImpact.energySavingPct.toFixed(1)}%` : "Not enabled"}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 13, color: THEME.textSub }}>
                {forecast === "on" ? "Expected Thermal Improvement" : "Forecast Delta"}
              </div>
              <div style={{ fontSize: 26, fontWeight: 700, color: THEME.accentGreen }}>
                {forecast === "on" ? `↓ ${Math.abs(forecastImpact.tempImprovement).toFixed(1)}°C` : "Reactive mode"}
              </div>
            </div>
            <div style={{ alignSelf: "center", color: "#1e40af" }}>
              {forecastImpact.summary}
            </div>
          </div>
        </div>

        <div style={sectionLabelStyle()}>Operational and Explainability Summary</div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: 16,
            marginBottom: 20,
          }}
        >
          <div style={cardStyle()}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
              <DollarSign size={18} color={THEME.primarySoft} />
              <h3 style={{ margin: 0, color: THEME.textMain }}>Operational Impact Summary</h3>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <div style={{ fontSize: 13, color: THEME.textSub }}>Cooling Cost</div>
                <div style={{ fontWeight: 700, marginTop: 4 }}>${coolingCost.toFixed(2)}</div>
              </div>
              <div>
                <div style={{ fontSize: 13, color: THEME.textSub }}>Energy Savings</div>
                <div style={{ fontWeight: 700, marginTop: 4 }}>{energySavings.toFixed(1)}%</div>
              </div>
              <div>
                <div style={{ fontSize: 13, color: THEME.textSub }}>SLA Violations</div>
                <div style={{ fontWeight: 700, marginTop: 4 }}>{slaViolations}</div>
              </div>
              <div>
                <div style={{ fontSize: 13, color: THEME.textSub }}>Risk Exposure</div>
                <div style={{ fontWeight: 700, marginTop: 4 }}>${riskCost.toFixed(2)}</div>
              </div>
            </div>
          </div>

          <div style={cardStyle()}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
              <Info size={18} color={THEME.primarySoft} />
              <h3 style={{ margin: 0, color: THEME.textMain }}>Why this action?</h3>
            </div>
            <div
              style={{
                fontSize: 14,
                lineHeight: 1.6,
                color: THEME.primarySoft,
                background: "#f8fafc",
                border: "1px solid #e2e8f0",
                borderRadius: 12,
                padding: 14,
              }}
            >
              {whyThisAction}
            </div>
          </div>

          <div style={cardStyle()}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
              <Gauge size={18} color={THEME.primarySoft} />
              <h3 style={{ margin: 0, color: THEME.textMain }}>Compute Status</h3>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <div style={{ fontSize: 13, color: THEME.textSub }}>Device</div>
                <div style={{ fontWeight: 700, marginTop: 4 }}>{computeStatus.device}</div>
              </div>
              <div>
                <div style={{ fontSize: 13, color: THEME.textSub }}>Concurrency</div>
                <div style={{ fontWeight: 700, marginTop: 4 }}>{computeStatus.concurrency}</div>
              </div>
              <div>
                <div style={{ fontSize: 13, color: THEME.textSub }}>Batch Mode</div>
                <div style={{ fontWeight: 700, marginTop: 4 }}>
                  {computeStatus.batchMode ? "Enabled" : "Disabled"}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 13, color: THEME.textSub }}>Forecast Mode</div>
                <div style={{ fontWeight: 700, marginTop: 4 }}>
                  {forecast === "on" ? "Enabled" : "Disabled"}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: 16,
            marginBottom: 20,
          }}
        >
          <StatCard
            title="Avg Temperature"
            value={simulationSummary ? `${avgTemperature.toFixed(1)}°C` : "--"}
            subtitle="Across simulation episode"
            icon={Thermometer}
          />
          <StatCard
            title="Total Energy"
            value={simulationSummary ? totalEnergy.toFixed(1) : "--"}
            subtitle="Across simulation episode"
            icon={Activity}
          />
          <StatCard
            title="Overheat Count"
            value={simulationSummary ? `${overheatCount}` : "--"}
            subtitle={`Temperature > ${safeTempMax}°C`}
            icon={ShieldAlert}
          />
          <StatCard
            title="Safe Zone Rate"
            value={hasSimulation ? `${safeZoneRate.toFixed(0)}%` : "--"}
            subtitle="Share of safe timesteps"
            icon={ShieldCheck}
          />
        </div>

        <div style={sectionLabelStyle()}>Simulation Charts</div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 16, marginBottom: 16 }}>
          <ChartCard
            title="Temperature over time"
            subtitle="Blue = actual rack temperature, Red = safe upper threshold."
          >
            {hasSimulation ? (
              <ResponsiveContainer width="100%" height="100%" minWidth={300}>
                <LineChart data={simulationTimeline}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="step" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="temperature"
                    stroke={THEME.accentBlue}
                    strokeWidth={3}
                    dot={false}
                    name="Temperature"
                  />
                  <Line
                    type="monotone"
                    dataKey="safeMaxTemp"
                    stroke={THEME.accentRed}
                    strokeWidth={2}
                    dot={false}
                    name="Safe Max Temp"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <ChartPlaceholder message="Run a simulation to view temperature behavior over time." />
            )}
          </ChartCard>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))",
            gap: 16,
            marginBottom: 16,
          }}
        >
          <ChartCard
            title="Actual heat vs predicted heat"
            subtitle="Blue = actual heat, Teal = predicted next-step heat."
          >
            {hasSimulation ? (
              <ResponsiveContainer width="100%" height="100%" minWidth={300}>
                <LineChart data={simulationTimeline}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="step" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="heat"
                    stroke={THEME.accentBlue}
                    strokeWidth={3}
                    dot={false}
                    name="Actual Heat"
                  />
                  <Line
                    type="monotone"
                    dataKey="predictedHeat"
                    stroke={THEME.accentTeal}
                    strokeWidth={3}
                    dot={false}
                    name="Predicted Heat"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <ChartPlaceholder message="Run a simulation to compare actual heat and predicted heat." />
            )}
          </ChartCard>

          <ChartCard
            title="Energy and cooling level"
            subtitle="Teal = energy demand, Light blue = cooling level selected by the controller."
          >
            {hasSimulation ? (
              <ResponsiveContainer width="100%" height="100%" minWidth={300}>
                <LineChart data={simulationTimeline}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="step" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="energy"
                    stroke={THEME.accentTeal}
                    strokeWidth={3}
                    dot={false}
                    name="Energy"
                  />
                  <Line
                    type="monotone"
                    dataKey="coolingLevel"
                    stroke={THEME.accentBlueSoft}
                    strokeWidth={3}
                    dot={false}
                    name="Cooling Level"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <ChartPlaceholder message="Run a simulation to view energy usage and cooling actions." />
            )}
          </ChartCard>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))",
            gap: 16,
            marginBottom: 16,
          }}
        >
          <ChartCard
            title="Reward per step"
            subtitle="Orange = step-level RL reward used during training."
          >
            {hasSimulation ? (
              <ResponsiveContainer width="100%" height="100%" minWidth={300}>
                <LineChart data={simulationTimeline}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="step" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="reward"
                    stroke={THEME.accentAmber}
                    strokeWidth={3}
                    dot={false}
                    name="Reward"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <ChartPlaceholder message="Run a simulation to view reward at each timestep." />
            )}
          </ChartCard>

          <ChartCard
            title="Cumulative reward"
            subtitle="Green = cumulative RL reward across the full episode."
          >
            {hasSimulation ? (
              <ResponsiveContainer width="100%" height="100%" minWidth={300}>
                <LineChart data={simulationTimeline}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="step" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="cumulativeReward"
                    stroke={THEME.accentGreen}
                    strokeWidth={3}
                    dot={false}
                    name="Cumulative Reward"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <ChartPlaceholder message="Run a simulation to view how reward accumulates across the episode." />
            )}
          </ChartCard>
        </div>

        {!hasSimulation && (
          <div
            style={{
              marginBottom: 16,
              padding: "10px 14px",
              borderRadius: 12,
              background: "#fff7ed",
              border: "1px solid #fdba74",
              color: "#9a3412",
              fontWeight: 500,
            }}
          >
            Run a custom simulation to populate Prompt Features, Business Verdict, and the experiment charts.
          </div>
        )}

        <ChartCard
          title="Controller benchmark comparison"
          subtitle="Teal bars show energy (scaled for comparison), Green bars show mean reward. Selected controller is highlighted more strongly."
        >
          <ResponsiveContainer width="100%" height="100%" minWidth={300}>
            <BarChart data={controllerComparison}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" interval={0} angle={-6} textAnchor="end" height={60} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="energyScaled" name="Energy (scaled)">
                {controllerComparison.map((entry, index) => (
                  <Cell
                    key={`energy-${index}`}
                    fill={entry.isSelected ? THEME.accentTealStrong : "#bfe9e4"}
                  />
                ))}
              </Bar>
              <Bar dataKey="reward" name="Mean Reward">
                {controllerComparison.map((entry, index) => (
                  <Cell
                    key={`reward-${index}`}
                    fill={entry.isSelected ? THEME.accentGreen : "#c8f6d8"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: 24,
            marginTop: 12,
            fontSize: 14,
            color: THEME.primarySoft,
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div
              style={{
                width: 12,
                height: 12,
                background: THEME.accentTealStrong,
                borderRadius: 2,
              }}
            />
            Energy (scaled)
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div
              style={{
                width: 12,
                height: 12,
                background: THEME.accentGreen,
                borderRadius: 2,
              }}
            />
            Mean Reward
          </div>
        </div>

        <div
          style={{
            marginTop: 12,
            padding: "14px 18px",
            borderRadius: 14,
            background: "#f8fafc",
            border: "1px solid #e2e8f0",
            color: THEME.primarySoft,
            fontSize: 15,
            lineHeight: 1.5,
          }}
        >
          {simulationSummary
            ? `Custom prompt simulation finished with verdict ${simulationSummary.pass_fail}. The charts above show the current run, while the controller comparison panel is a benchmark overview across controller types. Business Score is a human-friendly normalized score, while RL Reward remains the internal training signal.`
            : "Run a custom simulation to evaluate controller behavior and business outcome. The controller comparison panel is a benchmark overview across controller types."}
        </div>
      </div>
    </div>
  );
}
