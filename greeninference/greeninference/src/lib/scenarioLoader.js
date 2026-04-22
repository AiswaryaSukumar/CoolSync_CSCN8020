// Loads CoolSync scenario JSON files from /public/scenarios/ and converts to heat traces.
// Mirrors run_scenario() logic from CoolSync_Final.ipynb Cell 15.

import { PHYS, BURST_PARAMS } from "./physics";

const { Q_BASE_W, Q_RACK_MAX } = PHYS;

// Token-length → burst class (same as tokensToBurstClass but readable labels)
const TOKEN_CLASS_THRESHOLDS = [64, 256, 800];

function estimateClass(promptText) {
  const words  = (promptText || "").trim().split(/\s+/).length;
  const tokens = Math.max(1, Math.round(words * 1.35));
  if (tokens < TOKEN_CLASS_THRESHOLDS[0]) return 0;
  if (tokens < TOKEN_CLASS_THRESHOLDS[1]) return 1;
  if (tokens < TOKEN_CLASS_THRESHOLDS[2]) return 2;
  return 3;
}

// Build heat trace from scenario JSON events (additive accumulation, CoolSync logic)
export function buildHeatTrace(scenario) {
  const total  = Number(scenario.total_minutes);
  const heat   = new Array(total).fill(Q_BASE_W);
  const events = [...(scenario.events || [])].sort((a, b) => a.time_min - b.time_min);

  for (const ev of events) {
    const tStart  = Math.max(0, Math.min(Math.floor(ev.time_min), total - 1));
    const users   = Math.max(1, Number(ev.users ?? 1));
    const cls     = ev.force_class !== undefined ? Number(ev.force_class) : estimateClass(ev.prompt);
    const bp      = BURST_PARAMS[cls] ?? BURST_PARAMS[1];
    const Qrack   = Math.min(Q_BASE_W * (1 + (bp.amp - 1) * users), Q_RACK_MAX);
    const tEnd    = Math.min(tStart + bp.dur, total);
    const delta   = Qrack - Q_BASE_W;
    for (let i = tStart; i < tEnd; i++) {
      heat[i] = Math.min(heat[i] + delta, Q_RACK_MAX);
    }
  }
  return heat;
}

// Available CoolSync scenario presets (id → JSON filename mapping)
export const COOLSYNC_SCENARIOS = [
  {
    id:    "cs_steady_load",
    label: "Steady Load",
    file:  "scenario_steady_load.json",
    description: "30 min — low-variance short queries, tests baseline efficiency gap",
    model: "Small Model",
    workloadFlexibility: "flexible",
  },
  {
    id:    "cs_multi_user",
    label: "Multi-User Mixed",
    file:  "scenario_multi_user_mixed.json",
    description: "60 min — concurrent users with mixed prompt lengths",
    model: "LLM-70B (Dense)",
    workloadFlexibility: "urgent",
  },
  {
    id:    "cs_peak_hour",
    label: "Peak Hour",
    file:  "scenario_peak_hour.json",
    description: "60 min — peak-hour burst waves with concurrent users",
    model: "MoE Model",
    workloadFlexibility: "urgent",
  },
  {
    id:    "cs_stress_test",
    label: "Stress Test",
    file:  "scenario_stress_test.json",
    description: "45 min — overlapping VeryLong queries, ASHRAE breach expected",
    model: "LLM-70B (Dense)",
    workloadFlexibility: "urgent",
  },
  {
    id:    "cs_chaos_load",
    label: "Chaos Load",
    file:  "scenario_chaos_load.json",
    description: "120 min — random burst storm, 37 events, thermally chaotic",
    model: "LLM-70B (Dense)",
    workloadFlexibility: "flexible",
  },
];

// Fetch + parse a scenario JSON file, return { heatTrace, scenario }
export async function loadScenario(fileOrId) {
  const preset = COOLSYNC_SCENARIOS.find(s => s.id === fileOrId || s.file === fileOrId);
  const filename = preset?.file ?? fileOrId;
  const url = `/scenarios/${filename}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load scenario: ${url}`);
  const scenario  = await response.json();
  const heatTrace = buildHeatTrace(scenario);
  return { heatTrace, scenario, preset };
}
