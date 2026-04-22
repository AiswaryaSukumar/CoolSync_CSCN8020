import {
  buildSchemaFieldIndex,
  computeLayerConfidence,
  detectTelemetrySchema,
  getTelemetryAcceptedColumnsText,
} from "./telemetrySchema";
import { buildObservedOverheadAttribution } from "./overheadModel";

function splitCsvLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];

    if (ch === '"') {
      const next = line[i + 1];
      if (inQuotes && next === '"') {
        cur += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
      continue;
    }

    cur += ch;
  }

  out.push(cur);
  return out.map((s) => s.trim());
}

function normalizeHeader(h) {
  return String(h || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_");
}

function toNum(x) {
  if (x === undefined || x === null || String(x).trim() === "") return NaN;
  const n = Number(String(x).trim());
  return Number.isFinite(n) ? n : NaN;
}

function getConfidence(rows) {
  if (rows >= 20) return "HIGH";
  if (rows >= 10) return "MEDIUM";
  return "LOW";
}

function firstFinite(...values) {
  for (const value of values) {
    if (Number.isFinite(value)) return value;
  }
  return NaN;
}

function safeRound(value, digits = 3) {
  if (!Number.isFinite(value)) return 0;
  return Number(value.toFixed(digits));
}

function normalizePromptType(value) {
  return String(value || "").trim();
}

function safePositiveInt(value, fallback = 1) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.max(1, Math.round(n));
}

function getPhaseKey(point = {}) {
  const phase = String(point.phase || "").toUpperCase();
  if (phase === "PREFILL") return "prefill";
  if (phase === "DECODE") return "decode";
  return "baseline";
}

function normalizePoint(rawPoint = {}) {
  const gpuW = Number(rawPoint.gpuW);
  const cpuW = Number(rawPoint.cpuW);
  const dramW = Number(rawPoint.dramW);
  const nicW = Number(rawPoint.nicW);
  const rackKw = Number(rawPoint.rackKw);
  const serverKw = Number(rawPoint.serverKw);
  const facilityKw = Number(rawPoint.facilityKw);
  const coolingKw = Number(rawPoint.coolingKw);
  const pumpKw = Number(rawPoint.pumpKw);
  const otherOverheadKw = Number(rawPoint.otherOverheadKw);
  const requestCountRaw = Number(rawPoint.requestCount);
  const tokensPerRequestRaw = Number(rawPoint.tokensPerRequest);

  const componentItPowerW = [gpuW, cpuW, dramW, nicW]
    .filter((value) => Number.isFinite(value))
    .reduce((sum, value) => sum + value, 0);

  const inferredServerPowerW = Number.isFinite(serverKw) ? serverKw * 1000 : NaN;
  const itPowerW = firstFinite(
    componentItPowerW > 0 ? componentItPowerW : NaN,
    inferredServerPowerW,
    Number.isFinite(gpuW) ? gpuW : NaN
  );

  const overheadPowerW =
    (Number.isFinite(coolingKw) ? coolingKw * 1000 : 0) +
    (Number.isFinite(pumpKw) ? pumpKw * 1000 : 0) +
    (Number.isFinite(otherOverheadKw) ? otherOverheadKw * 1000 : 0);

  const totalPowerW = firstFinite(
    Number.isFinite(facilityKw) ? facilityKw * 1000 : NaN,
    Number.isFinite(rackKw) ? rackKw * 1000 : NaN,
    Number.isFinite(itPowerW) || overheadPowerW > 0
      ? (Number.isFinite(itPowerW) ? itPowerW : 0) + overheadPowerW
      : NaN
  );

  const requestCount = safePositiveInt(requestCountRaw, 1);
  const tokensPerRequest = Number.isFinite(tokensPerRequestRaw) ? tokensPerRequestRaw : NaN;
  const workloadTokens =
    Number.isFinite(tokensPerRequest) ? requestCount * tokensPerRequest : NaN;

  return {
    t: rawPoint.t,
    gpuW: Number.isFinite(gpuW) ? gpuW : NaN,
    cpuW: Number.isFinite(cpuW) ? cpuW : NaN,
    dramW: Number.isFinite(dramW) ? dramW : NaN,
    nicW: Number.isFinite(nicW) ? nicW : NaN,
    inlet: Number(rawPoint.inlet),
    coolingKw: Number.isFinite(coolingKw) ? coolingKw : NaN,
    facilityKw: Number.isFinite(facilityKw) ? facilityKw : NaN,
    rackKw: Number.isFinite(rackKw) ? rackKw : NaN,
    serverKw: Number.isFinite(serverKw) ? serverKw : NaN,
    pumpKw: Number.isFinite(pumpKw) ? pumpKw : NaN,
    otherOverheadKw: Number.isFinite(otherOverheadKw) ? otherOverheadKw : NaN,
    waterLpm: Number(rawPoint.waterLpm),
    itPowerW: Number.isFinite(itPowerW) ? itPowerW : NaN,
    totalPowerW: Number.isFinite(totalPowerW) ? totalPowerW : NaN,
    promptType: normalizePromptType(rawPoint.promptType),
    requestCount,
    tokensPerRequest,
    workloadTokens,
  };
}

function inferPhaseWindows(points) {
  const maxT = points[points.length - 1]?.t ?? 0;
  const prefillEnd = maxT * 0.17;
  const decodeEnd = maxT * 0.78;

  return points.map((point) => ({
    ...point,
    phase:
      point.t <= prefillEnd
        ? "PREFILL"
        : point.t <= decodeEnd
        ? "DECODE"
        : "BASELINE",
  }));
}

function getTelemetryWorkloadSummary(points = []) {
  const promptTypeCounts = {};
  let peakRequests = 0;
  let peakWorkloadTokens = 0;
  let totalRequests = 0;
  let totalWorkloadTokens = 0;
  let rowsWithPromptType = 0;
  let rowsWithRequestCount = 0;
  let rowsWithTokensPerRequest = 0;

  points.forEach((point) => {
    const promptType = String(point.promptType || "").trim();
    const requestCount = Number(point.requestCount);
    const tokensPerRequest = Number(point.tokensPerRequest);
    const workloadTokens = Number(point.workloadTokens);

    if (promptType) {
      promptTypeCounts[promptType] = (promptTypeCounts[promptType] || 0) + 1;
      rowsWithPromptType += 1;
    }

    if (Number.isFinite(requestCount)) {
      rowsWithRequestCount += 1;
      peakRequests = Math.max(peakRequests, requestCount);
      totalRequests += requestCount;
    }

    if (Number.isFinite(tokensPerRequest)) {
      rowsWithTokensPerRequest += 1;
    }

    if (Number.isFinite(workloadTokens)) {
      peakWorkloadTokens = Math.max(peakWorkloadTokens, workloadTokens);
      totalWorkloadTokens += workloadTokens;
    }
  });

  const dominantPromptType =
    Object.entries(promptTypeCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "";

  const hasPromptMix = rowsWithPromptType > 0;
  const hasBurstSignals = rowsWithRequestCount > 0;
  const hasTokenSignals = rowsWithTokensPerRequest > 0;

  let burstRisk = "LOW";
  if (peakRequests >= 10 || peakWorkloadTokens >= 20000) {
    burstRisk = "HIGH";
  } else if (peakRequests >= 5 || peakWorkloadTokens >= 8000) {
    burstRisk = "MEDIUM";
  }

  return {
    hasPromptMix,
    hasBurstSignals,
    hasTokenSignals,
    dominantPromptType,
    promptTypeCounts,
    peakRequests,
    totalRequests,
    peakWorkloadTokens,
    totalWorkloadTokens,
    burstRisk,
    rowsWithPromptType,
    rowsWithRequestCount,
    rowsWithTokensPerRequest,
  };
}

export function parseTelemetryCsv(text) {
  try {
    const raw = String(text || "").trim();
    if (!raw) {
      return {
        points: [],
        meta: {
          format: "unknown",
          rows: 0,
          error: "Empty file.",
          confidence: "INVALID",
          layerConfidence: {},
        },
      };
    }

    const lines = raw
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    if (lines.length < 2) {
      return {
        points: [],
        meta: {
          format: "unknown",
          rows: 0,
          error: "CSV needs a header + at least 1 row.",
          confidence: "INVALID",
          layerConfidence: {},
        },
      };
    }

    const header = splitCsvLine(lines[0]).map(normalizeHeader);
    const colIndex = Object.fromEntries(
      header.map((value, index) => [value, index])
    );
    const schema = detectTelemetrySchema(colIndex);

    if (!schema) {
      return {
        points: [],
        meta: {
          format: "unknown",
          rows: 0,
          error: `Unrecognized columns. Expected one of: ${getTelemetryAcceptedColumnsText().join(
            " OR "
          )}.`,
          confidence: "INVALID",
          layerConfidence: {},
        },
      };
    }

    const fieldIndex = buildSchemaFieldIndex(schema, colIndex);
    const points = [];

    for (let i = 1; i < lines.length; i++) {
      const row = splitCsvLine(lines[i]);
      if (row.length < header.length) continue;

      const t = toNum(row[fieldIndex.t]);
      const inlet = toNum(row[fieldIndex.inlet]);
      const gpuW = toNum(row[fieldIndex.gpuW]);
      const coolingKw = toNum(row[fieldIndex.coolingKw]);

      if ([t, inlet, gpuW, coolingKw].some((value) => Number.isNaN(value))) {
        continue;
      }

      const normalizedPoint = normalizePoint({
        t,
        gpuW,
        cpuW: toNum(row[fieldIndex.cpuW]),
        dramW: toNum(row[fieldIndex.dramW]),
        nicW: toNum(row[fieldIndex.nicW]),
        inlet,
        coolingKw,
        facilityKw: toNum(row[fieldIndex.facilityKw]),
        rackKw: toNum(row[fieldIndex.rackKw]),
        serverKw: toNum(row[fieldIndex.serverKw]),
        pumpKw: toNum(row[fieldIndex.pumpKw]),
        otherOverheadKw: toNum(row[fieldIndex.otherOverheadKw]),
        waterLpm: toNum(row[fieldIndex.waterLpm]),
        promptType:
          fieldIndex.promptType !== undefined
            ? row[fieldIndex.promptType]
            : "",
        requestCount:
          fieldIndex.requestCount !== undefined
            ? toNum(row[fieldIndex.requestCount])
            : NaN,
        tokensPerRequest:
          fieldIndex.tokensPerRequest !== undefined
            ? toNum(row[fieldIndex.tokensPerRequest])
            : NaN,
      });

      points.push(normalizedPoint);
    }

    if (points.length === 0) {
      return {
        points: [],
        meta: {
          format: schema.id,
          rows: 0,
          error:
            "No valid numeric rows found. Check for blanks, NaNs, and required telemetry columns.",
          confidence: "INVALID",
          layerConfidence: {},
        },
      };
    }

    points.sort((a, b) => a.t - b.t);
    const t0 = points[0].t;
    const normalized = inferPhaseWindows(
      points.map((point) => ({
        ...point,
        t: point.t - t0,
      }))
    );

    const workloadSummary = getTelemetryWorkloadSummary(normalized);

    return {
      points: normalized,
      meta: {
        format: schema.id,
        label: schema.label,
        rows: normalized.length,
        error: null,
        confidence: getConfidence(normalized.length),
        layerConfidence: computeLayerConfidence(normalized, lines.length - 1),
        columns: header,
        workloadSummary,
      },
    };
  } catch (e) {
    return {
      points: [],
      meta: {
        format: "unknown",
        rows: 0,
        error: `Parse failed: ${e?.message || "Unknown error"}`,
        confidence: "INVALID",
        layerConfidence: {},
      },
    };
  }
}

export function computeTelemetryMetrics(
  points,
  { grid_g_per_kWh = 110, overheadTarget = 0.27, tokens = 0 } = {}
) {
  const ps = Array.isArray(points) ? points : [];
  if (ps.length < 2) {
    return {
      tokens,
      whPerToken: 0,
      baseComputeWh: 0,
      coolingWhMeasured: 0,
      overheadCurrentMeasured: 0,
      totalWhCurrent: 0,
      totalWhTarget: 0,
      savedWh: 0,
      co2e_g_current: 0,
      co2e_g_target: 0,
      savedCo2e_g: 0,
      overheadCurrent: NaN,
      overheadTarget,
      grid_g_per_kWh,
      breakdownWh: {
        gpu: 0,
        cpu: 0,
        dram: 0,
        nic: 0,
        cooling: 0,
        otherOverhead: 0,
      },
      contributionPct: {
        gpu: 0,
        cpu: 0,
        dram: 0,
        nic: 0,
        cooling: 0,
        otherOverhead: 0,
      },
      workload: {
        totalRequests: 0,
        peakRequests: 0,
        totalWorkloadTokens: 0,
        peakWorkloadTokens: 0,
        dominantPromptType: "",
        burstRisk: "LOW",
      },
    };
  }

  const sorted = [...ps].sort((a, b) => a.t - b.t);

  let gpuWh = 0;
  let cpuWh = 0;
  let dramWh = 0;
  let nicWh = 0;
  let coolingWh = 0;
  let otherOverheadWh = 0;
  let totalWhObserved = 0;
  let dtMsSum = 0;
  let dtN = 0;

  const phaseEnergyWh = {
    prefillComputeWh: 0,
    decodeComputeWh: 0,
    baselineComputeWh: 0,
    prefillTotalWh: 0,
    decodeTotalWh: 0,
    baselineTotalWh: 0,
  };

  for (let i = 0; i < sorted.length - 1; i++) {
    const a = sorted[i];
    const b = sorted[i + 1];
    const dtMs = Math.max(0, (b.t ?? 0) - (a.t ?? 0));
    if (dtMs <= 0) continue;

    const dtHours = dtMs / 3_600_000;
    const gpuW = Number(a.gpuW);
    const cpuW = Number(a.cpuW);
    const dramW = Number(a.dramW);
    const nicW = Number(a.nicW);
    const coolingKw = Number(a.coolingKw);
    const pumpKw = Number(a.pumpKw);
    const otherOverheadKw = Number(a.otherOverheadKw);
    const totalPowerW = Number(a.totalPowerW);
    const phaseKey = getPhaseKey(a);
    const computeW =
      [gpuW, cpuW, dramW, nicW]
        .filter((value) => Number.isFinite(value))
        .reduce((sum, value) => sum + value, 0) || gpuW;

    if (Number.isFinite(gpuW)) gpuWh += gpuW * dtHours;
    if (Number.isFinite(cpuW)) cpuWh += cpuW * dtHours;
    if (Number.isFinite(dramW)) dramWh += dramW * dtHours;
    if (Number.isFinite(nicW)) nicWh += nicW * dtHours;
    if (Number.isFinite(coolingKw)) coolingWh += coolingKw * 1000 * dtHours;
    if (Number.isFinite(pumpKw)) otherOverheadWh += pumpKw * 1000 * dtHours;
    if (Number.isFinite(otherOverheadKw)) {
      otherOverheadWh += otherOverheadKw * 1000 * dtHours;
    }
    if (Number.isFinite(totalPowerW)) totalWhObserved += totalPowerW * dtHours;

    if (Number.isFinite(computeW)) {
      phaseEnergyWh[`${phaseKey}ComputeWh`] += computeW * dtHours;
    }
    if (Number.isFinite(totalPowerW)) {
      phaseEnergyWh[`${phaseKey}TotalWh`] += totalPowerW * dtHours;
    }

    dtMsSum += dtMs;
    dtN += 1;
  }

  const computeWh = gpuWh + cpuWh + dramWh + nicWh || gpuWh;
  const inferredTotalWh = computeWh + coolingWh + otherOverheadWh;
  const totalWhCurrent = totalWhObserved > 0 ? totalWhObserved : inferredTotalWh;
  const explicitOverheadWh = Math.max(0, totalWhCurrent - computeWh);
  const totalWhTarget = computeWh * (1 + overheadTarget);
  const savedWh = Math.max(0, totalWhCurrent - totalWhTarget);
  const overheadCurrentMeasured =
    computeWh > 0 ? explicitOverheadWh / computeWh : 0;

  const observedAttribution = buildObservedOverheadAttribution({
    computeWh,
    coolingWh,
    pumpWh: 0,
    otherOverheadWh,
    totalWhObserved: totalWhCurrent,
  });

  const co2e_g_current = (totalWhCurrent / 1000) * grid_g_per_kWh;
  const co2e_g_target = (totalWhTarget / 1000) * grid_g_per_kWh;
  const savedCo2e_g = Math.max(0, co2e_g_current - co2e_g_target);
  const whPerToken = tokens > 0 ? computeWh / tokens : 0;

  const breakdownWh = {
    gpu: safeRound(gpuWh),
    cpu: safeRound(cpuWh),
    dram: safeRound(dramWh),
    nic: safeRound(nicWh),
    cooling: safeRound(coolingWh),
    otherOverhead: safeRound(Math.max(0, explicitOverheadWh - coolingWh)),
  };

  const contributionPct = Object.fromEntries(
    Object.entries(breakdownWh).map(([key, value]) => [
      key,
      totalWhCurrent > 0
        ? Number(((value / totalWhCurrent) * 100).toFixed(1))
        : 0,
    ])
  );

  const workloadSummary = getTelemetryWorkloadSummary(sorted);

  return {
    tokens,
    whPerToken: safeRound(whPerToken, 4),
    baseComputeWh: safeRound(computeWh),
    coolingWhMeasured: safeRound(coolingWh),
    overheadCurrentMeasured: safeRound(overheadCurrentMeasured),
    totalWhCurrent: safeRound(totalWhCurrent),
    totalWhTarget: safeRound(totalWhTarget),
    savedWh: safeRound(savedWh),
    itEnergyWh: safeRound(observedAttribution.itEnergyWh),
    coolingEnergyWh: safeRound(observedAttribution.coolingEnergyWh),
    powerDeliveryLossWh: safeRound(observedAttribution.powerDeliveryLossWh),
    idleReserveWh: safeRound(observedAttribution.idleReserveWh),
    totalFacilityEnergyWh: safeRound(observedAttribution.totalFacilityEnergyWh),
    co2e_g_current: safeRound(co2e_g_current, 2),
    co2e_g_target: safeRound(co2e_g_target, 2),
    savedCo2e_g: safeRound(savedCo2e_g, 2),
    overheadCurrent: safeRound(overheadCurrentMeasured),
    overheadTarget,
    observedOverheadAttribution: observedAttribution,
    grid_g_per_kWh,
    breakdownWh,
    contributionPct,
    workload: {
      totalRequests: workloadSummary.totalRequests,
      peakRequests: workloadSummary.peakRequests,
      totalWorkloadTokens: safeRound(workloadSummary.totalWorkloadTokens, 0),
      peakWorkloadTokens: safeRound(workloadSummary.peakWorkloadTokens, 0),
      dominantPromptType: workloadSummary.dominantPromptType,
      burstRisk: workloadSummary.burstRisk,
      hasPromptMix: workloadSummary.hasPromptMix,
      hasBurstSignals: workloadSummary.hasBurstSignals,
      hasTokenSignals: workloadSummary.hasTokenSignals,
      promptTypeCounts: workloadSummary.promptTypeCounts,
    },
    meta: {
      dtMsAvg: dtN > 0 ? Math.round(dtMsSum / dtN) : null,
      rows: sorted.length,
      hasFacilityMetering: sorted.some((point) =>
        Number.isFinite(Number(point.facilityKw))
      ),
      hasComponentBreakdown: sorted.some((point) =>
        ["cpuW", "dramW", "nicW"].some((field) =>
          Number.isFinite(Number(point[field]))
        )
      ),
      hasPromptMix: workloadSummary.hasPromptMix,
      hasBurstSignals: workloadSummary.hasBurstSignals,
      hasTokenSignals: workloadSummary.hasTokenSignals,
    },
    phaseEnergyWh: Object.fromEntries(
      Object.entries(phaseEnergyWh).map(([key, value]) => [
        key,
        safeRound(value),
      ])
    ),
  };
}