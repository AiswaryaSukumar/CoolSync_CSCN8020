const TELEMETRY_SCHEMAS = [
  {
    id: "legacy_minimal",
    label: "Legacy MVP",
    columns: ["t", "gpuW", "inlet", "coolingKw"],
    aliases: {
      t: ["t"],
      gpuW: ["gpuw"],
      inlet: ["inlet"],
      coolingKw: ["coolingkw"],
      promptType: ["prompt_type", "prompttype", "prompt"],
      requestCount: ["request_count", "requestcount", "requests"],
      tokensPerRequest: [
        "tokens_per_request",
        "tokensperrequest",
        "avg_tokens",
        "tokens",
      ],
    },
  },
  {
    id: "canonical_minimal",
    label: "Canonical Minimal",
    columns: ["time_ms", "gpu_power_w", "inlet_temp_c", "cooling_kw"],
    aliases: {
      t: ["time_ms"],
      gpuW: ["gpu_power_w"],
      inlet: ["inlet_temp_c"],
      coolingKw: ["cooling_kw"],
      promptType: ["prompt_type", "prompttype", "prompt"],
      requestCount: ["request_count", "requestcount", "requests"],
      tokensPerRequest: [
        "tokens_per_request",
        "tokensperrequest",
        "avg_tokens",
        "tokens",
      ],
    },
  },
  {
    id: "full_stack_v1",
    label: "Full-Stack Telemetry",
    columns: [
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
    aliases: {
      t: ["time_ms", "t"],
      facilityKw: ["facility_kw"],
      rackKw: ["rack_kw"],
      serverKw: ["server_kw"],
      gpuW: ["gpu_power_w", "gpuw"],
      cpuW: ["cpu_power_w"],
      dramW: ["dram_power_w"],
      nicW: ["nic_power_w"],
      inlet: ["inlet_temp_c", "inlet"],
      coolingKw: ["cooling_kw", "coolingkw"],
      pumpKw: ["pump_kw"],
      otherOverheadKw: ["other_overhead_kw"],
      waterLpm: ["water_lpm"],
      promptType: ["prompt_type", "prompttype", "prompt"],
      requestCount: ["request_count", "requestcount", "requests"],
      tokensPerRequest: [
        "tokens_per_request",
        "tokensperrequest",
        "avg_tokens",
        "tokens",
      ],
    },
  },
];

const REQUIRED_FIELDS = ["t", "gpuW", "inlet", "coolingKw"];

const LAYER_FIELD_MAP = {
  facility: ["facilityKw", "rackKw", "serverKw"],
  rackServer: ["rackKw", "serverKw"],
  gpu: ["gpuW"],
  cpu: ["cpuW"],
  dram: ["dramW"],
  nic: ["nicW"],
  cooling: ["coolingKw", "pumpKw"],
  overhead: ["otherOverheadKw", "facilityKw", "coolingKw"],
  workload: ["requestCount", "tokensPerRequest"],
  promptMix: ["promptType"],
};

function getRowConfidence(validRows, totalRows) {
  if (!totalRows || validRows <= 0) return "INVALID";
  const coverage = validRows / totalRows;
  if (coverage >= 0.85) return "HIGH";
  if (coverage >= 0.55) return "MEDIUM";
  return "LOW";
}

function hasAnyAlias(colIndex = {}, aliases = []) {
  return aliases.some((alias) => alias in colIndex);
}

export function getTelemetrySchemas() {
  return TELEMETRY_SCHEMAS;
}

export function getTelemetryAcceptedColumnsText() {
  return TELEMETRY_SCHEMAS.map((schema) => schema.columns.join(", "));
}

export function detectTelemetrySchema(colIndex = {}) {
  for (const schema of TELEMETRY_SCHEMAS) {
    const matchesRequired = REQUIRED_FIELDS.every((field) =>
      hasAnyAlias(colIndex, schema.aliases[field] || [])
    );

    if (matchesRequired) return schema;
  }

  return null;
}

export function buildSchemaFieldIndex(schema, colIndex = {}) {
  const index = {};

  for (const [field, aliases] of Object.entries(schema.aliases)) {
    const match = aliases.find((alias) => alias in colIndex);
    if (match) index[field] = colIndex[match];
  }

  return index;
}

export function computeLayerConfidence(points = [], totalRows = 0) {
  const resolvedTotalRows = totalRows || points.length;
  const layerConfidence = {};

  Object.entries(LAYER_FIELD_MAP).forEach(([layer, fields]) => {
    const validRows = points.filter((point) =>
      fields.some((field) => {
        const value = point?.[field];

        if (field === "promptType") {
          return typeof value === "string" && value.trim().length > 0;
        }

        return Number.isFinite(Number(value));
      })
    ).length;

    layerConfidence[layer] = getRowConfidence(validRows, resolvedTotalRows);
  });

  return layerConfidence;
}