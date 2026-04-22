import Card from "./Card";

const LAYER_LABELS = [
  { key: "gpu", label: "GPU", tone: "bg-[#A8D64F]" },
  { key: "cpu", label: "CPU", tone: "bg-[#8FAC69]" },
  { key: "dram", label: "DRAM", tone: "bg-[#AFC49A]" },
  { key: "nic", label: "NIC", tone: "bg-[#BFD98A]" },
  { key: "cooling", label: "Cooling", tone: "bg-[#90A082]" },
  { key: "otherOverhead", label: "Other overhead", tone: "bg-[#BBC2B7]" },
];

export default function TelemetryBreakdownCard({
  metrics,
  layerConfidence = {},
}) {
  if (!metrics) return null;

  return (
    <Card className="demo-breakdown-card h-full">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-[#262626]">
            Uploaded data breakdown
          </div>
          <div className="mt-1 text-xs text-[#6F756E]">
            This shows how the uploaded energy is split across the main parts of the system.
          </div>
        </div>
        <div className="demo-signal-chip rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1 text-xs text-[#3A3A3A]">
          {metrics.meta?.hasComponentBreakdown ? "Full view" : "Partial view"}
        </div>
      </div>

      <div className="mt-4 space-y-3">
        {LAYER_LABELS.map(({ key, label, tone }) => (
          <div key={key}>
            <div className="flex items-center justify-between gap-3 text-xs text-[#3A3A3A]">
              <span>{label}</span>
              <span>
                {(metrics.breakdownWh?.[key] ?? 0).toFixed(3)} Wh |{" "}
                {(metrics.contributionPct?.[key] ?? 0).toFixed(1)}%
              </span>
            </div>
            <div className="mt-1 h-2 overflow-hidden rounded-full bg-[#DDE4DA]">
              <div
                className={`h-full rounded-full ${tone}`}
                style={{
                  width: `${Math.max(4, metrics.contributionPct?.[key] ?? 0)}%`,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
        {Object.entries(layerConfidence).map(([layer, confidence]) => (
          <div
            key={layer}
            className="rounded-xl border border-[#D3D9D0] bg-[#F6F7F3] px-3 py-2 text-[#3A3A3A]"
          >
            <div className="text-[10px] uppercase tracking-[0.14em] text-[#6F756E]">
              {layer}
            </div>
            <div className="mt-1 font-semibold text-[#262626]">{confidence}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

