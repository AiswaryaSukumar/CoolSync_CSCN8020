import Card from "./Card";

function MetricCell({ label, value, tone = "text-[#262626]" }) {
  return (
    <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
      <div className="text-xs text-[#6F756E]">{label}</div>
      <div className={`mt-1 text-sm font-semibold ${tone}`}>{value}</div>
    </div>
  );
}

function AttributionColumn({ title, subtitle, attribution, accent = "text-[#262626]" }) {
  if (!attribution) return null;

  return (
    <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
      <div className="text-sm font-semibold text-[#262626]">{title}</div>
      <div className="mt-1 text-xs text-[#6F756E]">{subtitle}</div>

      <div className="mt-3 grid grid-cols-1 gap-2">
        <MetricCell label="Computer energy" value={`${attribution.itEnergyWh.toFixed(3)} Wh`} />
        <MetricCell
          label="Cooling energy"
          value={`${attribution.coolingEnergyWh.toFixed(3)} Wh`}
          tone="text-[#6B7B48]"
        />
        <MetricCell
          label="Power loss"
          value={`${attribution.powerDeliveryLossWh.toFixed(3)} Wh`}
          tone="text-[#7A7F54]"
        />
        <MetricCell
          label="Idle power"
          value={`${attribution.idleReserveWh.toFixed(3)} Wh`}
          tone="text-[#5E7766]"
        />
        <MetricCell
          label="Total building energy"
          value={`${attribution.totalFacilityEnergyWh.toFixed(3)} Wh`}
          tone={accent}
        />
      </div>
    </div>
  );
}

export default function OverheadAttributionCard({
  heuristicAttribution,
  observedAttribution,
  calibratedAttribution,
}) {
  if (!heuristicAttribution && !observedAttribution && !calibratedAttribution) return null;

  return (
    <Card>
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-[#262626]">
            Where the energy goes
          </div>
          <div className="mt-1 text-xs text-[#6F756E]">
            This splits the energy into computer use, cooling, power loss, idle power, and total building energy.
          </div>
        </div>
        <div className="rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1 text-xs text-[#3A3A3A]">
          Estimated vs uploaded vs adjusted
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 xl:grid-cols-3">
        <AttributionColumn
          title="Estimated energy split"
          subtitle="Simple estimate from the simulation"
          attribution={heuristicAttribution}
          accent="text-[#7A7F54]"
        />
        <AttributionColumn
          title="Uploaded energy split"
          subtitle="Based on the uploaded data"
          attribution={observedAttribution}
          accent="text-[#6B7B48]"
        />
        <AttributionColumn
          title="Adjusted energy split"
          subtitle="Estimate changed to better match the site"
          attribution={calibratedAttribution}
          accent="text-[#6E7F6E]"
        />
      </div>
    </Card>
  );
}

