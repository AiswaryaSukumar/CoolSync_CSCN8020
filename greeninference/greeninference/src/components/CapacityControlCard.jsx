import Card from "./Card";

export default function CapacityControlCard({
  controls,
  onChange,
  capacityAssessment,
}) {
  if (!controls || !onChange || !capacityAssessment) return null;

  return (
    <Card className="demo-support-surface">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-[#262626]">
            Capacity controls
          </div>
          <div className="mt-1 text-xs text-[#6F756E]">
            Settings that control spare room, batching, and power limits.
          </div>
          <div className="mt-2 text-xs text-[#6F756E]">
            These settings change how smoothly work arrives. That affects waiting time,
            computer use, and temperature stability.
          </div>
        </div>
        <div className="demo-signal-chip rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1 text-xs text-[#3A3A3A]">
          {capacityAssessment.mode}
        </div>
      </div>

      <div className="demo-support-grid mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="flex items-center justify-between gap-3">
            <label className="text-xs text-[#6F756E]">Spare room kept free</label>
            <span className="text-xs font-semibold text-[#6B7B48]">
              {controls.headroomPct}%
            </span>
          </div>
          <input
            type="range"
            min="8"
            max="35"
            step="1"
            value={controls.headroomPct}
            onChange={(e) => onChange("headroomPct", Number(e.target.value))}
            className="mt-3 w-full accent-[#A8D64F]"
          />
        </div>

        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">How full to run the system</div>
          <select
            value={controls.oversubscriptionMode}
            onChange={(e) => onChange("oversubscriptionMode", e.target.value)}
            className="mt-2 w-full rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] p-2 text-sm text-[#262626]"
          >
            <option value="guarded">Careful</option>
            <option value="balanced">Balanced</option>
            <option value="aggressive">Aggressive</option>
          </select>
        </div>

        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Batching style</div>
          <select
            value={controls.batchingMode}
            onChange={(e) => onChange("batchingMode", e.target.value)}
            className="mt-2 w-full rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] p-2 text-sm text-[#262626]"
          >
            <option value="low">Small batches</option>
            <option value="adaptive">Auto batching</option>
            <option value="high">Large batches</option>
          </select>
        </div>

        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Power limit</div>
          <select
            value={controls.powerCapMode}
            onChange={(e) => onChange("powerCapMode", e.target.value)}
            className="mt-2 w-full rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] p-2 text-sm text-[#262626]"
          >
            <option value="disabled">Off</option>
            <option value="balanced">Balanced</option>
            <option value="strict">Tight limit</option>
          </select>
        </div>
      </div>

      <div className="demo-support-grid mt-4 grid grid-cols-1 gap-3 md:grid-cols-4">
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Work done each second</div>
          <div className="mt-1 text-lg font-bold text-[#6B7B48]">
            {capacityAssessment.throughputRps.toFixed(3)} req/s
          </div>
        </div>
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Waiting time</div>
          <div className="mt-1 text-lg font-bold text-[#7A7F54]">
            {capacityAssessment.queueDelayMs} ms
          </div>
        </div>
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Computer use</div>
          <div className="mt-1 text-lg font-bold text-[#5E7766]">
            {capacityAssessment.utilizationPct.toFixed(0)}%
          </div>
        </div>
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">System pressure</div>
          <div className="mt-1 text-lg font-bold text-[#6E7F6E]">
            {capacityAssessment.capacityPressure}
          </div>
        </div>
      </div>

      <div className="demo-support-grid mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Expected stability</div>
          <div className="mt-1 text-sm font-semibold text-[#262626]">
            {capacityAssessment.projectedStability.toFixed(1)}
          </div>
        </div>
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Load ratio</div>
          <div className="mt-1 text-sm font-semibold text-[#262626]">
            {capacityAssessment.oversubscriptionRatio.toFixed(2)}x
          </div>
        </div>
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Energy change</div>
          <div className="mt-1 text-sm font-semibold text-[#262626]">
            {capacityAssessment.energyDeltaPct > 0 ? "+" : ""}
            {capacityAssessment.energyDeltaPct.toFixed(1)}%
          </div>
        </div>
      </div>

      <div className="demo-support-grid mt-4 space-y-2 text-sm text-[#3A3A3A]">
        {capacityAssessment.recommendations.map((item) => (
          <div
            key={item}
            className="rounded-xl border border-[#D3D9D0] bg-[#F6F7F3] px-3 py-2"
          >
            {item}
          </div>
        ))}
      </div>
    </Card>
  );
}

