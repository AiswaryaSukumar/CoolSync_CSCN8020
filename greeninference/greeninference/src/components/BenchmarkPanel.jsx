import Card from "./Card";

function formatCapturedAt(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatCarbonImpact(value) {
  const amount = Number(value) || 0;
  if (amount === 0) return "below 1 mgCO2e";
  if (Math.abs(amount) < 0.01) return `${(amount * 1000).toFixed(1)} mgCO2e`;
  return `${amount.toFixed(2)} gCO2e`;
}

export default function BenchmarkPanel({
  benchmarkProfile,
  benchmarkResult,
  benchmarkHistory = [],
  onChangeProfile,
  onCaptureSnapshot,
  onClearHistory,
}) {
  if (!benchmarkProfile || !benchmarkResult) return null;

  return (
    <Card className="demo-support-surface">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-[#262626]">Standard test view</div>
          <div className="mt-1 text-xs text-[#6F756E]">
            This view helps compare common task types in a consistent way.
          </div>
        </div>
        <div className="demo-signal-chip rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1 text-xs text-[#3A3A3A]">
          {benchmarkResult.estimatedTag}
        </div>
      </div>

      <div className="mt-4">
        <label className="text-xs text-[#6F756E]">Test type</label>
        <select
          value={benchmarkProfile.id}
          onChange={(e) => onChangeProfile(e.target.value)}
          className="mt-1 w-full rounded-xl border border-[#D3D9D0] bg-[#F3F5F0] p-2 text-sm text-[#262626]"
        >
          <option value="short_qa">Short QA</option>
          <option value="long_context">Long summary</option>
          <option value="batch_brief">Batch task</option>
        </select>
      </div>

      <div className="demo-support-grid mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Speed goal</div>
          <div className="mt-1 text-sm font-semibold text-[#262626]">{benchmarkResult.qosLabel}</div>
        </div>
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Quality level</div>
          <div className="mt-1 text-sm font-semibold text-[#262626]">{benchmarkResult.qualityTier}</div>
        </div>
      </div>

      <div className="demo-support-grid mt-4 grid grid-cols-1 md:grid-cols-4 gap-3">
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Wh/request</div>
          <div className="mt-1 text-lg font-bold text-[#6B7B48]">{benchmarkResult.whPerRequest.toFixed(3)}</div>
        </div>
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">J/token</div>
          <div className="mt-1 text-lg font-bold text-[#5E7766]">{benchmarkResult.jPerToken.toFixed(3)}</div>
        </div>
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">gCO2e/request</div>
          <div className="mt-1 text-lg font-bold text-[#7B8F4B]">{formatCarbonImpact(benchmarkResult.co2ePerRequest)}</div>
        </div>
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Extra power ratio</div>
          <div className="mt-1 text-lg font-bold text-[#7A7F54]">{benchmarkResult.overheadRatio.toFixed(3)}</div>
        </div>
      </div>

      <div className="demo-support-surface mt-4 rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
        <div className="text-xs text-[#6F756E]">Note</div>
        <div className="mt-1 text-sm text-[#3A3A3A]">{benchmarkResult.comparabilityNote}</div>
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs text-[#6F756E]">
          Saved test runs help you compare the same task over time.
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onCaptureSnapshot}
            className="demo-signal-chip rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1.5 text-xs text-[#262626] transition-colors hover:bg-[#BFD98A]/45"
          >
            Save result
          </button>
          <button
            type="button"
            onClick={onClearHistory}
            className="demo-signal-chip rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1.5 text-xs text-[#3A3A3A] transition-colors hover:bg-[#F3F5F0]"
          >
            Clear list
          </button>
        </div>
      </div>

      <div className="demo-support-surface mt-4 rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
        <div className="text-xs text-[#6F756E]">Saved test results over time</div>
        {benchmarkHistory.length ? (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-xs text-[#3A3A3A]">
              <thead>
                <tr className="border-b border-[#D3D9D0] text-[#6F756E]">
                  <th className="pb-2 pr-3 font-medium">Time</th>
                  <th className="pb-2 pr-3 font-medium">Data source</th>
                  <th className="pb-2 pr-3 font-medium">Mode</th>
                  <th className="pb-2 pr-3 font-medium">Wh/request</th>
                  <th className="pb-2 pr-3 font-medium">gCO2e/request</th>
                  <th className="pb-2 pr-3 font-medium">Stability</th>
                </tr>
              </thead>
              <tbody>
                {benchmarkHistory.map((entry) => (
                  <tr key={entry.id} className="border-b border-[#E1E5DD]">
                    <td className="py-2 pr-3">{formatCapturedAt(entry.capturedAt)}</td>
                    <td className="py-2 pr-3">{entry.traceSource}</td>
                    <td className="py-2 pr-3">{entry.strategy}</td>
                    <td className="py-2 pr-3">{entry.whPerRequest.toFixed(3)}</td>
                    <td className="py-2 pr-3">{formatCarbonImpact(entry.co2ePerRequest)}</td>
                    <td className="py-2 pr-3">{entry.stabilityScore.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="mt-2 text-sm text-[#3A3A3A]">
            No saved results yet. Save a run to compare it later.
          </div>
        )}
      </div>
    </Card>
  );
}

