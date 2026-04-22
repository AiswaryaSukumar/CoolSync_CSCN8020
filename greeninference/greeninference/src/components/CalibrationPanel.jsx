import Card from "./Card";

function FactorRow({ label, value }) {
  return (
    <div className="rounded-xl border border-[#D3D9D0] bg-[#F6F7F3] px-3 py-2">
      <div className="text-[10px] uppercase tracking-[0.14em] text-[#6F756E]">{label}</div>
      <div className="mt-1 text-sm font-semibold text-[#262626]">{value}</div>
    </div>
  );
}

export default function CalibrationPanel({ calibrationState }) {
  if (!calibrationState) return null;
  const isSeeded = calibrationState.status === "SEEDED";

  return (
    <Card className="demo-calibration-card">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-[#262626]">Calibration</div>
          <div className="mt-1 text-xs text-[#6F756E]">
            {isSeeded
              ? "These are starting values before any uploaded data is used."
              : "These values are adjusted by comparing estimated results with uploaded data."}
          </div>
        </div>
        <div className="demo-signal-chip rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1 text-xs text-[#3A3A3A]">
          Confidence: <span className="ml-1 text-[#6B7B48]">{calibrationState.confidence}</span>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
        <FactorRow label="Error before" value={`${calibrationState.errors.beforePct.toFixed(1)}%`} />
        <FactorRow label="Error after" value={`${calibrationState.errors.afterPct.toFixed(1)}%`} />
        <FactorRow label="Difference left" value={`${calibrationState.residuals.whPerRequest.toFixed(3)} Wh`} />
      </div>

      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        <FactorRow
          label={isSeeded ? "Model starting value" : "Model adjustment"}
          value={`${calibrationState.factors.modelWhMultiplier.toFixed(3)}x`}
        />
        <FactorRow
          label={isSeeded ? "Start-phase value" : "Start-phase adjustment"}
          value={`${calibrationState.factors.prefillMultiplier.toFixed(3)}x`}
        />
        <FactorRow
          label={isSeeded ? "Output-phase value" : "Output-phase adjustment"}
          value={`${calibrationState.factors.decodeMultiplier.toFixed(3)}x`}
        />
        <FactorRow
          label={isSeeded ? "Site power value" : "Site power adjustment"}
          value={`${calibrationState.factors.siteOverheadMultiplier.toFixed(3)}x`}
        />
      </div>

      <div className="mt-4 rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
        <div className="text-xs text-[#6F756E]">Estimated result compared with uploaded result</div>
        <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
          <div>
            <div className="text-[#6F756E]">Wh/request</div>
            <div className="text-[#262626]">
              {calibrationState.baseline.estimatedWhPerRequest.toFixed(3)} to {calibrationState.baseline.observedWhPerRequest.toFixed(3)}
            </div>
          </div>
          <div>
            <div className="text-[#6F756E]">J/token</div>
            <div className="text-[#262626]">
              {calibrationState.baseline.estimatedJPerToken.toFixed(3)} to {calibrationState.baseline.observedJPerToken.toFixed(3)}
            </div>
          </div>
          <div>
            <div className="text-[#6F756E]">Extra building power</div>
            <div className="text-[#262626]">
              {calibrationState.baseline.estimatedOverhead.toFixed(3)} to {calibrationState.baseline.observedOverhead.toFixed(3)}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 space-y-2 text-sm text-[#3A3A3A]">
        {calibrationState.notes.map((note) => (
          <div key={note} className="rounded-xl border border-[#D3D9D0] bg-[#F6F7F3] px-3 py-2">
            {note}
          </div>
        ))}
      </div>
    </Card>
  );
}

