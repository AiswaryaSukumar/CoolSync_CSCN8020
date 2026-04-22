import Card from "./Card";

function formatCarbonImpact(value) {
  const amount = Number(value) || 0;
  if (amount === 0) return "below 1 mgCO2e";
  if (Math.abs(amount) < 0.01) return `${(amount * 1000).toFixed(1)} mgCO2e`;
  return `${amount.toFixed(2)} gCO2e`;
}

export default function RoutingRecommendationCard({ recommendation, signals }) {
  if (!recommendation || !signals) return null;

  return (
    <Card className="demo-support-surface">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-[#262626]">
            Where and when to run the work
          </div>
          <div className="mt-1 text-xs text-[#6F756E]">
            A simple suggestion based on electricity pollution, water stress,
            and how urgent the work is.
          </div>
        </div>
        <div className="demo-signal-chip rounded-full border border-[#D3D9D0] bg-[#DDE4DA] px-3 py-1 text-xs text-[#3A3A3A]">
          Level:
          <span className="ml-1 text-[#6B7B48]">
            {recommendation.recommendationLevel}
          </span>
        </div>
      </div>

      <div className="demo-support-surface mt-4 rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
        <div className="text-xs text-[#6F756E]">Suggested action</div>
        <div className="mt-1 text-lg font-bold text-[#262626]">
          {recommendation.recommendation}
        </div>
        <div className="mt-1 text-xs text-[#6F756E]">
          Action: {recommendation.action} | Current location:{" "}
          {recommendation.currentRegion}
        </div>
      </div>

      <div className="demo-support-grid mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Pollution for this task</div>
          <div className="mt-1 text-lg font-bold text-[#7A7F54]">
            {formatCarbonImpact(recommendation.currentCo2e)}
          </div>
        </div>
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Pollution saved by moving it</div>
          <div className="mt-1 text-lg font-bold text-[#6B7B48]">
            {formatCarbonImpact(recommendation.routeSavings)}
          </div>
        </div>
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Pollution saved by waiting</div>
          <div className="mt-1 text-lg font-bold text-[#5E7766]">
            {formatCarbonImpact(recommendation.deferSavings)}
          </div>
        </div>
      </div>

      <div className="demo-support-grid mt-4 grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Pollution risk</div>
          <div className="mt-1 font-semibold text-[#262626]">
            {signals.carbonRisk}
          </div>
        </div>
        <div className="rounded-2xl border border-[#D3D9D0] bg-[#F6F7F3] p-3">
          <div className="text-xs text-[#6F756E]">Water risk</div>
          <div className="mt-1 font-semibold text-[#262626]">
            {signals.waterRisk}
          </div>
        </div>
      </div>

      <div className="demo-support-grid mt-4 space-y-2 text-sm text-[#3A3A3A]">
        {recommendation.rationale.map((item) => (
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

