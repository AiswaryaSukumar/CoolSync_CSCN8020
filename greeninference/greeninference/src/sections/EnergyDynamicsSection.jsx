import React from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import SectionHeader from "../components/SectionHeader";
import Card from "../components/Card";
import SignalItem from "../components/SignalItem";
import StatusPill from "../components/StatusPill";

const timelineData = [
  { t: "0 ms", request: 0.2, cooling: 0.2 },
  { t: "80 ms", request: 0.92, cooling: 0.26 },
  { t: "160 ms", request: 0.8, cooling: 0.34 },
  { t: "240 ms", request: 0.55, cooling: 0.52 },
  { t: "320 ms", request: 0.34, cooling: 0.62 },
  { t: "400 ms", request: 0.22, cooling: 0.44 },
];

const signalCards = [
  {
    title: "Load spike",
    value: "Power demand rises first",
    text: "Demand rises before cooling reacts",
  },
  {
    title: "Cooling delay",
    value: "Lag window remains open",
    text: "Cooling peaks after the demand spike",
  },
  {
    title: "Recovery window",
    value: "Cooling catches up later",
    text: "Delay window closes after recovery begins",
  },
];

export default function EnergyDynamicsSection() {
  return (
    <section id="thermal" className="section-shell thermal-section">
      <div className="section-surface thermal-section__surface">
        <div className="section-inner section-inner--offset space-y-8 thermal-section__inner">
          <SectionHeader
            eyebrow="Thermal"
            title="Thermal Lag (Delayed Response)"
            subtitle="Cooling responds after compute spikes due to thermal inertia"
            actions={
              <div className="flex flex-wrap items-center gap-2">
                <StatusPill tone="warn">Status: Warning</StatusPill>
                <StatusPill tone="neutral">Lag pattern (Simulated)</StatusPill>
              </div>
            }
          />

          <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr] thermal-main-grid">
            <Card className="space-y-4 chart-container thermal-evidence-card">
              <div className="flex flex-wrap items-start justify-between gap-3 thermal-evidence-head">
                <div className="space-y-1 thermal-evidence-copy">
                  <div className="text-xs font-semibold uppercase tracking-[0.24em] text-[#6E7F71]">
                    Observed behavior
                  </div>
                  <div className="text-[1.05rem] font-semibold text-[#213428]">
                    Thermal Lag (Delayed Response)
                  </div>
                  <div className="text-sm text-[#55635B]">
                    Cooling response lags behind compute demand
                  </div>
                </div>
              </div>

              <div className="thermal-chart-legend" aria-label="Thermal chart legend">
                <span className="thermal-chart-legend__item thermal-chart-legend__item--compute">
                  Compute demand
                </span>
                <span className="thermal-chart-legend__item thermal-chart-legend__item--cooling">
                  Cooling response
                </span>
              </div>

              <div className="h-[312px] w-full thermal-evidence-chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={timelineData}
                    margin={{ top: 18, right: 18, left: 8, bottom: 24 }}
                  >
                    <defs>
                      <linearGradient id="requestArea" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#5A7B68" stopOpacity={0.28} />
                        <stop offset="95%" stopColor="#5A7B68" stopOpacity={0.04} />
                      </linearGradient>
                      <linearGradient id="coolingArea" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#C97C2B" stopOpacity={0.26} />
                        <stop offset="95%" stopColor="#C97C2B" stopOpacity={0.04} />
                      </linearGradient>
                    </defs>

                    <CartesianGrid
                      stroke="#DCE6DE"
                      strokeDasharray="3 3"
                      vertical={false}
                    />

                    <XAxis
                      dataKey="t"
                      tickLine={false}
                      axisLine={false}
                      tick={{ fill: "#6E7F71", fontSize: 12 }}
                      label={{
                        value: "Time after workload spike (ms)",
                        position: "insideBottom",
                        offset: -8,
                        style: {
                          fontSize: 12,
                          fill: "#5F6E64",
                          fontWeight: 500,
                        },
                      }}
                    />

                    <YAxis
                      width={52}
                      tickLine={false}
                      axisLine={false}
                      tick={{ fill: "#6E7F71", fontSize: 12 }}
                      tickFormatter={(value) => `${Math.round(value * 100)}%`}
                      label={{
                        value: "Load / Cooling (%)",
                        angle: -90,
                        position: "insideLeft",
                        dx: -4,
                        style: {
                          fontSize: 12,
                          fill: "#5F6E64",
                          fontWeight: 600,
                        },
                      }}
                    />

                    <Tooltip
                      cursor={{ stroke: "#A8B8AD", strokeDasharray: "4 4" }}
                      contentStyle={{
                        borderRadius: 16,
                        border: "1px solid rgba(106, 128, 114, 0.18)",
                        background: "rgba(248, 251, 248, 0.96)",
                        boxShadow: "0 14px 40px rgba(29, 56, 38, 0.08)",
                      }}
                      formatter={(value, name) => [
                        `${Math.round(Number(value) * 100)}%`,
                        name,
                      ]}
                    />

                    <ReferenceLine
                      x="160 ms"
                      stroke="#9C5516"
                      strokeOpacity={0.9}
                      strokeWidth={2.5}
                      strokeDasharray="4 4"
                      label={{
                        value: "Peak delay",
                        position: "top",
                        dy: -6,
                        fill: "#8A4710",
                        fontSize: 12,
                        fontWeight: 700,
                      }}
                    />

                    <ReferenceLine
                      x="320 ms"
                      stroke="#B7C5BA"
                      strokeOpacity={0.4}
                      strokeDasharray="3 3"
                    />

                    <Area
                      type="monotone"
                      dataKey="request"
                      name="Compute demand"
                      stroke="#5A7B68"
                      strokeWidth={2.2}
                      fill="url(#requestArea)"
                    />

                    <Area
                      type="monotone"
                      dataKey="cooling"
                      name="Cooling response"
                      stroke="#C97C2B"
                      strokeWidth={2.2}
                      fill="url(#coolingArea)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div className="thermal-evidence-notes">
                <div>Demand rises first</div>
                <div>Cooling peaks later</div>
                <div>Delay window remains open briefly</div>
              </div>
            </Card>

            <div className="thermal-side-stack">
              <Card className="space-y-3 thermal-state-card">
                <div className="flex items-start justify-between gap-3">
                  <div className="space-y-1">
                    <div className="text-xs font-semibold uppercase tracking-[0.24em] text-[#6E7F71]">
                      State
                    </div>
                    <div className="text-lg font-semibold text-[#213428]">
                      Cooling lag detected
                    </div>
                    <div className="text-sm text-[#55635B]">
                      Thermal lag is increasing overhead
                    </div>
                  </div>
                  <StatusPill tone="warn">Lag detected</StatusPill>
                </div>

                <div className="grid gap-3 sm:grid-cols-2 thermal-state-grid">
                  <div className="rounded-[18px] border border-[#DCE6DE] bg-white/70 px-4 thermal-state-metric">
                    <div className="text-xs uppercase tracking-[0.2em] text-[#7D8B82]">
                      Current state
                    </div>
                    <div className="mt-2 text-sm font-medium text-[#213428]">
                      Cooling response delayed
                    </div>
                  </div>

                  <div className="rounded-[18px] border border-[#DCE6DE] bg-white/70 px-4 thermal-state-metric">
                    <div className="text-xs uppercase tracking-[0.2em] text-[#7D8B82]">
                      Confidence
                    </div>
                    <div className="mt-2 text-sm font-medium text-[#213428]">
                      Simulated signal pattern
                    </div>
                  </div>
                </div>

                <div className="rounded-[18px] border border-[#DCE6DE] bg-white/70 px-4 thermal-state-metric thermal-state-severity">
                  <div className="text-xs uppercase tracking-[0.2em] text-[#7D8B82]">
                    Lag severity
                  </div>
                  <div className="mt-2 text-sm font-medium text-[#213428]">
                    Lag severity: Moderate
                  </div>
                </div>

                <div className="live-interpretation space-y-2 rounded-[18px] border border-[#E7EFE8] bg-[#F7FAF7] px-4">
                  <div className="text-xs uppercase tracking-[0.2em] text-[#7D8B82]">
                    Live interpretation
                  </div>
                  <SignalItem text="Cooling demand rises after the spike" />
                  <SignalItem text="Recovery window remains open" />
                  <SignalItem text="Cooling response is recovering" />
                </div>
              </Card>

              <Card className="space-y-4 control-card thermal-control-card">
                <div className="flex items-start justify-between gap-3">
                  <div className="space-y-2">
                    <div className="text-xs font-semibold uppercase tracking-[0.24em] text-[#6E7F71]">
                      Control
                    </div>
                    <div className="text-lg font-semibold text-[#182A1F] thermal-control-card__title">
                      Coordinated cooling active
                    </div>
                    <p className="text-xs text-[#55635B] thermal-control-card__note">
                      Coordinated mode recommended
                    </p>
                  </div>
                  <StatusPill tone="success">Mode: Coordinated</StatusPill>
                </div>

                <div className="thermal-decision-compare">
                  <div className="thermal-decision-compare__item thermal-decision-compare__item--reactive">
                    Reactive → lag persists
                  </div>
                  <div className="thermal-decision-compare__item thermal-decision-compare__item--coordinated">
                    Coordinated → lag reduced
                  </div>
                </div>
              </Card>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-3 insight-row thermal-insight-row">
            {signalCards.map((card) => (
              <Card key={card.title} className="space-y-3 insight-card thermal-insight-card">
                <div className="text-xs font-semibold uppercase tracking-[0.22em] text-[#6E7F71]">
                  Insight
                </div>
                <div className="space-y-1">
                  <div className="text-[1rem] font-semibold text-[#213428]">
                    {card.title}
                  </div>
                  <div className="text-sm font-medium text-[#4D5C53]">{card.value}</div>
                </div>
                <div className="text-sm text-[#55635B]">{card.text}</div>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}