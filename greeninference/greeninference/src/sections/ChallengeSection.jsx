import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertTriangle,
  EyeOff,
  Snowflake,
  TrendingUp,
} from "lucide-react";
import Card from "../components/Card";
import Section from "../components/Section";

const overheadData = [
  { name: "Reactive", overhead: 3, status: "Higher overhead" },
  { name: "Coordinated", overhead: 2, status: "Lower energy impact" },
];

const stateMetrics = [
  { label: "Current state", value: "Cooling delay → energy rising" },
  { label: "Priority", value: "Warning" },
  { label: "Evidence basis", value: "Research + simulation" },
  { label: "Cooling share", value: "30–50% of energy" },
];

const overviewSignals = [
  {
    title: "Lag can repeat",
    copy: "New spikes can reopen the lag window.",
    icon: TrendingUp,
    tone: "text-[#5F6B67]",
  },
  {
    title: "Late cooling risk",
    copy: "Cooling can arrive after heat has already risen.",
    icon: Snowflake,
    tone: "text-[#B4691F]",
  },
  {
    title: "Hidden extra energy",
    copy: "Small delays can accumulate into facility overhead.",
    icon: EyeOff,
    tone: "text-[#7A7F54]",
  },
];

export default function ChallengeSection() {
  return (
    <Section
      id="overview"
      className="challenge-section"
      eyebrow="Overview"
      title="Why cooling timing matters"
      subtitle="AI workload can rise quickly, while cooling responds later"
    >
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)]">
        <Card className="ds-card--secondary system-state-card">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-[0.72rem] font-semibold uppercase tracking-[0.12em] text-[#697770]">
                State
              </div>
              <div className="mt-2 text-xl font-semibold text-[#18212F]">
                Cooling lag is increasing extra energy use
              </div>
            </div>

            <div className="inline-flex items-center gap-2 rounded-full border border-[rgba(180,105,31,0.18)] bg-[#F8EFE6] px-3 py-1 text-xs font-semibold text-[#9A5D23]">
              <AlertTriangle size={14} />
              Status: Warning
            </div>
          </div>

          <div className="state-grid mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {stateMetrics.map((item, index) => (
              <div
                key={item.label}
                className={[
                  "state-metric rounded-[16px] border border-[rgba(95,104,96,0.08)] bg-[#F8FAF8] px-3 py-3",
                  index === 0 ? "state-metric--primary" : "state-metric--secondary",
                ].join(" ")}
              >
                <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#6A776F]">
                  {item.label}
                </div>
                <div className="mt-2 text-sm font-semibold text-[#243240]">{item.value}</div>
              </div>
            ))}
          </div>

          <div className="mt-4 rounded-[18px] border border-[rgba(95,104,96,0.08)] bg-[#F8FAF8] px-4 py-3">
            <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#6A776F]">
              Live interpretation
            </div>
            <div className="mt-2 text-sm font-semibold text-[#314355]">
              Workload rises first
              <br />
              Cooling responds later
              <br />
              Earlier coordination can reduce extra energy use
            </div>
          </div>
        </Card>

        <Card className="ds-card--secondary control-card">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="text-[0.72rem] font-semibold uppercase tracking-[0.12em] text-[#697770]">
                Comparison
              </div>
              <div className="mt-2 text-xl font-semibold text-[#18212F]">
                Reactive vs coordinated cooling
              </div>
              <div className="mt-2 max-w-xl text-sm font-medium text-[var(--color-text-body)]">
                Reactive mode starts cooling after the spike.
                <br />
                Coordinated mode starts earlier to reduce the lag window.
              </div>
            </div>

            <div className="inline-flex items-center gap-2 rounded-full border border-[rgba(95,104,96,0.12)] bg-[#F5F8F4] px-3 py-1 text-xs font-semibold text-[#5F6B67]">
              Simulated comparison
            </div>
          </div>

          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="rounded-[18px] border border-[rgba(180,105,31,0.16)] bg-[#FBF4EC] px-4 py-3">
              <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#6A776F]">
                Reactive cooling
              </div>
              <div className="mt-2 text-lg font-bold text-[#9A5D23]">
                Higher overhead
              </div>
              <div className="mt-1 text-xs text-[#5F6B67]">
                Cooling responds after the load spike
              </div>
            </div>

            <div className="rounded-[18px] border border-[rgba(104,173,63,0.16)] bg-[#F2F8EF] px-4 py-3">
              <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#6A776F]">
                Coordinated cooling
              </div>
              <div className="mt-2 text-lg font-bold text-[#4F7A32]">
                Lower energy impact
              </div>
              <div className="mt-1 text-xs text-[#5F6B67]">
                Cooling timing shifts earlier to reduce lag
              </div>
            </div>
          </div>

          <div className="mt-4 rounded-[18px] border border-[rgba(95,104,96,0.08)] bg-[#F8FAF8] px-4 py-3 text-sm font-medium text-[#314355]">
            Safe takeaway → the main issue is not only how much cooling is used, but when cooling starts.
          </div>
        </Card>
      </div>

      <Card className="ds-card--primary chart-panel mt-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-[0.72rem] font-semibold uppercase tracking-[0.12em] text-[#697770]">
              Evidence
            </div>
            <div className="mt-2 font-semibold text-[#18212F]">Cooling delay impact</div>
            <div className="mt-2 max-w-2xl text-sm font-medium text-[var(--color-text-body)]">
              This chart is a simple relative comparison.
              <br />
              It shows why earlier cooling timing can reduce extra energy use.
            </div>
          </div>

          <div className="flex flex-wrap gap-2 text-[11px] font-semibold">
            <span className="rounded-full border border-[rgba(180,105,31,0.16)] bg-[#F8EFE6] px-3 py-1 text-[#9A5D23]">
              Relative comparison
            </span>
            <span className="rounded-full border border-[rgba(95,104,96,0.12)] bg-[#F5F8F4] px-3 py-1 text-[#5F6B67]">
              Simulated view
            </span>
            <span className="rounded-full border border-[rgba(104,173,63,0.16)] bg-[#F2F8EF] px-3 py-1 text-[#4F7A32]">
              Cooling share: 30–50% | Research-based
            </span>
          </div>
        </div>

        <div className="chart-shell chart-container mt-5">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={overheadData}
              margin={{ top: 8, right: 12, left: 10, bottom: 24 }}
              barCategoryGap={36}
            >
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis
                dataKey="name"
                tick={{ fill: "#374151", fontSize: 13, fontWeight: 600 }}
                tickMargin={10}
                interval={0}
                axisLine={{ stroke: "#94A3B8", strokeWidth: 1.2 }}
                tickLine={false}
              />
              <YAxis
                domain={[0, 4]}
                width={54}
                tick={{ fill: "#334155", fontSize: 12, fontWeight: 500 }}
                axisLine={false}
                tickLine={false}
                label={{
                  value: "Relative overhead level",
                  angle: -90,
                  position: "insideLeft",
                  fill: "#475569",
                  fontSize: 12,
                  fontWeight: 600,
                  dx: -2,
                }}
              />
              <ReferenceLine
                y={2.5}
                stroke="#B4691F"
                strokeOpacity={0.35}
                strokeDasharray="4 4"
              />
              <Tooltip
                contentStyle={{
                  background: "#FFFFFF",
                  border: "1px solid #D7DED8",
                  borderRadius: "16px",
                  boxShadow: "0 12px 30px rgba(52, 68, 59, 0.10)",
                }}
                labelStyle={{ color: "#18212F", fontWeight: 600 }}
                itemStyle={{ color: "#334155" }}
                formatter={(value) => [value, "Relative overhead level"]}
              />
              <Bar dataKey="overhead" radius={[12, 12, 0, 0]} maxBarSize={84}>
                <Cell fill="#B4691F" />
                <Cell fill="#8BC34A" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-4 rounded-[18px] border border-[rgba(95,104,96,0.08)] bg-[#F8FAF8] px-4 py-3 text-sm text-[#314355]">
          Coordinated mode is shown lower here because the cooling response starts earlier.
        </div>
      </Card>

      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-3">
        {overviewSignals.map(({ title, copy, icon, tone }) => (
          <Card key={title} className="ds-card--secondary insight-card">
            <div className="text-[0.72rem] font-semibold uppercase tracking-[0.12em] text-[#697770]">
              Insight
            </div>
            <div className="mt-3 flex items-center gap-2">
              {React.createElement(icon, { className: tone, size: 18 })}
              <div className="font-semibold text-[#18212F]">{title}</div>
            </div>
            <p className="mt-3 text-sm font-medium text-[var(--color-text-body)]">{copy}</p>
          </Card>
        ))}
      </div>
    </Section>
  );
}