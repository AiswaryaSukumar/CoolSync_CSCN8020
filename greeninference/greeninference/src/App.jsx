import { Suspense, lazy } from "react";
import Navbar from "./components/Navbar";
import KpiCard from "./components/KpiCard";

const ChallengeSection = lazy(() => import("./sections/ChallengeSection"));
const EnergyDynamicsSection = lazy(() => import("./sections/EnergyDynamicsSection"));
const ArchitectureSection = lazy(() => import("./sections/ArchitectureSection"));
const PrototypeDemoSection = lazy(() => import("./sections/PrototypeDemoSection"));

export default function App() {
  const topKpis = [
    { label: "Workload", value: "Active", hint: "Simulated view", accent: "neutral" },
    {
      label: "Thermal Lag",
      value: "Cooling starts late",
      hint: "Simulated view",
      accent: "amber",
    },
    {
      label: "Cooling Share",
      value: "30–50%",
      hint: "Research-based",
      accent: "emerald",
    },
    {
      label: "Carbon View",
      value: <span className="whitespace-nowrap">Regional comparison</span>,
      hint: "Modeled view",
      accent: "neutral",
    },
    {
      label: "System State",
      value: "Lag detected",
      hint: "Modeled view",
      accent: "amber",
    },
  ];

  return (
    <div className="app-shell">
      <Navbar />

      <main className="app-main">
        <div className="app-container">
          <div className="kpi-strip" aria-label="System KPI strip">
            {topKpis.map((item) => (
              <KpiCard
                key={item.label}
                label={item.label}
                value={item.value}
                hint={item.hint}
                accent={item.accent}
                className={[
                  "kpi-strip__card ds-card--secondary",
                  item.label === "System State" ? "system-state-card" : "",
                ]
                  .join(" ")
                  .trim()}
                valueClassName="kpi-strip__value"
                hintClassName="kpi-strip__hint"
              />
            ))}
          </div>

          <Suspense
            fallback={
              <div className="ds-card rounded-[var(--radius-xl)] px-5 py-10 text-center text-sm text-[var(--color-text-muted)]">
                Loading the dashboard...
              </div>
            }
          >
            <ChallengeSection />
            <EnergyDynamicsSection />
            <ArchitectureSection />
            <PrototypeDemoSection />
          </Suspense>
        </div>
      </main>

      <footer className="app-footer">
        <div className="app-footer__inner">
          GreenInference demo | AI Cooling Decision Support | simulation and modeled views
        </div>
      </footer>
    </div>
  );
}