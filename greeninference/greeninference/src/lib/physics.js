// FOPDT thermal model + PIDv2 controllers
// Ported from CoolSync_final/train_models.py

// ─── Physical constants ───────────────────────────────────────────────────────
export const PHYS = {
  RHO_AIR:    1.225,
  CP_AIR:     1006.0,
  C_RACK:     200_000.0,
  T_SUPPLY:   18.0,
  T_SETPOINT: 22.0,
  T_RL:       23.0,
  T_DANGER:   25.0,
  T_CRITICAL: 27.0,
  T_AMB:      24.0,
  DEAD_STEPS: 3,
  DT_SIM:     60.0,
  MAX_FLOW:   4.0,
  FAN_MAX_W:  5_000.0,
  FAN_MIN:    0.20,
  Q_BASE_W:   12_000.0,
  Q_RACK_MAX: 36_000.0,
  H_AMB:      30.0,
  ETA_CHILLER: 0.50,
};

const {
  RHO_AIR, CP_AIR, C_RACK, T_SUPPLY, T_SETPOINT, T_RL,
  T_DANGER, T_CRITICAL, T_AMB, DEAD_STEPS, DT_SIM,
  MAX_FLOW, FAN_MAX_W, FAN_MIN, Q_BASE_W, Q_RACK_MAX,
  H_AMB, ETA_CHILLER,
} = PHYS;

// ─── Burst class lookup (calibrated from workload_timeseries.csv) ─────────────
export const BURST_PARAMS = {
  0: { amp: 1.50, dur: 3,  label: "Short" },
  1: { amp: 1.42, dur: 7,  label: "Medium" },
  2: { amp: 1.33, dur: 13, label: "Long" },
  3: { amp: 1.25, dur: 20, label: "VeryLong" },
};

export function tokensToBurstClass(tokens) {
  if (tokens < 64)  return 0;
  if (tokens < 256) return 1;
  if (tokens < 800) return 2;
  return 3;
}

// ─── Physics helpers ──────────────────────────────────────────────────────────
export function qCool(fanPct, T, Tsup = T_SUPPLY) {
  return RHO_AIR * MAX_FLOW * fanPct * CP_AIR * Math.max(T - Tsup, 0.0);
}

export function pFan(fanPct) {
  return FAN_MAX_W * Math.pow(fanPct, 3);
}

export function pChiller(QcoolW, TsupplyC, TambC = T_AMB) {
  const TsupK = TsupplyC + 273.15;
  const TambK = TambC + 273.15;
  const COP = (TsupK / Math.max(TambK - TsupK, 0.5)) * ETA_CHILLER;
  return QcoolW / Math.max(COP, 1.0);
}

export function rackStep(T, Qdelayed, fanPct, Tsup = T_SUPPLY, dt = DT_SIM) {
  const Qc  = qCool(fanPct, T, Tsup);
  const Ql  = H_AMB * (T - T_AMB);
  const dT  = ((Qdelayed - Qc - Ql) / C_RACK) * dt;
  const Tnew = Math.max(10.0, Math.min(55.0, T + dT));
  return { Tnew, Qc, Pfan: pFan(fanPct) };
}

// ─── Controllers ──────────────────────────────────────────────────────────────
export class PIDv2 {
  constructor({
    kp = 0.015, ki = 0.001, sp = T_SETPOINT,
    fanMin = FAN_MIN, fanMax = 1.0,
    windup = 400.0, deadband = 1.5,
  } = {}) {
    this.kp = kp; this.ki = ki;
    this.sp = sp; this.fanMin = fanMin; this.fanMax = fanMax;
    this.windup = windup; this.deadband = deadband;
    this._I = 0.0; this._fanOut = 0.61;
  }

  reset() { this._I = 0.0; this._fanOut = 0.61; }

  compute(T, dt = DT_SIM) {
    const e = T - this.sp;
    if (Math.abs(e) > this.deadband) {
      this._I = Math.max(-this.windup, Math.min(this.windup, this._I + e * dt));
    } else {
      this._I *= 0.7;
    }
    const fanRaw     = 0.61 + this.kp * e + this.ki * this._I;
    const fanClipped = Math.max(this.fanMin, Math.min(this.fanMax, fanRaw));
    const alpha      = T < T_DANGER ? 0.30 : 0.80;
    this._fanOut     = alpha * fanClipped + (1.0 - alpha) * this._fanOut;
    return this._fanOut;
  }
}

export class PIDv2Conservative {
  static T_SUPPLY_COLD = 16.0;

  constructor({
    kp = 0.015, ki = 0.001, sp = 20.0,
    fanMin = FAN_MIN, fanMax = 1.0,
    windup = 400.0, deadband = 0.5,
  } = {}) {
    this.kp = kp; this.ki = ki;
    this.sp = sp; this.fanMin = fanMin; this.fanMax = fanMax;
    this.windup = windup; this.deadband = deadband;
    this._I = 0.0;
  }

  reset() { this._I = 0.0; }

  compute(T, dt = DT_SIM) {
    const e = T - this.sp;
    if (Math.abs(e) > this.deadband) {
      this._I = Math.max(-this.windup, Math.min(this.windup, this._I + e * dt));
    } else {
      this._I *= 0.9;
    }
    const fanRaw = 0.61 + this.kp * e + this.ki * this._I;
    return Math.max(this.fanMin, Math.min(this.fanMax, fanRaw));
  }
}

// ─── Workload generator ───────────────────────────────────────────────────────
export function generateWorkload(nSteps, burstStart, burstClass, concurrentUsers = 1) {
  const heat = new Array(nSteps).fill(Q_BASE_W);
  const bp   = BURST_PARAMS[burstClass] ?? BURST_PARAMS[1];
  const Qrack = Math.min(Q_BASE_W * (1 + (bp.amp - 1) * concurrentUsers), Q_RACK_MAX);
  const end  = Math.min(burstStart + bp.dur, nSteps);
  for (let i = burstStart; i < end; i++) heat[i] = Qrack;
  return heat;
}

// ─── Core simulation loop ─────────────────────────────────────────────────────
function runLoop(heatTrace, getAction, T0, Tsup0) {
  const n      = heatTrace.length;
  const deadBuf = new Array(DEAD_STEPS).fill(heatTrace[0] * 0.5);
  let T    = T0;
  let cumE = 0.0;
  const rows = [];

  for (let step = 0; step < n; step++) {
    const Q = heatTrace[step];
    const { fan, tsup } = getAction(T, step);
    const { Tnew, Qc, Pfan } = rackStep(T, deadBuf[0], fan, tsup);
    deadBuf.shift(); deadBuf.push(Q);
    const Pchill = pChiller(Qc, tsup);
    const Ptotal = Pfan + Pchill;
    cumE += Ptotal * DT_SIM / 3600.0;

    // phase based on heat level relative to baseline
    let phase = "RECOVERY";
    if (Q > Q_BASE_W * 1.05) phase = "DECODE";
    else if (step < DEAD_STEPS) phase = "PREFILL";

    rows.push({
      t:           step * 60_000,
      step,
      inlet:       T,
      T_rack:      T,
      fan_pct:     fan,
      fanPct:      fan,
      P_fan_w:     Pfan,
      P_chiller_w: Pchill,
      P_total_w:   Ptotal,
      Q_gpu:       Q,
      heatW:       Q,
      gpuW:        Q,
      energy_wh:   cumE,
      T_supply:    tsup,
      tSupply:     tsup,
      coolingKw:   (Pfan + Pchill) / 1000,
      phase,
    });
    T = Tnew;
  }
  return rows;
}

// ─── Strategy runners ─────────────────────────────────────────────────────────
export function runPID(heatTrace) {
  const pid = new PIDv2();
  return runLoop(
    heatTrace,
    (T) => ({ fan: pid.compute(T), tsup: T_SUPPLY }),
    T_SETPOINT,
    T_SUPPLY,
  );
}

export function runPIDConservative(heatTrace) {
  const pid  = new PIDv2Conservative();
  const tsup = PIDv2Conservative.T_SUPPLY_COLD;
  return runLoop(
    heatTrace,
    (T) => ({ fan: pid.compute(T), tsup }),
    20.0,
    tsup,
  );
}

export function runCoordinated(heatTrace) {
  // Simulates LSTM+DQN lookahead benefit: fan ramps DEAD_STEPS before burst arrives.
  // Pre-cooling fires when predicted future heat exceeds threshold (= LSTM signal).
  const pid = new PIDv2();
  const n   = heatTrace.length;
  return runLoop(
    heatTrace,
    (T, step) => {
      const futureQ = heatTrace[Math.min(step + DEAD_STEPS, n - 1)];
      const pid_fan = pid.compute(T);
      if (futureQ > Q_BASE_W * 1.15 && T < T_DANGER) {
        return {
          fan:  Math.min(1.0, pid_fan + 0.15),
          tsup: 19.0,
        };
      }
      return { fan: pid_fan, tsup: T_SUPPLY };
    },
    T_SETPOINT,
    T_SUPPLY,
  );
}

// ─── High-level helpers for simulate.js ──────────────────────────────────────
export function buildPhysicsRun({ tokens = 128, strategy = "reactive" }) {
  const burstClass  = tokensToBurstClass(tokens);
  const bp          = BURST_PARAMS[burstClass];
  const nSteps      = Math.max(bp.dur + 20, 30); // enough pre- and post-burst steps
  const burstStart  = 8;

  const heatTrace = generateWorkload(nSteps, burstStart, burstClass, 1);

  let points;
  if (strategy === "coordinated") {
    points = runCoordinated(heatTrace);
  } else {
    points = runPID(heatTrace);
  }

  const energyWh    = points[points.length - 1]?.energy_wh ?? 0;
  const peakT       = Math.max(...points.map(p => p.T_rack));
  const breachSteps = points.filter(p => p.T_rack > T_DANGER).length;
  // Coordinated mode: LSTM prediction eliminates effective cooling lag
  const lagMs = strategy === "coordinated" ? 0 : DEAD_STEPS * 60_000;

  return { points, energyWh, peakT, breachSteps, lagMs };
}

export function buildConservativeRun({ tokens = 128 }) {
  const burstClass  = tokensToBurstClass(tokens);
  const bp          = BURST_PARAMS[burstClass];
  const nSteps      = Math.max(bp.dur + 20, 30);
  const burstStart  = 8;

  const heatTrace = generateWorkload(nSteps, burstStart, burstClass, 1);
  const points    = runPIDConservative(heatTrace);

  const energyWh    = points[points.length - 1]?.energy_wh ?? 0;
  const peakT       = Math.max(...points.map(p => p.T_rack));
  const breachSteps = points.filter(p => p.T_rack > T_DANGER).length;

  return { points, energyWh, peakT, breachSteps, lagMs: DEAD_STEPS * 60_000 };
}
