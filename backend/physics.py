"""
CoolSync physics engine — ported from CoolSync_final/train_models.py.
Single source of truth for server-side simulation (mirrors src/lib/physics.js).
"""
from __future__ import annotations
import numpy as np
from collections import deque

# ─── Constants ───────────────────────────────────────────────────────────────
RHO_AIR    = 1.225
CP_AIR     = 1006.0
C_RACK     = 200_000.0
T_SUPPLY   = 18.0
T_SETPOINT = 22.0
T_RL       = 23.0
T_DANGER   = 25.0
T_CRITICAL = 27.0
T_AMB      = 24.0
DEAD_STEPS = 3
DT_SIM     = 60.0
MAX_FLOW   = 4.0
FAN_MAX_W  = 5_000.0
FAN_MIN    = 0.20
Q_BASE_W   = 12_000.0
Q_RACK_MAX = 36_000.0
H_AMB      = 30.0
ETA_CHILLER = 0.50

FAN_LEVELS  = np.array([0.30, 0.45, 0.60, 0.75, 1.00])
TSUP_LEVELS = np.array([16.0, 19.0])
JOINT_ACTIONS = [(float(f), float(ts)) for f in FAN_LEVELS for ts in TSUP_LEVELS]
N_ACTIONS   = len(JOINT_ACTIONS)   # 10

BURST_PARAMS = {
    0: dict(amp=1.50, dur=3,  label="Short"),
    1: dict(amp=1.42, dur=7,  label="Medium"),
    2: dict(amp=1.33, dur=13, label="Long"),
    3: dict(amp=1.25, dur=20, label="VeryLong"),
}

# ─── Physics helpers ──────────────────────────────────────────────────────────
def q_cool(fan_pct: float, T: float, T_sup: float = T_SUPPLY) -> float:
    return RHO_AIR * MAX_FLOW * fan_pct * CP_AIR * max(T - T_sup, 0.0)

def p_fan(fan_pct: float) -> float:
    return FAN_MAX_W * fan_pct ** 3

def p_chiller(Q_cool_W: float, T_supply_C: float, T_amb_C: float = T_AMB) -> float:
    T_sup_K = T_supply_C + 273.15
    T_amb_K = T_amb_C    + 273.15
    COP = T_sup_K / max(T_amb_K - T_sup_K, 0.5) * ETA_CHILLER
    return Q_cool_W / max(COP, 1.0)

def rack_step(T: float, Q_delayed: float, fan_pct: float,
              T_sup: float = T_SUPPLY, dt: float = DT_SIM):
    Qc  = q_cool(fan_pct, T, T_sup)
    Ql  = H_AMB * (T - T_AMB)
    dT  = (Q_delayed - Qc - Ql) / C_RACK * dt
    return float(np.clip(T + dT, 10.0, 55.0)), Qc, p_fan(fan_pct)

# ─── Controllers ──────────────────────────────────────────────────────────────
class PIDv2:
    def __init__(self, kp=0.015, ki=0.001, sp=T_SETPOINT,
                 fan_min=FAN_MIN, fan_max=1.0, windup=400.0, deadband=1.5):
        self.kp, self.ki = kp, ki
        self.sp = sp; self.fan_min = fan_min; self.fan_max = fan_max
        self.windup = windup; self.deadband = deadband
        self._I = 0.0; self._fan_out = 0.61

    def reset(self): self._I = 0.0; self._fan_out = 0.61

    def compute(self, T, dt=DT_SIM):
        e = T - self.sp
        if abs(e) > self.deadband:
            self._I = float(np.clip(self._I + e * dt, -self.windup, self.windup))
        else:
            self._I *= 0.7
        fan_raw     = 0.61 + self.kp * e + self.ki * self._I
        fan_clipped = float(np.clip(fan_raw, self.fan_min, self.fan_max))
        alpha       = 0.30 if T < T_DANGER else 0.80
        self._fan_out = alpha * fan_clipped + (1.0 - alpha) * self._fan_out
        return self._fan_out


class PIDv2Conservative:
    T_SUPPLY_COLD = 16.0

    def __init__(self, kp=0.015, ki=0.001, sp=20.0,
                 fan_min=FAN_MIN, fan_max=1.0, windup=400.0, deadband=0.5):
        self.kp, self.ki = kp, ki
        self.sp = sp; self.fan_min = fan_min; self.fan_max = fan_max
        self.windup = windup; self.deadband = deadband
        self._I = 0.0

    def reset(self): self._I = 0.0

    def compute(self, T, dt=DT_SIM):
        e = T - self.sp
        if abs(e) > self.deadband:
            self._I = float(np.clip(self._I + e * dt, -self.windup, self.windup))
        else:
            self._I *= 0.9
        fan_raw = 0.61 + self.kp * e + self.ki * self._I
        return float(np.clip(fan_raw, self.fan_min, self.fan_max))

# ─── Workload generator ───────────────────────────────────────────────────────
def generate_workload(n_steps: int, burst_start: int = 8,
                      burst_class: int = 1, concurrent_users: int = 1):
    heat = np.full(n_steps, Q_BASE_W, dtype=float)
    bp   = BURST_PARAMS[burst_class]
    Q_rack = min(Q_BASE_W * (1 + (bp["amp"] - 1) * concurrent_users), Q_RACK_MAX)
    end  = min(burst_start + bp["dur"], n_steps)
    heat[burst_start:end] = Q_rack
    return heat

# ─── Simulation runners ───────────────────────────────────────────────────────
def _run_loop(heat_trace, get_action, T0=T_SETPOINT, tsup0=T_SUPPLY):
    buf   = deque([heat_trace[0] * 0.5] * DEAD_STEPS, maxlen=DEAD_STEPS)
    T     = T0
    cum_e = 0.0
    rows  = []
    n     = len(heat_trace)

    for step, Q in enumerate(heat_trace):
        fan, tsup = get_action(T, step)
        T_new, Qc, Pf = rack_step(T, buf[0], fan, tsup)
        buf.append(Q)
        Pc     = p_chiller(Qc, tsup)
        Pt     = Pf + Pc
        cum_e += Pt * DT_SIM / 3600.0

        if   Q > Q_BASE_W * 1.05: phase = "DECODE"
        elif step < DEAD_STEPS:    phase = "PREFILL"
        else:                      phase = "RECOVERY"

        rows.append({
            "t":           step * 60_000,
            "step":        step,
            "T_rack":      round(T, 4),
            "inlet":       round(T, 4),
            "fan_pct":     round(fan, 4),
            "P_fan_w":     round(Pf, 2),
            "P_chiller_w": round(Pc, 2),
            "P_total_w":   round(Pt, 2),
            "Q_gpu":       round(Q, 1),
            "gpuW":        round(Q, 1),
            "heatW":       round(Q, 1),
            "energy_wh":   round(cum_e, 4),
            "T_supply":    round(tsup, 2),
            "tSupply":     round(tsup, 2),
            "coolingKw":   round((Pf + Pc) / 1000, 4),
            "phase":       phase,
        })
        T = T_new

    return rows


def run_pid(heat_trace):
    pid = PIDv2()
    return _run_loop(heat_trace, lambda T, _: (pid.compute(T), T_SUPPLY),
                     T0=T_SETPOINT, tsup0=T_SUPPLY)


def run_pid_conservative(heat_trace):
    pid  = PIDv2Conservative()
    tsup = PIDv2Conservative.T_SUPPLY_COLD
    return _run_loop(heat_trace, lambda T, _: (pid.compute(T), tsup),
                     T0=20.0, tsup0=tsup)


def run_coordinated(heat_trace):
    """Simulates LSTM lookahead: fan ramps DEAD_STEPS before burst arrives."""
    pid = PIDv2()
    n   = len(heat_trace)

    def get_action(T, step):
        future_Q = heat_trace[min(step + DEAD_STEPS, n - 1)]
        pid_fan  = pid.compute(T)
        if future_Q > Q_BASE_W * 1.15 and T < T_DANGER:
            return min(1.0, pid_fan + 0.15), 19.0
        return pid_fan, T_SUPPLY

    return _run_loop(heat_trace, get_action, T0=T_SETPOINT, tsup0=T_SUPPLY)


def run_dqn(heat_trace, dqn_model, lstm_model, joint_actions, device):
    """Full DQN+LSTM inference using loaded PyTorch models."""
    import torch
    ACT_HOLD   = 2
    LOOKBACK   = DEAD_STEPS
    buf        = deque([heat_trace[0] * 0.5] * DEAD_STEPS, maxlen=DEAD_STEPS)
    T, fan, T_sup = T_RL, 0.65, 19.0
    cum_e     = 0.0
    T_hist    = [T]
    Q_hist    = [float(heat_trace[0])]
    fan_hist  = [fan]
    hold_counter = 0
    rows = []
    n    = len(heat_trace)

    def lstm_predict(T_h, Q_h, f_h):
        if len(T_h) < LOOKBACK:
            return (T_h[-1] - 22.0) / 5.0
        seq_T   = np.array([(t - 22.0) / 5.0  for t in T_h[-LOOKBACK:]])
        seq_Q   = np.array([(q - Q_BASE_W) / Q_BASE_W for q in Q_h[-LOOKBACK:]])
        seq_fan = np.array(f_h[-LOOKBACK:])
        x = torch.tensor(
            np.stack([seq_T, seq_Q, seq_fan], axis=1),
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            return float(lstm_model(x).item())

    def make_obs(T_cur, Q_now, fan_p, q_buf_cur, step, total, T_pred_n, tsup_p):
        buf_list    = list(q_buf_cur)
        t_err       = (T_cur - T_RL) / 5.0
        q_now       = (Q_now - Q_BASE_W) / Q_BASE_W
        f_prev      = (fan_p - 0.5) / 0.5
        q_mean      = (float(np.mean(buf_list)) - Q_BASE_W) / Q_BASE_W
        q_trnd      = (Q_now - buf_list[-1]) / Q_BASE_W
        above       = max(0.0, T_cur - T_RL) / 5.0
        below       = max(0.0, T_RL - T_cur) / 5.0
        phase       = step / max(total - 1, 1)
        q_lag       = (buf_list[0] - Q_BASE_W) / Q_BASE_W
        T_pred_d    = (T_pred_n * 5.0) - (T_cur - 22.0)
        tsup_norm   = (tsup_p - 19.0) / 3.0
        T_sup_K     = tsup_p + 273.15
        cop         = T_sup_K / max((T_AMB + 273.15) - T_sup_K, 0.5) * ETA_CHILLER
        cop_norm    = (cop - 3.0) / 2.0
        return np.array([
            t_err, q_now, f_prev, q_mean, q_trnd,
            above, below, phase, q_lag,
            T_pred_n, T_pred_d / 5.0,
            tsup_norm, cop_norm,
        ], dtype=np.float32)

    import torch
    for step, Q in enumerate(heat_trace):
        T_pred = lstm_predict(T_hist, Q_hist, fan_hist)
        obs    = make_obs(T, Q, fan, buf, step, n, T_pred, T_sup)

        if hold_counter <= 0 or T >= T_DANGER:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                a = int(dqn_model(obs_t).argmax())
            new_fan, new_tsup = joint_actions[a]
            if new_fan != fan or new_tsup != T_sup:
                fan, T_sup = new_fan, new_tsup
                hold_counter = ACT_HOLD
            else:
                hold_counter = max(hold_counter - 1, 0)
        else:
            hold_counter -= 1

        T_new, Qc, Pf = rack_step(T, buf[0], fan, T_sup)
        buf.append(Q)
        Pc     = p_chiller(Qc, T_sup)
        Pt     = Pf + Pc
        cum_e += Pt * DT_SIM / 3600.0

        phase = "DECODE" if Q > Q_BASE_W * 1.05 else ("PREFILL" if step < DEAD_STEPS else "RECOVERY")

        rows.append({
            "t":           step * 60_000,
            "step":        step,
            "T_rack":      round(T, 4),
            "inlet":       round(T, 4),
            "fan_pct":     round(fan, 4),
            "P_fan_w":     round(Pf, 2),
            "P_chiller_w": round(Pc, 2),
            "P_total_w":   round(Pt, 2),
            "Q_gpu":       round(Q, 1),
            "gpuW":        round(Q, 1),
            "heatW":       round(Q, 1),
            "energy_wh":   round(cum_e, 4),
            "T_supply":    round(T_sup, 2),
            "tSupply":     round(T_sup, 2),
            "coolingKw":   round((Pf + Pc) / 1000, 4),
            "T_pred":      round(T_pred * 5.0 + 22.0, 3),
            "phase":       phase,
        })
        T = T_new
        T_hist.append(T_new)
        Q_hist.append(Q)
        fan_hist.append(fan)

    return rows


def summarise(rows):
    energy_wh    = rows[-1]["energy_wh"] if rows else 0.0
    peak_T       = max(r["T_rack"] for r in rows) if rows else 0.0
    breach_steps = sum(1 for r in rows if r["T_rack"] > T_DANGER)
    fan_diffs    = [abs(rows[i]["fan_pct"] - rows[i-1]["fan_pct"])
                    for i in range(1, len(rows))]
    fan_var      = float(np.mean(fan_diffs)) if fan_diffs else 0.0
    return {
        "energy_wh":    round(energy_wh, 3),
        "peak_T":       round(peak_T, 3),
        "breach_steps": breach_steps,
        "fan_var":      round(fan_var, 5),
    }
