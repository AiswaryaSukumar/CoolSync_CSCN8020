"""
CoolSync Final — Model Training Script
=======================================
Trains two models for the CoolSync+ inference system:

  1. LSTMPredictor  — predicts rack temperature θ=3 min ahead
                      gives DQN "look-ahead" that PID never has
  2. DQNAgent       — Deep Q-Network cooling controller
                      operates at T~23°C with pre-cooling capability

Run once before opening CoolSync_Final.ipynb:
    python train_models.py

Output: models/lstm.pt  and  models/dqn.pt
"""

import os, pickle, time
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)
torch.manual_seed(42)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS  (θ = 3 min dead time, burst 13–36 kW; per-user heat scales)
# ═══════════════════════════════════════════════════════════════════════════════
RHO_AIR    = 1.225          # kg/m3
CP_AIR     = 1006.0         # J/(kg·K)
C_RACK     = 200_000.0      # J/K — reduced for faster thermal dynamics (demo space)
T_SUPPLY   = 18.0           # °C    CRAC cold-aisle supply
T_SETPOINT = 22.0           # °C    PID target
T_RL       = 23.0           # °C    DQN operating point (vs PID 22°C: 1°C higher, less energy)
T_DANGER   = 25.0           # °C    warning threshold (PID should cross this during bursts)
T_CRITICAL = 27.0           # °C    ASHRAE A1 hard limit
T_AMB      = 24.0           # °C    datacenter ambient
DEAD_STEPS = 3              # steps  θ=3 min — realistic duct lag; PID cannot pre-empt 3-min bursts
DT_SIM     = 60.0           # s per step
MAX_FLOW   = 4.0            # m3/s
FAN_MAX_W  = 5_000.0        # W
FAN_MIN    = 0.20
Q_BASE_W   = 12_000.0       # W baseline rack heat (8-server × 4-GPU inference rack)
Q_RACK_MAX = 36_000.0       # W physical TDP cap (3× baseline; 8 servers at peak draw)
H_AMB      = 30.0           # W/°C — ambient heat loss coeff (thermal_simulation.csv)

# Phase B: joint action space (fan × T_supply) — 10 combined HVAC actions
# Basis: cold_source_control_dataset.csv → AHU_Usage 31–70%, Inlet_Temperature 15–28°C
# Fan equilibrium at 12kW baseline ≈ 0.608 → 0.60 level added to prevent chattering
# T_sup=22°C removed: provides 0 cooling at T≤22°C → causes idle oscillation
FAN_LEVELS    = np.array([0.30, 0.45, 0.60, 0.75, 1.00])
TSUP_LEVELS   = np.array([16.0, 19.0])
JOINT_ACTIONS = [(f, ts) for f in FAN_LEVELS for ts in TSUP_LEVELS]
N_ACTIONS     = len(JOINT_ACTIONS)   # 10

# Multi-user rack utilisation scaling
# A fully-loaded rack (32 GPU) requires ~16 concurrent medium requests
USERS_PER_FULL_RACK = 16

# Chiller COP model: Carnot × η=0.50 (scroll chiller, cold_source_control_dataset.csv)
ETA_CHILLER = 0.50

# LSTM hyper-params
LOOKBACK   = DEAD_STEPS      # history window (= θ steps)
LSTM_DIM   = 3               # features: [T_norm, Q_norm, fan]
LSTM_HIDDEN = 48
OBS_DIM    = 13              # 11 (Phase A) + 2 (tsup_norm, cop_norm)

# DQN hyper-params
DQN_HIDDEN  = 64
LR_DQN      = 3e-4
GAMMA       = 0.99
BATCH_SIZE  = 256
BUF_MAX     = 40_000
N_EP_DQN    = 6_000
EP_STEPS    = 80
EPS_START   = 1.0
EPS_END     = 0.05
EPS_DECAY   = 8_000          # steps over which eps decays linearly

# LSTM training
N_EP_LSTM   = 3_000          # simulation episodes to collect LSTM data
LR_LSTM     = 1e-3
LSTM_EPOCHS = 30
LSTM_BATCH  = 512

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def q_cool(fan_pct, T, T_sup=T_SUPPLY):
    return RHO_AIR * MAX_FLOW * fan_pct * CP_AIR * max(T - T_sup, 0.0)

def p_fan(fan_pct):
    return FAN_MAX_W * fan_pct ** 3

def p_chiller(Q_cool_W, T_supply_C, T_amb_C=T_AMB):
    """COP-based chiller power. Carnot × η=0.50 (scroll chiller).
    Validated: T_supply=18°C, Q_cool=15kW → P_chiller≈0.62kW ✓"""
    T_sup_K = T_supply_C + 273.15
    T_amb_K = T_amb_C    + 273.15
    COP = T_sup_K / max(T_amb_K - T_sup_K, 0.5) * ETA_CHILLER
    return Q_cool_W / max(COP, 1.0)

def rack_step(T, Q_delayed, fan_pct, T_sup=T_SUPPLY, dt=DT_SIM):
    Qc  = q_cool(fan_pct, T, T_sup)
    Ql  = H_AMB * (T - T_AMB)
    dT  = (Q_delayed - Qc - Ql) / C_RACK * dt
    return float(np.clip(T + dT, 10.0, 55.0)), Qc, p_fan(fan_pct)

def reward_fn(T, fan_pct, T_pred_norm=None, Q_cool_W=0.0, T_sup=T_SUPPLY,
             fan_prev=0.65, tsup_prev=T_SUPPLY):
    # Fan energy penalty (cube law)
    r = -3.0 * fan_pct ** 3
    if   T > T_CRITICAL:   r -= 500.0
    elif T > 26.0:          r -= 80.0 * (T - 26.0) ** 2
    elif T > T_DANGER:      r -= 12.0 * (T - T_DANGER) ** 2
    # Gaussian temperature reward peaked at T_RL=23°C (sigma=1.2°C)
    # Flat +2.0 caused chattering: T=21↔24 gave same reward as stable T=23
    # Gaussian gap: stable=2.34 vs chatter_avg=0.47 → 1.87 pts advantage
    if T < T_DANGER:
        r += 3.0 * float(np.exp(-0.5 * ((T - T_RL) / 1.2) ** 2))
    if T < 20.0: r -= 2.0 * (20.0 - T)
    if T_pred_norm is not None:
        T_pred_C = T_pred_norm * 5.0 + 22.0
        if T_pred_C > T_DANGER - 0.5 and T < T_DANGER:
            r += 5.0
    # Smoothness penalty — only in safe zone (T < T_DANGER=25°C)
    # Safe zone: penalise unnecessary switching (2.0 × Δfan)
    #   e.g. chattering 0.60↔0.75: Δ=0.15 → -0.30/step → suppressed
    # Danger zone (T >= T_DANGER): no penalty → DQN responds freely
    # Safety: largest valid Δfan=0.70, penalty=-1.40 << ASHRAE -500 → bursts unaffected
    if T < T_DANGER:
        r -= 2.0 * abs(fan_pct - fan_prev)
        r -= 2.0 * abs(T_sup - tsup_prev) / 6.0  # was 0.3 — T_sup switching causes 7× cooling swings
    return r

# ═══════════════════════════════════════════════════════════════════════════════
# LSTM PREDICTOR  (predicts T at t+θ from the last LOOKBACK steps)
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMPredictor(nn.Module):
    """LSTM that predicts T(t + DEAD_STEPS) from the last LOOKBACK observations."""
    def __init__(self, input_size=LSTM_DIM, hidden=LSTM_HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):          # x: (B, LOOKBACK, input_size)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # predict from last hidden state


def generate_lstm_data(n_episodes=N_EP_LSTM, rng_seed=42):
    """
    Roll out random FOPDT episodes to build (X_seq, y_T_future) pairs.
    X_seq : (N, LOOKBACK, 3)  — normalized [T, Q, fan] sequences
    y_T   : (N,)              — normalized T after DEAD_STEPS steps
    """
    rng   = np.random.default_rng(rng_seed)
    Xs, ys = [], []
    hist_len = LOOKBACK + DEAD_STEPS + 1  # need both history and future

    for _ in range(n_episodes):
        T    = rng.uniform(20.0, 27.0)
        buf  = deque([Q_BASE_W * rng.uniform(0.8, 1.5)] * DEAD_STEPS, maxlen=DEAD_STEPS)
        fan  = rng.uniform(0.2, 0.8)
        T_hist, Q_hist, fan_hist = [], [], []

        n_steps = hist_len + rng.integers(20, 60)
        burst_start = rng.integers(5, n_steps - 20)
        burst_amp   = rng.uniform(1.1, 3.0)   # covers 13.2–36 kW (1-user to 6-user Short)
        burst_dur   = rng.integers(3, 26)

        for s in range(n_steps):
            if burst_start <= s < burst_start + burst_dur:
                Q_now = Q_BASE_W * burst_amp
            else:
                Q_now = Q_BASE_W * rng.uniform(0.8, 1.2)

            T_hist.append(T)
            Q_hist.append(Q_now)
            fan_hist.append(fan)

            fan = float(np.clip(fan + rng.uniform(-0.1, 0.1), FAN_MIN, 1.0))
            T, _, _ = rack_step(T, buf[0], fan)  # use delayed heat first
            buf.append(Q_now)                     # then enqueue current heat

        # build sequences: every valid window → future T
        T_arr   = np.array(T_hist)
        Q_arr   = np.array(Q_hist)
        fan_arr = np.array(fan_hist)

        for i in range(LOOKBACK, len(T_arr) - DEAD_STEPS):
            seq_T   = (T_arr[i - LOOKBACK: i] - 22.0) / 5.0
            seq_Q   = (Q_arr[i - LOOKBACK: i] - Q_BASE_W) / Q_BASE_W
            seq_fan = fan_arr[i - LOOKBACK: i]
            X = np.stack([seq_T, seq_Q, seq_fan], axis=1)  # (LOOKBACK, 3)
            y = (T_arr[i + DEAD_STEPS - 1] - 22.0) / 5.0  # normalized future T
            Xs.append(X)
            ys.append(y)

    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm():
    print('\n' + '='*60)
    print(f'Training LSTM Temperature Predictor  (θ={DEAD_STEPS} min look-ahead)')
    print('='*60)
    t0 = time.time()

    X, y = generate_lstm_data()
    print(f'  Collected {len(X):,} (sequence, target) pairs from {N_EP_LSTM} episodes')

    # 80/20 split
    n = len(X)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    X_tr, y_tr = X[idx[:split]], y[idx[:split]]
    X_te, y_te = X[idx[split:]], y[idx[split:]]

    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr).unsqueeze(1)

    model = LSTMPredictor()
    opt   = optim.Adam(model.parameters(), lr=LR_LSTM)
    loss_fn = nn.MSELoss()

    for epoch in range(LSTM_EPOCHS):
        model.train()
        perm   = torch.randperm(len(X_tr_t))
        losses = []
        for i in range(0, len(X_tr_t), LSTM_BATCH):
            batch_idx = perm[i: i + LSTM_BATCH]
            pred = model(X_tr_t[batch_idx])
            loss = loss_fn(pred, y_tr_t[batch_idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.from_numpy(X_te))
                val_mse  = loss_fn(val_pred, torch.from_numpy(y_te).unsqueeze(1)).item()
            # convert normalized MSE back to °C
            val_mae_C = float(torch.abs(val_pred.squeeze() -
                              torch.from_numpy(y_te)).mean()) * 5.0
            print(f'  Epoch {epoch+1:3d}/{LSTM_EPOCHS}  '
                  f'train_loss={np.mean(losses):.4f}  '
                  f'val_MAE={val_mae_C:.3f}°C')

    path = os.path.join(MODELS_DIR, 'lstm.pt')
    torch.save({'state_dict': model.state_dict(),
                'lookback': LOOKBACK, 'hidden': LSTM_HIDDEN}, path)
    print(f'\n  Saved → {path}   ({time.time()-t0:.0f}s)')
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# DQN AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class DQNNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=DQN_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x): return self.net(x)


class ReplayBuffer:
    def __init__(self, maxlen=BUF_MAX):
        self.buf = []
        self.ptr = 0
        self.maxlen = maxlen

    def push(self, obs, act, rew, nobs, done):
        item = (obs, act, rew, nobs, done)
        if len(self.buf) < self.maxlen:
            self.buf.append(item)
        else:
            self.buf[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.maxlen

    def sample(self, n):
        idx = np.random.choice(len(self.buf), n, replace=False)
        batch = [self.buf[i] for i in idx]
        obs  = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        act  = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rew  = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        nobs = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        done = torch.tensor([float(b[4]) for b in batch], dtype=torch.float32)
        return obs, act, rew, nobs, done

    def __len__(self): return len(self.buf)


def make_obs(T, Q_now, fan_prev, q_buf, step, total_steps, T_pred_norm, tsup_prev=T_SUPPLY):
    buf     = list(q_buf)
    t_err   = (T - T_RL) / 5.0
    q_now   = (Q_now - Q_BASE_W) / Q_BASE_W
    f_prev  = (fan_prev - 0.5) / 0.5
    q_mean  = (np.mean(buf) - Q_BASE_W) / Q_BASE_W
    q_trend = (Q_now - buf[-1]) / Q_BASE_W
    above   = max(0.0, T - T_RL) / 5.0
    below   = max(0.0, T_RL - T) / 5.0
    phase   = step / max(total_steps - 1, 1)
    q_lag   = (buf[0] - Q_BASE_W) / Q_BASE_W
    T_pred_delta = (T_pred_norm * 5.0) - (T - 22.0)
    # Phase B: supply temp + COP
    tsup_norm = (tsup_prev - 19.0) / 3.0
    T_sup_K   = tsup_prev + 273.15
    T_amb_K   = T_AMB + 273.15
    cop       = T_sup_K / max(T_amb_K - T_sup_K, 0.5) * ETA_CHILLER
    cop_norm  = (cop - 3.0) / 2.0
    return np.array([t_err, q_now, f_prev, q_mean, q_trend,
                     above, below, phase, q_lag,
                     T_pred_norm, T_pred_delta / 5.0,
                     tsup_norm, cop_norm], dtype=np.float32)


def lstm_predict(lstm_model, T_hist, Q_hist, fan_hist):
    """Run LSTM inference; return normalized predicted T."""
    if len(T_hist) < LOOKBACK:
        return (T_hist[-1] - 22.0) / 5.0  # fallback: current T
    seq_T   = np.array([(t - 22.0) / 5.0 for t in T_hist[-LOOKBACK:]])
    seq_Q   = np.array([(q - Q_BASE_W) / Q_BASE_W for q in Q_hist[-LOOKBACK:]])
    seq_fan = np.array(fan_hist[-LOOKBACK:])
    x = torch.tensor(np.stack([seq_T, seq_Q, seq_fan], axis=1),
                     dtype=torch.float32).unsqueeze(0)  # (1, LOOKBACK, 3)
    with torch.no_grad():
        return float(lstm_model(x).item())


def train_dqn(lstm_model):
    print('\n' + '='*60)
    print(f'Training DQN Controller  ({N_EP_DQN} episodes x {EP_STEPS} steps, θ={DEAD_STEPS}min)')
    print('='*60)
    t0 = time.time()

    online  = DQNNet()
    target  = DQNNet()
    target.load_state_dict(online.state_dict())
    opt     = optim.Adam(online.parameters(), lr=LR_DQN)
    buf     = ReplayBuffer()
    loss_fn = nn.MSELoss()

    eps         = EPS_START
    total_steps = 0
    best_reward = -float('inf')
    path_best   = os.path.join(MODELS_DIR, 'dqn_best.pt')
    ep_rewards  = []
    lstm_model.eval()

    rng = np.random.default_rng(0)

    # Backup Phase A model before overwriting
    path_phaseA = os.path.join(MODELS_DIR, 'dqn_phaseA.pt')
    path_phaseB = os.path.join(MODELS_DIR, 'dqn.pt')
    if os.path.exists(path_phaseB) and not os.path.exists(path_phaseA):
        import shutil; shutil.copy2(path_phaseB, path_phaseA)
        print(f'  Phase A model backed up → {path_phaseA}')

    for ep in range(N_EP_DQN):
        T    = T_RL + rng.normal(0, 1.5)
        fan  = 0.65 + rng.uniform(-0.1, 0.1)
        T_sup = float(rng.choice(TSUP_LEVELS))
        Q    = Q_BASE_W * rng.uniform(0.8, 1.2)
        q_buf = deque([Q] * DEAD_STEPS, maxlen=DEAD_STEPS)
        T_hist, Q_hist, fan_hist = [T], [Q], [fan]

        burst_start = rng.integers(5, 40)
        burst_amp   = rng.uniform(1.1, 3.0)   # covers 13.2–36 kW full range
        burst_dur   = rng.integers(3, 26)
        ep_r = 0.0

        for st in range(EP_STEPS):
            if burst_start <= st < burst_start + burst_dur:
                Q = Q_BASE_W * burst_amp
            elif st == burst_start + burst_dur:
                Q = Q_BASE_W * rng.uniform(0.9, 1.1)
            else:
                Q = float(np.clip(Q + rng.normal(0, 300), Q_BASE_W*0.7, Q_RACK_MAX))

            T_pred = lstm_predict(lstm_model, T_hist, Q_hist, fan_hist)
            obs    = make_obs(T, Q, fan, q_buf, st, EP_STEPS, T_pred, tsup_prev=T_sup)

            # ε-greedy action
            if rng.random() < eps:
                a = rng.integers(N_ACTIONS)
            else:
                with torch.no_grad():
                    a = int(online(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax())
            fan_prev_step = fan
            tsup_prev_step = T_sup
            fan, T_sup = JOINT_ACTIONS[a]
            fan, T_sup = float(fan), float(T_sup)

            T_new, Qc, _ = rack_step(T, q_buf[0], fan, T_sup=T_sup)  # use delayed heat first
            q_buf.append(Q)                                            # then enqueue current heat
            r    = reward_fn(T_new, fan, T_pred_norm=T_pred, Q_cool_W=Qc, T_sup=T_sup,
                             fan_prev=fan_prev_step, tsup_prev=tsup_prev_step)
            done = (st == EP_STEPS - 1)

            T_hist.append(T_new); Q_hist.append(Q); fan_hist.append(fan)
            T_pred_next = lstm_predict(lstm_model, T_hist, Q_hist, fan_hist)
            nobs = make_obs(T_new, Q, fan, q_buf, st+1, EP_STEPS, T_pred_next, tsup_prev=T_sup)

            buf.push(obs, a, r, nobs, done)
            ep_r += r
            T     = T_new
            total_steps += 1

            # linear epsilon decay
            eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * total_steps / EPS_DECAY)

            if len(buf) >= BATCH_SIZE:
                obs_b, act_b, rew_b, nobs_b, done_b = buf.sample(BATCH_SIZE)
                with torch.no_grad():
                    q_next = target(nobs_b).max(dim=1).values
                    q_tgt  = rew_b + (1.0 - done_b) * GAMMA * q_next
                q_pred = online(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
                loss   = loss_fn(q_pred, q_tgt)
                opt.zero_grad(); loss.backward(); opt.step()

            # sync target network every 500 steps
            if total_steps % 500 == 0:
                target.load_state_dict(online.state_dict())

        ep_rewards.append(ep_r)
        if (ep + 1) % 500 == 0:
            mr  = np.mean(ep_rewards[-200:])
            marker = ''
            if mr > best_reward:
                best_reward = mr
                ckpt = {'online_state_dict': online.state_dict(),
                        'target_state_dict': target.state_dict(),
                        'obs_dim': OBS_DIM, 'n_actions': N_ACTIONS,
                        'hidden': DQN_HIDDEN,
                        'best_reward': best_reward,
                        'best_ep': ep + 1,
                        'joint_actions': [(float(f), float(ts)) for f, ts in JOINT_ACTIONS]}
                torch.save(ckpt, path_best)
                marker = '  ** best **'
            print(f'  ep {ep+1:5d}/{N_EP_DQN}  eps={eps:.3f}  '
                  f'mean_reward(200)={mr:+.1f}  buf={len(buf)}{marker}')

    # Save final model
    torch.save({'online_state_dict': online.state_dict(),
                'target_state_dict': target.state_dict(),
                'obs_dim': OBS_DIM, 'n_actions': N_ACTIONS,
                'hidden': DQN_HIDDEN,
                'joint_actions': [(float(f), float(ts)) for f, ts in JOINT_ACTIONS]},
               path_phaseB)
    # Overwrite dqn.pt with best checkpoint so notebook always loads best policy
    import shutil
    shutil.copy2(path_best, path_phaseB)
    print(f'\n  Saved → {path_phaseB} (best ep, reward={best_reward:+.1f})   ({time.time()-t0:.0f}s)')
    return online


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('CoolSync Final - Training Pipeline')
    print(f'  Dead time theta = {DEAD_STEPS} min')
    print(f'  Burst range : 1.10x-3.00x ({1.10*Q_BASE_W/1000:.0f}-{3.00*Q_BASE_W/1000:.0f} kW)  [per-user amp, Q_RACK_MAX={Q_RACK_MAX/1000:.1f} kW]')
    print(f'  PID max cooling (22C, fan=100%) = '
          f'{RHO_AIR*MAX_FLOW*CP_AIR*(T_SETPOINT-T_SUPPLY)/1000:.1f} kW')
    print(f'  DQN max cooling (25C, fan=100%) = '
          f'{RHO_AIR*MAX_FLOW*CP_AIR*(T_RL-T_SUPPLY)/1000:.1f} kW')
    t_total = time.time()

    lstm_path = os.path.join(MODELS_DIR, 'lstm.pt')
    if os.path.exists(lstm_path):
        print('\n[SKIP] lstm.pt already exists -- loading saved model.')
        lstm_model = LSTMPredictor()
        ck = torch.load(lstm_path, map_location='cpu')
        lstm_model.load_state_dict(ck['state_dict'])
        lstm_model.eval()
    else:
        lstm_model = train_lstm()
    dqn_model  = train_dqn(lstm_model)

    print('\n' + '='*60)
    print(f'All models saved to  {MODELS_DIR}')
    print(f'Total training time  {(time.time()-t_total)/60:.1f} min')
    print('Run CoolSync_Final.ipynb for inference.')
    print('='*60)
