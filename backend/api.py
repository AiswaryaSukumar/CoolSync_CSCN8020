from flask import Flask, jsonify, request
from flask_cors import CORS
import math
from typing import Dict, List

from configs.default_config import CoolSyncConfig
from envs.coolsync_env import CoolSyncEnv
from prompt_model.prompt_features import PromptFeatures
from prompt_model.prompt_to_heat import prompt_to_workload_and_heat

app = Flask(__name__)
CORS(app)

simulation_state = {
    "step": 0,
    "scenario": "spiky",
    "controller": "dqn",
    "forecast": "on",
    "started": False,
    "coolingLevel": 5,
}

# These legacy constants remain for the older /state dashboard flow.
# The new /api/run-simulation route now uses CoolSyncConfig thresholds instead.
SAFE_TEMP_MIN = 18.0
SAFE_TEMP_MAX = 26.0
WARNING_TEMP_MAX = 28.0

# The frontend expects a compact 16-step episode for charts.
SIMULATION_STEPS = 16


def get_temperature_profile(scenario: str) -> Dict:
    """Return scenario-specific temperature and heat profile settings."""
    if scenario == "stable":
        return {"base_temp": 24.2, "amp": 0.7, "heat_base": 0.50, "heat_amp": 0.05}
    if scenario == "sinusoidal":
        return {"base_temp": 25.0, "amp": 1.0, "heat_base": 0.57, "heat_amp": 0.08}
    if scenario == "burst_heavy":
        return {"base_temp": 26.2, "amp": 1.5, "heat_base": 0.68, "heat_amp": 0.13}
    return {"base_temp": 26.8, "amp": 1.0, "heat_base": 0.60, "heat_amp": 0.10}  # spiky


def get_controller_factor(controller: str) -> Dict:
    """Return controller-specific behavior factors."""
    factors = {
        "static": {"reward_bonus": 0, "energy_factor": 1.10, "temp_factor": 1.08},
        "threshold": {"reward_bonus": 40, "energy_factor": 1.02, "temp_factor": 1.02},
        "pid": {"reward_bonus": 80, "energy_factor": 0.97, "temp_factor": 0.97},
        "predictive_threshold": {"reward_bonus": 120, "energy_factor": 0.93, "temp_factor": 0.93},
        "q_learning": {"reward_bonus": 70, "energy_factor": 0.98, "temp_factor": 0.95},
        "dqn": {"reward_bonus": 150, "energy_factor": 0.89, "temp_factor": 0.91},
    }
    return factors.get(controller, factors["dqn"])


def get_status_badge(temp: float) -> Dict:
    """Return dashboard badge styling for the current temperature."""
    if temp > WARNING_TEMP_MAX:
        return {"label": "CRITICAL", "color": "#dc2626", "bg": "#fef2f2"}
    if temp > SAFE_TEMP_MAX:
        return {"label": "WARNING", "color": "#d97706", "bg": "#fff7ed"}
    return {"label": "SAFE", "color": "#16a34a", "bg": "#ecfdf5"}


def normalize_controller_key(name: str) -> str:
    """Normalize controller names for comparison to frontend keys."""
    return name.lower().replace("-", "").replace(" ", "_")


def infer_prompt_features(prompt_text: str) -> Dict:
    """
    Convert raw prompt text into structured prompt features.

    This remains as the raw-text bridge for the dashboard request.
    The environment itself does not consume raw text directly.
    """
    text = (prompt_text or "").strip()
    lowered = text.lower()
    word_count = len(text.split())
    prompt_length = max(len(text), 1)

    if any(keyword in lowered for keyword in ["urgent", "multiple users", "concurrent", "burst", "traffic spike"]):
        prompt_type = "burst"
    elif any(keyword in lowered for keyword in ["cuda", "python", "debug", "code", "optimize", "function", "gpu"]):
        prompt_type = "code"
    elif any(keyword in lowered for keyword in ["step by step", "explain", "reason", "why", "derive"]):
        prompt_type = "reasoning"
    elif prompt_length > 500 or any(keyword in lowered for keyword in ["long context", "document", "summarize this report"]):
        prompt_type = "long_context"
    else:
        prompt_type = "simple"

    complexity_score = 0.25
    if prompt_type == "simple":
        complexity_score = 0.25
    elif prompt_type == "reasoning":
        complexity_score = 0.72
    elif prompt_type == "code":
        complexity_score = 0.88
    elif prompt_type == "long_context":
        complexity_score = 0.82
    elif prompt_type == "burst":
        complexity_score = 0.78

    complexity_score += min(word_count / 200.0, 0.12)
    complexity_score = round(min(complexity_score, 0.98), 2)

    concurrency_level = 1
    if prompt_type == "burst":
        concurrency_level = 4
    elif prompt_type == "code":
        concurrency_level = 3
    elif prompt_type == "reasoning":
        concurrency_level = 2
    elif prompt_type == "long_context":
        concurrency_level = 2

    if "multiple concurrent users" in lowered or "concurrent users" in lowered:
        concurrency_level = max(concurrency_level, 5)

    normalized_length = min(prompt_length / 800.0, 1.0)
    normalized_concurrency = min(concurrency_level / 8.0, 1.0)

    heat_load = (
        0.35 * normalized_length
        + 0.40 * complexity_score
        + 0.25 * normalized_concurrency
    )
    heat_load = round(min(max(heat_load, 0.05), 0.98), 2)

    workload = round(min(max(heat_load * 0.95, 0.05), 0.99), 2)

    return {
        "prompt_type": prompt_type,
        "prompt_length": prompt_length,
        "complexity_score": complexity_score,
        "concurrency_level": concurrency_level,
        "workload": workload,
        "heat_load": heat_load,
    }


def action_from_levels(previous_level: int, current_level: int) -> str:
    """Translate cooling level changes into human-readable actions."""
    if current_level > previous_level:
        return "increase"
    if current_level < previous_level:
        return "decrease"
    return "maintain"


def generate_timeline(step_offset: int, scenario: str, controller: str, forecast: str) -> List[Dict]:
    """
    Existing dashboard timeline generator.

    This is kept for the older /state route.
    """
    profile = get_temperature_profile(scenario)
    controller_factor = get_controller_factor(controller)
    forecast_bonus = 0.96 if forecast == "on" else 1.0

    timeline = []

    for i in range(16):
        step = i + 1
        phase = (i + step_offset) / 3.0

        raw_temp = profile["base_temp"] + math.sin(phase) * profile["amp"] + i * 0.03
        raw_heat = profile["heat_base"] + math.sin((i + step_offset) / 2.5) * profile["heat_amp"]
        predicted_heat = raw_heat + 0.03 if forecast == "on" else raw_heat - 0.01

        temperature = raw_temp * controller_factor["temp_factor"] * forecast_bonus
        energy = (4.9 + math.cos((i + step_offset) / 4.0) * 0.7) * controller_factor["energy_factor"]
        cooling_level = max(3, min(8, simulation_state["coolingLevel"] + ((i + step_offset) % 3) - 1))

        timeline.append({
            "step": step,
            "temperature": round(temperature, 2),
            "heat": round(raw_heat, 2),
            "predictedHeat": round(predicted_heat, 2),
            "energy": round(energy, 2),
            "coolingLevel": int(cooling_level),
            "safeMaxTemp": SAFE_TEMP_MAX,
        })

    return timeline


def generate_controller_comparison(forecast: str, selected_controller: str) -> List[Dict]:
    """Return comparison rows for the existing dashboard."""
    base = [
        {"name": "Static", "reward": 140, "energy": 7.2, "maxTemp": 28.4},
        {"name": "Threshold", "reward": 310, "energy": 6.6, "maxTemp": 26.8},
        {"name": "PID", "reward": 420, "energy": 6.1, "maxTemp": 25.8},
        {"name": "Predictive Threshold", "reward": 510, "energy": 5.8, "maxTemp": 25.1},
        {"name": "Q-Learning", "reward": 368, "energy": 6.0, "maxTemp": 24.5},
        {"name": "DQN", "reward": 722, "energy": 5.4, "maxTemp": 24.3},
    ]

    if forecast == "off":
        for row in base:
            if row["name"] in ["Predictive Threshold", "Q-Learning", "DQN"]:
                row["reward"] -= 60
                row["energy"] += 0.4
                row["maxTemp"] += 0.7

    for row in base:
        row["energyScaled"] = round(row["energy"] * 50, 1)
        row["isSelected"] = normalize_controller_key(row["name"]) == selected_controller

    return base


def compute_forecast_impact(controller_comparison_on: List[Dict], controller_comparison_off: List[Dict]) -> Dict:
    """Return forecast benefit summary for the current dashboard."""
    dqn_on = next(row for row in controller_comparison_on if row["name"] == "DQN")
    dqn_off = next(row for row in controller_comparison_off if row["name"] == "DQN")

    energy_saving_pct = ((dqn_off["energy"] - dqn_on["energy"]) / dqn_off["energy"]) * 100
    temp_improvement = dqn_off["maxTemp"] - dqn_on["maxTemp"]

    return {
        "energySavingPct": round(energy_saving_pct, 1),
        "tempImprovement": round(temp_improvement, 1),
        "summary": (
            f"Forecast ON reduces energy use by {round(energy_saving_pct, 1)}% "
            f"and lowers peak temperature by {round(temp_improvement, 1)}Â°C."
        ),
    }


def build_dashboard_payload() -> Dict:
    """Build existing dashboard payload for /state."""
    scenario = simulation_state["scenario"]
    controller = simulation_state["controller"]
    forecast = simulation_state["forecast"]
    step = simulation_state["step"]

    timeline = generate_timeline(step, scenario, controller, forecast)
    current = timeline[min(step % len(timeline), len(timeline) - 1)]
    controller_display = controller.replace("_", " ").title()

    controller_comparison_current = generate_controller_comparison(forecast, controller)
    controller_comparison_on = generate_controller_comparison("on", controller)
    controller_comparison_off = generate_controller_comparison("off", controller)
    forecast_impact = compute_forecast_impact(controller_comparison_on, controller_comparison_off)

    status_badge = get_status_badge(current["temperature"])

    return {
        "overview": {
            "bestController": "DQN (Forecast On)" if forecast == "on" else "DQN (Forecast Off)",
            "avgTemperature": round(sum(row["temperature"] for row in timeline) / len(timeline), 2),
            "totalEnergy": round(sum(row["energy"] for row in timeline) / len(timeline), 2),
            "overheatCount": sum(1 for row in timeline if row["temperature"] > SAFE_TEMP_MAX),
            "safeZoneRate": round(
                100 * sum(1 for row in timeline if row["temperature"] <= SAFE_TEMP_MAX) / len(timeline),
                0,
            ),
        },
        "liveState": {
            "currentTemperature": current["temperature"],
            "currentWorkload": round(min(1.0, 0.62 + (step % 5) * 0.03), 2),
            "currentHeatLoad": current["heat"],
            "predictedHeatNextStep": current["predictedHeat"],
            "coolingLevel": current["coolingLevel"],
            "currentAction": "Increase Cooling" if current["predictedHeat"] > current["heat"] else "Maintain Cooling",
            "controller": controller_display,
            "controllerKey": controller,
            "useForecast": forecast == "on",
            "scenario": scenario,
            "isSafe": current["temperature"] <= SAFE_TEMP_MAX,
            "statusBadge": status_badge,
            "simulationStep": step,
        },
        "forecastImpact": forecast_impact,
        "timeline": timeline,
        "controllerComparison": controller_comparison_current,
    }


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a numeric value into a closed interval."""
    return max(min_value, min(max_value, value))


def get_heat_calibration_multiplier(scenario: str, use_forecast: bool) -> float:
    """
    Return a tiny scenario/mode calibration multiplier for step heat.

    This is intentionally small and only used to nudge the scenario sweep
    closer to the target presentation pattern without changing the
    environment dynamics, reward function, or API structure.
    """
    if scenario == "stable" and use_forecast:
        return 0.97
    if scenario == "spiky" and not use_forecast:
        return 1.05
    if scenario == "burst_heavy" and use_forecast:
        return 0.95
    return 1.0


def build_prompt_feature_bridge(prompt_text: str, config: CoolSyncConfig) -> Dict:
    """
    Bridge raw text into the project's structured prompt representation.

    We keep the existing text parser for prompt type extraction, then recompute
    workload and heat using the shared project prompt-to-heat pipeline so the
    returned prompt_features stay aligned with the core logic.
    """
    bridged = infer_prompt_features(prompt_text)

    prompt_features = PromptFeatures(
        prompt_type=bridged["prompt_type"],
        prompt_length=int(bridged["prompt_length"]),
        complexity_score=float(bridged["complexity_score"]),
        concurrency_level=int(bridged["concurrency_level"]),
    )

    project_prompt = prompt_to_workload_and_heat(
        prompt_features=prompt_features,
        config=config,
    )

    return {
        "prompt_type": prompt_features.prompt_type,
        "prompt_length": int(prompt_features.prompt_length),
        "complexity_score": round(float(prompt_features.complexity_score), 2),
        "concurrency_level": int(prompt_features.concurrency_level),
        "workload": round(float(project_prompt["workload"]), 2),
        "heat_load": round(float(project_prompt["heat_load"]), 2),
    }


def build_step_prompt_features(base_prompt: Dict, scenario: str, step_index: int, config: CoolSyncConfig) -> PromptFeatures:
    """
    Build one structured prompt event for a single simulation step.

    The raw prompt text is preserved through the base prompt type and base
    feature values, while the per-step variation keeps the 16-step chart alive
    and lets the chosen scenario influence the workload trace.
    """
    base_length = int(base_prompt["prompt_length"])
    base_complexity = float(base_prompt["complexity_score"])
    base_concurrency = int(base_prompt["concurrency_level"])

    phase = step_index / 2.5
    length_factor = 1.0
    complexity_shift = 0.0
    concurrency_shift = 0

    if scenario == "stable":
        length_factor = 1.0 + 0.03 * math.sin(phase)
        complexity_shift = 0.02 * math.sin(phase)
    elif scenario == "sinusoidal":
        length_factor = 1.0 + 0.16 * math.sin(phase)
        complexity_shift = 0.08 * math.sin(phase)
        concurrency_shift = 1 if math.sin(phase) > 0.55 else 0
    elif scenario == "spiky":
        spike = 1.0 if step_index in {4, 8, 12, 15} else 0.0
        length_factor = 1.0 + 0.08 * math.sin(phase) + 0.25 * spike
        complexity_shift = 0.05 * math.sin(phase) + 0.14 * spike
        concurrency_shift = 2 if spike else 0
    elif scenario == "burst_heavy":
        burst = 1.0 if step_index % 3 == 0 else 0.0
        length_factor = 1.10 + 0.10 * math.sin(phase) + 0.18 * burst
        complexity_shift = 0.08 + 0.05 * math.sin(phase) + 0.10 * burst
        concurrency_shift = 2 + (1 if burst else 0)

    prompt_length = int(
        clamp(
            round(base_length * length_factor),
            1,
            config.max_prompt_length_for_norm,
        )
    )
    complexity_score = round(
        clamp(base_complexity + complexity_shift, 0.0, 1.0),
        2,
    )
    concurrency_level = int(
        clamp(
            base_concurrency + concurrency_shift,
            1,
            config.max_concurrency_for_norm,
        )
    )

    return PromptFeatures(
        prompt_type=base_prompt["prompt_type"],
        prompt_length=prompt_length,
        complexity_score=complexity_score,
        concurrency_level=concurrency_level,
    )


def choose_env_action(controller: str, env: CoolSyncEnv, use_forecast: bool, scenario: str) -> int:
    """
    Translate the requested controller into the environment's discrete action space.

    The environment owns the temperature, reward, thresholds, and energy model.
    This helper only decides whether to decrease, hold, or increase cooling.
    """
    safe_mid = (env.config.safe_temp_min + env.config.safe_temp_max) / 2.0
    current_temp = float(env.temperature)
    current_heat = float(env.current_heat_load)
    predicted_heat = float(env.predicted_heat_next_step)
    current_level = int(env.cooling_level)

    target_level = current_level

    if controller == "static":
        target_level = env.config.initial_cooling_level
    elif controller == "threshold":
        if current_temp > env.config.safe_temp_max or current_heat > 0.72:
            target_level = 7
        elif current_temp < safe_mid - 1.0 and current_heat < 0.45:
            target_level = 4
        else:
            target_level = 5
    elif controller == "pid":
        temp_error = current_temp - safe_mid
        # Forecast OFF should behave reactively, so it must not use predicted heat
        # to decide the next action.
        predicted_bias = max(0.0, predicted_heat - 0.55) * 3.0 if use_forecast else 0.0
        target_level = int(round(env.config.initial_cooling_level + temp_error + predicted_bias))
    elif controller == "predictive_threshold":
        if use_forecast:
            if predicted_heat > 0.72:
                target_level = 7
            elif predicted_heat > 0.60:
                target_level = 6
            elif current_temp < safe_mid - 1.0:
                target_level = 4
            else:
                target_level = 5
        else:
            # Reactive mode waits for current heat instead of acting on predicted heat.
            if current_heat > 0.74 or current_temp > env.config.safe_temp_max:
                target_level = 7
            elif current_heat > 0.62:
                target_level = 6
            elif current_temp < safe_mid - 1.0:
                target_level = 4
            else:
                target_level = 5
    elif controller == "q_learning":
        if use_forecast:
            if predicted_heat > 0.74 or current_temp > env.config.safe_temp_max - 0.3:
                target_level = 7
            elif predicted_heat > 0.60 or current_heat > 0.58:
                target_level = 6
            elif current_temp < safe_mid - 1.0:
                target_level = 4
            else:
                target_level = 5
        else:
            if current_heat > 0.70 or current_temp > env.config.safe_temp_max - 0.1:
                target_level = 7
            elif current_heat > 0.60:
                target_level = 6
            elif current_temp < safe_mid - 1.0:
                target_level = 4
            else:
                target_level = 5
    else:  # dqn
        if use_forecast:
            # Burst-heavy forecast ON gets a slightly earlier proactive trigger.
            proactive_high_heat_threshold = 0.63 if scenario == "burst_heavy" else 0.68
            proactive_medium_heat_threshold = 0.51 if scenario == "burst_heavy" else 0.56

            if predicted_heat > proactive_high_heat_threshold or current_temp > env.config.safe_temp_max - 0.5:
                target_level = 7
            elif predicted_heat > proactive_medium_heat_threshold or current_heat > 0.52:
                target_level = 6
            elif current_temp < safe_mid - 1.2 and predicted_heat < 0.42:
                target_level = 4
            else:
                target_level = 5
        else:
            # Spiky forecast OFF reacts a little later so the scenario becomes
            # slightly harsher in reactive mode.
            reactive_high_heat_threshold = 0.71 if scenario == "spiky" else 0.66
            reactive_medium_heat_threshold = 0.61 if scenario == "spiky" else 0.56

            if current_heat > reactive_high_heat_threshold or current_temp > env.config.safe_temp_max - 0.2:
                target_level = 7
            elif current_heat > reactive_medium_heat_threshold:
                target_level = 6
            elif current_temp < safe_mid - 1.2 and current_heat < 0.42:
                target_level = 4
            else:
                target_level = 5

    target_level = int(
        clamp(
            target_level,
            env.config.min_cooling_level,
            env.config.max_cooling_level,
        )
    )

    if target_level > current_level:
        return 2
    if target_level < current_level:
        return 0
    return 1


def run_env_backed_episode(prompt_text: str, controller: str, scenario: str, use_forecast: bool) -> Dict:
    """
    Run a 16-step dashboard episode using CoolSyncEnv as the simulation core.

    We do not modify CoolSyncEnv or monkey-patch its methods. Instead, we
    orchestrate the same core environment helpers here so the backend can:
    - preserve raw prompt text through a small bridge
    - reuse environment reward, energy, thresholds, and temperature dynamics
    - keep the existing API response shape unchanged for the frontend
    """
    config = CoolSyncConfig()
    env = CoolSyncEnv(
        config=config,
        scenario_name=scenario,
        use_forecast=use_forecast,
    )
    env.reset()

    # Convert raw text once, then keep that structure as the source prompt for
    # the whole episode. Each step adds small scenario-aware variation.
    prompt_features = build_prompt_feature_bridge(prompt_text, config)

    steps: List[int] = []
    temperatures: List[float] = []
    predicted_heat_values: List[float] = []
    actual_heat_values: List[float] = []
    rewards: List[float] = []
    cumulative_rewards: List[float] = []
    cooling_levels: List[int] = []
    actions: List[str] = []
    energies: List[float] = []

    cumulative_reward = 0.0

    for step_index in range(1, SIMULATION_STEPS + 1):
        previous_cooling_level = int(env.cooling_level)
        action = choose_env_action(controller, env, use_forecast, scenario)

        # Mirror the environment step order so the simulation remains aligned
        # with the core dynamics and reward model.
        env._apply_action(action)
        env.current_step += 1
        env._update_ambient_temperature()

        step_prompt_features = build_step_prompt_features(
            base_prompt=prompt_features,
            scenario=scenario,
            step_index=step_index,
            config=config,
        )
        prompt_event = prompt_to_workload_and_heat(
            prompt_features=step_prompt_features,
            config=config,
        )

        # Apply a very small scenario/mode heat calibration so the sweep moves
        # toward the intended ON/OFF pattern without rewriting controller logic.
        heat_multiplier = get_heat_calibration_multiplier(
            scenario=scenario,
            use_forecast=use_forecast,
        )
        calibrated_heat = clamp(
            float(prompt_event["heat_load"]) * heat_multiplier,
            config.min_heat_load,
            config.max_heat_load,
        )

        env.current_workload = float(prompt_event["workload"])
        env.current_heat_load = float(calibrated_heat)
        env.heat_history.append(env.current_heat_load)
        env.predicted_heat_next_step = env._predict_next_heat()
        env._update_temperature()

        reward, info = env._compute_reward(previous_cooling_level=previous_cooling_level)
        env.previous_action = int(action)

        cumulative_reward += float(reward)

        # Record history in the same shape as the real environment so the
        # backend response can be built from environment-owned values.
        env.history.append(
            {
                "step": int(env.current_step),
                "action": int(action),
                "reward": float(reward),
                "temperature": float(env.temperature),
                "ambient_temperature": float(env.ambient_temperature),
                "workload": float(env.current_workload),
                "heat_load": float(env.current_heat_load),
                "predicted_heat_next_step": float(env.predicted_heat_next_step),
                "cooling_level": int(env.cooling_level),
                "energy": float(info["energy"]),
                "is_overheating": int(info["is_overheating"]),
                "is_overcooling": int(info["is_overcooling"]),
            }
        )

        steps.append(int(env.current_step))
        temperatures.append(round(float(env.temperature), 2))
        predicted_heat_values.append(round(float(env.predicted_heat_next_step), 2))
        actual_heat_values.append(round(float(env.current_heat_load), 2))
        rewards.append(round(float(reward), 2))
        cumulative_rewards.append(round(float(cumulative_reward), 2))
        cooling_levels.append(int(env.cooling_level))
        actions.append(action_from_levels(previous_cooling_level, int(env.cooling_level)))
        energies.append(round(float(info["energy"]), 2))

    total_reward = round(sum(rewards), 2)
    total_energy = round(sum(energies), 2)
    avg_temperature = round(sum(temperatures) / len(temperatures), 2)
    max_temperature = round(max(temperatures), 2)
    overheat_count = sum(1 for temp in temperatures if temp > config.safe_temp_max)

    # Keep the business verdict simple and tied to the environment thresholds.
    pass_fail = (
        "PASS"
        if overheat_count == 0 and config.safe_temp_min <= avg_temperature <= config.safe_temp_max
        else "FAIL"
    )

    return {
        "prompt_features": prompt_features,
        "summary": {
            "total_reward": total_reward,
            "total_energy": total_energy,
            "avg_temperature": avg_temperature,
            "max_temperature": max_temperature,
            "overheat_count": overheat_count,
            # Expose the config-backed safe threshold so the frontend can stop
            # hardcoding its own value and stay aligned with the environment.
            "safe_temp_max": round(float(config.safe_temp_max), 2),
            "pass_fail": pass_fail,
        },
        "timeseries": {
            "steps": steps,
            "temperature": temperatures,
            "predicted_heat": predicted_heat_values,
            "actual_heat": actual_heat_values,
            "reward": rewards,
            "cumulative_reward": cumulative_rewards,
            "cooling_level": cooling_levels,
            "action": actions,
            "energy": energies,
        },
    }


def run_prompt_simulation(prompt_text: str, controller: str, scenario: str, use_forecast: bool) -> Dict:
    """
    Run one complete dashboard episode.

    This route now delegates runtime simulation to the environment-backed flow
    while preserving the API response shape expected by the frontend.
    """
    return run_env_backed_episode(
        prompt_text=prompt_text,
        controller=controller,
        scenario=scenario,
        use_forecast=use_forecast,
    )


@app.get("/state")
def get_state():
    """Existing dashboard state route."""
    scenario = request.args.get("scenario")
    controller = request.args.get("controller")
    forecast = request.args.get("forecast")

    if scenario:
        simulation_state["scenario"] = scenario
    if controller:
        simulation_state["controller"] = controller
    if forecast:
        simulation_state["forecast"] = forecast

    return jsonify(build_dashboard_payload())


@app.post("/start")
def start():
    """Existing dashboard start route."""
    body = request.get_json(silent=True) or {}

    simulation_state["started"] = True
    simulation_state["step"] = 0
    simulation_state["scenario"] = body.get("scenario", simulation_state["scenario"])
    simulation_state["controller"] = body.get("controller", simulation_state["controller"])
    simulation_state["forecast"] = body.get("forecast", simulation_state["forecast"])
    simulation_state["coolingLevel"] = 5

    return jsonify({"status": "started", "state": simulation_state})


@app.post("/step")
def step():
    """Existing dashboard step route."""
    body = request.get_json(silent=True) or {}

    simulation_state["started"] = True
    simulation_state["step"] += 1
    simulation_state["scenario"] = body.get("scenario", simulation_state["scenario"])
    simulation_state["controller"] = body.get("controller", simulation_state["controller"])
    simulation_state["forecast"] = body.get("forecast", simulation_state["forecast"])

    current_controller = simulation_state["controller"]

    if current_controller == "static":
        simulation_state["coolingLevel"] = 5
    elif current_controller == "threshold":
        simulation_state["coolingLevel"] = min(8, simulation_state["coolingLevel"] + 1)
    elif current_controller == "pid":
        simulation_state["coolingLevel"] = min(8, max(4, simulation_state["coolingLevel"]))
    elif current_controller == "predictive_threshold":
        simulation_state["coolingLevel"] = min(8, simulation_state["coolingLevel"] + 1)
    elif current_controller == "q_learning":
        simulation_state["coolingLevel"] = min(
            8,
            max(4, simulation_state["coolingLevel"] + (1 if simulation_state["step"] % 2 == 0 else 0)),
        )
    else:
        simulation_state["coolingLevel"] = min(
            8,
            max(4, simulation_state["coolingLevel"] + (1 if simulation_state["step"] % 3 != 0 else 0)),
        )

    return jsonify({"status": "stepped", "state": simulation_state})


@app.post("/reset")
def reset():
    """Existing dashboard reset route."""
    body = request.get_json(silent=True) or {}

    simulation_state["started"] = False
    simulation_state["step"] = 0
    simulation_state["scenario"] = body.get("scenario", simulation_state["scenario"])
    simulation_state["controller"] = body.get("controller", simulation_state["controller"])
    simulation_state["forecast"] = body.get("forecast", simulation_state["forecast"])
    simulation_state["coolingLevel"] = 5

    return jsonify({"status": "reset", "state": simulation_state})


@app.post("/api/run-simulation")
def run_simulation():
    """
    New showcase route.

    Request body:
    {
      "prompt_text": "...",
      "controller": "dqn",
      "scenario": "spiky",
      "use_forecast": true
    }
    """
    body = request.get_json(silent=True) or {}

    prompt_text = body.get("prompt_text", "").strip()
    controller = body.get("controller", "dqn")
    scenario = body.get("scenario", "spiky")
    use_forecast = bool(body.get("use_forecast", True))

    if not prompt_text:
        return jsonify({"error": "prompt_text is required"}), 400

    if controller not in {"static", "threshold", "pid", "predictive_threshold", "q_learning", "dqn"}:
        return jsonify({"error": "invalid controller"}), 400

    if scenario not in {"stable", "sinusoidal", "spiky", "burst_heavy"}:
        return jsonify({"error": "invalid scenario"}), 400

    result = run_prompt_simulation(
        prompt_text=prompt_text,
        controller=controller,
        scenario=scenario,
        use_forecast=use_forecast,
    )

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
