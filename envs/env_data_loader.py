# envs/env_data_loader.py

import pandas as pd
import numpy as np


class EnvDataLoader:
    """
    Loads stage4_cooling_control_norm.csv and provides
    real data to drive the RL environment simulation.

    All values in stage4 are z-score normalized:
        mean = 0, std = 1

    We use them as relative signals not absolute values.
    The environment maintains its own temperature state
    in real degrees via the physics equation.

    The workload signal from this dataset drives
    how fast temperature rises at each timestep.
    """

    # Map 5 original actions to 3 simplified actions
    ACTION_MAP = {
        'Increase Chiller': 2,  # increase cooling
        'Boost All':        2,  # increase cooling
        'Maintain':         1,  # maintain cooling
        'Reduce AHU':       0,  # decrease cooling
        'Eco Mode':         0,  # decrease cooling
    }

    def __init__(self, csv_path: str):
        self.df = self._load(csv_path)
        print(f"Environment data loaded : {len(self.df)} rows")
        print(f"Period                  : "
              f"{self.df['Timestamp'].iloc[0]} to "
              f"{self.df['Timestamp'].iloc[-1]}")

    def _load(self, path: str) -> pd.DataFrame:
        """Load and prepare stage4 dataset."""
        df = pd.read_csv(path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)

        # Map 5 actions to 3
        df['action_simplified'] = df[
            'Cooling_Strategy_Action'
        ].map(self.ACTION_MAP)

        return df

    def get_workload(self, idx: int) -> float:
        """
        Get normalized workload for this timestep.

        The raw value is a z-score (mean=0, std=1).
        We map it to 0-1 range using:
            (z + 3) / 6

        This assumes z-scores fall mostly in -3 to +3
        which covers 99.7% of normal distribution.

        Returns float between 0.0 and 1.0
        """
        raw = float(
            self.df['Server_Workload(%)'].iloc[
                idx % len(self.df)
            ]
        )
        return float(np.clip((raw + 3) / 6, 0.0, 1.0))

    def get_ambient_offset(self, idx: int) -> float:
        """
        Get ambient temperature offset for this timestep.

        Used to add realistic variation to ambient temp.
        Scaled to small degree offset range (-2 to +2).

        Returns float between -2.0 and 2.0
        """
        raw = float(
            self.df['Ambient_Temperature(°C)'].iloc[
                idx % len(self.df)
            ]
        )
        return float(np.clip(raw * 0.5, -2.0, 2.0))

    def get_real_action(self, idx: int) -> int:
        """
        Get the actual action taken in the real system.
        Returns simplified action (0, 1, or 2).

        Used for comparison and analysis only.
        The RL agent makes its own decisions.
        """
        return int(
            self.df['action_simplified'].iloc[
                idx % len(self.df)
            ]
        )

    def get_cooling_power(self, idx: int) -> float:
        """
        Get normalized cooling power for this timestep.
        Used for energy analysis and validation.
        """
        raw = float(
            self.df['Cooling_Unit_Power_Consumption(kW)'].iloc[
                idx % len(self.df)
            ]
        )
        return float(np.clip((raw + 3) / 6, 0.0, 1.0))

    def __len__(self) -> int:
        return len(self.df)