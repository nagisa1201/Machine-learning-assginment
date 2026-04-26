import csv
import os
import time
from collections import deque
import matplotlib
import numpy as np


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return float(default)
            return float(arr[0])
        return float(value)
    except Exception:
        return float(default)


class TrainingEvaluator:
    def __init__(
        self,
        out_dir="evaluation_logs",
        run_name=None,
        live_plot=True,
        save_plot=True,
        save_csv=True,
        plot_every_steps=50,
        table_every_steps=50,
        dpi=180,
        headless=None,
    ):
        self.out_dir = out_dir
        self.run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.out_dir, self.run_name)
        self.live_plot = live_plot
        self.save_plot = save_plot
        self.save_csv = save_csv
        self.plot_every_steps = max(1, int(plot_every_steps))
        self.table_every_steps = max(1, int(table_every_steps))
        self.dpi = int(dpi)

        if headless is None:
            headless = os.environ.get("DISPLAY", "") == ""

        self.headless = bool(headless)
        if self.headless:
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        self.plt = plt
        if self.live_plot and (not self.headless):
            self.plt.ion()

        os.makedirs(self.run_dir, exist_ok=True)

        self.global_step = 0
        self.current_episode = -1

        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_actions = []
        self.episode_track_pos = []
        self.episode_speed = []
        self.episode_damage = []
        self.episode_offtrack_count = 0
        self.episode_overlap_count = 0
        self.episode_distance_start = None
        self.episode_distance_last = None
        self.episode_start_time = None

        self.episode_history = []
        self.step_history = []
        self.train_history = []

        self.window = 200
        self.recent_rewards = deque(maxlen=self.window)
        self.recent_q = deque(maxlen=self.window)
        self.recent_critic_loss = deque(maxlen=self.window)
        self.recent_actor_loss = deque(maxlen=self.window)

        self._init_plot()
        self._init_csv()

    def _init_plot(self):
        self.fig, axes = self.plt.subplots(3, 2, figsize=(16, 10), dpi=self.dpi)
        self.axes = axes.flatten()
        titles = [
            "Episode Total Reward",
            "Episode Length",
            "Average Target Q-Value",
            "Critic / Actor Loss",
            "Control Smoothness",
            "Safety & Physical Quality",
        ]
        for axis, title in zip(self.axes, titles):
            axis.set_title(title)
            axis.grid(True, alpha=0.3)

        self.lines = {
            "reward": self.axes[0].plot([], [], label="ep_reward", color="#1f77b4")[0],
            "length": self.axes[1].plot([], [], label="ep_steps", color="#ff7f0e")[0],
            "q": self.axes[2].plot([], [], label="avg_target_q", color="#2ca02c")[0],
            "critic_loss": self.axes[3].plot([], [], label="critic_loss", color="#d62728")[0],
            "actor_loss": self.axes[3].plot([], [], label="actor_loss", color="#9467bd")[0],
            "steer_jerk": self.axes[4].plot([], [], label="steer_jerk", color="#8c564b")[0],
            "steer_var": self.axes[4].plot([], [], label="steer_var", color="#e377c2")[0],
            "offtrack": self.axes[5].plot([], [], label="offtrack_count", color="#7f7f7f")[0],
            "damage": self.axes[5].plot([], [], label="damage_delta", color="#17becf")[0],
            "center_mse": self.axes[5].plot([], [], label="center_mse", color="#bcbd22")[0],
        }

        for axis in self.axes:
            axis.legend(loc="best", fontsize=8)

        self.fig.suptitle("DDPG TORCS Evaluation Dashboard", fontsize=14)
        self.fig.tight_layout(rect=[0, 0, 1, 0.97])

    def _init_csv(self):
        self.episode_csv_path = os.path.join(self.run_dir, "episode_metrics.csv")
        self.step_csv_path = os.path.join(self.run_dir, "step_metrics.csv")
        self.train_csv_path = os.path.join(self.run_dir, "train_metrics.csv")

        if not self.save_csv:
            self.episode_writer = None
            self.step_writer = None
            self.train_writer = None
            self.episode_csv_file = None
            self.step_csv_file = None
            self.train_csv_file = None
            return

        self.episode_csv_file = open(self.episode_csv_path, "w", newline="")
        self.step_csv_file = open(self.step_csv_path, "w", newline="")
        self.train_csv_file = open(self.train_csv_path, "w", newline="")

        self.episode_writer = csv.DictWriter(
            self.episode_csv_file,
            fieldnames=[
                "episode",
                "episode_total_reward",
                "episode_length",
                "avg_target_q",
                "avg_critic_loss",
                "avg_actor_loss",
                "steer_variance",
                "steer_jerk",
                "throttle_brake_overlap",
                "distance_traveled",
                "track_completion_pct",
                "avg_speed",
                "lap_time",
                "center_mse",
                "collision_rate",
                "damage_accumulation",
                "offtrack_count",
            ],
        )
        self.episode_writer.writeheader()

        self.step_writer = csv.DictWriter(
            self.step_csv_file,
            fieldnames=[
                "global_step",
                "episode",
                "step",
                "reward",
                "episode_reward",
                "speed",
                "track_pos",
                "steer",
                "accel",
                "brake",
                "offtrack",
                "damage_delta",
                "target_q_mean",
                "critic_loss",
                "actor_loss",
            ],
        )
        self.step_writer.writeheader()

        self.train_writer = csv.DictWriter(
            self.train_csv_file,
            fieldnames=["global_step", "episode", "critic_loss", "actor_loss", "target_q_mean"],
        )
        self.train_writer.writeheader()

    def on_episode_start(self, episode_idx, init_info=None):
        self.current_episode = int(episode_idx)
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_actions = []
        self.episode_track_pos = []
        self.episode_speed = []
        self.episode_damage = []
        self.episode_offtrack_count = 0
        self.episode_overlap_count = 0
        self.episode_start_time = time.time()

        init_info = init_info or {}
        start_distance = _safe_float(init_info.get("distRaced", init_info.get("distFromStart", 0.0)))
        self.episode_distance_start = start_distance
        self.episode_distance_last = start_distance

    def on_step(
        self,
        episode_idx,
        step_idx,
        action,
        reward,
        obs,
        done,
        info,
        target_q_mean=None,
        critic_loss=None,
        actor_loss=None,
    ):
        info = info or {}
        self.global_step += 1
        self.episode_steps += 1
        self.episode_reward += _safe_float(reward)

        steer = _safe_float(action[0] if len(action) > 0 else 0.0)
        accel = _safe_float(action[1] if len(action) > 1 else 0.0)
        brake = _safe_float(action[2] if len(action) > 2 else 0.0)

        speed = _safe_float(getattr(obs, "speedX", info.get("speedX", 0.0)))
        track_pos = _safe_float(getattr(obs, "trackPos", info.get("trackPos", 0.0)))
        damage = _safe_float(info.get("damage", 0.0))
        damage_delta = _safe_float(info.get("damage_delta", 0.0))

        offtrack = int(info.get("offtrack", abs(track_pos) > 1.0))
        if offtrack:
            self.episode_offtrack_count += 1

        if accel > 0.1 and brake > 0.1:
            self.episode_overlap_count += 1

        self.episode_actions.append([steer, accel, brake])
        self.episode_track_pos.append(track_pos)
        self.episode_speed.append(speed)
        self.episode_damage.append(damage)

        current_distance = _safe_float(info.get("distRaced", info.get("distFromStart", self.episode_distance_last)))
        self.episode_distance_last = current_distance

        step_row = {
            "global_step": self.global_step,
            "episode": int(episode_idx),
            "step": int(step_idx),
            "reward": _safe_float(reward),
            "episode_reward": self.episode_reward,
            "speed": speed,
            "track_pos": track_pos,
            "steer": steer,
            "accel": accel,
            "brake": brake,
            "offtrack": offtrack,
            "damage_delta": damage_delta,
            "target_q_mean": _safe_float(target_q_mean, np.nan),
            "critic_loss": _safe_float(critic_loss, np.nan),
            "actor_loss": _safe_float(actor_loss, np.nan),
        }
        self.step_history.append(step_row)

        if self.step_writer is not None:
            self.step_writer.writerow(step_row)

        if (self.global_step % self.table_every_steps) == 0:
            self.print_step_table(step_row)

        if (self.global_step % self.plot_every_steps) == 0:
            self.update_plot()

        if done:
            self.on_episode_end(episode_idx, info)

    def on_train_update(self, episode_idx, critic_loss=None, actor_loss=None, target_q_mean=None):
        entry = {
            "global_step": self.global_step,
            "episode": int(episode_idx),
            "critic_loss": _safe_float(critic_loss, np.nan),
            "actor_loss": _safe_float(actor_loss, np.nan),
            "target_q_mean": _safe_float(target_q_mean, np.nan),
        }
        self.train_history.append(entry)

        if np.isfinite(entry["critic_loss"]):
            self.recent_critic_loss.append(entry["critic_loss"])
        if np.isfinite(entry["actor_loss"]):
            self.recent_actor_loss.append(entry["actor_loss"])
        if np.isfinite(entry["target_q_mean"]):
            self.recent_q.append(entry["target_q_mean"])

        if self.train_writer is not None:
            self.train_writer.writerow(entry)

    def on_episode_end(self, episode_idx, info=None):
        info = info or {}
        ep = int(episode_idx)
        actions = np.asarray(self.episode_actions, dtype=np.float32) if self.episode_actions else np.zeros((1, 3), dtype=np.float32)
        track_pos_arr = np.asarray(self.episode_track_pos, dtype=np.float32) if self.episode_track_pos else np.zeros((1,), dtype=np.float32)
        speed_arr = np.asarray(self.episode_speed, dtype=np.float32) if self.episode_speed else np.zeros((1,), dtype=np.float32)

        steer = actions[:, 0]
        steer_var = float(np.var(steer))
        steer_jerk = float(np.mean(np.abs(np.diff(steer)))) if steer.size > 1 else 0.0

        overlap_ratio = float(self.episode_overlap_count) / max(1, self.episode_steps)

        distance_traveled = float(self.episode_distance_last - self.episode_distance_start)
        track_len = _safe_float(info.get("track_len", np.nan), np.nan)
        if np.isfinite(track_len) and track_len > 0:
            track_completion = float((distance_traveled / track_len) * 100.0)
        else:
            track_completion = np.nan

        avg_speed = float(np.mean(speed_arr))
        lap_time = _safe_float(info.get("lastLapTime", info.get("curLapTime", np.nan)), np.nan)

        center_mse = float(np.mean(np.square(track_pos_arr)))
        damage_accum = _safe_float(info.get("damage", self.episode_damage[-1] if self.episode_damage else 0.0))
        damage_delta = _safe_float(info.get("damage_delta", 0.0))

        collision_count = int(info.get("collision", damage_delta > 0.0))
        collision_rate = float(collision_count) / max(1, self.episode_steps)

        avg_target_q = float(np.nanmean([x.get("target_q_mean", np.nan) for x in self.train_history if x.get("episode") == ep])) if self.train_history else np.nan
        avg_critic_loss = float(np.nanmean([x.get("critic_loss", np.nan) for x in self.train_history if x.get("episode") == ep])) if self.train_history else np.nan
        avg_actor_loss = float(np.nanmean([x.get("actor_loss", np.nan) for x in self.train_history if x.get("episode") == ep])) if self.train_history else np.nan

        row = {
            "episode": ep,
            "episode_total_reward": float(self.episode_reward),
            "episode_length": int(self.episode_steps),
            "avg_target_q": avg_target_q,
            "avg_critic_loss": avg_critic_loss,
            "avg_actor_loss": avg_actor_loss,
            "steer_variance": steer_var,
            "steer_jerk": steer_jerk,
            "throttle_brake_overlap": overlap_ratio,
            "distance_traveled": distance_traveled,
            "track_completion_pct": track_completion,
            "avg_speed": avg_speed,
            "lap_time": lap_time,
            "center_mse": center_mse,
            "collision_rate": collision_rate,
            "damage_accumulation": damage_accum,
            "offtrack_count": int(self.episode_offtrack_count),
        }

        self.episode_history.append(row)
        self.recent_rewards.append(row["episode_total_reward"])

        if self.episode_writer is not None:
            self.episode_writer.writerow(row)

        self.print_episode_table(row)
        self.update_plot(force=True)

    def print_step_table(self, row):
        template = (
            "[STEP] g:{global_step:7d} ep:{episode:4d} st:{step:5d} "
            "r:{reward:8.3f} ep_r:{episode_reward:9.3f} v:{speed:7.3f} "
            "tp:{track_pos:7.3f} a:[{steer:6.3f},{accel:6.3f},{brake:6.3f}] "
            "Q:{target_q_mean:9.4f} Lc:{critic_loss:9.4f} La:{actor_loss:9.4f}"
        )
        print(template.format(**row))

    def print_episode_table(self, row):
        print("=" * 148)
        print(
            "| EP | Reward | Len | AvgQ | CriticLoss | ActorLoss | SteerVar | Jerk | Overlap | Dist | Complete% | AvgSpeed | LapTime | CenterMSE | CollRate | Damage | OffTrack |"
        )
        print("-" * 148)
        print(
            "| {episode:2d} | {episode_total_reward:7.2f} | {episode_length:4d} | {avg_target_q:7.3f} | "
            "{avg_critic_loss:10.4f} | {avg_actor_loss:9.4f} | {steer_variance:8.5f} | {steer_jerk:5.3f} | "
            "{throttle_brake_overlap:7.3f} | {distance_traveled:7.2f} | {track_completion_pct:9.2f} | "
            "{avg_speed:8.3f} | {lap_time:7.3f} | {center_mse:9.5f} | {collision_rate:8.4f} | "
            "{damage_accumulation:6.1f} | {offtrack_count:8d} |".format(**row)
        )
        print("=" * 148)

    def update_plot(self, force=False):
        if (not force) and ((self.global_step % self.plot_every_steps) != 0):
            return

        if not self.episode_history:
            return

        episodes = np.asarray([x["episode"] for x in self.episode_history], dtype=np.int32)

        reward = np.asarray([x["episode_total_reward"] for x in self.episode_history], dtype=np.float32)
        length = np.asarray([x["episode_length"] for x in self.episode_history], dtype=np.float32)
        q = np.asarray([x["avg_target_q"] for x in self.episode_history], dtype=np.float32)
        critic = np.asarray([x["avg_critic_loss"] for x in self.episode_history], dtype=np.float32)
        actor = np.asarray([x["avg_actor_loss"] for x in self.episode_history], dtype=np.float32)
        jerk = np.asarray([x["steer_jerk"] for x in self.episode_history], dtype=np.float32)
        var = np.asarray([x["steer_variance"] for x in self.episode_history], dtype=np.float32)
        offtrack = np.asarray([x["offtrack_count"] for x in self.episode_history], dtype=np.float32)
        damage = np.asarray([x["damage_accumulation"] for x in self.episode_history], dtype=np.float32)
        center_mse = np.asarray([x["center_mse"] for x in self.episode_history], dtype=np.float32)

        self.lines["reward"].set_data(episodes, reward)
        self.lines["length"].set_data(episodes, length)
        self.lines["q"].set_data(episodes, q)
        self.lines["critic_loss"].set_data(episodes, critic)
        self.lines["actor_loss"].set_data(episodes, actor)
        self.lines["steer_jerk"].set_data(episodes, jerk)
        self.lines["steer_var"].set_data(episodes, var)
        self.lines["offtrack"].set_data(episodes, offtrack)
        self.lines["damage"].set_data(episodes, damage)
        self.lines["center_mse"].set_data(episodes, center_mse)

        for axis in self.axes:
            axis.relim()
            axis.autoscale_view()

        if self.live_plot and (not self.headless):
            self.plt.draw()
            self.plt.pause(0.001)

        if self.save_plot:
            png_path = os.path.join(self.run_dir, "dashboard_latest.png")
            svg_path = os.path.join(self.run_dir, "dashboard_latest.svg")
            self.fig.savefig(png_path, dpi=300, bbox_inches="tight")
            self.fig.savefig(svg_path, format="svg", bbox_inches="tight")

    def close(self):
        self.update_plot(force=True)
        if self.episode_csv_file is not None:
            self.episode_csv_file.close()
        if self.step_csv_file is not None:
            self.step_csv_file.close()
        if self.train_csv_file is not None:
            self.train_csv_file.close()

        if self.live_plot and (not self.headless):
            self.plt.ioff()
            self.plt.show()
        else:
            self.plt.close(self.fig)
