import copy
import datetime
import os
import sys
import time

import matplotlib.pyplot as plt
import tqdm
import yaml

from f2d.entity import robot_loader, user_loader
from f2d.floorplan import floorplan_loader
from f2d.utils import utils


class Simulation(object):
    def __init__(self, config_path, config_overrides=None):
        # Load experiment config
        self.config = load_yaml_config(config_path, config_overrides=config_overrides)

        # Simulation parameters
        sim_config = self.config["simulation"]
        self.start_ts = sim_config["start_ts"]
        self.duration = datetime.timedelta(**sim_config["duration"])
        self.end_ts = self.start_ts + self.duration
        self.sim_delta = datetime.timedelta(**sim_config["delta"])
        self.speedup = sim_config["speedup"]
        assert self.speedup is None or isinstance(self.speedup, int)
        self.seed = sim_config["seed"]
        self.show_progress = sim_config["show_progress"]
        # Visualization parameters
        self.visualize = sim_config["visualization"]["visualize"]
        self._viz_n_steps = sim_config["visualization"]["every_n_steps"]
        # Parameters for saving simulation outputs
        self._get_outputs_params(sim_config["outputs"])

        # Compute the time to pause between each step in the run loop.
        if self.speedup is None:
            self.wall_time_per_step = None
        else:
            sim_time_per_step_in_seconds = self.sim_delta.total_seconds()
            self.wall_time_per_step = sim_time_per_step_in_seconds / self.speedup  # In seconds

        # Get randomizer for reproducibility.
        self.randomizer = utils.get_randomizer(self.seed)

        # Create floorplan
        # The floorplan is assumed to be completely static and does not
        # change when an episode is reset.
        self.fp = floorplan_loader.get_floorplan(self.config["floorplan"])

        # TODO: Create robot and user objects here, instead of in reset? Then only reset them there?
        self.users = user_loader.get_users(self.config["users"], self.fp, self.randomizer)
        self.robot = robot_loader.get_robot(self.config["robot"], self.fp, self.randomizer)
        user_names = list(self.users.keys())

    def reset(self):
        """Starts a new episode in the simulation."""
        # Note that we do not reset the RandomState, so that each episode
        # in a Simulation instance's lifetime is distinct for variety. If
        # desired, we can provide for a hard reset (or create a new
        # instance) to allow for this.

        # Define the state associated with a simulation episode.
        self.state = {
            "sim_ts": copy.copy(self.start_ts),
            "sim_steps": 0,
            "sim_complete": self.start_ts >= self.end_ts,
            "start_wall_time": time.time(),
            "elapsed_wall_time": 0.0,
            # TODO: update these attributes as needed.
            "users": [],
            "robot": None,
        }
        self._max_sim_steps = int(self.duration / self.sim_delta)

        # TODO: Reset stateful objects such as robot, user, etc.
        for user in self.users.values():
            user.reset(self.state["sim_ts"])
        self.robot.reset(self.state["sim_ts"])

        # Visualization-related
        if self.visualize:
            self.create_viz()

    def step(self):
        """Takes the next step within the episode."""
        self.state["sim_ts"] += self.sim_delta
        self.state["sim_steps"] += 1
        self.state["sim_complete"] = self.state["sim_ts"] >= self.end_ts
        self.state["elapsed_wall_time"] = time.time() - self.state["start_wall_time"]

        # Update user, robot, etc.
        for user in self.users.values():
            user.step(self.sim_delta, self.state["sim_ts"])
        self.robot.step(self.sim_delta, self.state["sim_ts"], users=self.users)

        if self.visualize:
            if self.state["sim_steps"] % self._viz_n_steps == 0:
                self.update_viz()

    def run(self, reset=True):
        """Starts a new (or resumes) episode."""
        if reset:
            self.reset()
        disable_tqdm = not self.show_progress
        with tqdm.tqdm(total=self._max_sim_steps, disable=disable_tqdm) as pbar:
            while not self.state["sim_complete"]:
                self.step()
                if self.save_outputs and self.state["sim_steps"] % self.save_every_n_steps == 0:
                    pass
                pbar.update(1)
                if self.wall_time_per_step:
                    time.sleep(self.wall_time_per_step)
        print("Run complete:")
        sim_duration = self.state["sim_ts"] - self.start_ts
        print("      sim duration: {}".format(sim_duration))
        print(" elapsed wall time: {:.1f}s".format(self.state["elapsed_wall_time"]))
        print("           speedup: {:.1f}".format(self.get_effective_speedup()))

    def create_viz(self):
        plt.ion()  # Interactive plotting on
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(right=0.8)  # Leave space for the legend on the right

        # Draw the floorplan first
        self.fp.draw(ax)

        # TODO: Draw circles corresponding to robot and user(s) here.
        for user in self.users.values():
            user.draw(ax)
        self.robot.draw(ax)

        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

        # Store all viz objects that may require updating afterwards.
        self._viz_objects = {"fig": fig, "ax": ax}
        self.update_viz()

    def update_viz(self):
        # TODO: Make updates based on state.
        fig, ax = self._viz_objects["fig"], self._viz_objects["ax"]
        ax.set_title(self._get_viz_title())

        self.fp.draw_update(ax)

        # TODO: Update circles corresponding to robot and user(s) here.
        for user in self.users.values():
            user.draw_update(ax)
        self.robot.draw_update(ax)

        # Update canvas in interactive mode.
        fig.canvas.draw()
        fig.canvas.flush_events()

    def _get_viz_title(self):
        if not self.state:
            return ""
        step_str = "sim_step={}".format(self.state["sim_steps"])
        title_str = "{:>20}, sim_time={}\nwall_time={:.1f}s".format(
            step_str, self.state["sim_ts"], self.state["elapsed_wall_time"]
        )
        title_str += ", speedup={:.1f}".format(self.get_effective_speedup())
        return title_str

    def get_effective_speedup(self):
        elapsed_sim_time = (self.state["sim_ts"] - self.start_ts).total_seconds()
        speedup = elapsed_sim_time / (self.state["elapsed_wall_time"] + 1e-6)
        return speedup

    def _get_outputs_params(self, outputs_config):
        self.save_outputs = outputs_config["save"]
        if self.save_outputs:
            save_dir = outputs_config["output_dir"]
            if not os.path.exists(save_dir):
                raise ValueError(f"Output directory {save_dir} does not exist!")
            prefix = outputs_config["prefix"]
            if not prefix:
                time_now = datetime.datetime.now()
                prefix = time_now.strftime("%Y_%m_%d_%H_%M_%S")
            save_dir = os.path.join(save_dir, prefix)
            if os.path.exists(save_dir):
                raise ValueError(f"Experiment directory {save_dir} already exists!")
            self.save_dir = save_dir
            os.mkdir(self.save_dir)
            print(f"[Simulation] Saving outputs to {self.save_dir}")
            self.save_every_n_steps = outputs_config["every_n_steps"]


def load_yaml_config(config_path, config_overrides=None):
    if not os.path.exists(config_path):
        raise ValueError(f"Invalid config path: f{config_path}")
    if config_overrides is not None:
        # TODO: Enable config overrides for large-scale experiments.
        raise ValueError("config_overrides not yet supported!")
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    return config


def main():
    config_path = sys.argv[1]  # e.g. 'f2d/configs/config_simple.yaml'
    # config_path = 'f2d/configs/config_dev.yaml'
    print(f"Loading configuration at path: {config_path}")
    sim = Simulation(config_path)
    input("Config loaded successfully! Press ENTER to begin the simulation.")
    # sim.reset()
    sim.run()
    input("Simulation has ended, press ENTER to exit.")


if __name__ == "__main__":
    main()
