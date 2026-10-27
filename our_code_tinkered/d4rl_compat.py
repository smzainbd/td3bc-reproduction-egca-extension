# =============================================================================
# COMP8221 Assignment 2 — Student Compatibility Layer (Section 3)
# Authors: Sudipta Saha Niloy (49139568), S M Zain (60801131)
# Role: Downloads UC Berkeley D4RL v0 HDF5 datasets, implements qlearning_dataset()
# timeout handling, and wraps Gymnasium MuJoCo v4 environments with D4RL normalization.
# =============================================================================
"""Drop-in replacement for `gym` + `d4rl` as used by main.py (added for COMP8221 A2).

The original stack (gym 0.17, mujoco-py 1.50, d4rl) does not install on modern Python.
This module downloads the original D4RL hdf5 files and evaluates in gymnasium's MuJoCo
v4 environments (official `mujoco` bindings, same models/observations as the v3 envs
D4RL was collected with), exposing the old gym API that main.py expects.

Usage:
	python d4rl_compat.py                      # self-check
	python d4rl_compat.py hopper-medium-v2 ... # pre-download datasets
"""
import os
import sys
import urllib.request

import gymnasium
import h5py
import numpy as np

GYM_IDS = {"halfcheetah": "HalfCheetah-v4", "hopper": "Hopper-v4", "walker2d": "Walker2d-v4"}

# D4RL REF_MIN_SCORE / REF_MAX_SCORE (identical for v0 and v2)
REF_SCORES = {
	"halfcheetah": (-280.178953, 12135.0),
	"hopper": (-20.272305, 3234.3),
	"walker2d": (1.629008, 4592.3),
}

BASE_URL = "http://rail.eecs.berkeley.edu/datasets/offline_rl"
DATASET_DIR = os.environ.get("D4RL_DATASET_DIR", os.path.expanduser("~/.d4rl/datasets"))


def dataset_url(domain, dataset_type, version):
	if version == "v0":
		name = {"medium-replay": "mixed"}.get(dataset_type, dataset_type).replace("-", "_")
		# The v0 walker medium-replay file is walker_mixed.hdf5, not walker2d_mixed.hdf5.
		file_domain = "walker" if domain == "walker2d" and name == "mixed" else domain
		return f"{BASE_URL}/gym_mujoco/{file_domain}_{name}.hdf5"
	return f"{BASE_URL}/gym_mujoco_{version}/{domain}_{dataset_type.replace('-', '_')}-{version}.hdf5"


class D4RLEnv:
	"""Old-gym-API wrapper: seed(), reset() -> obs, step() -> (obs, r, done, info)."""

	def __init__(self, env_name):
		task, self.version = env_name.rsplit("-", 1)
		self.domain, self.dataset_type = task.split("-", 1)
		self.env = gymnasium.make(GYM_IDS[self.domain])
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space
		self.max_episode_steps = self.env.spec.max_episode_steps
		self._seed = None

	def seed(self, seed):
		self._seed = seed

	def reset(self):
		obs, _ = self.env.reset(seed=self._seed)
		self._seed = None
		return obs

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		return obs, reward, terminated or truncated, info

	def get_normalized_score(self, score):
		lo, hi = REF_SCORES[self.domain]
		return (score - lo) / (hi - lo)

	def get_dataset(self):
		url = dataset_url(self.domain, self.dataset_type, self.version)
		path = os.path.join(DATASET_DIR, os.path.basename(url))
		if not os.path.exists(path):
			os.makedirs(DATASET_DIR, exist_ok=True)
			print(f"Downloading {url}")
			tmp = f"{path}.{os.getpid()}.part"
			urllib.request.urlretrieve(url, tmp)
			os.replace(tmp, path)  # atomic: no half-written file if interrupted
		with h5py.File(path, "r") as f:
			keys = ["observations", "actions", "rewards", "terminals", "timeouts"]
			return {k: f[k][:] for k in keys if k in f}


def make_env(env_name):
	return D4RLEnv(env_name)


def qlearning_dataset(env, dataset=None):
	"""Same semantics as d4rl.qlearning_dataset(env, terminate_on_end=False):
	transitions ending an episode by time-limit are dropped (their next obs belongs to
	the next episode); next_observations is the following row."""
	d = env.get_dataset() if dataset is None else dataset
	terminals, timeouts = d["terminals"], d.get("timeouts")
	n = d["rewards"].shape[0]
	keep = np.zeros(n - 1, dtype=bool)
	step = 0
	for i in range(n - 1):
		final = timeouts[i] if timeouts is not None else step == env.max_episode_steps - 1
		if final:
			step = 0
			continue
		keep[i] = True
		step = 1 if terminals[i] else step + 1  # matches d4rl's reset-then-increment
	idx = np.nonzero(keep)[0]
	return {
		"observations": d["observations"][idx],
		"actions": d["actions"][idx],
		"next_observations": d["observations"][idx + 1],
		"rewards": d["rewards"][idx],
		"terminals": terminals[idx].astype(bool),
	}


def _self_check():
	class Stub:
		max_episode_steps = 3

	# rows: ep A = 0,1,2 (timeout at 2) | ep B = 3,4 (terminal at 4) | row 5 = last row (always dropped)
	obs = np.arange(6, dtype=np.float32)[:, None]
	data = {
		"observations": obs, "actions": obs, "rewards": obs[:, 0],
		"terminals": np.array([0, 0, 0, 0, 1, 0], bool),
		"timeouts": np.array([0, 0, 1, 0, 0, 0], bool),
	}
	out = qlearning_dataset(Stub(), data)
	assert out["observations"][:, 0].tolist() == [0, 1, 3, 4]
	assert out["next_observations"][:, 0].tolist() == [1, 2, 4, 5]
	assert out["terminals"].tolist() == [False, False, False, True]

	# no timeouts key (some v0 files): fall back to step counter, episode length 3
	del data["timeouts"]
	out = qlearning_dataset(Stub(), data)
	assert out["observations"][:, 0].tolist() == [0, 1, 3, 4]

	env = make_env("hopper-medium-replay-v2")
	assert env.dataset_type == "medium-replay" and env.version == "v2"
	assert dataset_url("hopper", "medium-replay", "v0").endswith("gym_mujoco/hopper_mixed.hdf5")
	assert dataset_url("walker2d", "medium-replay", "v0").endswith("gym_mujoco/walker_mixed.hdf5")
	assert dataset_url("walker2d", "medium", "v0").endswith("gym_mujoco/walker2d_medium.hdf5")
	assert dataset_url("hopper", "medium-expert", "v2").endswith("gym_mujoco_v2/hopper_medium_expert-v2.hdf5")
	assert abs(env.get_normalized_score(3234.3) - 1.0) < 1e-9
	env.seed(0)
	o1 = env.reset()
	env.seed(0)
	assert np.allclose(o1, env.reset()), "seeded reset must be deterministic"
	_, _, done, _ = env.step(env.action_space.sample())
	assert isinstance(done, bool)
	print("d4rl_compat self-check passed")


if __name__ == "__main__":
	if len(sys.argv) > 1:
		for name in sys.argv[1:]:
			make_env(name).get_dataset()
	else:
		_self_check()
