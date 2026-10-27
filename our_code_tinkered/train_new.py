# =============================================================================
# COMP8221 Assignment 2 — Student Work for Section 4 (EG-CA Training Driver)
# Authors: Sudipta Saha Niloy (49139568), S M Zain (60801131)
# Role: Resumable multi-job training driver for Section 4 variants, logging
# w_joint_cv and gate_mean diagnostics across 1,000,000 gradient steps.
# =============================================================================
"""A2: resumable training driver for Notebook 2 using TD3_BC_new.py and utils_new.py.

Supports Vanilla TD3+BC ("td3bc"), Scalar Advantage-Weighted TD3+BC ("aw_td3bc"),
Epistemic-Gated Coordinate Attribution ("eg_ca_td3bc"), Shuffled Controls
("shuffled_coord", "shuffled_joint"), and Ablation Variants
("eg_ca_td3bc_nogate", "eg_ca_td3bc_tau05", "eg_ca_td3bc_tau20") under the paper's protocol:
1e6 steps, 10 eval episodes every 5000 steps (or --eval_freq with dense 5000-step evals
in the final 50k steps), score = mean of the final 10 evaluations.

Single run:  python train_new.py --algo eg_ca_td3bc --env hopper-medium-replay-v0 --seed 0
Many runs:   from train_new import live; live([[...args...], ...], jobs=2)
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch

# T4 does no tensor-core fp32. L4 does, and it would change the arithmetic, so keep full fp32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import TD3_BC_new as TD3_BC
import utils_new as utils
from d4rl_compat import make_env, qlearning_dataset
from main import eval_policy

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results_a2")
CKPT_DIR = os.path.join(HERE, "checkpoints")
LOG_DIR = os.path.join(HERE, "logs")
DENSE_EVAL = 5000
# args that change the experiment; non-default values go into the run name
TAGGED = ["alpha", "normalize", "max_timesteps", "tau_aw", "beta_ood"]


def build_parser():
	p = argparse.ArgumentParser()
	p.add_argument("--algo", default="td3bc", choices=list(TD3_BC.VALID_ALGOS))
	p.add_argument("--env", default="hopper-medium-v0")
	p.add_argument("--seed", default=0, type=int)
	p.add_argument("--max_timesteps", default=1_000_000, type=int)
	p.add_argument("--eval_freq", default=DENSE_EVAL, type=int)
	p.add_argument("--eval_episodes", default=10, type=int)
	p.add_argument("--ckpt_freq", default=50_000, type=int)
	p.add_argument("--batch_size", default=256, type=int)
	p.add_argument("--alpha", default=2.5, type=float)
	p.add_argument("--normalize", default=True, type=lambda s: s.lower() in ("true", "1"))
	# Epistemic-Gated Coordinate Attribution & AW hyperparameters
	p.add_argument("--tau_aw", default=1.0, type=float)
	p.add_argument("--beta_ood", default=1.0, type=float)
	return p


def run_name(args):
	p = build_parser()
	keys = TAGGED if args.algo != "td3bc" else TAGGED[:3]  # EG-CA args don't affect td3bc runs
	suffix = "".join(f"_{k}{getattr(args, k)}" for k in keys if getattr(args, k) != p.get_default(k))
	return f"{args.algo}_{args.env}_{args.seed}{suffix}"


def _modules(policy):
	return {k: v for k, v in vars(policy).items() if isinstance(v, (torch.nn.Module, torch.optim.Optimizer))}


def save_checkpoint(path, policy, t, log):
	state = {k: v.state_dict() for k, v in _modules(policy).items()}
	state.update(total_it=policy.total_it, t=t, log=log, torch_rng=torch.get_rng_state(), np_rng=np.random.get_state())
	if torch.cuda.is_available():
		state["cuda_rng"] = torch.cuda.get_rng_state_all()
	tmp = path + ".tmp"
	torch.save(state, tmp)
	os.replace(tmp, path)  # atomic: a disconnect mid-save keeps the previous checkpoint


def load_checkpoint(path, policy):
	# CPU, not the GPU: map_location to CUDA turns the RNG state into a CUDA tensor and
	# torch.set_rng_state then raises "RNG state must be a torch.ByteTensor".
	state = torch.load(path, map_location="cpu", weights_only=False)
	for k, v in _modules(policy).items():
		v.load_state_dict(state[k])
	policy.total_it = state["total_it"]
	torch.set_rng_state(state["torch_rng"].cpu().byte())
	np.random.set_state(state["np_rng"])
	if "cuda_rng" in state and torch.cuda.is_available():
		torch.cuda.set_rng_state_all([t.cpu().byte() for t in state["cuda_rng"]])
	return state["t"], state["log"]


def make_policy(args, state_dim, action_dim, max_action):
	kwargs = dict(
		state_dim=state_dim,
		action_dim=action_dim,
		max_action=max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2 * max_action,
		noise_clip=0.5 * max_action,
		policy_freq=2,
		alpha=args.alpha,
		algo=args.algo,
		tau_aw=args.tau_aw,
		beta_ood=args.beta_ood,
	)
	return TD3_BC.TD3_BC(**kwargs)


def train(args):
	name = run_name(args)
	result_path = os.path.join(RESULTS_DIR, name + ".npz")
	if os.path.exists(result_path):
		print(f"{name}: already finished")
		return
	os.makedirs(RESULTS_DIR, exist_ok=True)
	os.makedirs(CKPT_DIR, exist_ok=True)
	ckpt_path = os.path.join(CKPT_DIR, name + ".pt")

	env = make_env(args.env)
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])
	policy = make_policy(args, state_dim, action_dim, max_action)

	buffer = utils.ReplayBuffer(state_dim, action_dim)
	buffer.convert_D4RL(qlearning_dataset(env))
	mean, std = buffer.normalize_states() if args.normalize else (0, 1)

	t0, log = 0, {"steps": [], "scores": [], "diag": []}
	if os.path.exists(ckpt_path):
		t0, log = load_checkpoint(ckpt_path, policy)
		print(f"{name}: resumed at step {t0}")

	start = time.time()
	for t in range(t0, args.max_timesteps):
		policy.train(buffer, args.batch_size)
		step = t + 1
		final_window = step > args.max_timesteps - 10 * DENSE_EVAL
		if step % args.eval_freq == 0 or (final_window and step % DENSE_EVAL == 0):
			log["steps"].append(step)
			log["scores"].append(eval_policy(policy, args.env, args.seed, mean, std, eval_episodes=args.eval_episodes))
			if hasattr(policy, "diagnostics"):
				log["diag"].append(policy.diagnostics(buffer))
			rate = (step - t0) / (time.time() - start)
			print(f"{name}: step {step}  score {log['scores'][-1]:.1f}  {rate:.0f} steps/s", flush=True)
		if step % args.ckpt_freq == 0 and step < args.max_timesteps:
			save_checkpoint(ckpt_path, policy, step, log)

	diag = {k: np.array([d[k] for d in log["diag"]]) for k in (log["diag"][0] if log["diag"] else {})}
	np.savez(
		result_path, steps=np.array(log["steps"]), scores=np.array(log["scores"]),
		final_score=np.mean(log["scores"][-10:]), args=str(vars(args)), **{f"diag_{k}": v for k, v in diag.items()},
	)
	if os.path.exists(ckpt_path):
		os.remove(ckpt_path)
	print(f"{name}: done, final-10 mean {np.mean(log['scores'][-10:]):.1f}")


def _log_tail(name):
	"""Last progress line of a running job, so a quiet cell is not mistaken for a hang."""
	path = os.path.join(LOG_DIR, name + ".log")
	try:
		with open(path, "rb") as f:
			f.seek(0, os.SEEK_END)
			f.seek(max(0, f.tell() - 4000))
			lines = f.read().decode("utf-8", "replace").splitlines()
	except OSError:
		return ""
	for line in reversed(lines):
		text = line.strip()
		if text and not text.startswith("File ") and "DeprecationWarning" not in text and not text.startswith("logger."):
			return text
	return ""


def _status_lines(runs):
	parser = build_parser()
	lines = []
	for argv in runs:
		name = run_name(parser.parse_args([str(a) for a in argv]))
		done = os.path.exists(os.path.join(RESULTS_DIR, name + ".npz"))
		tail = "" if done else _log_tail(name)
		if done:
			lines.append(f"done      {name}")
		elif tail and any(s in tail for s in ("Error", "Traceback", "Interrupt")):
			lines.append(f"failed    {tail}")
		elif tail:
			lines.append(f"running   {tail}")
		elif os.path.exists(os.path.join(CKPT_DIR, name + ".pt")):
			lines.append(f"paused    {name}")
		else:
			lines.append(f"waiting   {name}")
	return lines


def run_all(runs, jobs=4, poll=30, on_tick=None):
	"""Run each argument list in `runs` as `python train_new.py ...`, `jobs` at a time.
	Finished runs are skipped and interrupted runs resume, so re-running the cell is safe."""
	os.makedirs(LOG_DIR, exist_ok=True)
	parser = build_parser()
	pending = []
	for argv in runs:
		name = run_name(parser.parse_args([str(a) for a in argv]))
		if not os.path.exists(os.path.join(RESULTS_DIR, name + ".npz")):
			pending.append((name, [str(a) for a in argv]))
	if on_tick is None:
		print(f"{len(runs) - len(pending)} finished, {len(pending)} to run, {jobs} at a time")
	# Package layout: authors_code/ + our_code_tinkered/. Child processes need both on PYTHONPATH.
	pkg_root = os.path.dirname(HERE)
	code_paths = os.pathsep.join([
		os.path.join(pkg_root, "authors_code"),
		os.path.join(pkg_root, "our_code_tinkered"),
		HERE,
	])
	env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
	env["PYTHONPATH"] = code_paths + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
	active = {}
	start = time.time()
	try:
		if on_tick:
			on_tick()
		while pending or active:
			while pending and len(active) < jobs:
				name, argv = pending.pop(0)
				log = open(os.path.join(LOG_DIR, name + ".log"), "a")
				active[name] = (subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "train_new.py"), *argv],
					cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT), log)
				if on_tick is None:
					print(f"[{(time.time() - start) / 60:6.1f} min] start {name}", flush=True)
			time.sleep(poll)
			if on_tick:
				on_tick()
			else:
				for name in list(active):
					line = _log_tail(name)
					if line:
						print(f"[{(time.time() - start) / 60:6.1f} min] {line}", flush=True)
			for name in list(active):
				proc, log = active[name]
				if proc.poll() is not None:
					log.close()
					del active[name]
					if on_tick is None:
						status = "done" if proc.returncode == 0 else f"FAILED ({proc.returncode}), see logs/{name}.log"
						print(f"[{(time.time() - start) / 60:6.1f} min] {status} {name}", flush=True)
	finally:
		# stopping the cell stops the jobs too; they resume from their checkpoints on the next run_all
		for name, (proc, log) in active.items():
			proc.terminate()
			proc.wait()
			log.close()
			print(f"stopped {name}")


def live(runs, jobs=2, every=10):
	"""Run `runs` and redraw their status in the notebook every `every` seconds."""
	try:
		from IPython.display import clear_output
	except ImportError:
		run_all(runs, jobs=jobs, poll=every)
		return

	def show():
		clear_output(wait=True)
		print(time.strftime("%H:%M:%S"), flush=True)
		print("\n".join(_status_lines(runs)), flush=True)

	run_all(runs, jobs=jobs, poll=every, on_tick=show)
	show()


def benchmark(argv, steps=2000, warmup=200):
	"""Milliseconds per training step (no evaluation) for train_new.py arguments, one CPU thread like run_all jobs."""
	args = build_parser().parse_args([str(a) for a in argv])
	env = make_env(args.env)
	state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
	policy = make_policy(args, state_dim, action_dim, float(env.action_space.high[0]))
	buffer = utils.ReplayBuffer(state_dim, action_dim)
	buffer.convert_D4RL(qlearning_dataset(env))
	buffer.normalize_states()
	threads = torch.get_num_threads()
	torch.set_num_threads(1)
	for _ in range(warmup):
		policy.train(buffer, args.batch_size)
	if torch.cuda.is_available():
		torch.cuda.synchronize()
	start = time.time()
	for _ in range(steps):
		policy.train(buffer, args.batch_size)
	if torch.cuda.is_available():
		torch.cuda.synchronize()
	torch.set_num_threads(threads)
	return (time.time() - start) / steps * 1000


def _bench_entry(argv, steps, queue):
	os.environ["OMP_NUM_THREADS"] = "1"
	queue.put(1000.0 / benchmark(argv, steps, warmup=50))  # steps/s, measured while the other jobs run


def choose_jobs(argv, candidates=(1, 2, 4, 8), steps=400):
	"""How many copies of this run to keep on this machine at once."""
	import multiprocessing as mp

	ctx = mp.get_context("spawn")
	best_jobs, best_total, alone_ms = 1, 0.0, None
	for jobs in candidates:
		queue = ctx.Queue()
		procs = [ctx.Process(target=_bench_entry, args=(list(argv), steps, queue)) for _ in range(jobs)]
		for proc in procs:
			proc.start()
		rates = [queue.get() for _ in procs]
		for proc in procs:
			proc.join()
		if any(proc.exitcode for proc in procs):
			raise RuntimeError(f"concurrency probe failed at {jobs} jobs")
		total = sum(rates)
		alone_ms = alone_ms if alone_ms is not None else 1000.0 / rates[0]
		print(f"{jobs} at a time: {total:.0f} steps/s total ({total / jobs:.0f} each)", flush=True)
		if total > best_total * 1.05:
			best_jobs, best_total = jobs, total
		else:
			break
	print(f"using {best_jobs} at a time", flush=True)
	return best_jobs, alone_ms


def result(argv):
	"""npz contents of the finished run with these train_new.py arguments, or None if it hasn't finished."""
	path = os.path.join(RESULTS_DIR, run_name(build_parser().parse_args([str(a) for a in argv])) + ".npz")
	if not os.path.exists(path):
		return None
	with np.load(path, allow_pickle=False) as f:
		return {k: f[k] for k in f.files}


if __name__ == "__main__":
	train(build_parser().parse_args())
