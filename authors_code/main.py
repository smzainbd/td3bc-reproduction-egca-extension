# =============================================================================
# COMP8221 Assignment 2 — Authors' Reference Implementation (Fujimoto & Gu, 2021)
# Source: https://github.com/sfujim/TD3_BC (Provides eval_policy rollout evaluator)
# =============================================================================
# MODIFIED for COMP8221 A2 (all changes marked "A2:"):
#   1. gym/d4rl replaced by d4rl_compat (original stack no longer installs).
#   2. --alpha / --normalize parsed with real types (were strings; "--normalize False" was truthy).
#   3. Non-default alpha/normalize appended to the results file name so ablations don't overwrite.
import numpy as np
import torch
import argparse
import os

from d4rl_compat import make_env, qlearning_dataset  # A2: was `import gym, d4rl`
import utils
import TD3_BC


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
_eval_envs = {}


def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	# A2: reuse the env. seed() is called every evaluation, so the episodes match a fresh env.
	eval_env = _eval_envs.get(env_name)
	if eval_env is None:
		eval_env = _eval_envs[env_name] = make_env(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--env", default="hopper-medium-v0")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5, type=float)  # A2: type added
	parser.add_argument("--normalize", default=True, type=lambda s: s.lower() in ("true", "1"))  # A2
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	if args.alpha != 2.5: file_name += f"_alpha{args.alpha}"  # A2
	if not args.normalize: file_name += "_nonorm"  # A2
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = make_env(args.env)  # A2

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	}

	# Initialize policy
	policy = TD3_BC.TD3_BC(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(qlearning_dataset(env))  # A2
	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1
	
	evaluations = []
	for t in range(int(args.max_timesteps)):
		policy.train(replay_buffer, args.batch_size)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			evaluations.append(eval_policy(policy, args.env, args.seed, mean, std))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
