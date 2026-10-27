# =============================================================================
# COMP8221 Assignment 2 — Student Work for Section 4 (EG-CA TD3+BC Implementation)
# Authors: Sudipta Saha Niloy (49139568), S M Zain (60801131)
# Role: Implements Epistemic-Gated Coordinate Attribution (EG-CA) TD3+BC
# (`eg_ca_td3bc`, `eg_ca_td3bc_nogate`), custom scalar advantage (`aw_td3bc`),
# within-sample shuffled-joint negative control (`shuffled_joint`), diagnostics,
# and automated numerical equivalence self-checks (`_self_check()`).
# =============================================================================
import copy
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALID_ALGOS = (
	"td3bc",
	"aw_td3bc",
	"eg_ca_td3bc",
	"shuffled_coord",
	"shuffled_joint",
	"eg_ca_td3bc_nogate",
	"eg_ca_td3bc_tau05",
	"eg_ca_td3bc_tau20",
)


def _arg_from_argv(flag, default, cast=str):
	if flag in sys.argv:
		idx = sys.argv.index(flag)
		if idx + 1 < len(sys.argv):
			return cast(sys.argv[idx + 1])
	return default


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)

		self.max_action = max_action

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3_BC(object):
	"""TD3+BC extended with Epistemic-Gated Coordinate Attribution (EG-CA).

	Supported `algo` modes:
	  - "td3bc": Standard Fujimoto & Gu (2021) uniform BC penalty.
	  - "aw_td3bc": Scalar advantage-weighted BC (with optional beta_ood=0).
	  - "eg_ca_td3bc": Proposed Epistemic-Gated Coordinate Attribution TD3+BC (Idea 1 + Idea 2).
	  - "shuffled_coord": Causal control permuting coordinate weights across batch samples.
	  - "shuffled_joint": Causal control permuting coordinate weights across joints within each sample.
	  - "eg_ca_td3bc_nogate": Ablation removing the epistemic disagreement gate (beta_ood = 0.0).
	  - "eg_ca_td3bc_tau05": Ablation with sharp temperature (tau_aw = 0.5).
	  - "eg_ca_td3bc_tau20": Ablation with soft temperature (tau_aw = 2.0).
	"""

	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
		algo=None,
		tau_aw=None,
		beta_ood=None,
	):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha

		self.algo = algo if algo is not None else _arg_from_argv("--algo", "td3bc", str)
		self.tau_aw = float(tau_aw) if tau_aw is not None else _arg_from_argv("--tau_aw", 1.0, float)
		self.beta_ood = float(beta_ood) if beta_ood is not None else _arg_from_argv("--beta_ood", 1.0, float)

		# Apply preset overrides for named ablation variants
		if self.algo == "eg_ca_td3bc_nogate":
			self.beta_ood = 0.0
		elif self.algo == "eg_ca_td3bc_tau05":
			self.tau_aw = 0.5
		elif self.algo == "eg_ca_td3bc_tau20":
			self.tau_aw = 2.0

		if self.algo not in VALID_ALGOS:
			raise ValueError(f"Unknown algo: {self.algo}. Valid options: {VALID_ALGOS}")

		self.total_it = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		with torch.no_grad():
			return self.actor(state).cpu().data.numpy().flatten()

	def _compute_gate(self, state, action, q1_pi, q2_pi):
		"""Epistemic disagreement gate in (0,1]; 1 = trust critic, 0 = fall back to TD3+BC weight 1."""
		with torch.no_grad():
			q1_data, q2_data = self.critic(state, action)
			dis_pi = (q1_pi.detach() - q2_pi.detach()).abs()            # [B,1]
			dis_data = (q1_data - q2_data).abs()                         # [B,1]
			ood = (dis_pi - dis_data).clamp(min=0.0)                     # [B,1]
			gate = torch.exp(-self.beta_ood * ood / (dis_data.mean() + 1e-6))  # (0,1]; 1 = trust critic
		return gate, dis_pi, dis_data, ood

	def _compute_weights(self, state, action, pi):
		"""Compute normalized BC weights `w` for the active algorithm mode."""
		if self.algo == "aw_td3bc":
			with torch.no_grad():
				q1_pi, q2_pi = self.critic(state, pi.detach())
				q1_data, q2_data = self.critic(state, action)
				q_min_pi = torch.min(q1_pi, q2_pi)
				q_min_data = torch.min(q1_data, q2_data)
				adv = q_min_data - q_min_pi
				gate, _, _, _ = self._compute_gate(state, action, q1_pi, q2_pi)
				z = (adv - adv.mean()) / (adv.std() + 1e-6)
				w_raw = torch.exp((z / self.tau_aw).clamp(-5.0, 5.0))
				w_raw = w_raw / (w_raw.mean() + 1e-8)
				w = gate * w_raw + (1.0 - gate) * 1.0
			return w

		# Coordinate-attribution modes: eg_ca_td3bc, shuffled_coord, shuffled_joint, and ablations
		pi_grad = pi.detach().requires_grad_(True)
		q1_pi, q2_pi = self.critic(state, pi_grad)
		q_min_pi = torch.min(q1_pi, q2_pi)
		g = torch.autograd.grad(q_min_pi.sum(), pi_grad, create_graph=False)[0].detach()

		with torch.no_grad():
			gate, _, _, _ = self._compute_gate(state, action, q1_pi, q2_pi)
			c = g * (action - pi.detach())                               # [B,D] Taylor attribution
			z = (c - c.mean()) / (c.std() + 1e-6)
			w_raw = torch.exp((z / self.tau_aw).clamp(-5.0, 5.0))
			w_raw = w_raw / (w_raw.mean() + 1e-8)

			if self.algo == "shuffled_coord":
				perm = torch.randperm(w_raw.size(0), device=w_raw.device)
				w_raw = w_raw[perm]
			elif self.algo == "shuffled_joint":
				perm_j = torch.rand(w_raw.shape, device=w_raw.device).argsort(dim=1)
				w_raw = torch.gather(w_raw, 1, perm_j)

			w = gate * w_raw + (1.0 - gate) * 1.0

		return w

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor loss
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha / Q.abs().mean().detach()

			if self.algo == "td3bc":
				bc_loss = F.mse_loss(pi, action)
			else:
				w = self._compute_weights(state, action, pi)
				bc_loss = (w * (pi - action).pow(2)).mean()

			actor_loss = -lmbda * Q.mean() + bc_loss

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def diagnostics(self, replay_buffer, batch_size=1024):
		"""Record critic disagreement, gate, and BC weight statistics during evaluation checkpoints."""
		state, action, *_ = replay_buffer.sample(batch_size)
		with torch.no_grad():
			pi = self.actor(state)
			q1_pi, q2_pi = self.critic(state, pi)

		if self.algo == "td3bc":
			with torch.no_grad():
				q1_data, q2_data = self.critic(state, action)
				dis_pi = (q1_pi - q2_pi).abs()
				dis_data = (q1_data - q2_data).abs()
				ood_pen = (dis_pi - dis_data).clamp(min=0.0)
			w_cv, w_max, w_joint_cv, gate_mean = 0.0, 1.0, 0.0, 1.0
		else:
			gate, dis_pi, dis_data, ood_pen = self._compute_gate(state, action, q1_pi, q2_pi)
			gate_mean = gate.mean().item()
			w = self._compute_weights(state, action, pi)
			w_cv = (w.std() / (w.mean() + 1e-8)).item()
			w_max = w.max().item()
			w_joint_cv = (w.std(dim=1) / (w.mean(dim=1) + 1e-8)).mean().item() if w.shape[1] > 1 else 0.0

		return {
			"dis_pi": dis_pi.mean().item(),
			"dis_data": dis_data.mean().item(),
			"ood_pen": ood_pen.mean().item(),
			"w_cv": w_cv,
			"w_max": w_max,
			"w_joint_cv": w_joint_cv,
			"gate_mean": gate_mean,
		}

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)


def _self_check():
	rng_state = torch.get_rng_state()
	cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
	try:
		torch.manual_seed(0)
		s = torch.randn(16, 11, device=device)
		a = torch.randn(16, 3, device=device).clamp(-1.0, 1.0)
		r = torch.randn(16, 1, device=device)
		nd = torch.ones(16, 1, device=device)

		class StubBuf:
			def sample(self, batch_size):
				return s, a, s, r, nd

		# Check 1: Training update and diagnostics across all valid modes
		for mode in VALID_ALGOS:
			pol = TD3_BC(11, 3, 1.0, algo=mode, tau_aw=1.0, beta_ood=1.0)
			if mode == "eg_ca_td3bc_nogate":
				assert pol.beta_ood == 0.0
			elif mode == "eg_ca_td3bc_tau05":
				assert pol.tau_aw == 0.5
			elif mode == "eg_ca_td3bc_tau20":
				assert pol.tau_aw == 2.0
			before = [p.clone() for p in pol.actor.parameters()]
			for _ in range(4):
				pol.train(StubBuf(), batch_size=16)
			assert any(not torch.equal(b, p) for b, p in zip(before, pol.actor.parameters())), f"actor did not update for {mode}"
			act = pol.select_action(np.zeros(11, dtype=np.float32))
			assert act.shape == (3,)
			diag = pol.diagnostics(StubBuf(), batch_size=16)
			assert "w_cv" in diag and "w_joint_cv" in diag and "gate_mean" in diag

		# Check 3a: Exact equivalence with authors' TD3_BC in td3bc mode
		import TD3_BC as _orig
		torch.manual_seed(0)
		old = _orig.TD3_BC(11, 3, 1.0)
		torch.manual_seed(0)
		new = TD3_BC(11, 3, 1.0, algo="td3bc")
		for step_i in range(10):
			torch.manual_seed(100 + step_i)
			old.train(StubBuf(), batch_size=16)
			torch.manual_seed(100 + step_i)
			new.train(StubBuf(), batch_size=16)
		for p_old, p_new in zip(old.actor.parameters(), new.actor.parameters()):
			assert torch.allclose(p_old, p_new, atol=1e-6), "actor param mismatch with authors' TD3_BC"
		for p_old, p_new in zip(old.critic.parameters(), new.critic.parameters()):
			assert torch.allclose(p_old, p_new, atol=1e-6), "critic param mismatch with authors' TD3_BC"

		# Check 3b: nogate (beta_ood=0) weights match full-method with beta_ood=0
		pol_full = TD3_BC(11, 3, 1.0, algo="eg_ca_td3bc", beta_ood=1.0)
		pol_nogate = TD3_BC(11, 3, 1.0, algo="eg_ca_td3bc_nogate")
		pol_nogate.actor.load_state_dict(pol_full.actor.state_dict())
		pol_nogate.critic.load_state_dict(pol_full.critic.state_dict())
		w_nogate = pol_nogate._compute_weights(s, a, pol_nogate.actor(s))
		pol_full.beta_ood = 0.0
		w_full_zero_beta = pol_full._compute_weights(s, a, pol_full.actor(s))
		assert torch.allclose(w_nogate, w_full_zero_beta, atol=1e-6), "nogate must match beta_ood=0 exactly"
		# When ood == 0, gate is exactly 1.0
		gate_zero, _, _, _ = pol_full._compute_gate(s, a, torch.zeros(16, 1, device=device), torch.zeros(16, 1, device=device))
		assert torch.allclose(gate_zero, torch.ones_like(gate_zero), atol=1e-6)

		# Check 3c: w.mean() ≈ 1 and w > 0 for all non-td3bc modes
		for mode in VALID_ALGOS:
			if mode == "td3bc":
				continue
			pol = TD3_BC(11, 3, 1.0, algo=mode)
			w = pol._compute_weights(s, a, pol.actor(s))
			assert 0.5 < w.mean().item() < 1.5, f"{mode} weight mean {w.mean().item()} far from 1"
			assert (w > 0).all(), f"{mode} produced non-positive weights"

		# Check 3d: utils_new vs utils normalization equivalence
		import utils as _orig_utils
		import utils_new as _new_utils
		rb_orig = _orig_utils.ReplayBuffer(11, 3, max_size=32)
		rb_new = _new_utils.ReplayBuffer(11, 3, max_size=32)
		np_data = {
			"observations": np.random.randn(20, 11).astype(np.float32),
			"actions": np.random.randn(20, 3).astype(np.float32),
			"next_observations": np.random.randn(20, 11).astype(np.float32),
			"rewards": np.random.randn(20).astype(np.float32),
			"terminals": np.zeros(20, dtype=bool),
		}
		rb_orig.convert_D4RL(np_data)
		rb_new.convert_D4RL(np_data)
		mean_orig, std_orig = rb_orig.normalize_states()
		mean_new, std_new = rb_new.normalize_states()
		assert np.allclose(mean_orig, mean_new, atol=1e-4), "normalize_states mean mismatch"
		assert np.allclose(std_orig, std_new, atol=1e-4), "normalize_states std mismatch"
		assert np.allclose(rb_orig.state, rb_new.state.cpu().numpy(), atol=1e-4), "normalized states mismatch"

		print("TD3_BC_new self-check passed (all equivalence checks passed)")
	finally:
		torch.set_rng_state(rng_state)
		if cuda_rng is not None:
			torch.cuda.set_rng_state_all(cuda_rng)


if __name__ == "__main__":
	_self_check()
