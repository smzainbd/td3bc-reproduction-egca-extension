"""A2 Section 4 (our work, not the authors'): uncertainty- and advantage-weighted TD3+BC.

Built on the authors' TD3_BC class (actor, optimisers, target-noise, delayed updates and
soft target updates are inherited unchanged). Changes:
  1. Critic: ensemble of N Q-networks (vectorised). Bootstrap target = min over 2 randomly
     chosen target heads (REDQ in-target minimisation), so with N=2 it equals TD3's min(Q1, Q2)
     and the ensemble adds uncertainty information rather than extra pessimism.
  2. Actor value term: Q_lcb = mean_i Q_i(s, pi(s)) - beta * std_i Q_i(s, pi(s)),
     lambda = alpha / mean|Q_lcb| as in the paper.
  3. Per-sample BC weight w(s, a), detached and normalised to batch mean 1 (so alpha keeps its
     meaning and only the *distribution* of BC strength changes):
        w_unc = 1 + gamma_w * std(s, pi(s)) / mean_batch std      (trust Q less where critics disagree)
        w_adv = exp(z(A(s, a)) / adv_tau),  A = Q(s, a) - Q(s, pi(s)), z = batch z-score
                                                                  (imitate dataset actions that look good)
     shuffle_w=True permutes w across the batch: same weight distribution, random assignment.
     It is the control showing whether *which* sample gets the weight matters.
With beta = gamma_w = adv_tau = 0 this is TD3+BC with an N-head critic (actor uses the head mean).
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import TD3_BC
from TD3_BC import device


class VectorizedLinear(nn.Module):
	"""N independent linear layers applied in one batched matmul: [N, B, in] -> [N, B, out]."""

	def __init__(self, n, in_features, out_features):
		super().__init__()
		bound = 1 / math.sqrt(in_features)  # same distribution as nn.Linear's default init
		self.weight = nn.Parameter(torch.empty(n, in_features, out_features).uniform_(-bound, bound))
		self.bias = nn.Parameter(torch.empty(n, 1, out_features).uniform_(-bound, bound))

	def forward(self, x):
		return torch.baddbmm(self.bias, x, self.weight)


class EnsembleCritic(nn.Module):
	"""N copies of the authors' Q architecture (256-256-1, ReLU). Returns Q of shape [N, B]."""

	def __init__(self, state_dim, action_dim, n):
		super().__init__()
		self.n = n
		self.l1 = VectorizedLinear(n, state_dim + action_dim, 256)
		self.l2 = VectorizedLinear(n, 256, 256)
		self.l3 = VectorizedLinear(n, 256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1).unsqueeze(0).expand(self.n, -1, -1)
		q = F.relu(self.l1(sa))
		q = F.relu(self.l2(q))
		return self.l3(q).squeeze(-1)


class TD3_BC_UW(TD3_BC.TD3_BC):
	def __init__(self, state_dim, action_dim, max_action, n_critics=10, beta=0.0, gamma_w=0.0,
			adv_tau=0.0, shuffle_w=False, **kwargs):
		super().__init__(state_dim, action_dim, max_action, **kwargs)
		self.critic = EnsembleCritic(state_dim, action_dim, n_critics).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.n_critics = n_critics
		self.beta = beta
		self.gamma_w = gamma_w
		self.adv_tau = adv_tau
		self.shuffle_w = shuffle_w

	def bc_weights(self, state, action, q_pi):
		"""Per-sample BC weights from the detached ensemble Q at pi(s) (q_pi: [N, B]), batch mean 1."""
		with torch.no_grad():
			w = torch.ones(state.shape[0], device=device)
			if self.gamma_w > 0:
				std = q_pi.std(0)
				w = w * (1 + self.gamma_w * std / (std.mean() + 1e-6))
			if self.adv_tau > 0:
				adv = self.critic(state, action).mean(0) - q_pi.mean(0)
				z = (adv - adv.mean()) / (adv.std() + 1e-6)
				w = w * torch.exp(z / self.adv_tau).clamp(max=100.)
			if self.shuffle_w:
				w = w[torch.randperm(w.shape[0], device=device)]
			return w / w.mean()

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
			target_all = self.critic_target(next_state, next_action)
			heads = torch.randperm(self.n_critics, device=device)[:2]
			target_Q = target_all[heads].min(0).values.unsqueeze(1)
			target_Q = reward + not_done * self.discount * target_Q

		current_Q = self.critic(state, action)
		# per-head MSE summed over heads (= authors' mse(Q1) + mse(Q2) when N = 2)
		critic_loss = ((current_Q - target_Q.T) ** 2).mean(1).sum()

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		if self.total_it % self.policy_freq == 0:
			pi = self.actor(state)
			q_pi = self.critic(state, pi)
			Q = q_pi.mean(0) - self.beta * q_pi.std(0)
			lmbda = self.alpha / Q.abs().mean().detach()
			w = self.bc_weights(state, action, q_pi.detach())
			bc = ((pi - action) ** 2).mean(1)  # weighted version of the authors' F.mse_loss(pi, action)

			actor_loss = -lmbda * Q.mean() + (w * bc).mean()

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	@torch.no_grad()
	def diagnostics(self, replay_buffer, batch_size=4096):
		"""Is ensemble std an out-of-distribution signal? Compare it at dataset, policy and random actions."""
		state, action, *_ = replay_buffer.sample(batch_size)
		pi = self.actor(state)
		rand = torch.empty_like(action).uniform_(-self.max_action, self.max_action)
		out = {}
		for name, a in (("data", action), ("pi", pi), ("rand", rand)):
			q = self.critic(state, a)
			out[f"q_{name}"] = q.mean().item()
			out[f"std_{name}"] = q.std(0).mean().item()
		out["pi_data_dist"] = (pi - action).norm(dim=1).mean().item()
		w = self.bc_weights(state, action, self.critic(state, pi))
		out["w_max"] = w.max().item()
		out["w_cv"] = (w.std() / w.mean()).item()
		return out


def _self_check():
	torch.manual_seed(0)
	critic = EnsembleCritic(3, 2, 5).to(device)
	s, a = torch.randn(7, 3, device=device), torch.randn(7, 2, device=device)
	q = critic(s, a)
	assert q.shape == (5, 7)
	# each head must equal an ordinary MLP built from that head's weights
	k = 3
	h = F.relu(torch.cat([s, a], 1) @ critic.l1.weight[k] + critic.l1.bias[k])
	h = F.relu(h @ critic.l2.weight[k] + critic.l2.bias[k])
	assert torch.allclose(q[k], (h @ critic.l3.weight[k] + critic.l3.bias[k]).squeeze(-1), atol=1e-5)

	pol = TD3_BC_UW(3, 2, 1.0, n_critics=5, gamma_w=1.0, adv_tau=1.0, policy_noise=0.2, noise_clip=0.5)
	w = pol.bc_weights(s, a, pol.critic(s, pol.actor(s)))
	assert w.shape == (7,) and torch.isclose(w.mean(), torch.tensor(1.0, device=device)) and (w > 0).all()
	pol.gamma_w = pol.adv_tau = 0.0
	assert torch.allclose(pol.bc_weights(s, a, pol.critic(s, pol.actor(s))), torch.ones(7, device=device))

	class Buf:
		def sample(self, n):
			return s, a, s, torch.zeros(7, 1, device=device), torch.ones(7, 1, device=device)
	before = [p.clone() for p in pol.actor.parameters()]
	for _ in range(4):
		pol.train(Buf())
	assert any(not torch.equal(b, p) for b, p in zip(before, pol.actor.parameters())), "actor must update"
	print("TD3_BC_UW self-check passed")


if __name__ == "__main__":
	_self_check()
