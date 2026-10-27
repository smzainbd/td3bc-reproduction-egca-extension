# =============================================================================
# COMP8221 Assignment 2 — Student Work for Section 4 (GPU-Preloaded ReplayBuffer)
# Authors: Sudipta Saha Niloy (49139568), S M Zain (60801131)
# Role: Preloads D4RL transitions directly onto CUDA tensors and performs in-place
# device standardization for Notebook 2 (Section 4).
# =============================================================================
import numpy as np
import torch


class ReplayBuffer(object):
	"""GPU-preloaded ReplayBuffer for offline RL (D4RL).

	Loads all dataset arrays directly into device (VRAM) tensors in convert_D4RL()
	to eliminate per-step CPU-to-GPU transfer overhead during training.
	normalize_states() normalizes state and next_state tensors in-place on the GPU,
	returning (mean, std) as CPU NumPy arrays for compatibility with eval_policy().
	"""

	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.state_dim = state_dim
		self.action_dim = action_dim

		# "cuda" and "cuda:0" are the same GPU but do not compare equal, so pin the index.
		if torch.cuda.is_available():
			self.device = torch.device("cuda", torch.cuda.current_device())
		else:
			self.device = torch.device("cpu")

		self.state = None
		self.action = None
		self.next_state = None
		self.reward = None
		self.not_done = None

	def _ensure_allocated(self):
		if self.state is None:
			self.state = torch.zeros((self.max_size, self.state_dim), dtype=torch.float32, device=self.device)
			self.action = torch.zeros((self.max_size, self.action_dim), dtype=torch.float32, device=self.device)
			self.next_state = torch.zeros((self.max_size, self.state_dim), dtype=torch.float32, device=self.device)
			self.reward = torch.zeros((self.max_size, 1), dtype=torch.float32, device=self.device)
			self.not_done = torch.zeros((self.max_size, 1), dtype=torch.float32, device=self.device)

	def add(self, state, action, next_state, reward, done):
		self._ensure_allocated()
		self.state[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
		self.action[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
		self.next_state[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
		self.reward[self.ptr] = float(reward)
		self.not_done[self.ptr] = 1.0 - float(done)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, size=(batch_size,), device=self.device)
		return (
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind],
		)

	def convert_D4RL(self, dataset):
		self.state = torch.tensor(dataset["observations"], dtype=torch.float32, device=self.device)
		self.action = torch.tensor(dataset["actions"], dtype=torch.float32, device=self.device)
		self.next_state = torch.tensor(dataset["next_observations"], dtype=torch.float32, device=self.device)
		self.reward = torch.tensor(dataset["rewards"].reshape(-1, 1), dtype=torch.float32, device=self.device)
		self.not_done = torch.tensor(1.0 - dataset["terminals"].reshape(-1, 1), dtype=torch.float32, device=self.device)
		self.size = self.state.shape[0]
		self.ptr = self.size % self.max_size

	def normalize_states(self, eps=1e-3):
		mean = self.state.mean(0, keepdim=True)
		std = self.state.std(0, keepdim=True, unbiased=False) + eps
		self.state.sub_(mean).div_(std)
		self.next_state.sub_(mean).div_(std)
		return mean.cpu().numpy(), std.cpu().numpy()


def _self_check():
	obs = np.random.randn(32, 11).astype(np.float32)
	act = np.random.randn(32, 3).astype(np.float32)
	nxt = np.random.randn(32, 11).astype(np.float32)
	rew = np.random.randn(32).astype(np.float32)
	term = np.random.randint(0, 2, size=(32,)).astype(bool)

	rb = ReplayBuffer(11, 3, max_size=64)
	rb.convert_D4RL({
		"observations": obs,
		"actions": act,
		"next_observations": nxt,
		"rewards": rew,
		"terminals": term,
	})
	assert rb.state.device == rb.device
	mean, std = rb.normalize_states()
	assert isinstance(mean, np.ndarray) and isinstance(std, np.ndarray)
	assert mean.shape == (1, 11) and std.shape == (1, 11)
	batch = rb.sample(8)
	assert len(batch) == 5 and all(t.device == rb.device for t in batch)
	assert batch[0].shape == (8, 11) and batch[1].shape == (8, 3) and batch[3].shape == (8, 1)
	print("utils_new self-check passed")


if __name__ == "__main__":
	_self_check()
