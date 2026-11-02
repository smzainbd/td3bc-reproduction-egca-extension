# COMP8221 Assignment 2 — Project Option 2 (Cutting-Edge Research in DRL)

**Title:** Empirical Evaluation and Coordinate-Wise Credit Attribution in Minimalist Offline Reinforcement Learning  
**Students:** S M Zain , Sudipta Saha Niloy
**Selected Paper:** *A Minimalist Approach to Offline Reinforcement Learning* (Fujimoto & Gu, NeurIPS 2021\)

---

## Folder layout

```
td3bc-reproduction-egca-extension/
├── README.md
├── requirements.txt
├── authors_code/          # original Fujimoto & Gu TD3+BC reference files
├── our_code_tinkered/     # our compatibility layer, runners, and EG-CA extension
├── notebooks/             # Section 3 and Section 4 experiment notebooks
└── figures/               # report figures
```

---

## 1\. Project report

- `2026S2_COMP8221_Assignment2_49139568_60801131_Niloy_Zain.pdf`

## 2\. `authors_code/` — original reference (Fujimoto & Gu, 2021\)

Source: [https\://github.com/sfujim/TD3\_BC](https://github.com/sfujim/TD3_BC)

- `TD3_BC.py` — reference TD3+BC actor-critic (adds `with torch.no_grad():` in `select_action()`).  
- `utils.py` — CPU NumPy replay buffer and `normalize_states()`.  
- `main.py` — entry point and `eval_policy()` (imports `d4rl_compat` from `our_code_tinkered/`).  
- `run_experiments.sh`, `LICENSE` — original repository script and license.

## 3\. `our_code_tinkered/` — our reproduction & extension code

### Section 3 (reproduction & ablations)

- `d4rl_compat.py` — D4RL v0 HDF5 loader, Gymnasium MuJoCo v4 wrapper, D4RL score normalization.  
- `train.py` — resumable training driver with GPU `DeviceBuffer` and atomic RNG checkpoints.

### Section 4 (EG-CA TD3+BC)

- `utils_new.py` — GPU-preloaded replay buffer.  
- `TD3_BC_new.py` — EG-CA (`eg_ca_td3bc`, `eg_ca_td3bc_nogate`), scalar AW (`aw_td3bc`), `shuffled_joint`, diagnostics, `_self_check()`.  
- `train_new.py` — multi-job driver for Section 4 runs.  
- `TD3_BC_UW.py` — earlier unused ensemble prototype still imported by `train.py` (not used in the final EG-CA experiments).

## 4\. `notebooks/`

- `A2_notebook1_reproduction.ipynb` — 6-task D4RL v0 reproduction \+ 2 hopper-medium-v0 ablations.  
- `A2_notebook2_extension.ipynb` — Stage 1 EG-CA comparisons, stopping rule, diagnostics.

## 5\. `figures/`

- `td3bc_flowchart.png`, `egca_architecture.png`  
- `repro_curves.png`, `repro_ablation.png`  
- `stage1_curves.png`, `stage1_diagnostics.png`

---

## How to run

Install once from the package root:

```sh
pip install -r requirements.txt
```

Put both code folders on `PYTHONPATH` (needed because `main.py` lives in `authors_code/` while `d4rl_compat.py` lives in `our_code_tinkered/`):

```sh
# Linux / macOS / Colab
export PYTHONPATH="$PWD/authors_code:$PWD/our_code_tinkered:$PYTHONPATH"

# Windows PowerShell
$env:PYTHONPATH = "$PWD\authors_code;$PWD\our_code_tinkered;$env:PYTHONPATH"
```

Self-checks:

```sh
python our_code_tinkered/d4rl_compat.py
python our_code_tinkered/TD3_BC_new.py
```

Example CLI runs (from the package root, with `PYTHONPATH` set as above):

```sh
ENV=hopper-medium-replay-v0
python our_code_tinkered/train.py --algo td3bc --env hopper-medium-v0 --seed 0
python our_code_tinkered/train_new.py --algo eg_ca_td3bc --env $ENV --seed 0
python our_code_tinkered/train_new.py --algo eg_ca_td3bc_nogate --env $ENV --seed 0
python our_code_tinkered/train_new.py --algo aw_td3bc --beta_ood 0.0 --env $ENV --seed 0
python our_code_tinkered/train_new.py --algo shuffled_joint --env $ENV --seed 0
```

### Notebooks (Colab)

1. Upload or open this whole package folder on Drive.  
2. Open `notebooks/A2_notebook1_reproduction.ipynb` then `notebooks/A2_notebook2_extension.ipynb`.  
3. In the setup cell, set `os.chdir(...)` to this package root (the folder that contains `authors_code/` and `our_code_tinkered/`).  
4. Run cells top to bottom. The setup cell adds both code folders to `sys.path` automatically.

---

## Generative AI acknowledgement

Generative AI tools (Claude, Gemini, ChatGPT) were used for polishing code, mathematical formatting, and proofreading. Both authors independently executed, verified, and critically analysed all implementations, training logs, and empirical results.
