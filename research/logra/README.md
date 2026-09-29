# LoGRA transfer benchmark

This isolated benchmark uses the original frozen controlled-math-v2 grader.
`historical_math_grader.py` is an unmodified copy of
`rpga/experiments/controlled_math_v2_20260914/code/third_party/molt/examples/python/utils/math_grader.py`
(SHA-256 `0b61a3258bacc70c4c698ae9276489cb63b33da5837aaeec945fecf57fdd7fde`). It retains the original Apache-2.0 header.

Run `research/logra/run_experiment.py` with a config selecting
`data.default.env_name=historical_math` and `env.historical_math.num_workers=8`.
The standard NeMo RL math environment is unchanged.

The main algorithm lives in `nemo_rl/models/automodel/logra.py` and `logra_probe.py`.
The paired benchmark uses full-weight policy synchronization, matching the old controls.
