# vmc_pgcs/config.py

import jax
jax.config.update("jax_enable_x64", True)

# Whether to collect all many-body configurations from the random walks
COLLECT_CONFIGS = False
