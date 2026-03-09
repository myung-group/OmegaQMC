# vmc_pgcs/config.py

import warnings

import jax
jax.config.update("jax_enable_x64", True)

# Whether to collect all many-body configurations from the random walks
COLLECT_CONFIGS = False


def _custom_formatwarning(msg, category, filename, lineno, line=None):
    # return f"⚠️\t{filename}:{lineno}: {msg}\n"
    return f"⚠️\t{msg}\n"


warnings.formatwarning = _custom_formatwarning
