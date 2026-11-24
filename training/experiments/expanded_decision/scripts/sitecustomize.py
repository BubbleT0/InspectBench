"""Experiment-local workaround for broken hard-lock release on the login host."""

import filelock


# Cache export is single-process at dataset-builder initialization. The installed
# UnixFileLock can lose its fd before release; SoftFileLock preserves exclusion
# without touching the shared environment or frozen training dependencies.
filelock.FileLock = filelock.SoftFileLock
