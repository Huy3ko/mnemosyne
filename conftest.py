"""
Session-wide environment isolation for the repository test suite.

The deployment exports environment variables that describe the RUNNING
system: the data directory and the bank map.  A test suite must not inherit
them.  Two concrete consequences made that a correctness problem, not a
tidiness one:

* ``MNEMOSYNE_BANKS_MAP`` (and the map inside the data directory) turned
  the per-chat bank routing ON.  Tests that expect a write to land in the
  private profile storage instead saw it land in a routed bank -- and the
  routed bank was a bank of the PRODUCTION deployment, so the test wrote
  into production data.
* ``MNEMOSYNE_DATA_DIR`` pointed at the production data directory, so a
  test that does not pin its own directory wrote into production storage.

Both are neutralized here, before any test module is imported.  Tests that
want a specific value set it themselves (``monkeypatch.setenv``), which
overrides whatever this module does.

The data directory is redirected to a per-session temporary directory rather
than removed: the suite resolves a default data directory from ``HOME``, and
the real home directory is not a place a test suite should write to either.
"""

import os
import tempfile

# The routing map: a deployment-exported path must not decide how the
# repository's own tests route.  A missing map means routing is off, which
# is the default behaviour the upstream tests were written against.
os.environ.pop("MNEMOSYNE_BANKS_MAP", None)

# The data directory: point it at a throwaway location instead of the
# deployment's.  Popping it would send the suite to the real HOME.
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="mnemosyne-test-data-")
os.environ["MNEMOSYNE_DATA_DIR"] = _TEST_DATA_DIR
