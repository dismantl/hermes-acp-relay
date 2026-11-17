#!/bin/sh
# Install the relay and bridge into the hermes-agent image's venv the way the
# production relay image does (--no-deps, so the image's hermes-agent and ACP
# SDK stay as shipped), add the test tools, and run the suite.
set -eu

python=/opt/hermes/.venv/bin/python
# The image ships uv through its package manager, off PATH.
uv=$("$python" -c 'import pm; print(pm.uv_launcher("uv"))')

"$uv" pip install --python "$python" --no-deps ./relay ./bridge
"$uv" pip install --python "$python" "pytest>=8" "pytest-asyncio>=0.23"
exec "$python" -m pytest "$@"
