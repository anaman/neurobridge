"""Bridge between the TH21A headband and an LLM agent.

Public surface (frozen by docs/INTERFACES.md section 4):

* :class:`~neurobridge.agent.state.StateEstimate` - one honest snapshot.
* :class:`~neurobridge.agent.state.StateEstimator` - rolling single-channel estimator.
* :class:`~neurobridge.agent.state.IntentDetector` - blink / jaw-clench commands.
* :func:`~neurobridge.agent.bridge.create_app` - the FastAPI app.
* :func:`~neurobridge.agent.bridge.serve` - headless runner that prints its URL.
* :func:`~neurobridge.agent.bridge.state_to_agent_json` - the wire payload.

Nothing here touches hardware or the network at import time, so importing this
package is always safe; the device is only opened by ``serve()``/the app's
lifespan, in a background thread.
"""

from .bridge import create_app, serve, state_to_agent_json  # noqa: F401
from .state import IntentDetector, StateEstimate, StateEstimator  # noqa: F401

__all__ = [
    "StateEstimate",
    "StateEstimator",
    "IntentDetector",
    "create_app",
    "serve",
    "state_to_agent_json",
]
