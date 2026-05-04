"""
models/pomdp_device_agent.py  —  Device-level POMDP Gating Agent

Semantically identical to BeliefStatePOMDPAgent in pomdp_sensor_agent.py,
but the `num_sensors` dimension represents **devices** (groups of sensors)
rather than individual sensors.

Gate semantics
--------------
  p_hard[b, d]  ∈ {0, 1}  — whether device d is active for sample b
  Expansion to channels is done externally by the trainer using
  `channels_per_device`, e.g. [6, 6] for two 6-channel devices.

This module simply re-exports BeliefStatePOMDPAgent under the alias
BeliefStatePOMDPDeviceAgent to make call-site intent clear without
duplicating architecture code.
"""

from models.pomdp_sensor_agent import (  # noqa: F401
    AgentOutput,
    BeliefStatePOMDPAgent as BeliefStatePOMDPDeviceAgent,
    RegretTracker,
)

__all__ = [
    "BeliefStatePOMDPDeviceAgent",
    "AgentOutput",
    "RegretTracker",
]
