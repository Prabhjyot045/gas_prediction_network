# Block 5: Actuator (merged into Interface block)

The actuator block has been merged into `blocks/interface/`. All vent routing and policy logic now lives in the environment interface.

See [blocks/interface/README.md](../interface/README.md) for the combined documentation.

## Files

- `controller.py` — backward-compat redirect to `blocks/interface/interface.py`
- `__init__.py` — points to interface block

## Quick Reference

```python
# Preferred:
from blocks.interface.interface import EnvironmentInterface

# Backward compat:
from blocks.actuator.controller import DamperController  # alias for EnvironmentInterface
```
