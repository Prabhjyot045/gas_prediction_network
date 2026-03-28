# Block 3: Network (merged into Sensor block)

The network block has been merged into `blocks/sensor/`. All topology, placement, and graph functionality now lives alongside the inference code.

See [blocks/sensor/README.md](../sensor/README.md) for the combined documentation.

## Files

- `sensor_network.py` — backward-compat redirect to `blocks/sensor/sensor_network.py`
- `placement.py` — backward-compat redirect to `blocks/sensor/placement.py`
- `test_network.py` — topology tests (imports from `blocks/sensor/`)

## Quick Reference

```python
# Both work:
from blocks.sensor import SensorNetwork
from blocks.network import SensorNetwork  # backward compat
```
