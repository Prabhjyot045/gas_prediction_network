# Actuator block merged into blocks/interface/.
# See blocks/interface/interface.py for EnvironmentInterface (replaces DamperController).
#
# Import redirect for backward compatibility:
from blocks.interface.interface import EnvironmentInterface as DamperController  # noqa: F401
from blocks.interface.interface import VentAction as DamperAction  # noqa: F401
