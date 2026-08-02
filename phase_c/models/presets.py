from __future__ import annotations

from phase_c.models.config import ModelConfig


MODEL_PRESETS = {
    "debug": ModelConfig("debug", 2, 64, 1),
    "L1_H32": ModelConfig("L1_H32", 1, 32, 1),
    "L1_H64": ModelConfig("L1_H64", 1, 64, 1),
    "L1_H128": ModelConfig("L1_H128", 1, 128, 2),
    "L1_H256": ModelConfig("L1_H256", 1, 256, 4),
    "L2_H32": ModelConfig("L2_H32", 2, 32, 1),
    "L2_H64": ModelConfig("L2_H64", 2, 64, 1),
    "L2_H128": ModelConfig("L2_H128", 2, 128, 2),
    "L2_H256": ModelConfig("L2_H256", 2, 256, 4),
    "L4_H32": ModelConfig("L4_H32", 4, 32, 1),
    "L4_H64": ModelConfig("L4_H64", 4, 64, 1),
    "L4_H128": ModelConfig("L4_H128", 4, 128, 2),
    "L4_H256": ModelConfig("L4_H256", 4, 256, 4),
    "L8_H32": ModelConfig("L8_H32", 8, 32, 1),
    "L8_H64": ModelConfig("L8_H64", 8, 64, 1),
    "L8_H128": ModelConfig("L8_H128", 8, 128, 2),
    "L8_H256": ModelConfig("L8_H256", 8, 256, 4),
    "L16_H32": ModelConfig("L16_H32", 16, 32, 1),
    "L16_H64": ModelConfig("L16_H64", 16, 64, 1),
    "L16_H128": ModelConfig("L16_H128", 16, 128, 2),
    "L16_H256": ModelConfig("L16_H256", 16, 256, 4),
    "120m_legacy": ModelConfig("120m_legacy", 12, 896, 14),
}
