"""Contract cho bảng Health Index trên train FD001."""

from __future__ import annotations


HEALTH_INDEX_COLUMNS = ("split", "engine_id", "cycle", "health_index")
SELECTED_ENGINE_COLUMNS = ("split", "engine_id", "fold", "n_cycles", "selection_rank")
