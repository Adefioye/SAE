from __future__ import annotations

import logging
import time

from sae_seed_similarity.utils import monitored_operation


def test_monitored_operation_logs_heartbeat_and_completion(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="sae_seed_similarity"):
        with monitored_operation("synthetic slow step", heartbeat_seconds=0.01):
            time.sleep(0.04)

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        message.startswith("Started: synthetic slow step") for message in messages
    )
    assert any(
        message.startswith("Still running: synthetic slow step")
        and "peak_ram=" in message
        and "avg_cpu=" in message
        for message in messages
    )
    assert any(
        message.startswith("Completed: synthetic slow step")
        and "elapsed=" in message
        for message in messages
    )
