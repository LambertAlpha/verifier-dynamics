"""Shared helpers for the E005b-0 scripts: config checks, thread setup, logging, memory."""

import json
import resource
import socket
import sys
from pathlib import Path
from typing import Any

import torch

from vdyn import provenance
from vdyn.e005b import model as mdl
from vdyn.e005b import task as tk

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "e005b" / "pilot.toml"


def load_config() -> dict[str, Any]:
    cfg = provenance.load_config(CONFIG)
    m = cfg["model"]
    c = mdl.GPTConfig()
    assert (m["layers"], m["d"], m["heads"], m["mlp"], m["ctx"]) == (
        c.layers,
        c.d,
        c.heads,
        c.mlp,
        c.ctx,
    )
    assert cfg["grpo"]["max_new"] == tk.MAX_NEW and cfg["device"] == "cpu"
    assert (cfg["data"]["dev"], cfg["data"]["test"]) == (tk.DEV_MIN, tk.TEST_MIN)
    torch.set_num_threads(cfg["threads"])
    return cfg


def run_extra(cfg: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return {"host": socket.gethostname(), "device": cfg["device"],
            "threads": torch.get_num_threads(), "argv": sys.argv, **kw}  # fmt: skip


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # macOS: bytes


class JsonlLog:
    def __init__(self, path: Path) -> None:
        self.f = path.open("a")

    def write(self, rec: dict[str, Any]) -> None:
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()
