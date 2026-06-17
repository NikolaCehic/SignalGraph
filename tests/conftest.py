from __future__ import annotations

import pytest

from signalgraph.config import ProjectPaths
from signalgraph.graph import GraphBuilder
from signalgraph.sample_data import ensure_sample_corpus


@pytest.fixture
def sample_project(tmp_path):
    paths = ProjectPaths(tmp_path)
    ensure_sample_corpus(paths, force=True)
    graph = GraphBuilder(paths).build(use_sample_if_empty=False)
    return paths, graph
