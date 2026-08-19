#!/usr/bin/env python3
"""Verify the immutable ContextWorld original-baseline result archive."""

from __future__ import annotations

import argparse
import json

from contextworld.benchmarks.original_baseline_archive import (
    audit_archived_original_baseline_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = audit_archived_original_baseline_matrix()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
