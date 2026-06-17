from __future__ import annotations

import sys

from scripts.prepare_dataset import main as prepare_main


def main() -> None:
    """Convenience entrypoint.

    Equivalent command:
    `python scripts/prepare_dataset.py --label-type hard`
    """

    if "--label-type" not in sys.argv:
        sys.argv.extend(["--label-type", "hard"])
    prepare_main()


if __name__ == "__main__":
    main()
