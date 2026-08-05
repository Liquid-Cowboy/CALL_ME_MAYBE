import json
from pathlib import Path


def dump_json(entry: list, output: str | None) -> None:
    if not output:
        return

    path = Path(output)
    path.parent.mkdir(exist_ok=True, parents=True)

    with path.open('w+', encoding='utf-8') as f:
        json.dump(entry, f, indent=4)
