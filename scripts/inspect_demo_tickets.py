"""Inspect the local synthetic-ticket and audit tables without exposing an API."""

import argparse
import json
from pathlib import Path

from app.tools.demo_ticket import DemoTicketService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/novatel_demo_tickets.sqlite3"))
    args = parser.parse_args()
    service = DemoTicketService(args.database)
    print(
        json.dumps(
            {"tickets": service.list_tickets(), "audit_events": service.list_audit_events()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
