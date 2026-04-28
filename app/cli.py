from __future__ import annotations

import argparse
import shutil
import subprocess  # nosec B404
from pathlib import Path


def _run(cmd: list[str]) -> int:
    return subprocess.call(cmd)  # nosec B603


def cmd_build() -> int:
    return _run(["python", "-m", "build"])


def cmd_start(app: str, host: str, port: int) -> int:
    return _run(["uvicorn", app, "--host", host, "--port", str(port)])


def cmd_test() -> int:
    return _run(["pytest", "-q"])


def cmd_init(name: str) -> int:
    src = Path("realms/_realm_template")
    dst = Path("realms") / name
    if not src.exists():
        print("Template not found: realms/_realm_template")
        return 1
    if dst.exists():
        print(f"Realm already exists: {dst}")
        return 1
    shutil.copytree(src, dst)
    cfg = dst / "configs" / "domain.yaml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace("your_realm_name", name), encoding="utf-8"
    )
    print(f"Created realm scaffold at {dst}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="realm", description="RealmForge CLI")
    parser.add_argument("--init", metavar="REALM_NAME", help="Create a new realm from template")
    parser.add_argument("--build", action="store_true", help="Build package artifacts")
    parser.add_argument("--start", action="store_true", help="Start default API app")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--app", default="app.serving.app:app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.init:
        return cmd_init(args.init)
    if args.build:
        return cmd_build()
    if args.start:
        return cmd_start(args.app, args.host, args.port)
    if args.test:
        return cmd_test()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
