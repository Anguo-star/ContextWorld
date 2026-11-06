#!/usr/bin/env python3
"""Start one explicitly named long-running worker in its own session."""
import argparse
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid-file', type=Path, required=True)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command and Path(command[0]).is_absolute()
    if args.pid_file.exists():
        pid = int(args.pid_file.read_text())
        try:
            state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        except FileNotFoundError:
            state = None
        if state not in {None, 'Z', 'X'}:
            raise SystemExit(f'Worker {pid} already exists; not starting a duplicate')
    child = os.fork()
    if child:
        os.waitpid(child, 0)
        for _ in range(100):
            if args.pid_file.exists():
                print(f'Started worker PID {args.pid_file.read_text().strip()}', flush=True)
                return
            time.sleep(.05)
        raise SystemExit('Worker did not write its PID file')
    os.setsid()
    if os.fork():
        os._exit(0)
    with open(os.devnull, 'rb') as stream:
        os.dup2(stream.fileno(), 0)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open('ab', buffering=0) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
    args.pid_file.write_text(str(os.getpid()) + '\n')
    os.execv(command[0], command)


if __name__ == '__main__':
    main()
