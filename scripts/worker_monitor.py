#!/usr/bin/env python3
"""
Worker monitoring system for parallel agent development.

Creates structured log files and status tracking for worker agents
working on GitHub issues via git worktrees.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from enum import Enum


class WorkerStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkerState:
    """State of a worker agent."""
    worker_id: str
    issue_number: int
    issue_title: str
    branch_name: str
    worktree_path: str
    status: str
    started_at: str
    updated_at: str
    completed_at: Optional[str] = None
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    error_message: Optional[str] = None
    current_phase: Optional[str] = None
    progress_notes: list = None

    def __post_init__(self):
        if self.progress_notes is None:
            self.progress_notes = []


class WorkerMonitor:
    """Manages worker status and logging."""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.environ.get('FRESH_AIR_ROOT', '/Users/bedwards/fresh-air')
        self.base_dir = Path(base_dir)
        self.workers_dir = self.base_dir / '.workers'
        self.workers_dir.mkdir(exist_ok=True)
        self.status_file = self.workers_dir / 'status.json'
        self.load_status()

    def load_status(self) -> dict:
        """Load current worker status from file."""
        if self.status_file.exists():
            with open(self.status_file) as f:
                self.status = json.load(f)
        else:
            self.status = {'workers': {}, 'history': []}
        return self.status

    def save_status(self):
        """Save current worker status to file."""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2, default=str)

    def register_worker(
        self,
        worker_id: str,
        issue_number: int,
        issue_title: str,
        branch_name: str,
        worktree_path: str
    ) -> WorkerState:
        """Register a new worker."""
        now = datetime.now().isoformat()
        state = WorkerState(
            worker_id=worker_id,
            issue_number=issue_number,
            issue_title=issue_title,
            branch_name=branch_name,
            worktree_path=worktree_path,
            status=WorkerStatus.RUNNING.value,
            started_at=now,
            updated_at=now,
            current_phase="initializing"
        )
        self.status['workers'][worker_id] = asdict(state)
        self.save_status()

        # Create worker-specific log file
        log_file = self.workers_dir / f'{worker_id}.log'
        self._log(worker_id, f"Worker started for Issue #{issue_number}: {issue_title}")
        self._log(worker_id, f"Branch: {branch_name}")
        self._log(worker_id, f"Worktree: {worktree_path}")

        return state

    def update_worker(
        self,
        worker_id: str,
        status: Optional[WorkerStatus] = None,
        phase: Optional[str] = None,
        note: Optional[str] = None,
        pr_number: Optional[int] = None,
        pr_url: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Update worker state."""
        if worker_id not in self.status['workers']:
            raise ValueError(f"Unknown worker: {worker_id}")

        worker = self.status['workers'][worker_id]
        now = datetime.now().isoformat()
        worker['updated_at'] = now

        if status:
            worker['status'] = status.value
            if status in (WorkerStatus.COMPLETED, WorkerStatus.FAILED, WorkerStatus.CANCELLED):
                worker['completed_at'] = now

        if phase:
            worker['current_phase'] = phase
            self._log(worker_id, f"Phase: {phase}")

        if note:
            worker['progress_notes'].append({'time': now, 'note': note})
            self._log(worker_id, note)

        if pr_number:
            worker['pr_number'] = pr_number

        if pr_url:
            worker['pr_url'] = pr_url
            self._log(worker_id, f"PR created: {pr_url}")

        if error:
            worker['error_message'] = error
            self._log(worker_id, f"ERROR: {error}")

        self.save_status()

    def complete_worker(self, worker_id: str, pr_number: int = None, pr_url: str = None):
        """Mark worker as completed."""
        self.update_worker(
            worker_id,
            status=WorkerStatus.COMPLETED,
            phase="completed",
            pr_number=pr_number,
            pr_url=pr_url,
            note="Worker completed successfully"
        )

        # Move to history
        worker = self.status['workers'].pop(worker_id)
        self.status['history'].append(worker)
        self.save_status()

    def fail_worker(self, worker_id: str, error: str):
        """Mark worker as failed."""
        self.update_worker(
            worker_id,
            status=WorkerStatus.FAILED,
            phase="failed",
            error=error
        )

        # Move to history
        worker = self.status['workers'].pop(worker_id)
        self.status['history'].append(worker)
        self.save_status()

    def _log(self, worker_id: str, message: str):
        """Write to worker-specific log file."""
        log_file = self.workers_dir / f'{worker_id}.log'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

    def get_active_workers(self) -> list:
        """Get list of currently active workers."""
        return [
            w for w in self.status['workers'].values()
            if w['status'] == WorkerStatus.RUNNING.value
        ]

    def get_worker_log(self, worker_id: str) -> str:
        """Read worker log file."""
        log_file = self.workers_dir / f'{worker_id}.log'
        if log_file.exists():
            return log_file.read_text()
        return ""

    def print_status(self):
        """Print current worker status summary."""
        active = self.get_active_workers()
        print(f"\n{'='*60}")
        print(f"WORKER STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        if not active:
            print("No active workers")
        else:
            print(f"Active workers: {len(active)}\n")
            for w in active:
                print(f"  [{w['worker_id'][:8]}] Issue #{w['issue_number']}: {w['issue_title'][:40]}")
                print(f"    Status: {w['status']} | Phase: {w['current_phase']}")
                print(f"    Started: {w['started_at']}")
                if w['progress_notes']:
                    latest = w['progress_notes'][-1]
                    print(f"    Latest: {latest['note'][:50]}...")
                print()

        completed = [h for h in self.status['history'] if h['status'] == 'completed']
        failed = [h for h in self.status['history'] if h['status'] == 'failed']

        if completed:
            print(f"Completed: {len(completed)}")
            for w in completed[-3:]:  # Show last 3
                print(f"  - Issue #{w['issue_number']}: {w['issue_title'][:40]}")
                if w.get('pr_url'):
                    print(f"    PR: {w['pr_url']}")

        if failed:
            print(f"\nFailed: {len(failed)}")
            for w in failed[-3:]:
                print(f"  - Issue #{w['issue_number']}: {w['error_message'][:50]}")

        print(f"{'='*60}\n")


def main():
    """CLI interface for worker monitor."""
    import argparse
    parser = argparse.ArgumentParser(description='Worker monitoring system')
    parser.add_argument('command', choices=['status', 'log', 'list'],
                       help='Command to run')
    parser.add_argument('--worker', '-w', help='Worker ID for log command')
    args = parser.parse_args()

    monitor = WorkerMonitor()

    if args.command == 'status':
        monitor.print_status()
    elif args.command == 'list':
        print(json.dumps(monitor.status, indent=2))
    elif args.command == 'log':
        if not args.worker:
            print("Error: --worker required for log command")
            sys.exit(1)
        print(monitor.get_worker_log(args.worker))


if __name__ == '__main__':
    main()
