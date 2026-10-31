"""Archive #86/#165/#166/#170: reset generated rotations and retained samples.

Run after stopping serving/calibration. Move the whole automatic cache aside,
including pair manifests and capture rows, so recipe migration cannot reuse it.
Model weights and explicit imported PTs are never removed. No NPU is accessed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/target.json')
    parser.add_argument('--apply', action='store_true', help='move cache to backup; default only previews')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    overrides = [key for key in ('rotation_artifact', 'rotation_k_path', 'rotation_v_path')
                 if config.get(key)]
    if os.environ.get('OSCAR_ROTATION_ARTIFACT'):
        overrides.append('OSCAR_ROTATION_ARTIFACT')
    if overrides:
        raise ValueError('Explicit rotation sources are configured: ' + ', '.join(overrides)
                         + '. Remove these overrides before resetting the automatic cache;'
                         ' supplied PT files are left untouched.')
    cache = ROOT / 'artifacts' / 'rotations'
    backups = ROOT / 'artifacts' / 'rotation_cache_backups'
    if cache.parent.is_symlink() or cache.is_symlink() or backups.is_symlink():
        raise ValueError('Expected local cache directories, not symlinks')
    if not cache.exists():
        print(f'No automatic rotation cache: {cache}')
        return 0
    if not cache.is_dir():
        raise ValueError(f'Expected a cache directory: {cache}')
    print(f'Automatic cache (all models/recipes, PTs, manifests and captures): {cache}')
    if not args.apply:
        print('Preview only. Stop serving/calibration, then rerun with --apply.')
        return 0
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    destination = backups / f'rotations-{stamp}-{os.getpid()}'
    if destination.exists():
        raise ValueError(f'Backup already exists: {destination}')
    cache.rename(destination)
    print(f'Cache reset. Backup: {destination}')
    print('Restart install_serve.sh --variant candidate to generate fresh calibration.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f'Rotation cache reset failed: {exc}', file=sys.stderr)
        sys.exit(1)
