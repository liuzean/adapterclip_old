"""Experiment-I JSON writes tolerant of short Windows sharing conflicts.

Only advisory progress may be skipped. Configurations, completion markers and
result metadata remain critical and raise if bounded retries cannot save them.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_WARNED=set()
_NEXT_PROGRESS_ATTEMPT={}
RETRY_DELAYS=(.01,.03,.06,.10,.20,.30,.50)


def _warn_once(path,error):
    key=str(path)
    if key not in _WARNED:
        _WARNED.add(key)
        print(f'进度文件暂时无法更新，继续运行；保留上次进度。{path}: {error}',file=sys.stderr,flush=True)


def write_json(path,data,*,critical=True):
    """Close a unique sibling temporary file, then atomically replace target.

    Retries are bounded (at most 1.2 seconds total delay). Windows can reject
    replacing a file while a status reader or another process has it open.
    No in-place/truncating fallback is used. Invalid data always raises.
    """
    path=Path(path)
    text=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)
    key=str(path.resolve())
    if not critical and time.monotonic()<_NEXT_PROGRESS_ATTEMPT.get(key,0):
        return False
    temp=None
    try:
        descriptor,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
        temp=Path(name)
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            stream.write(text)
        for attempt in range(len(RETRY_DELAYS)+1):
            try:
                os.replace(temp,path)
                _NEXT_PROGRESS_ATTEMPT.pop(key,None)
                return True
            except PermissionError:
                if attempt==len(RETRY_DELAYS):raise
                time.sleep(RETRY_DELAYS[attempt])
    except OSError as error:
        if critical:raise
        _NEXT_PROGRESS_ATTEMPT[key]=time.monotonic()+5
        _warn_once(path,error)
        return False
    finally:
        if temp is not None:
            try:temp.unlink(missing_ok=True)
            except OSError:pass


def write_progress(path,data):
    """A stale progress display must never turn into a failed training job."""
    return write_json(path,data,critical=False)
