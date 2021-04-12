import os
from contextlib import contextmanager
from shutil import rmtree
from tempfile import mkdtemp
from typing import Optional


@contextmanager
def open_tmp_dir(dir: Optional[str] = None):
    tmpdir = mkdtemp()

    yield_dir = tmpdir
    if dir:
        user_dir_path = os.path.join(tmpdir, dir)
        os.mkdir(user_dir_path)
        yield_dir = user_dir_path

    yield yield_dir

    rmtree(tmpdir)
