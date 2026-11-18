import json
import os

import numpy as np
import pytest
import torch

from research import m2b_io as io_


def test_durable_write_leaves_no_tmp_and_roundtrips(tmp_path):
    io_.durable_write_json(tmp_path / "a" / "x.json", {"k": 1})
    io_.durable_write_bytes(tmp_path / "a" / "b.bin", b"abc")
    io_.durable_write_npz(tmp_path / "a" / "p.npz", ids=np.arange(3), y=np.arange(3))
    io_.durable_write_torch(tmp_path / "a" / "t.pt", {"w": torch.ones(2)})
    assert json.loads((tmp_path / "a" / "x.json").read_text()) == {"k": 1}
    assert (tmp_path / "a" / "b.bin").read_bytes() == b"abc"
    assert list(np.load(tmp_path / "a" / "p.npz")["ids"]) == [0, 1, 2]
    assert torch.load(tmp_path / "a" / "t.pt")["w"].tolist() == [1.0, 1.0]
    assert not list((tmp_path / "a").glob("*.tmp"))


def test_crash_before_rename_keeps_old_file(tmp_path, monkeypatch):
    target = tmp_path / "x.json"
    io_.durable_write_json(target, {"v": 1})

    def boom(*a, **k):
        raise OSError("simulated crash before rename")
    monkeypatch.setattr(io_.os, "replace", boom)
    with pytest.raises(OSError):
        io_.durable_write_json(target, {"v": 2})
    monkeypatch.undo()
    assert json.loads(target.read_text()) == {"v": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_writer_exception_cleans_temp(tmp_path):
    def bad(stream):
        stream.write(b"partial")
        raise ValueError("x")
    with pytest.raises(ValueError):
        io_._durable(tmp_path / "y.bin", bad)
    assert not (tmp_path / "y.bin").exists() and not list(tmp_path.glob("*.tmp"))


def test_lock_blocks_second_holder_and_cleans_up(tmp_path):
    lock = tmp_path / "locks" / "u.lock"
    with io_.unit_lock(lock):
        assert lock.exists()
        with pytest.raises(io_.LockHeldError):
            with io_.unit_lock(lock):
                pass
        assert lock.exists()  # the failed second attempt did not remove the first lock
    assert not lock.exists()


def test_dead_pid_lock_is_stale_and_renamed(tmp_path):
    lock = tmp_path / "u.lock"
    lock.write_text(json.dumps({"pid": 2 ** 22 + 12345, "host": io_.socket.gethostname(), "start": "x", "create_time": 1.0}))
    with io_.unit_lock(lock):
        pass
    assert len(list(tmp_path.glob("u.lock.stale-*"))) == 1


def test_reused_pid_with_different_create_time_is_stale(tmp_path):
    lock = tmp_path / "u.lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "host": io_.socket.gethostname(), "start": "x",
                                "create_time": io_._pid_start(os.getpid()) - 100.0}))
    with io_.unit_lock(lock):
        pass
    assert len(list(tmp_path.glob("u.lock.stale-*"))) == 1


def test_create_time_within_tolerance_is_live(tmp_path):
    lock = tmp_path / "u.lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "host": io_.socket.gethostname(), "start": "x",
                                "create_time": io_._pid_start(os.getpid()) + 0.5}))
    with pytest.raises(io_.LockHeldError):
        with io_.unit_lock(lock):
            pass


def test_archive_move_renames_and_never_deletes(tmp_path):
    f = tmp_path / "unit.json"
    f.write_text("a")
    d = tmp_path / "unitdir"
    d.mkdir()
    (d / "inner.txt").write_text("b")
    first = io_.archive_move(f, tmp_path / "archive", "tag1")
    second = io_.archive_move(d, tmp_path / "archive", "tag1")
    assert not f.exists() and not d.exists()
    assert first.read_text() == "a" and (second / "inner.txt").read_text() == "b"
    f.write_text("again")
    third = io_.archive_move(f, tmp_path / "archive", "tag1")
    assert third != first and first.read_text() == "a" and third.read_text() == "again"


def test_sha256_file(tmp_path):
    (tmp_path / "f").write_bytes(b"abc")
    assert io_.sha256_file(tmp_path / "f") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
