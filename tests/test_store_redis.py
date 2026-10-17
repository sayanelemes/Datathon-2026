"""Очередь Redis: задача, поставленная API, выполняется отдельным воркером; прерванная — возобновляется.
Нужен redis-server (пропускается, если его нет)."""
import os
import shutil
import socket
import subprocess
import time

import pytest

redis = pytest.importorskip("redis")
pytestmark = pytest.mark.skipif(not shutil.which("redis-server"), reason="нет redis-server")


@pytest.fixture()
def redis_url(tmp_path):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    p = subprocess.Popen(["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"redis://127.0.0.1:{port}/0"
    for _ in range(50):
        try:
            redis.Redis.from_url(url).ping()
            break
        except Exception:
            time.sleep(0.1)
    yield url
    p.terminate()


def test_redis_queue_roundtrip(tmp_path, redis_url):
    import store
    st = store.RedisStore(str(tmp_path / "jobs"), redis_url, "t1")
    vid = tmp_path / "a.mov"
    vid.write_bytes(b"1")
    st.create("abcd1234", str(vid), time.time(), "в очереди")
    assert st.qsize() == 1 and st.get("abcd1234")["status"] == "processing"
    assert st.dequeue(1) == "abcd1234"
    st.save_result({"job_id": "abcd1234", "status": "done", "verdict": "no_violation", "violations": []})
    assert st.get("abcd1234")["verdict"] == "no_violation"
    st.patch_result("abcd1234", annotated_video_url="/results/abcd1234_annotated.mp4")
    assert st.get("abcd1234")["annotated_video_url"].endswith(".mp4")


def test_redis_resume_after_crash(tmp_path, redis_url):
    import store
    st = store.RedisStore(str(tmp_path / "jobs"), redis_url, "t2")
    vid = tmp_path / "b.mov"
    vid.write_bytes(b"1")
    st.create("feed0001", str(vid), time.time(), "в очереди")
    assert st.dequeue(1) == "feed0001"                       # воркер взял задачу и «упал»
    st.update("feed0001", _worker_ts=time.time() - 300)
    assert st.resume() == 1 and st.qsize() == 1
    assert st.resume() == 0                                   # повторно не ставится
    st2 = store.RedisStore(str(tmp_path / "jobs"), redis_url, "t2")
    assert st2.dequeue(1) == "feed0001"
    st2.update("feed0001", _worker_ts=time.time())           # живой воркер — не трогаем
    assert st2.resume() == 0
