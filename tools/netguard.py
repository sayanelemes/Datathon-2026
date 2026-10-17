"""Доказательство локальности: блокировка сети в процессе анализа + наблюдение за соединениями vLLM/сервиса.

1. install()            — в ЭТОМ процессе любое соединение и DNS-запрос не на localhost запрещены и записываются.
2. Sampler              — пока идёт анализ, раз в 0,3 с читает /proc/net/tcp(6) и записывает исходящие соединения
                          процессов vLLM / Ollama / веб-сервиса на внешние адреса.
3. listen_addr(port)    — на каком адресе слушает порт (vLLM должен слушать только 127.0.0.1).
4. proc_env(name, key)  — переменная окружения процесса (HF_HUB_OFFLINE у vLLM).
"""
import ipaddress
import os
import socket
import threading
import time
import traceback

BLOCKED = []                      # [(адрес, где в коде)]
PROXIES = {}                      # снятые переменные прокси: через localhost-прокси трафик мог бы уйти наружу
_orig = {}


def _local(host):
    if host is None:
        return True
    host = str(host).strip("[]").lower()
    if host in ("localhost", "localhost.localdomain", ""):
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def _where():
    for fr in reversed(traceback.extract_stack()[:-3]):
        if "netguard" not in fr.filename:
            return f"{os.path.basename(fr.filename)}:{fr.lineno}"
    return "?"


def install():
    if _orig:
        return
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if os.environ.get(k):
            PROXIES[k] = os.environ.pop(k)

    def connect(self, address, *a):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _local(address[0]):
            BLOCKED.append((f"{address[0]}:{address[1]}", _where()))
            raise OSError(101, f"netguard: внешнее соединение запрещено ({address[0]})")
        return _orig["connect"](self, address, *a)

    def connect_ex(self, address, *a):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _local(address[0]):
            BLOCKED.append((f"{address[0]}:{address[1]}", _where()))
            return 101
        return _orig["connect_ex"](self, address, *a)

    def getaddrinfo(host, *a, **k):
        if not _local(host if not isinstance(host, bytes) else host.decode()):
            BLOCKED.append((f"DNS {host}", _where()))
            raise socket.gaierror(-2, f"netguard: DNS-запрос запрещён ({host})")
        return _orig["getaddrinfo"](host, *a, **k)

    _orig.update(connect=socket.socket.connect, connect_ex=socket.socket.connect_ex, getaddrinfo=socket.getaddrinfo)
    socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, getaddrinfo


# ---------------- /proc ----------------
def _hex_addr(s):
    ip, port = s.split(":")
    port = int(port, 16)
    if len(ip) == 8:
        addr = socket.inet_ntop(socket.AF_INET, bytes.fromhex(ip)[::-1])
    else:
        b = bytes.fromhex(ip)
        addr = socket.inet_ntop(socket.AF_INET6, b"".join(b[i:i + 4][::-1] for i in range(0, 16, 4)))
    return addr, port


def _sockets():
    """[(local, lport, remote, rport, state, inode)]"""
    out = []
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = open(f).read().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            p = ln.split()
            try:
                (la, lp), (ra, rp) = _hex_addr(p[1]), _hex_addr(p[2])
            except Exception:  # noqa
                continue
            out.append((la, lp, ra, rp, p[3], p[9]))
    return out


def _inode_pids():
    m = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                link = os.readlink(f"/proc/{pid}/fd/{fd}")
                if link.startswith("socket:["):
                    m[link[8:-1]] = int(pid)
        except OSError:
            continue
    return m


def cmdline(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


WATCH = ("vllm", "uvicorn", "ollama", "selftest", "app:app", "enginecore")


def _ppid(pid):
    try:
        return int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return 0


def watched(pid, me):
    """Процесс — наш (vLLM, его дочерние EngineCore/воркеры, сервис, сам selftest)?"""
    for _ in range(12):
        if pid in (0, 1):
            return False
        if pid == me or any(w in cmdline(pid).lower() for w in WATCH):
            return True
        pid = _ppid(pid)
    return False
SERVER_PORTS = {8000, 8001, 8888, 11434, 22}       # входящие подключения к нам (жюри, браузер, Jupyter) — не исходящие


def _ext(addr):
    try:
        ip = ipaddress.ip_address(addr)
        return not (ip.is_loopback or ip.is_unspecified)
    except ValueError:
        return False


class Sampler(threading.Thread):
    def __init__(self, every=0.3):
        super().__init__(daemon=True)
        self.every, self.stop_ev, self.found, self.samples = every, threading.Event(), {}, 0

    def run(self):
        me = os.getpid()
        while not self.stop_ev.is_set():
            socks = [s for s in _sockets() if s[4] in ("01", "02") and _ext(s[2]) and s[1] not in SERVER_PORTS]
            if socks:
                ip = _inode_pids()
                for la, lp, ra, rp, st, ino in socks:
                    pid = ip.get(ino)
                    if pid is None:
                        continue
                    if watched(pid, me):
                        self.found.setdefault(f"{ra}:{rp}", cmdline(pid)[:80])
            self.samples += 1
            self.stop_ev.wait(self.every)

    def stop(self):
        self.stop_ev.set()
        self.join(2)


def listen_addr(port):
    return sorted({la for la, lp, ra, rp, st, ino in _sockets() if st == "0A" and lp == port})


def proc_env(name, key):
    for pid in os.listdir("/proc"):
        if pid.isdigit() and name in cmdline(pid).lower():
            try:
                env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
            except OSError:
                continue
            for e in env:
                if e.startswith(key.encode() + b"="):
                    return e.split(b"=", 1)[1].decode()
            return "(не задана)"
    return None


def port_pid(port):
    """PID процесса, который слушает порт (vLLM — 8001)."""
    inodes = {ino for la, lp, ra, rp, st, ino in _sockets() if st == "0A" and lp == port}
    m = _inode_pids()
    for ino in inodes:
        if ino in m:
            return m[ino]
    return None


def pid_env(pid, key):
    try:
        env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
    except OSError:
        return None
    for e in env:
        if e.startswith(key.encode() + b"="):
            return e.split(b"=", 1)[1].decode()
    return "(не задана)"
