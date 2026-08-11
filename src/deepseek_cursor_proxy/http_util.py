"""HTTP helpers for reliable upstream connections on Windows DNS."""

from __future__ import annotations

import http.client
import socket
from typing import Any
from urllib.request import HTTPHandler, HTTPSHandler, Request, build_opener


def create_connection_prefer_ipv4(
    address: tuple[str, int],
    timeout: float | None = socket._GLOBAL_DEFAULT_TIMEOUT,  # noqa: SLF001
    source_address: tuple[str, int] | None = None,
) -> socket.socket:
    """Like socket.create_connection, but try IPv4 before dual-stack.

    On some Windows setups, getaddrinfo(host, …, AF_UNSPEC) fails for hosts
    that only have A records behind a CNAME (notably api.deepseek.com), while
    an explicit AF_INET lookup succeeds.
    """
    host, port = address
    errors: list[OSError] = []
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            addrinfo = socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)
        except OSError as exc:
            errors.append(exc)
            continue
        for af, socktype, proto, _canon, sockaddr in addrinfo:
            sock = socket.socket(af, socktype, proto)
            try:
                if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:  # noqa: SLF001
                    sock.settimeout(timeout)
                if source_address is not None:
                    sock.bind(source_address)
                sock.connect(sockaddr)
                return sock
            except OSError as exc:
                errors.append(exc)
                sock.close()
    if errors:
        raise errors[-1]
    raise socket.gaierror(socket.EAI_NONAME, f"getaddrinfo failed for {host}")


class PreferIPv4HTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = create_connection_prefer_ipv4


class PreferIPv4HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = create_connection_prefer_ipv4


class PreferIPv4HTTPHandler(HTTPHandler):
    def http_open(self, req: Request):  # type: ignore[override]
        return self.do_open(PreferIPv4HTTPConnection, req)


class PreferIPv4HTTPSHandler(HTTPSHandler):
    def https_open(self, req: Request):  # type: ignore[override]
        return self.do_open(PreferIPv4HTTPSConnection, req)


_UPSTREAM_OPENER = build_opener(PreferIPv4HTTPHandler, PreferIPv4HTTPSHandler)


def upstream_urlopen(request: Request, timeout: float | None = None):
    """urlopen that prefers IPv4 when resolving upstream hosts."""
    return _UPSTREAM_OPENER.open(request, timeout=timeout)
