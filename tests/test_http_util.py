from __future__ import annotations

import socket
import unittest
from unittest.mock import patch

from deepseek_cursor_proxy.http_util import create_connection_prefer_ipv4


class CreateConnectionPreferIPv4Tests(unittest.TestCase):
    def test_uses_ipv4_when_unspec_would_fail(self) -> None:
        """Reproduce the Windows dual-stack failure: AF_UNSPEC fails, AF_INET works."""
        ipv4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.10", 443))

        def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            if family in (0, socket.AF_UNSPEC):
                raise socket.gaierror(socket.EAI_NONAME, "getaddrinfo failed")
            if family == socket.AF_INET:
                return [ipv4]
            raise socket.gaierror(socket.EAI_NONAME, "no ipv6")

        created: list[tuple] = []

        class FakeSocket:
            def __init__(self, af, socktype, proto):
                created.append((af, socktype, proto))

            def settimeout(self, _timeout):
                return None

            def connect(self, address):
                self.peer = address

            def close(self):
                return None

        with (
            patch("socket.getaddrinfo", side_effect=fake_getaddrinfo),
            patch("socket.socket", FakeSocket),
        ):
            sock = create_connection_prefer_ipv4(("api.deepseek.com", 443), timeout=5)

        self.assertEqual(created, [(socket.AF_INET, socket.SOCK_STREAM, 6)])
        self.assertEqual(sock.peer, ("203.0.113.10", 443))


if __name__ == "__main__":
    unittest.main()
