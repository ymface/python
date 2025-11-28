# server.py
import socket
import threading
import struct
import sys

HOST = socket.gethostbyname(socket.gethostname())
PORT = 9999

HEADER_FMT = '!4sQ'  # 4-byte type, 8-byte uint64 size
HEADER_SIZE = struct.calcsize(HEADER_FMT)

TYPE_IMAGE = b'IMG0'

class RelayServer:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.sock = None
        self.clients = []  # list of (conn, addr)
        self.lock = threading.Lock()

    def start(self):
        print(f"[Server] Starting relay on {self.host}:{self.port}")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(2)
        threading.Thread(target=self.accept_loop, daemon=True).start()
        print("[Server] Ready. Waiting for up to 2 clients...")

    def accept_loop(self):
        while True:
            conn, addr = self.sock.accept()
            with self.lock:
                if len(self.clients) >= 2:
                    print(f"[Server] Rejecting extra client {addr}")
                    conn.sendall(b'REJ')
                    conn.close()
                    continue
                self.clients.append((conn, addr))
                print(f"[Server] Client connected: {addr} (total {len(self.clients)})")
                threading.Thread(target=self.client_loop, args=(conn, addr), daemon=True).start()

    def remove_client(self, conn):
        with self.lock:
            self.clients = [c for c in self.clients if c[0] != conn]
            try:
                conn.close()
            except:
                pass
            print(f"[Server] Client removed. Remaining: {len(self.clients)}")

    def get_peer(self, conn):
        with self.lock:
            peers = [c for c in self.clients if c[0] != conn]
            return peers[0][0] if peers else None

    def client_loop(self, conn, addr):
        try:
            while True:
                header = self.recv_all(conn, HEADER_SIZE)
                if not header:
                    break
                msg_type, size = struct.unpack(HEADER_FMT, header)
                # receive payload
                payload = self.recv_all(conn, size)
                if payload is None:
                    break
                # forward to peer if exists
                peer = self.get_peer(conn)
                if peer:
                    try:
                        peer.sendall(header + payload)
                    except Exception as e:
                        print(f"[Server] Forward error to peer: {e}")
                else:
                    # no peer: optionally buffer or ignore
                    print("[Server] No peer yet; dropping message")
        except Exception as e:
            print(f"[Server] Client loop error {addr}: {e}")
        finally:
            print(f"[Server] Client disconnected: {addr}")
            self.remove_client(conn)

    def recv_all(self, conn, n):
        data = b''
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

if __name__ == '__main__':
    server = RelayServer()
    server.start()
    try:
        while True:
            cmd = input()
            if cmd.lower() in ('q','quit','exit'):
                print("[Server] Shutting down")
                break
    except KeyboardInterrupt:
        print("[Server] KeyboardInterrupt, exiting")
    finally:
        if server.sock:
            server.sock.close()
        sys.exit(0)
