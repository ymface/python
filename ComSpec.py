import struct

from PIL import Image, ImageTk
import pickle
import cv2

HEADER_SIZE = struct.calcsize("Q") 
BLOCK_SIZE = 4096
SERVER_PORT = 9999
SERVER_IP = "127.0.0.1"

OUTPUT_MS_LIMIT = int(1000 / 33)

def recv_data(sock):
    header = sock.recv(HEADER_SIZE)
    size = struct.unpack("Q", header)[0]

    block_cnt = int(size / BLOCK_SIZE)
    leftover = int(size % BLOCK_SIZE)

    data = b""
    for i in range(block_cnt):
        data += sock.recv(BLOCK_SIZE)
    data += sock.recv(leftover)

    return header, size, data

def recv_imagetk(sock):
    _, _, data = recv_data(sock)

    img = pickle.loads(data)
    
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(img)

    return imgtk

def send_data(sock, data: bytes):
    sock.sendall(struct.pack("Q", len(data)) + data)
