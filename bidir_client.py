# 성공
from ComSpec import *

import tkinter as tk
from tkinter import Tk, Label, Button
import cv2
from PIL import Image, ImageTk
import socket
import threading

root = Tk()
root.title("Simple video streamming")
root.minsize(800, 600)

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture("resources/sample_1.mp4")

cap_lbl = Label(root)
decoded_lbl = Label(root)

buttonA = Button(root, text="Takeoff", command=lambda: None)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def recv_stream():
    global root
    global decoded_lbl

    while True:
        imgtk = recv_imagetk(client_socket)

        decoded_lbl.imgtk = imgtk
        decoded_lbl.configure(image=imgtk)

recv_thread = threading.Thread(target=recv_stream, args=(), daemon=True)

def run_app():
    global root
    global cap
    global cap_lbl
    global decoded_lbl
    global buttonA

    try:
        buttonA.pack(side='bottom', pady=10)
        
        cap_lbl.pack(anchor="center", side=tk.LEFT, pady=15)
        decoded_lbl.pack(anchor="center", side=tk.LEFT, pady=15)

        client_socket.connect((SERVER_IP, SERVER_PORT))
        recv_thread.start()
        video_stream()

        root.mainloop()

    except Exception as e:
        print(f"!!! Error on running app: {e}")
    finally:
        cleanup()



def video_stream():
    global root
    global cap
    global cap_lbl
    global buttonA
    global client_socket

    # cam read
    ret, frame = cap.read()

    if not ret: # local video stream if cam fails
        if cap2.get(cv2.CAP_PROP_POS_FRAMES) == cap2.get(cv2.CAP_PROP_FRAME_COUNT):
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap2.read()

    cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2img = cv2.resize(cv2img, (640, 480))

    ok, encoded = cv2.imencode('.jpg', cv2img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])

    img = Image.fromarray(cv2img)
    imgtk = ImageTk.PhotoImage(image=img)

    cap_lbl.imgtk = imgtk
    cap_lbl.configure(image=imgtk)

    send_data(client_socket, pickle.dumps(encoded))

    root.after(OUTPUT_MS_LIMIT, video_stream)

def cleanup() -> None:
    global root
    global cap
    global cap_lbl
    global decoded_lbl
    global buttonA

    try:
        print("Cleanning up...")
        cap.release()
        root.quit()
        exit()
    except Exception as e:
        print(f"!!! Error on cleanup: {e}")

run_app()
