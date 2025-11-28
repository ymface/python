# client.py
from extra import add_record_controls, handle_record_frame
from extra import compare_image_quality
from extra import compute_psnr, compute_ssim_y
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import threading
import socket
import struct
import time
import json
import os
import sys
import io
import matplotlib.pyplot as plt
import numpy as np


# -----------------------
# Configuration / Constants
# -----------------------
DEFAULT_SERVER_HOST = '192.168.56.1'  # 필요시 UI에서 변경
SERVER_PORT = 9999

HEADER_FMT = '!4sQ'
HEADER_SIZE = struct.calcsize(HEADER_FMT)

TYPE_VIDEO      = b'VID0'
TYPE_FILE_HDR   = b'FHD0'
TYPE_FILE_CHUNK = b'FCH0'
TYPE_TEXT       = b'TEX0'

TYPE_IMAGE = b'IMG0'

# -----------------------
# Helper functions
# -----------------------
def safe_send_all(sock, data: bytes) -> bool:
    try:
        sock.sendall(data)
        return True
    except Exception as e:
        print("Send failed:", e)
        return False

# -----------------------
# Main App
# -----------------------
class VideoChatClient:
    def __init__(self, root):
        self.root = root
        self.root.title("1:1 Video Chat + File Transfer + Text Chat")
        # 요구대로 전체 프레임 크기: 1200x720
        self.root.geometry("1200x720")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        # Network
        self.sock = None
        self.running = False
        self.recv_thread = None
        self.recv_lock = threading.Lock()

        # Camera
        self.cap = None
        self.sending_video = False
        self.capture_thread = None

        # Incoming file state
        self._incoming_file = None

        # UI image references (keep to avoid GC)
        self._local_img_ref = None
        self._remote_img_ref = None

        # quality / filter
        self.compression_quality = 50
        self.filter_mode = "None"

        # Build UI
        self.setup_gui()

    def setup_gui(self):
        control_frame = tk.Frame(self.root, bg="#f0f0f0", pady=8, bd=2, relief=tk.GROOVE)
        control_frame.pack(fill=tk.X)

        # Connection group
        group_conn = tk.LabelFrame(control_frame, text="Connection", padx=6, pady=6)
        group_conn.pack(side=tk.LEFT, padx=6)

        tk.Label(group_conn, text="Server IP:").pack(side=tk.LEFT)
        self.ip_entry = tk.Entry(group_conn, width=18)
        self.ip_entry.insert(0, DEFAULT_SERVER_HOST)
        self.ip_entry.pack(side=tk.LEFT, padx=6)

        tk.Button(group_conn, text="Connect", command=self.connect_server, bg="#8ee58e").pack(side=tk.LEFT, padx=6)
        tk.Button(group_conn, text="Disconnect", command=self.disconnect_server, bg="#f5a3a3").pack(side=tk.LEFT, padx=6)

       # Mode and file
        group_mode = tk.LabelFrame(control_frame, text="Source", padx=6, pady=6)
        group_mode.pack(side=tk.LEFT, padx=6)

        self.combo_mode = ttk.Combobox(
            group_mode,
            values=["Camera", "Video File", "Image File", "Audio Visualizer"],
            state="readonly", width=16
        )
        self.combo_mode.current(0)
        self.combo_mode.pack(side=tk.LEFT, padx=6)

        tk.Button(group_mode, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT, padx=6)
        tk.Button(group_mode, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT, padx=6)

        tk.Button(group_mode, text="Load File", command=self.load_file).pack(side=tk.LEFT, padx=6)
        tk.Button(group_mode, text="Send File(Q)", command=self.compress_and_send_with_quality).pack(side=tk.LEFT, padx=6)

        # ✔ 녹화 버튼은 반드시 여기!
        add_record_controls(group_mode, self)

        # Quality Test group
        group_quality = tk.LabelFrame(control_frame, text="Quality Test", padx=6, pady=6)
        group_quality.pack(side=tk.LEFT, padx=6)

        tk.Button(group_quality, text="Compare Quality",
                  command=self.open_quality_compare_window).pack(side=tk.LEFT, padx=6)

        # Filter group
        group_effect = tk.LabelFrame(control_frame, text="Quality / Filter", padx=6, pady=6)
        group_effect.pack(side=tk.LEFT, padx=6)

        tk.Label(group_effect, text="Quality:").pack(side=tk.LEFT)
        self.scale_quality = tk.Scale(group_effect, from_=10, to=95,
                                      orient=tk.HORIZONTAL, showvalue=0, length=140,
                                      command=self.update_quality)
        self.scale_quality.set(self.compression_quality)
        self.scale_quality.pack(side=tk.LEFT, padx=6)

        tk.Label(group_effect, text="Filter:").pack(side=tk.LEFT, padx=6)
        self.combo_filter = ttk.Combobox(group_effect,
                                         values=["None", "Gray", "Canny(Edge)", "Inverse", "Face Detect", "Blur"],
                                         state="readonly", width=14)
        self.combo_filter.current(0)
        self.combo_filter.pack(side=tk.LEFT, padx=6)
        self.combo_filter.bind("<<ComboboxSelected>>", self.change_filter)


        # Main area: 3분할 (left 40% / middle 40% / right 20%)
        main_frame = tk.Frame(self.root, bg="#202020")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left - Local
        left_frame = tk.Frame(main_frame, bg="black", bd=2, relief=tk.SUNKEN)
        left_frame.place(relx=0.0, rely=0.0, relwidth=0.4, relheight=1.0)
        tk.Label(left_frame, text="[ Local Source ]", fg="yellow", bg="black").pack(anchor=tk.NW)
        self.lbl_local = tk.Label(left_frame, bg="black")
        self.lbl_local.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Middle - Remote
        middle_frame = tk.Frame(main_frame, bg="black", bd=2, relief=tk.SUNKEN)
        middle_frame.place(relx=0.4, rely=0.0, relwidth=0.4, relheight=1.0)
        tk.Label(middle_frame, text="[ Remote Received ]", fg="cyan", bg="black").pack(anchor=tk.NW)
        self.lbl_remote = tk.Label(middle_frame, bg="black")
        self.lbl_remote.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Right - Chat
        right_frame = tk.Frame(main_frame, bg="#1a1a1a", bd=2, relief=tk.SUNKEN)
        right_frame.place(relx=0.8, rely=0.0, relwidth=0.2, relheight=1.0)
        tk.Label(right_frame, text="Chat", fg="white", bg="#222").pack(fill=tk.X)

        self.chat_box = tk.Text(right_frame, bg="#111", fg="white", wrap=tk.WORD, state='disabled')
        self.chat_box.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        entry_frame = tk.Frame(right_frame, bg="#222")
        entry_frame.pack(fill=tk.X, padx=6, pady=6)
        self.chat_entry = tk.Entry(entry_frame, bg="#333", fg="white")
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,6))
        tk.Button(entry_frame, text="Send", bg="#444", fg="white", command=self.send_chat).pack(side=tk.RIGHT)

        # Bottom status bar
        self.status_bar = tk.Label(self.root, text="Ready.", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#ddd", font=("Arial", 10))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # -----------------------
    # Network functions
    # -----------------------
    def connect_server(self):
        ip = self.ip_entry.get().strip()
        if not ip:
            messagebox.showwarning("Input", "서버 IP를 입력하세요.")
            return
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((ip, SERVER_PORT))
            self.sock.settimeout(None)
            self.append_chat_system(f"Connected to {ip}:{SERVER_PORT}")
            self.running = True
            # start recv thread
            self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
            self.recv_thread.start()
        except Exception as e:
            messagebox.showerror("Connect failed", str(e))
            self.sock = None

    def disconnect_server(self):
        self.append_chat_system("Disconnecting...")
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except: pass
            self.sock = None

    def send_bytes(self, ttype, data_bytes):
        if not self.sock:
            return False
        header = struct.pack(HEADER_FMT, ttype, len(data_bytes))
        return safe_send_all(self.sock, header + data_bytes)

    def recv_all(self, n):
        data = b''
        while len(data) < n:
            try:
                packet = self.sock.recv(n - len(data))
            except Exception as e:
                print("recv error:", e)
                return None
            if not packet:
                return None
            data += packet
        return data

    def recv_loop(self):
        try:
            while self.running and self.sock:
                header = self.recv_all(HEADER_SIZE)
                if not header:
                    break
                ttype, size = struct.unpack(HEADER_FMT, header)
                payload = self.recv_all(size)
                if payload is None:
                    break

                # dispatch based on type
                if ttype == TYPE_VIDEO:
                    # update remote image safely via main thread
                    self.root.after(0, self._update_remote_image, payload)
                elif ttype == TYPE_TEXT:
                    text = payload.decode('utf-8', errors='replace')
                    self.root.after(0, self.append_chat, f"Peer: {text}")
                elif ttype == TYPE_IMAGE:
                    self.root.after(0, self._update_remote_image, payload)
                elif ttype == TYPE_FILE_HDR:
                    try:
                        meta = json.loads(payload.decode('utf-8'))
                        fname = meta.get('filename', 'received.bin')
                        fsize = meta.get('filesize', 0)
                        # prepare incoming file
                        self._incoming_file = {
                            'name': fname,
                            'size': fsize,
                            'received': 0,
                            'fp': open(f"recv_{os.path.basename(fname)}", 'wb')
                        }
                        self.root.after(0, self.append_chat_system, f"Incoming file: {fname} ({fsize} bytes)")
                    except Exception as e:
                        print("file header parse error:", e)
                elif ttype == TYPE_FILE_CHUNK:
                    if self._incoming_file:
                        info = self._incoming_file
                        info['fp'].write(payload)
                        info['received'] += len(payload)

                        if info['received'] >= info['size']:
                            info['fp'].close()
                            name = info['name']

                            # ⬇⬇ 여기 추가 ⬇⬇
                            if name.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                                img = cv2.imread(f"recv_{os.path.basename(name)}")
                                self.root.after(0, self._update_remote_image, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                            self._incoming_file = None
                            self.root.after(0, self.append_chat_system,
                                f"File received: recv_{os.path.basename(name)}")
                else:
                    print("Unknown type:", ttype)
        except Exception as e:
            print("Receive loop error:", e)
        finally:
            print("Receiver exiting")
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
            self.running = False
            self.root.after(0, self.append_chat_system, "Disconnected from server")

    # -----------------------
    # Camera capture / display
    # -----------------------
    def start_camera(self):
        if self.cap:
            # 이미 열려 있으면 무시
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "카메라를 열 수 없습니다.")
            self.cap = None
            return
        self.sending_video = True
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        self.append_chat_system("Camera started")

    def stop_camera(self):
        self.sending_video = False
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

        # ✔ 내 화면에서도 꺼진 표시
        self.lbl_local.configure(image="")
        self._local_img_ref = None

        # ✔ 상대 화면에서도 꺼지게: 빈 프레임 보내기
        if self.sock:
            black = np.zeros((480, 640, 3), dtype=np.uint8)   # 검은 화면
            ok, jpg = cv2.imencode(".jpg", black, [cv2.IMWRITE_JPEG_QUALITY, 30])
            if ok:
                self.send_bytes(TYPE_VIDEO, jpg.tobytes())

        self.append_chat_system("Camera stopped")


    def capture_loop(self):
        while self.sending_video and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # apply selected filter
            frame = self.apply_filter(frame)

            # show local in UI
            try:
                self.root.after(0, self._update_local_image, frame.copy())
            except Exception:
                pass

            # encode to jpeg with slider quality
            try:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.compression_quality]
                success, jpg = cv2.imencode('.jpg', frame, encode_param)
                if success:
                    data = jpg.tobytes()
                    # send
                    if self.sock:
                        self.send_bytes(TYPE_VIDEO, data)
            except Exception as e:
                print("Encode/send error:", e)

            time.sleep(1/20)  # 20 FPS

    def _update_local_image(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((440, 560))
            imgtk = ImageTk.PhotoImage(img)
            self._local_img_ref = imgtk
            self.lbl_local.configure(image=imgtk)
        except Exception as e:
            print("show_local_frame error:", e)

        # 녹화는 여기서 실행해야 함 (try블록 바깥)
        handle_record_frame(self, frame)



    def _update_remote_image(self, jpeg_bytes):
        try:
            img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
            img = img.resize((440, 560))
            imgtk = ImageTk.PhotoImage(img)
            self._remote_img_ref = imgtk
            self.lbl_remote.configure(image=imgtk)
        except Exception as e:
            print("show_remote_frame error:", e)

    def apply_filter(self, frame):
        mode = self.combo_filter.get() if self.combo_filter else "None"
        try:
            if mode == "Gray":
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            elif mode == "Canny(Edge)":
                g = cv2.Canny(frame, 50, 150)
                return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            elif mode == "Inverse":
                return cv2.bitwise_not(frame)
            elif mode == "Blur":
                return cv2.GaussianBlur(frame, (15, 15), 0)
            elif mode == "Face Detect":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "User", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                return frame
        except Exception as e:
            print("Filter error:", e)
        return frame

    # -----------------------
    # Load File 버튼을 눌렀을 때 곧바로 품질 비교 + 압축 + 전송까지 실행
    # -----------------------
    def compress_and_send_with_quality(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.jpg *.png *.jpeg *.bmp")]
        )
        if not path:
            return

        # 1) 원본 로드
        original = cv2.imread(path)
        if original is None:
            messagebox.showerror("Error", "이미지 로드 실패")
            return

        # 2) Q 값으로 압축
        Q = self.compression_quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Q]
        ok, encoded_img = cv2.imencode(".jpg", original, encode_param)

        if not ok:
            messagebox.showerror("Error", "이미지 압축 실패")
            return

        compressed = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

        # 3) 파일 크기 계산
        original_size = os.path.getsize(path)
        compressed_size = len(encoded_img)

        # 4) PSNR / SSIM 계산
        psnr_val = compute_psnr(original, compressed)
        ssim_val = compute_ssim_y(original, compressed)

        # 5) 그래프 시각화
        fig, ax = plt.subplots()
        ax.bar(["Original", "Compressed"], [original_size, compressed_size])
        ax.set_title(f"File Size Comparison (Q={Q})\nPSNR={psnr_val:.2f} dB / SSIM={ssim_val:.4f}")
        ax.set_ylabel("Bytes")
        plt.show()

        # 6) 서버로 파일 전송
        if not self.sock:
            messagebox.showwarning("Not Connected", "서버 연결 후 다시 시도하세요.")
            return

        fname = f"compressed_Q{Q}.jpg"
        meta = {"filename": fname, "filesize": compressed_size}

        self.send_bytes(TYPE_FILE_HDR, json.dumps(meta).encode("utf-8"))
        self.send_bytes(TYPE_FILE_CHUNK, encoded_img.tobytes())

        self.append_chat_system(
            f"[전송 완료] {fname}\n"
            f"원본: {original_size} bytes → 압축: {compressed_size} bytes\n"
            f"PSNR={psnr_val:.2f} dB / SSIM={ssim_val:.4f}"
        )
    def load_file(self):
        """
        이미지/동영상 파일을 불러오고,
        현재 Quality(Q) 값으로 압축 → 품질 비교 → 그래프 출력 → 서버로 전송까지 수행.
        """

        mode = self.combo_mode.get()

        # -------------------------------
        # 1) 이미지 품질 비교 + 압축 전송
        # -------------------------------
        if mode == "Image File":

            # 파일 선택
            path = filedialog.askopenfilename(
            filetypes=[("Image", "*.jpg *.png *.bmp *.jpeg")]
            )
            if not path:
                return

            # 원본 이미지 로딩
            original = cv2.imread(path)
            if original is None:
                messagebox.showerror("Error", "이미지 로드 실패")
                return

            # 이미지 미리보기
            self.root.after(0, self._update_local_image, original.copy())

            # Q 값으로 JPEG 압축
            Q = self.compression_quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Q]
            ok, encoded_img = cv2.imencode(".jpg", original, encode_param)
            if not ok:
                messagebox.showerror("Error", "이미지 압축 실패")
                return

            compressed = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

            # 파일 크기 비교
            original_size = os.path.getsize(path)
            compressed_size = len(encoded_img)

            # PSNR / SSIM 계산
            psnr_val = compute_psnr(original, compressed)
            ssim_val = compute_ssim_y(original, compressed)

            # 그래프 시각화
            fig, ax = plt.subplots()
            ax.bar(["Original", "Compressed"], [original_size, compressed_size])
            ax.set_title(
            f"File Size Comparison (Q={Q})\n"
            f"PSNR={psnr_val:.2f} dB / SSIM={ssim_val:.4f}"
            )
            ax.set_ylabel("Bytes")
            plt.show()

            # 서버로 전송
            if not self.sock:
                messagebox.showwarning("Not Connected", "서버에 연결 후 다시 시도")
                return

            fname = f"compressed_Q{Q}.jpg"
            meta = {"filename": fname, "filesize": compressed_size}

            self.send_bytes(TYPE_IMAGE, encoded_img.tobytes())

            self.append_chat_system(
                f"이미지(Q={Q}) 압축·전송 완료\n"
                f"원본 {original_size} bytes → 압축 {compressed_size} bytes\n"
                f"PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.4f}"
            )

            return

        # -------------------------------
        # 2) 동영상 ZIP 압축 + 전송
        # -------------------------------
        elif mode == "Video File":

            # 파일 선택
            path = filedialog.askopenfilename(
                filetypes=[("Video", "*.mp4 *.avi *.mkv")]
            )
            if not path:
                return

            # zip 압축 import (extra.py)
            from extra import zip_compress_file

            # ZIP 압축 생성
            zip_path = zip_compress_file(path)

            size = os.path.getsize(zip_path)

            # 서버로 메타데이터 전송
            meta = {"filename": os.path.basename(zip_path), "filesize": size}
            self.send_bytes(TYPE_FILE_HDR, json.dumps(meta).encode("utf-8"))

            # ZIP 파일 내용 전송
            with open(zip_path, "rb") as f:
                self.send_bytes(TYPE_FILE_CHUNK, f.read())

            # 채팅창 출력
            self.append_chat_system(
                f"동영상 ZIP 압축 후 전송 완료: {os.path.basename(zip_path)} ({size} bytes)"
            )

            return

        # -------------------------------
        # 3) 잘못된 모드
        # -------------------------------
        else:
            messagebox.showinfo("Load File", "Image File 또는 Video File 모드를 선택하세요.")

    def select_and_send_file(self):
        if not self.sock:
            messagebox.showwarning("Not connected", "서버에 연결 후 시도하세요.")
            return
        path = filedialog.askopenfilename()
        if not path:
            return
        try:
            filesize = os.path.getsize(path)
            meta = {'filename': os.path.basename(path), 'filesize': filesize}
            # send metadata header
            self.send_bytes(TYPE_FILE_HDR, json.dumps(meta).encode('utf-8'))
            # send in chunks
            CHUNK = 32 * 1024
            with open(path, 'rb') as f:
                while True:
                    chunk = f.read(CHUNK)
                    if not chunk:
                        break
                    self.send_bytes(TYPE_FILE_CHUNK, chunk)
                    # tiny sleep to avoid hogging network
                    time.sleep(0.001)
            self.append_chat_system(f"File sent: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("File send failed", str(e))

    def open_quality_compare_window(self):
        # 새 창 열기
        win = tk.Toplevel(self.root)
        win.title("PSNR / SSIM 품질 비교")
        win.geometry("420x200")

        # 파일 선택 안내
        tk.Label(win, text="원본 이미지 선택").pack()
        orig_btn = tk.Button(win, text="원본 선택", command=lambda: self._select_orig(win))
        orig_btn.pack(pady=4)

        tk.Label(win, text="비교 대상 이미지 선택").pack()
        cmp_btn = tk.Button(win, text="전송 후 이미지 선택", command=lambda: self._select_cmp(win))
        cmp_btn.pack(pady=4)

        # 결과 표시창
        self.quality_result = tk.Label(win, text="\n결과가 여기에 표시됩니다.")
        self.quality_result.pack(pady=6)

        # 실행 버튼
        run_btn = tk.Button(win, text="비교 실행", command=self._run_quality_compare)
        run_btn.pack(pady=6)

        # 내부 용 변수
        self._orig_image_path = None
        self._cmp_image_path = None

    def _select_orig(self, win):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.png *.jpg *.jpeg")])
        if path:
            self._orig_image_path = path
            self.quality_result.config(text=f"원본: {os.path.basename(path)} 선택됨")

    def _select_cmp(self, win):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.png *.jpg *.jpeg")])
        if path:
            self._cmp_image_path = path
            self.quality_result.config(text=f"비교 대상: {os.path.basename(path)} 선택됨")

    def _run_quality_compare(self):
        if not self._orig_image_path or not self._cmp_image_path:
            messagebox.showerror("Error", "원본과 비교 이미지를 모두 선택하세요.")
            return

        try:
            psnr, ssim = compare_image_quality(self._orig_image_path, self._cmp_image_path)
            self.quality_result.config(
                text=f"PSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"품질 비교 중 오류 발생: {e}")


    # -----------------------
    # Chat functions
    # -----------------------
    def append_chat(self, text: str):
        # always call from main thread; if called from worker thread, wrap with after
        try:
            self.chat_box.configure(state='normal')
            self.chat_box.insert(tk.END, text + '\n')
            self.chat_box.configure(state='disabled')
            self.chat_box.see(tk.END)
        except Exception as e:
            print("append_chat error:", e)

    def append_chat_system(self, text: str):
        self.append_chat("[SYSTEM] " + text)

    def send_chat(self):
        text = self.chat_entry.get().strip()
        if not text:
            return
        self.append_chat(f"You: {text}")
        self.chat_entry.delete(0, tk.END)
        if self.sock:
            try:
                self.send_bytes(TYPE_TEXT, text.encode('utf-8'))
            except Exception as e:
                print("Chat send error:", e)
                self.append_chat_system("Failed to send chat")

    # -----------------------
    # UI callbacks
    # -----------------------
    def update_quality(self, val):
        try:
            self.compression_quality = int(val)
            self.append_chat_system(f"Quality set to {self.compression_quality}")
        except:
            pass

    def change_filter(self, event):
        self.filter_mode = self.combo_filter.get()

    def on_mode_change(self, event):
        # Just indicate change; actual actions are via Load/Start
        self.append_chat_system(f"Mode changed to {self.combo_mode.get()}")

    # -----------------------
    # Cleanup / Close
    # -----------------------
    def close(self):
        # stop everything and quit
        self.append_chat_system("Closing application...")
        self.running = False
        self.sending_video = False
        try:
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=0.5)
        except:
            pass
        try:
            if self.recv_thread and self.recv_thread.is_alive():
                self.recv_thread.join(timeout=0.5)
        except:
            pass
        try:
            if self.cap:
                self.cap.release()
        except:
            pass
        try:
            if self.sock:
                self.sock.close()
        except:
            pass
        try:
            self.root.destroy()
        except:
            pass
        sys.exit(0)


if __name__ == '__main__':
    root = tk.Tk()
    app = VideoChatClient(root)
    root.mainloop()
