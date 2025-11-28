import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import socket
import threading
import struct
import pickle
import time
import os
import math

# ==========================================
#  설정 및 상수 (Configuration)
# ==========================================
SERVER_IP = '127.0.0.1'  # 기본 아이피 (로컬호스트)
SERVER_PORT = 9999            # 포트 번호
HEADER_SIZE = struct.calcsize("Q")
BLOCK_SIZE = 4096

class VideoNetProUltimate:
    def __init__(self, root):
        self.root = root
        self.root.title("VideoNet Pro Ultimate - Multi-Mode Edition")
        self.root.geometry("1100x800")
        
        # --- 상태 변수 ---
        self.cap = None           # 비디오 캡처 객체
        self.static_image = None  # 이미지 모드용 변수
        self.audio_sim_step = 0   # 오디오 시각화용 스텝
        
        self.is_running = False
        self.is_sending = False
        self.client_socket = None
        
        # 모드 관리
        self.mode = "CAMERA" # CAMERA, VIDEO, IMAGE, AUDIO_VIS
        self.file_path = None
        
        # 효과 및 압축
        self.filter_mode = "None"
        self.compression_quality = 50 
        
        # 통계
        self.total_sent_bytes = 0
        self.start_time = time.time()

        # --- GUI 초기화 ---
        self.setup_gui()

    def setup_gui(self):
        # 1. 컨트롤 패널 (상단)
        control_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10, bd=2, relief=tk.GROOVE)
        control_frame.pack(fill=tk.X)

        # (1) 접속 설정
        group_conn = tk.LabelFrame(control_frame, text="1. Connection", padx=5, pady=5)
        group_conn.pack(side=tk.LEFT, padx=5)
        
        tk.Label(group_conn, text="Target IP:").pack(side=tk.LEFT)
        self.ip_entry = tk.Entry(group_conn, width=12)
        self.ip_entry.insert(0, SERVER_IP)
        self.ip_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Button(group_conn, text="Connect(Tx)", command=self.connect_to_server, bg="#ccffcc").pack(side=tk.LEFT, padx=2)

        # (2) 소스 모드 선택 (T4, T5)
        group_mode = tk.LabelFrame(control_frame, text="2. Source Mode (T4/T5)", padx=5, pady=5)
        group_mode.pack(side=tk.LEFT, padx=5)
        
        self.combo_mode = ttk.Combobox(group_mode, values=["Camera", "Video File", "Image File", "Audio Visualizer"], state="readonly", width=13)
        self.combo_mode.current(0)
        self.combo_mode.bind("<<ComboboxSelected>>", self.change_mode)
        self.combo_mode.pack(side=tk.LEFT)
        
        tk.Button(group_mode, text="Load File", command=self.load_file).pack(side=tk.LEFT, padx=5)

        # (3) 품질 및 효과 (T2, T3, T6)
        group_effect = tk.LabelFrame(control_frame, text="3. Controls (T2/T3/T6)", padx=5, pady=5)
        group_effect.pack(side=tk.LEFT, padx=5)
        
        tk.Label(group_effect, text="Quality:").pack(side=tk.LEFT)
        self.scale_quality = tk.Scale(group_effect, from_=5, to=100, orient=tk.HORIZONTAL, showvalue=0, command=self.update_quality)
        self.scale_quality.set(50)
        self.scale_quality.pack(side=tk.LEFT)
        
        tk.Label(group_effect, text="Effect:").pack(side=tk.LEFT, padx=5)
        self.combo_filter = ttk.Combobox(group_effect, values=["None", "Gray", "Canny(Edge)", "Inverse", "Face Detect", "Blur"], state="readonly", width=10)
        self.combo_filter.current(0)
        self.combo_filter.bind("<<ComboboxSelected>>", self.change_filter)
        self.combo_filter.pack(side=tk.LEFT)

        # 2. 비디오 화면 (중앙) - 2분할 (Local / Remote)
        video_frame = tk.Frame(self.root, bg="#202020")
        video_frame.pack(fill=tk.BOTH, expand=True)

        # Local View
        self.frame_local = tk.Frame(video_frame, bg="black", bd=2, relief=tk.RIDGE)
        self.frame_local.place(relx=0.0, rely=0.0, relwidth=0.5, relheight=1.0)
        tk.Label(self.frame_local, text="[Local Source]", fg="yellow", bg="black").pack(anchor=tk.NW)
        self.lbl_local = tk.Label(self.frame_local, bg="black")
        self.lbl_local.pack(expand=True)

        # Remote View
        self.frame_remote = tk.Frame(video_frame, bg="black", bd=2, relief=tk.RIDGE)
        self.frame_remote.place(relx=0.5, rely=0.0, relwidth=0.5, relheight=1.0)
        tk.Label(self.frame_remote, text="[Remote Received]", fg="cyan", bg="black").pack(anchor=tk.NW)
        self.lbl_remote = tk.Label(self.frame_remote, bg="black")
        self.lbl_remote.pack(expand=True)

        # 3. 정보창 (하단) - 전송률 모니터링 (T3)
        self.status_bar = tk.Label(self.root, text="System Ready.", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#ddd", font=("Arial", 10))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ==========================================
    #  로직 구현 (Logic)
    # ==========================================

    def update_quality(self, val):
        self.compression_quality = int(val)

    def change_filter(self, event):
        self.filter_mode = self.combo_filter.get()

    def change_mode(self, event):
        selection = self.combo_mode.get()
        if selection == "Camera":
            self.mode = "CAMERA"
            self.cap = cv2.VideoCapture(0)
        elif selection == "Video File":
            self.mode = "VIDEO"
            messagebox.showinfo("Info", "Please click 'Load File' to select a video.")
        elif selection == "Image File":
            self.mode = "IMAGE"
            messagebox.showinfo("Info", "Please click 'Load File' to select an image.")
        elif selection == "Audio Visualizer":
            self.mode = "AUDIO_VIS"
            # 오디오 파일 로드 대신, 간단히 시각화 모드로 진입 (파일 로드도 가능하게 확장 가능)
            # 여기서는 T4의 '음성 파일 전송'을 '음성 파형의 시각적 전송'으로 해석하여 구현
            pass

    def load_file(self):
        if self.mode == "VIDEO":
            path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mkv")])
            if path:
                self.file_path = path
                self.cap = cv2.VideoCapture(path)
                self.status_bar.config(text=f"Loaded Video: {os.path.basename(path)}")
        elif self.mode == "IMAGE":
            path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.bmp")])
            if path:
                self.file_path = path
                # 이미지는 한 번 읽어서 메모리에 둠
                img = cv2.imread(path)
                self.static_image = cv2.resize(img, (320, 240))
                self.status_bar.config(text=f"Loaded Image: {os.path.basename(path)}")
        elif self.mode == "AUDIO_VIS":
            messagebox.showinfo("Audio", "Audio Visualizer Mode Enabled. (Generating Synthetic Waves)")

    # --- 그래픽 효과 생성 (T6) ---
    def apply_effects(self, frame):
        if frame is None: return None
        
        # T6: 다양한 효과 적용
        if self.filter_mode == "Gray":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif self.filter_mode == "Canny(Edge)":
            frame = cv2.Canny(frame, 50, 150)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif self.filter_mode == "Inverse":
            frame = cv2.bitwise_not(frame)
        elif self.filter_mode == "Blur":
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
        elif self.filter_mode == "Face Detect":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "User", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    # --- 오디오 시각화 생성기 (T4: 음성 모드 시각화) ---
    def generate_audio_visualizer(self):
        # 640x480 검은 배경 생성
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 가상의 파형 생성 (오디오 파일을 직접 분석하려면 librosa 등이 필요하지만, 여기선 시뮬레이션)
        self.audio_sim_step += 0.2
        points = []
        for x in range(640):
            y = int(240 + 100 * math.sin(x * 0.02 + self.audio_sim_step) * math.cos(x * 0.01))
            points.append((x, y))
        
        # 파형 그리기
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
        
        # 텍스트
        cv2.putText(frame, "Audio Visualizer Mode", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 시각화 바 (Equalizer 효과)
        for i in range(10):
            h = int(abs(math.sin(self.audio_sim_step + i)) * 150)
            cv2.rectangle(frame, (50 + i*50, 400), (80 + i*50, 400-h), (0, 100+h, 255), -1)
            
        return frame

    # --- 메인 비디오 루프 (소스 획득 및 처리) ---
    def start_local_loop(self):
        self.is_running = True
        self.process_stream()
        self.recv_stream()

    def process_stream(self):
        if not self.is_running: return

        frame = None
        
        # 1. 소스에서 프레임 획득
        if self.mode == "CAMERA" or self.mode == "VIDEO":
            if self.cap and self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if ret:
                    frame = raw_frame
                    # 동영상 반복 재생
                    if self.mode == "VIDEO" and self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    # 카메라 읽기 실패 시
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "No Signal", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        elif self.mode == "IMAGE":
            if self.static_image is not None:
                frame = self.static_image.copy() # 정지 영상
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No Image Loaded", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        elif self.mode == "AUDIO_VIS":
            frame = self.generate_audio_visualizer()

        # 2. 전처리 및 효과 적용
        if frame is not None:
            # 리사이징 (전송 효율)
            frame = cv2.resize(frame, (640, 480)) 
            # 효과 적용
            frame = self.apply_effects(frame)

            # 3. 로컬 화면 표시 (Tkinter)
            self.display_frame(frame, self.lbl_local)

            # 4. 네트워크 전송 (T3: 압축 및 전송)
            if self.is_sending and self.client_socket:
                self.send_packet(frame)

        # 약 30 FPS
        self.root.after(33, self.process_stream)

    def display_frame(self, frame, label_widget):
        # BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        
        # 화면 크기에 맞춰 썸네일 조정
        img.thumbnail((500, 400)) 
        imgtk = ImageTk.PhotoImage(image=img)
        label_widget.imgtk = imgtk
        label_widget.configure(image=imgtk)

    # --- 네트워크 전송 (송신) ---
    def send_packet(self, frame):
        try:
            # [압축] JPEG 인코딩 (H.263 I-frame 유사) - T2
            # Quality 값에 따라 압축률 변동 (T3 실험 요소)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.compression_quality]
            result, encoded_img = cv2.imencode('.jpg', frame, encode_param)
            
            data = pickle.dumps(encoded_img)
            size = len(data)
            
            # [전송] 헤더(크기) + 데이터
            self.client_socket.sendall(struct.pack("Q", size) + data)
            
            # 전송 통계 업데이트 (T3)
            self.total_sent_bytes += size
            elapsed = time.time() - self.start_time
            kbps = (self.total_sent_bytes / 1024) / elapsed if elapsed > 0 else 0
            
            self.status_bar.config(text=f"[TX] Packet: {size/1024:.1f}KB | Speed: {kbps:.1f} KB/s | Quality: {self.compression_quality}")
            
        except Exception as e:
            print(f"Send Error: {e}")
            self.is_sending = False
            self.client_socket.close()
    
    # remote 수신
    def recv_stream(self):
        try:
            header = self.client_socket.recv(HEADER_SIZE)
            
            data_size = struct.unpack("Q", header)[0]
            if data_size == 0: return
            
            block_cnt = int(data_size / BLOCK_SIZE)
            leftover = data_size % BLOCK_SIZE
            data = b""
            for i in range(block_cnt):
                data += self.client_socket.recv(BLOCK_SIZE)
            data += self.client_socket.recv(leftover)
            
            # while len(data) < data_size:
            #     data += self.client_socket.recv(BLOCK_SIZE)
            
            data = data[:data_size]
            
            # [복원] JPEG 디코딩
            encoded= pickle.loads(data)
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            
            # 원격 화면 표시
            # 메인 스레드가 아니므로 invoke 사용 권장하지만 간단한 데모라 직접 호출
            if img is not None:
                self.display_frame(img, self.lbl_remote)
            self.root.after(5, self.recv_stream)
        except Exception as e:
            print(f"Rx Error: {e}")
            
    # --- 클라이언트 접속 ---
    def connect_to_server(self):
        ip = self.ip_entry.get()
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((ip, SERVER_PORT))
            self.is_sending = True
            self.start_time = time.time()
            self.total_sent_bytes = 0
            
            # 송신 시작 시 로컬 루프도 시작
            if self.mode == "CAMERA":
                self.cap = cv2.VideoCapture(0)
            self.start_local_loop()
            
            messagebox.showinfo("Success", f"Connected to {ip}")
        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {e}")

    def cleanup(self):
        self.is_running = False
        self.is_sending = False
        if self.cap: self.cap.release()
        if self.client_socket: self.client_socket.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoNetProUltimate(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()
