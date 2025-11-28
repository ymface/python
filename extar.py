# extra.py
"""
VideoNet 확장 기능 모듈

- 녹화 기능(웹캠/전송 중 영상 로컬 저장)
- Tkinter용 녹화 버튼 생성 헬퍼
- 1:1 통신 서버 실행 헬퍼
- 파일 압축 + 품질 비교 (PSNR/SSIM) 함수
"""

import os
import time
import math
import zipfile

import cv2
import numpy as np

try:
    import tkinter as tk
except ImportError:
    # 서버 쪽에서는 tkinter 없을 수도 있으니 무시
    tk = None


# ===============================
# 1. 녹화 관련 (웹캠 → 파일 저장)
# ===============================

class VideoRecorder:
    """
    간단한 비디오 녹화기.
    - 첫 프레임이 들어올 때 자동으로 VideoWriter 생성
    - app.capture_loop에서 매 프레임마다 write() 호출해 주면 됨
    """
    def __init__(self, out_dir="records", fps=20):
        self.out_dir = out_dir
        self.fps = fps
        self.writer = None
        self.recording = False
        self.frame_size = None

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

    def _create_writer(self, frame):
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.out_dir, f"record_{timestamp}.avi")
        self.writer = cv2.VideoWriter(filename, fourcc, self.fps, self.frame_size)
        print("[Recorder] Start recording to:", filename)
        return filename

    def start(self):
        self.recording = True
        print("[Recorder] Recording flag ON")

    def stop(self):
        self.recording = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print("[Recorder] Recording stopped / writer released")

    def write(self, frame):
        """
        capture_loop에서 매 프레임마다 호출.
        start()가 켜져 있을 때만 기록.
        """
        if not self.recording:
            return

        if self.writer is None:
            # 첫 프레임 들어오면 writer 생성
            self._create_writer(frame)

        self.writer.write(frame)


def add_record_controls(parent_frame, app):
    """
    client.VideoChatClient.setup_gui() 안에서 호출해서
    'Start Rec', 'Stop Rec' 버튼을 추가하는 헬퍼.

    parent_frame: Tkinter Frame (예: group_mode)
    app: VideoChatClient 인스턴스
    """
    if tk is None:
        return

    # VideoRecorder 인스턴스를 app에 달아둔다.
    if not hasattr(app, "_recorder"):
        app._recorder = VideoRecorder()

    def on_start():
        app.append_chat_system("Recording started")
        app._recorder.start()

    def on_stop():
        app._recorder.stop()
        app.append_chat_system("Recording stopped")

    tk.Button(parent_frame, text="Start Rec",
              command=on_start, bg="#ffd27f").pack(side="left", padx=4)
    tk.Button(parent_frame, text="Stop Rec",
              command=on_stop, bg="#ff9f9f").pack(side="left", padx=4)


def handle_record_frame(app, frame):
    """
    client.capture_loop() 안에서 프레임을 로컬 저장하고 싶을 때 호출.
    """
    rec = getattr(app, "_recorder", None)
    if rec is not None:
        rec.write(frame)


# =========================================
# 2. 1대1 통신 서버 실행 헬퍼 (RelayServer 래핑)
# =========================================

def run_relay_server(host="0.0.0.0", port=9999):
    """
    server.py의 RelayServer를 import해서 실행하는 헬퍼.
    python -m extra 처럼 단독 실행도 가능하도록 설계.
    """
    from server import RelayServer  # 순환 import 방지 위해 함수 안에서 import

    server = RelayServer(host=host, port=port)
    server.start()
    print(f"[Extra] Relay server running on {host}:{port}. Type Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Extra] Server stopped by KeyboardInterrupt.")


# ======================================
# 3. 파일 압축 + 품질 비교 도우미 함수들
#    (T3/T4/T5 리포트용)
# ======================================

def zip_compress_file(in_path, out_zip=None):
    """
    일반 파일을 zip으로 압축.
    - in_path: 원본 파일 경로
    - out_zip: 결과 zip 이름 (None이면 자동 생성)
    return: out_zip 경로
    """
    if out_zip is None:
        base = os.path.basename(in_path)
        out_zip = f"{base}.zip"

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(in_path, arcname=os.path.basename(in_path))
    print(f"[Extra] Compressed {in_path} -> {out_zip}")
    return out_zip


# ---------- PSNR / SSIM (이미지 품질) ----------

def compute_psnr(img_ref, img_cmp):
    diff = img_ref.astype(np.float64) - img_cmp.astype(np.float64)
    mse = np.mean(diff ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)


def bgr2y(img):
    B = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    R = img[:, :, 2].astype(np.float64)
    return 0.299 * R + 0.587 * G + 0.114 * B


def compute_ssim_y(img1, img2, ksize=11, sigma=1.5):
    """
    간단한 Y채널 기반 SSIM 구현 (과제용)
    """
    import cv2

    Y1 = bgr2y(img1)
    Y2 = bgr2y(img2)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(Y1, (ksize, ksize), sigma)
    mu2 = cv2.GaussianBlur(Y2, (ksize, ksize), sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(Y1 * Y1, (ksize, ksize), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(Y2 * Y2, (ksize, ksize), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(Y1 * Y2, (ksize, ksize), sigma) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def compare_image_quality(orig_path, cmp_path):
    """
    원본 이미지와 압축/전송 후 이미지를 읽어
    PSNR / SSIM 값을 리턴.

    return: (psnr, ssim)
    """
    img1 = cv2.imread(orig_path)
    img2 = cv2.imread(cmp_path)

    if img1 is None or img2 is None:
        raise ValueError("이미지 로딩 실패")

    # 크기 다르면 리사이즈
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    psnr = compute_psnr(img1, img2)
    ssim = compute_ssim_y(img1, img2)
    print(f"[Extra] Quality {os.path.basename(orig_path)} vs {os.path.basename(cmp_path)}")
    print(f"        PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")
    return psnr, ssim


if __name__ == "__main__":
    # python extra.py 로 실행하면 서버 테스트용으로 동작하게 해둠
    run_relay_server()
