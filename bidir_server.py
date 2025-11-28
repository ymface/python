from ComSpec import *
import socket
import threading

import tkinter as tk

class Server:
    def __init__(self):
        self.server_ip = socket.gethostbyname(socket.gethostname())
        self.server_thread = None
        self.client_0_thread = None
        self.client_1_thread = None
        self.client_0_socket = None
        self.client_1_socket = None

        self.root = None
        self.client_0_lbl = None
        self.client_1_lbl = None
        self.frame_cnt_0 = 0
        self.frame_cnt_1 = 0

    def setup_gui(self):
        self.root = tk.Tk()
        frame_0 = tk.Frame(self.root, pady=20)
        frame_0.pack(side=tk.TOP)
        tk.Label(frame_0, text="Client 0: ").pack(side=tk.LEFT)
        self.client_0_lbl = tk.Label(frame_0, text="None")
        self.client_0_lbl.pack(side=tk.LEFT)

        frame_1 = tk.Frame(self.root, pady = 20)
        frame_1.pack(side=tk.TOP)
        tk.Label(frame_1, text="Client 1:").pack(side=tk.LEFT)
        self.client_1_lbl = tk.Label(frame_1, text="None")
        self.client_1_lbl.pack(side=tk.LEFT)
    
    def start(self):
        self.setup_gui()
        self.server_thread = threading.Thread(target=self.server_listener, args=(), daemon=True)
        self.server_thread.start()

        self.root.mainloop()

    def wait_for_end(self):
        while True: continue
    
    def server_listener(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.server_ip, SERVER_PORT))
        server_socket.listen(2)
        print("# server start on {}:{}".format(self.server_ip, SERVER_PORT))

        while True:
            if (self.client_0_thread != None) and (self.client_1_thread != None): continue

            client_socket, addr = server_socket.accept()
            
            if not self.client_0_thread:
                self.client_0_socket = client_socket
                self.client_0_thread = threading.Thread(target=self.client_listener, args=(client_socket, 0), daemon=True)
                self.client_0_thread.start()
                print("# client 0 accepted")
                continue
            if not self.client_1_thread:
                self.client_1_socket = client_socket
                self.client_1_thread = threading.Thread(target=self.client_listener, args=(client_socket, 1), daemon=True)
                self.client_1_thread.start()
                print("# client 1 accepted")
                continue

            client_socket.close()
    
    def client_listener(self, client_socket, client_number):
        while True:
            header, size, data = recv_data(client_socket)
            full_data = header + data

            self.dispatch_to_others(full_data, client_number)
            
            if client_number == 0:
                self.frame_cnt_0 += 1
                self.client_0_lbl.configure(text=str(self.frame_cnt_0))
            else:
                self.frame_cnt_1 += 1
                self.client_1_lbl.configure(text=str(self.frame_cnt_1))

    def dispatch_to_others(self, full_data, my_client_number):
        target_socket = None
        if my_client_number == 0:
            target_socket = self.client_1_socket
        else:
            target_socket = self.client_0_socket
        
        if target_socket == None: 
            return
        else:
            target_socket.sendall(full_data)


if __name__ == "__main__":
    server = Server()
    server.start()
