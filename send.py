# send_command.py
import socket

JETSON_IP = "192.168.x.x"  # <-- Replace with actual Jetson IP
PORT = 9000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((JETSON_IP, PORT))
    s.sendall(b'release\n')
    response = s.recv(1024)

print("Jetson response:", response.decode())