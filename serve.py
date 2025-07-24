# jetson_server.py
import socket
import serial

# Connect to Arduino
arduino = serial.Serial('/dev/ttyACM0', 9600)
print("Arduino ready.")

# Create TCP server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 9000))  # Listen on all interfaces
server.listen(1)
print("Waiting for incoming connection...")

while True:
    client, addr = server.accept()
    print(f"Connected from {addr}")
    data = client.recv(1024).decode().strip()
    print(f"Received: {data}")

    if data == "release":
        arduino.write(b'release\n')
        client.sendall(b"Triggered\n")
    else:
        client.sendall(b"Unknown command\n")

    client.close()