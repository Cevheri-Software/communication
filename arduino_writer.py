import serial
import time

# Arduino'nun bağlı olduğu portu belirt
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)  # Arduino'nun reset atması için kısa süre bekle

# Servoya açı gönder (örneğin 90 derece)
angle = 90
arduino.write(f'{angle}\n'.encode())

arduino.close()
