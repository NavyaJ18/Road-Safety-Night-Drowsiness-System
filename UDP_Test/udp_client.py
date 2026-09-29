import socket
import time
import json

SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sequence_number = 0

while True:

    sequence_number += 1

    packet = {
        "device_id": "ESP32_01",
        "sensor_id": "heart_rate",
        "sequence": sequence_number,
        "timestamp": time.time(),
        "value": 75,
        "unit": "bpm"
    }

    message = json.dumps(packet)

    client_socket.sendto(
        message.encode("utf-8"),
        (SERVER_IP, SERVER_PORT)
    )

    print("Sent:", packet)

    time.sleep(0.1)