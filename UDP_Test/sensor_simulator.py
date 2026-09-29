import socket
import time
import json
import random

SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sensors = {
    "ESP32_01": {
        "sensor_id": "heart_rate",
        "value": 75,
        "unit": "bpm"
    },

    "ESP32_02": {
        "sensor_id": "spo2",
        "value": 98,
        "unit": "%"
    },

    "ESP32_03": {
        "sensor_id": "gps_speed",
        "value": 55,
        "unit": "km/h"
    }
}

sequence_numbers = {
    "ESP32_01": 0,
    "ESP32_02": 0,
    "ESP32_03": 0
}

while True:

    for device_id, sensor in sensors.items():

        sequence_numbers[device_id] += 1

        # Simulate small changes in sensor readings
        if sensor["sensor_id"] == "heart_rate":
            value = random.randint(72, 80)

        elif sensor["sensor_id"] == "spo2":
            value = random.randint(96, 99)

        elif sensor["sensor_id"] == "gps_speed":
            value = random.randint(50, 60)

        packet = {
            "device_id": device_id,
            "sensor_id": sensor["sensor_id"],
            "sequence": sequence_numbers[device_id],
            "timestamp": time.time(),
            "value": value,
            "unit": sensor["unit"]
        }

        message = json.dumps(packet)

        client_socket.sendto(
            message.encode("utf-8"),
            (SERVER_IP, SERVER_PORT)
        )

        print(
            f"Sent | {device_id} | "
            f"{sensor['sensor_id']} | "
            f"{value} {sensor['unit']}"
        )

    time.sleep(0.5)