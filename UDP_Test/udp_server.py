import socket
import json
import time
import csv
import os

HOST = "0.0.0.0"
PORT = 5000

DATA_FILE = "sensor_data.tsv"

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((HOST, PORT))

print("=" * 60)
print("              UDP SENSOR AGGREGATOR")
print("=" * 60)
print(f"Listening on UDP port {PORT}")
print(f"Recording data to: {DATA_FILE}")
print()


# ---------------------------------------
# Create data file if it doesn't exist
# ---------------------------------------

if not os.path.exists(DATA_FILE):

    with open(DATA_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file, delimiter="\t")

        writer.writerow([
            "Receive_Timestamp",
            "Device_ID",
            "Sensor_ID",
            "Sequence",
            "Source_Timestamp",
            "Value",
            "Unit",
            "Latency_ms",
            "Packet_Loss"
        ])


# ---------------------------------------
# Store sequence numbers
# ---------------------------------------

last_sequence = {}


while True:

    try:

        data, address = server_socket.recvfrom(1024)

        packet = json.loads(data.decode("utf-8"))

        # ---------------------------------------
        # Read packet
        # ---------------------------------------

        device_id = packet["device_id"]
        sensor_id = packet["sensor_id"]
        sequence = packet["sequence"]
        source_timestamp = packet["timestamp"]
        value = packet["value"]
        unit = packet["unit"]

        receive_time = time.time()

        # ---------------------------------------
        # Calculate latency
        # ---------------------------------------

        latency = (
            receive_time - source_timestamp
        ) * 1000

        # ---------------------------------------
        # Calculate packet loss
        # ---------------------------------------

        packet_loss = 0

        if device_id in last_sequence:

            previous_sequence = last_sequence[device_id]

            if sequence > previous_sequence + 1:

                packet_loss = (
                    sequence - previous_sequence - 1
                )

                print(
                    f"WARNING: {packet_loss} packet(s) "
                    f"lost from {device_id}"
                )

            elif sequence <= previous_sequence:

                print(
                    f"WARNING: duplicate/out-of-order "
                    f"packet from {device_id}"
                )

        last_sequence[device_id] = sequence

        # ---------------------------------------
        # Record data
        # ---------------------------------------

        receive_timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(receive_time)
        )

        with open(
            DATA_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(
                file,
                delimiter="\t"
            )

            writer.writerow([
                receive_timestamp,
                device_id,
                sensor_id,
                sequence,
                source_timestamp,
                value,
                unit,
                round(latency, 2),
                packet_loss
            ])

        # ---------------------------------------
        # Terminal output
        # ---------------------------------------

        print(
            f"{device_id:<12} | "
            f"{sensor_id:<15} | "
            f"{str(value):<8} {unit:<8} | "
            f"Seq: {sequence:<5} | "
            f"Latency: {latency:.2f} ms"
        )

    except KeyboardInterrupt:

        print("\nUDP server stopped.")
        break

    except Exception as e:

        print("Invalid packet:", e)