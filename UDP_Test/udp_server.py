
import socket
import json
import time
import csv
import os
import math

HOST = "0.0.0.0"
PORT = 5000

DATA_FILE = "sensor_data.tsv"

# ---------------------------------------
# Packet delivery statistics
# ---------------------------------------

total_received = 0
total_lost = 0
total_rejected = 0
total_duplicates = 0
total_out_of_order = 0


def validate_packet(packet):
    """Validate the structure and data types of a sensor packet."""

    if not isinstance(packet, dict):
        raise ValueError("Packet must be a JSON object")

    required_fields = [
        "device_id",
        "sensor_id",
        "sequence",
        "timestamp",
        "value",
        "unit"
    ]

    for field in required_fields:
        if field not in packet:
            raise ValueError(f"Missing required field: {field}")

    if not isinstance(packet["device_id"], str) or not packet["device_id"].strip():
        raise ValueError("Invalid device_id")

    if not isinstance(packet["sensor_id"], str) or not packet["sensor_id"].strip():
        raise ValueError("Invalid sensor_id")

    if not isinstance(packet["unit"], str):
        raise ValueError("Invalid unit")

    if type(packet["sequence"]) is not int or packet["sequence"] < 0:
        raise ValueError("Sequence must be a non-negative integer")

    if isinstance(packet["timestamp"], bool):
        raise ValueError("Invalid timestamp")

    if not isinstance(packet["timestamp"], (int, float)):
        raise ValueError("Timestamp must be numeric")

    if not math.isfinite(packet["timestamp"]):
        raise ValueError("Timestamp must be finite")

    if isinstance(packet["value"], bool):
        raise ValueError("Invalid sensor value")

    if not isinstance(packet["value"], (int, float)):
        raise ValueError("Sensor value must be numeric")

    if not math.isfinite(packet["value"]):
        raise ValueError("Sensor value must be finite")

    return True


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
# Store sequence numbers per sensor
# ---------------------------------------

last_sequence = {}


# ---------------------------------------
# Display packet delivery statistics
# ---------------------------------------

def display_statistics():

    total_expected = total_received + total_lost

    if total_expected > 0:
        delivery_rate = (
            total_received / total_expected
        ) * 100
    else:
        delivery_rate = 0

    print("\n" + "=" * 60)
    print("             NETWORK STATISTICS")
    print("=" * 60)

    print(f"Valid packets received : {total_received}")
    print(f"Packets lost           : {total_lost}")
    print(f"Duplicate packets      : {total_duplicates}")
    print(f"Out-of-order packets   : {total_out_of_order}")
    print(f"Rejected packets       : {total_rejected}")
    print(f"Packet delivery rate   : {delivery_rate:.2f}%")

    print("=" * 60 + "\n")


while True:

    try:

        data, address = server_socket.recvfrom(1024)

        try:
            packet = json.loads(data.decode("utf-8"))
            validate_packet(packet)

        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:

            total_rejected += 1
            print("Invalid packet:", e)
            continue

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

        total_received += 1

        # ---------------------------------------
        # Calculate latency
        # ---------------------------------------

        latency = (
            receive_time - source_timestamp
        ) * 1000

        # ---------------------------------------
        # Calculate packet loss per sensor
        # ---------------------------------------

        packet_loss = 0

        sensor_key = (device_id, sensor_id)

        if sensor_key not in last_sequence:

            last_sequence[sensor_key] = sequence

        else:

            previous_sequence = last_sequence[sensor_key]

            if sequence > previous_sequence + 1:

                packet_loss = (
                    sequence - previous_sequence - 1
                )

                total_lost += packet_loss

                print(
                    f"WARNING: {packet_loss} packet(s) "
                    f"lost from {device_id} - {sensor_id}"
                )

                last_sequence[sensor_key] = sequence

            elif sequence > previous_sequence:

                last_sequence[sensor_key] = sequence

            else:

                if sequence == previous_sequence:
                    total_duplicates += 1
                else:
                    total_out_of_order += 1

                print(
                    f"WARNING: duplicate/out-of-order "
                    f"packet from {device_id} - {sensor_id}"
                )

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
        display_statistics()
        break

    except Exception as e:

        print("Invalid packet:", e)
        total_rejected += 1
