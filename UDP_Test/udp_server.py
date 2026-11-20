
import socket
import json
import time
import csv
import os
import math

# ---------------------------------------
# Server configuration
# ---------------------------------------

HOST = "0.0.0.0"
PORT = 5000

STALE_TIMEOUT = 3.0

# Always save data inside the UDP_Test folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "sensor_data.tsv")

# ---------------------------------------
# Runtime statistics
# ---------------------------------------

total_received = 0
total_lost = 0
total_rejected = 0
total_duplicates = 0
total_out_of_order = 0

# ---------------------------------------
# Sensor tracking
# ---------------------------------------

last_sequence = {}
last_arrival_time = {}
last_interarrival = {}
sensor_jitter = {}

last_seen_time = {}
stale_sensors = set()


# ---------------------------------------
# Validate incoming packets
# ---------------------------------------

def validate_packet(packet):

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
            raise ValueError(f"Missing field: {field}")

    if not isinstance(packet["device_id"], str) or not packet["device_id"].strip():
        raise ValueError("Invalid device_id")

    if not isinstance(packet["sensor_id"], str) or not packet["sensor_id"].strip():
        raise ValueError("Invalid sensor_id")

    if not isinstance(packet["unit"], str):
        raise ValueError("Invalid unit")

    if type(packet["sequence"]) is not int or packet["sequence"] < 0:
        raise ValueError("Invalid sequence number")

    timestamp = packet["timestamp"]
    value = packet["value"]

    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise ValueError("Timestamp must be numeric")

    if not math.isfinite(timestamp):
        raise ValueError("Timestamp must be finite")

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Sensor value must be numeric")

    if not math.isfinite(value):
        raise ValueError("Sensor value must be finite")


# ---------------------------------------
# Calculate inter-arrival jitter
# ---------------------------------------

def calculate_jitter(sensor_key, arrival_time):

    jitter = sensor_jitter.get(sensor_key, 0.0)

    if sensor_key in last_arrival_time:

        interarrival = (
            arrival_time - last_arrival_time[sensor_key]
        )

        if sensor_key in last_interarrival:

            variation = abs(
                interarrival - last_interarrival[sensor_key]
            )

            jitter += (variation * 1000 - jitter) / 16

        last_interarrival[sensor_key] = interarrival

    last_arrival_time[sensor_key] = arrival_time
    sensor_jitter[sensor_key] = jitter

    return jitter


# ---------------------------------------
# Detect stale sensors
# ---------------------------------------

def check_stale_sensors():

    current_time = time.monotonic()

    for sensor_key, last_seen in last_seen_time.items():

        elapsed = current_time - last_seen

        if elapsed > STALE_TIMEOUT and sensor_key not in stale_sensors:

            stale_sensors.add(sensor_key)

            device_id, sensor_id = sensor_key

            print(
                f"\nWARNING: STALE SENSOR DETECTED | "
                f"{device_id} | {sensor_id} | "
                f"No data for {elapsed:.2f} seconds"
            )


# ---------------------------------------
# Display final statistics
# ---------------------------------------

def display_statistics():

    average_jitter = (
        sum(sensor_jitter.values()) / len(sensor_jitter)
        if sensor_jitter else 0
    )

    print("\n" + "=" * 60)
    print("             NETWORK STATISTICS")
    print("=" * 60)

    print(f"Valid packets received : {total_received}")
    print(f"Packets lost (inferred): {total_lost}")
    print(f"Duplicate packets      : {total_duplicates}")
    print(f"Out-of-order packets   : {total_out_of_order}")
    print(f"Rejected packets       : {total_rejected}")
    print(f"Average jitter         : {average_jitter:.2f} ms")
    print(f"Total sensors detected : {len(last_seen_time)}")
    print(f"Stale sensors          : {len(stale_sensors)}")

    print("=" * 60)


# ---------------------------------------
# Create TSV file if necessary
# ---------------------------------------

if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:

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
# Start UDP server
# ---------------------------------------

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((HOST, PORT))
server_socket.settimeout(1.0)

print("=" * 60)
print("              UDP SENSOR AGGREGATOR")
print("=" * 60)
print(f"Listening on UDP port {PORT}")
print(f"Recording data to: {DATA_FILE}")
print(f"Sensor stale timeout: {STALE_TIMEOUT} seconds")
print()


# ---------------------------------------
# Main packet receiving loop
# ---------------------------------------

try:

    while True:

        try:

            data, address = server_socket.recvfrom(1024)

            # Parse and validate packet
            try:

                packet = json.loads(data.decode("utf-8"))
                validate_packet(packet)

            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:

                total_rejected += 1
                print("Invalid packet:", e)
                continue

            # Read packet fields
            device_id = packet["device_id"]
            sensor_id = packet["sensor_id"]
            sequence = packet["sequence"]
            source_timestamp = packet["timestamp"]
            value = packet["value"]
            unit = packet["unit"]

            receive_time = time.time()
            arrival_time = time.monotonic()

            total_received += 1

            # Calculate latency
            latency = (
                receive_time - source_timestamp
            ) * 1000

            # Identify individual sensor
            sensor_key = (device_id, sensor_id)

            # Calculate jitter
            jitter = calculate_jitter(
                sensor_key,
                arrival_time
            )

            # Update sensor activity
            last_seen_time[sensor_key] = arrival_time

            if sensor_key in stale_sensors:

                stale_sensors.remove(sensor_key)

                print(
                    f"\nSENSOR RECOVERED | "
                    f"{device_id} | {sensor_id}"
                )

            # ---------------------------------------
            # Sequence number and packet-loss tracking
            # ---------------------------------------

            packet_loss = 0

            if sensor_key not in last_sequence:

                last_sequence[sensor_key] = sequence

            else:

                previous_sequence = last_sequence[sensor_key]

                if sequence > previous_sequence + 1:

                    packet_loss = sequence - previous_sequence - 1

                    total_lost += packet_loss

                    print(
                        f"WARNING: {packet_loss} packet(s) lost "
                        f"from {device_id} - {sensor_id}"
                    )

                    last_sequence[sensor_key] = sequence

                elif sequence > previous_sequence:

                    last_sequence[sensor_key] = sequence

                elif sequence == previous_sequence:

                    total_duplicates += 1

                    print(
                        f"WARNING: Duplicate packet from "
                        f"{device_id} - {sensor_id}"
                    )

                else:

                    total_out_of_order += 1

                    print(
                        f"WARNING: Out-of-order packet from "
                        f"{device_id} - {sensor_id}"
                    )

            # ---------------------------------------
            # Record data in TSV
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

                writer = csv.writer(file, delimiter="\t")

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
                f"Latency: {latency:.2f} ms | "
                f"Jitter: {jitter:.2f} ms"
            )

        except socket.timeout:

            check_stale_sensors()

        except Exception as e:

            print("Processing error:", e)

except KeyboardInterrupt:

    print("\nUDP server stopped.")

finally:

    server_socket.close()
    display_statistics()
