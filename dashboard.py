import streamlit as st
import pandas as pd
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Drowsiness Detection Dashboard",
    layout="wide"
)

# ============================================================
# DASHBOARD VIEW SELECTION
# ============================================================

st.sidebar.header("Dashboard View")

dashboard_view = st.sidebar.radio(
    "Select View",
    [
        "🧠 Drowsiness Detection",
        "📡 Sensor Network"
    ]
)


# ============================================================
# SENSOR NETWORK DASHBOARD
# ============================================================

if dashboard_view == "📡 Sensor Network":

    SENSOR_FILE = os.path.join("UDP_Test", "sensor_data.tsv")

    st.title("📡 Sensor Network Dashboard")
    st.caption(
        "Live UDP sensor monitoring, network status and historical records"
    )

    @st.fragment(run_every="1s")
    def sensor_network_dashboard():

        if not os.path.exists(SENSOR_FILE):
            st.warning(
                f"Sensor data file '{SENSOR_FILE}' not found."
            )
            st.info(
                "Start udp_server.py first. The file will be created "
                "automatically when sensor data is received."
            )
            return

        try:
            sensor_df = pd.read_csv(
                SENSOR_FILE,
                sep="\t",
                engine="python",
                on_bad_lines="skip"
            )
        except Exception as e:
            st.error(f"Error reading sensor data: {e}")
            return

        sensor_df.columns = (
            sensor_df.columns.astype(str).str.strip()
        )

        if sensor_df.empty:
            st.info("Waiting for sensor data...")
            return

        sensor_numeric_columns = [
            "Sequence",
            "Source_Timestamp",
            "Value",
            "Latency_ms",
            "Packet_Loss"
        ]

        for column in sensor_numeric_columns:
            if column in sensor_df.columns:
                sensor_df[column] = pd.to_numeric(
                    sensor_df[column],
                    errors="coerce"
                )

        if "Receive_Timestamp" in sensor_df.columns:
            sensor_df["Receive_Timestamp"] = pd.to_datetime(
                sensor_df["Receive_Timestamp"],
                errors="coerce"
            )

        if "Device_ID" in sensor_df.columns:
            latest_sensor_data = (
                sensor_df
                .sort_values("Sequence")
                .groupby("Device_ID", dropna=False)
                .tail(1)
                .copy()
            )
        else:
            latest_sensor_data = sensor_df.tail(1).copy()

        current_time = pd.Timestamp.now()

        if "Receive_Timestamp" in latest_sensor_data.columns:
            latest_sensor_data["Age_ms"] = (
                current_time -
                latest_sensor_data["Receive_Timestamp"]
            ).dt.total_seconds() * 1000
        else:
            latest_sensor_data["Age_ms"] = 0

        def get_sensor_status(age):
            if pd.isna(age):
                return "⚪ UNKNOWN"
            if age < 2000:
                return "🟢 LIVE"
            if age < 5000:
                return "🟡 DELAYED"
            return "🔴 STALE"

        latest_sensor_data["Status"] = (
            latest_sensor_data["Age_ms"].apply(get_sensor_status)
        )

        st.subheader("🟢 Live Sensor Feed")

        live_columns = [
            "Device_ID",
            "Sensor_ID",
            "Sequence",
            "Value",
            "Unit",
            "Age_ms",
            "Latency_ms",
            "Packet_Loss",
            "Status"
        ]

        available_live_columns = [
            column
            for column in live_columns
            if column in latest_sensor_data.columns
        ]

        if available_live_columns:
            live_data = latest_sensor_data[
                available_live_columns
            ].copy()

            if "Age_ms" in live_data.columns:
                live_data["Age_ms"] = live_data["Age_ms"].round(1)

            if "Latency_ms" in live_data.columns:
                live_data["Latency_ms"] = (
                    live_data["Latency_ms"].round(2)
                )

            st.dataframe(
                live_data,
                use_container_width=True,
                hide_index=True
            )

        st.subheader("📊 System Summary")

        summary_col1, summary_col2, summary_col3, summary_col4 = (
            st.columns(4)
        )

        with summary_col1:
            st.metric(
                "Connected Sensors",
                len(latest_sensor_data)
            )

        with summary_col2:
            st.metric(
                "Total Records",
                len(sensor_df)
            )

        with summary_col3:
            if "Latency_ms" in sensor_df.columns:
                avg_latency = sensor_df["Latency_ms"].mean()
                if pd.notna(avg_latency):
                    st.metric(
                        "Average Latency",
                        f"{avg_latency:.2f} ms"
                    )
                else:
                    st.metric("Average Latency", "N/A")
            else:
                st.metric("Average Latency", "N/A")

        with summary_col4:
            if "Packet_Loss" in sensor_df.columns:
                total_packet_loss = int(
                    sensor_df["Packet_Loss"].fillna(0).sum()
                )
                st.metric(
                    "Packets Lost",
                    total_packet_loss
                )
            else:
                st.metric("Packets Lost", 0)

        st.divider()
        st.subheader("📈 Sensor Analytics")

        if "Sensor_ID" in sensor_df.columns:

            sensors = sensor_df["Sensor_ID"].dropna().unique()

            for sensor in sensors:

                sensor_history = sensor_df[
                    sensor_df["Sensor_ID"] == sensor
                ].copy()

                if "Receive_Timestamp" in sensor_history.columns:
                    sensor_history = sensor_history.sort_values(
                        "Receive_Timestamp"
                    )

                st.write(
                    f"### {str(sensor).replace('_', ' ').title()}"
                )

                if (
                    "Receive_Timestamp" in sensor_history.columns
                    and "Value" in sensor_history.columns
                ):

                    chart_data = sensor_history[
                        ["Receive_Timestamp", "Value"]
                    ].dropna()

                    if not chart_data.empty:

                        chart_data = chart_data.set_index(
                            "Receive_Timestamp"
                        )

                        st.line_chart(chart_data["Value"])

        if (
            "Receive_Timestamp" in sensor_df.columns
            and "Latency_ms" in sensor_df.columns
        ):

            st.divider()
            st.subheader("📡 Network Latency")

            latency_data = sensor_df[
                ["Receive_Timestamp", "Latency_ms"]
            ].dropna()

            if not latency_data.empty:

                latency_data = latency_data.set_index(
                    "Receive_Timestamp"
                )

                st.line_chart(latency_data["Latency_ms"])

        st.divider()
        st.subheader("📋 Recorded Sensor Data")

        st.dataframe(
            sensor_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )

        st.divider()
        st.subheader("Download Sensor Data")

        sensor_download = sensor_df.to_csv(
            sep="\t",
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Sensor Data",
            data=sensor_download,
            file_name="sensor_data.tsv",
            mime="text/tab-separated-values",
            key="download_sensor_data"
        )

    sensor_network_dashboard()
    st.stop()

st.title("🚗 Driver Drowsiness Detection Dashboard")


# ============================================================
# DATA FILE
# ============================================================

CSV_FILE = "drowsiness_rainbow.tsv"


# ============================================================
# CHECK IF FILE EXISTS
# ============================================================

if not os.path.exists(CSV_FILE):

    st.error(
        f"Data file '{CSV_FILE}' not found."
    )

    st.info(
        "Make sure drowsiness_rainbow.tsv is in the same "
        "folder as dashboard.py"
    )

    st.stop()


# ============================================================
# READ TAB-SEPARATED DATA
# ============================================================

try:

    df = pd.read_csv(
        CSV_FILE,
        sep="\t",
        engine="python",
        on_bad_lines="skip"
    )

except Exception as e:

    st.error(
        f"Error reading data file: {e}"
    )

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# DISPLAY AVAILABLE COLUMNS
# ============================================================

st.sidebar.header("Dashboard Controls")

st.sidebar.write("Available Columns:")

st.sidebar.write(
    list(df.columns)
)


# ============================================================
# TIMESTAMP PROCESSING
# ============================================================

if "Timestamp" in df.columns:

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [

    "EAR",

    "Left_EAR",

    "Right_EAR",

    "Average_EAR",

    "MAR",

    "MAR_Threshold",

    "PERCLOS",

    "PERCLOS (%)",

    "Blink_Rate"

]


for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# REMOVE COMPLETELY EMPTY ROWS
# ============================================================

df = df.dropna(
    how="all"
)


# ============================================================
# CHECK DATA
# ============================================================

if df.empty:

    st.warning(
        "No valid data available in the data file."
    )

    st.stop()


# ============================================================
# LATEST VALUES
# ============================================================

latest = df.iloc[-1]


# ============================================================
# LIVE DETECTION PARAMETERS
# ============================================================

st.subheader(
    "Live Detection Parameters"
)


# ============================================================
# METRIC COLUMNS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)


# ============================================================
# AVERAGE EAR
# ============================================================

with col1:

    if "Average_EAR" in df.columns:

        ear_value = latest["Average_EAR"]

        if pd.notna(ear_value):

            st.metric(
                "Average EAR",
                f"{ear_value:.3f}"
            )

        else:

            st.metric(
                "Average EAR",
                "N/A"
            )

    elif "EAR" in df.columns:

        ear_value = latest["EAR"]

        if pd.notna(ear_value):

            st.metric(
                "EAR",
                f"{ear_value:.3f}"
            )

        else:

            st.metric(
                "EAR",
                "N/A"
            )

    else:

        st.metric(
            "EAR",
            "Not Available"
        )


# ============================================================
# MAR
# ============================================================

with col2:

    if "MAR" in df.columns:

        mar_value = latest["MAR"]

        if pd.notna(mar_value):

            st.metric(
                "MAR",
                f"{mar_value:.3f}"
            )

        else:

            st.metric(
                "MAR",
                "N/A"
            )

    else:

        st.metric(
            "MAR",
            "Not Available"
        )


# ============================================================
# PERCLOS
# ============================================================

with col3:

    if "PERCLOS" in df.columns:

        perclos_value = latest["PERCLOS"]

        if pd.notna(perclos_value):

            st.metric(
                "PERCLOS",
                f"{perclos_value:.2f}%"
            )

        else:

            st.metric(
                "PERCLOS",
                "N/A"
            )

    elif "PERCLOS (%)" in df.columns:

        perclos_value = latest["PERCLOS (%)"]

        if pd.notna(perclos_value):

            st.metric(
                "PERCLOS",
                f"{perclos_value:.2f}%"
            )

        else:

            st.metric(
                "PERCLOS",
                "N/A"
            )

    else:

        st.metric(
            "PERCLOS",
            "Not Available"
        )


# ============================================================
# BLINK RATE
# ============================================================

with col4:

    if "Blink_Rate" in df.columns:

        blink_rate_value = latest["Blink_Rate"]

        if pd.notna(blink_rate_value):

            st.metric(
                "Blink Rate",
                f"{blink_rate_value:.1f}/min"
            )

        else:

            st.metric(
                "Blink Rate",
                "N/A"
            )

    else:

        st.metric(
            "Blink Rate",
            "Not Available"
        )


# ============================================================
# TOTAL RECORDS
# ============================================================

with col5:

    st.metric(
        "Total Records",
        len(df)
    )


# ============================================================
# BLINK RATE STATUS
# ============================================================

if "Blink_Rate" in df.columns:

    blink_rate_value = latest["Blink_Rate"]

    if pd.notna(blink_rate_value):

        if blink_rate_value < 10:

            st.error(
                f"⚠️ LOW BLINK RATE: "
                f"{blink_rate_value:.1f} blinks/min"
            )

        else:

            st.success(
                f"✓ NORMAL BLINK RATE: "
                f"{blink_rate_value:.1f} blinks/min"
            )


# ============================================================
# CURRENT DETECTION STATUS
# ============================================================

st.divider()

st.subheader(
    "Current Detection Status"
)


status_col1, status_col2, status_col3, status_col4 = st.columns(4)


# ============================================================
# EYE STATUS
# ============================================================

with status_col1:

    if "Eye_Status" in df.columns:

        st.metric(
            "Eye Status",
            str(latest["Eye_Status"])
        )

    else:

        st.metric(
            "Eye Status",
            "Not Available"
        )


# ============================================================
# YAWN STATUS
# ============================================================

with status_col2:

    if "Yawn_Status" in df.columns:

        st.metric(
            "Yawn Status",
            str(latest["Yawn_Status"])
        )

    else:

        st.metric(
            "Yawn Status",
            "Not Available"
        )


# ============================================================
# BLINK STATUS
# ============================================================

with status_col3:

    if "Blink_Status" in df.columns:

        st.metric(
            "Blink Status",
            str(latest["Blink_Status"])
        )

    else:

        st.metric(
            "Blink Status",
            "Not Available"
        )


# ============================================================
# DROWSINESS STATUS
# ============================================================

with status_col4:

    if "Drowsiness_Status" in df.columns:

        st.metric(
            "Drowsiness Status",
            str(latest["Drowsiness_Status"])
        )

    else:

        st.metric(
            "Drowsiness Status",
            "Not Available"
        )


# ============================================================
# CURRENT VALUES TABLE
# ============================================================

st.divider()

st.subheader(
    "Current Parameter Values"
)


current_columns = [
    "Timestamp",
    "Average_EAR",
    "MAR",
    "MAR_Threshold",
    "PERCLOS",
    "Blink_Rate",
    "Eye_Status",
    "Yawn_Status",
    "Blink_Status",
    "Drowsiness_Status"
]


available_current_columns = [

    column

    for column in current_columns

    if column in df.columns

]


if available_current_columns:

    current_data = pd.DataFrame(
        [latest[available_current_columns]]
    )

    st.dataframe(
        current_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DETECTION THRESHOLDS
# ============================================================

st.divider()

st.subheader(
    "Detection Thresholds"
)


threshold_col1, threshold_col2, threshold_col3, threshold_col4 = st.columns(4)


# ============================================================
# EAR THRESHOLD
# ============================================================

with threshold_col1:

    st.write("### EAR")

    st.write(
        "Threshold: **0.30**"
    )

    st.write(
        "EAR ≥ 0.30 → Eyes Open"
    )

    st.write(
        "0.18 ≤ EAR < 0.30 → Partially Closed"
    )

    st.write(
        "EAR < 0.18 → Eyes Closed"
    )


# ============================================================
# MAR THRESHOLD
# ============================================================

with threshold_col2:

    st.write("### MAR")

    if "MAR_Threshold" in df.columns:

        current_mar_threshold = latest["MAR_Threshold"]

        if pd.notna(current_mar_threshold):

            st.write(
                f"Yawn Threshold: "
                f"**{current_mar_threshold:.2f}**"
            )

        else:

            st.write(
                "Yawn Threshold: **0.60**"
            )

    else:

        st.write(
            "Yawn Threshold: **0.60**"
        )

    st.write(
        "MAR ≥ 0.60 → Yawning"
    )


# ============================================================
# PERCLOS THRESHOLD
# ============================================================

with threshold_col3:

    st.write("### PERCLOS")

    st.write(
        "Window: **90 seconds**"
    )

    st.write(
        "Threshold: **80%**"
    )

    st.write(
        "PERCLOS ≥ 80% → Drowsiness"
    )


# ============================================================
# BLINK RATE THRESHOLD
# ============================================================

with threshold_col4:

    st.write("### BLINK RATE")

    st.write(
        "Threshold: **10 blinks/min**"
    )

    st.write(
        "≥ 10/min → Normal"
    )

    st.write(
        "< 10/min → Low Blink Rate"
    )


# ============================================================
# EAR ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "EAR Analysis"
)


if "Average_EAR" in df.columns:

    if "Timestamp" in df.columns:

        chart_data = df[
            [
                "Timestamp",
                "Average_EAR"
            ]
        ].dropna()

        if not chart_data.empty:

            chart_data = chart_data.set_index(
                "Timestamp"
            )

            st.line_chart(
                chart_data
            )

    else:

        st.line_chart(
            df["Average_EAR"]
        )


elif "EAR" in df.columns:

    if "Timestamp" in df.columns:

        chart_data = df[
            [
                "Timestamp",
                "EAR"
            ]
        ].dropna()

        if not chart_data.empty:

            chart_data = chart_data.set_index(
                "Timestamp"
            )

            st.line_chart(
                chart_data
            )

    else:

        st.line_chart(
            df["EAR"]
        )


else:

    st.info(
        "EAR column not found in data file."
    )


# ============================================================
# MAR ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "MAR Analysis"
)


if "MAR" in df.columns:

    if "Timestamp" in df.columns:

        chart_data = df[
            [
                "Timestamp",
                "MAR"
            ]
        ].dropna()

        if not chart_data.empty:

            chart_data = chart_data.set_index(
                "Timestamp"
            )

            st.line_chart(
                chart_data
            )

    else:

        st.line_chart(
            df["MAR"]
        )


else:

    st.info(
        "MAR column not found in data file."
    )


# ============================================================
# PERCLOS ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "PERCLOS Analysis"
)


perclos_column = None


if "PERCLOS" in df.columns:

    perclos_column = "PERCLOS"


elif "PERCLOS (%)" in df.columns:

    perclos_column = "PERCLOS (%)"


if perclos_column is not None:

    if "Timestamp" in df.columns:

        chart_data = df[
            [
                "Timestamp",
                perclos_column
            ]
        ].dropna()

        if not chart_data.empty:

            chart_data = chart_data.set_index(
                "Timestamp"
            )

            st.line_chart(
                chart_data
            )

    else:

        st.line_chart(
            df[perclos_column]
        )


else:

    st.info(
        "PERCLOS column not found in data file."
    )


# ============================================================
# BLINK RATE ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "Blink Rate Analysis"
)


if "Blink_Rate" in df.columns:

    if "Timestamp" in df.columns:

        chart_data = df[
            [
                "Timestamp",
                "Blink_Rate"
            ]
        ].dropna()

        if not chart_data.empty:

            chart_data = chart_data.set_index(
                "Timestamp"
            )

            st.line_chart(
                chart_data
            )

    else:

        st.line_chart(
            df["Blink_Rate"]
        )


else:

    st.info(
        "Blink_Rate column not found in data file."
    )


# ============================================================
# INDIVIDUAL EYE EAR
# ============================================================

st.divider()

st.subheader(
    "Individual Eye EAR"
)


if (
    "Left_EAR" in df.columns
    and
    "Right_EAR" in df.columns
):

    if "Timestamp" in df.columns:

        chart_data = df[
            [
                "Timestamp",
                "Left_EAR",
                "Right_EAR"
            ]
        ].dropna()

        if not chart_data.empty:

            chart_data = chart_data.set_index(
                "Timestamp"
            )

            st.line_chart(
                chart_data
            )

    else:

        st.line_chart(
            df[
                [
                    "Left_EAR",
                    "Right_EAR"
                ]
            ]
        )


else:

    st.info(
        "Left_EAR or Right_EAR column not found."
    )


# ============================================================
# DROWSINESS STATUS OVER TIME
# ============================================================

st.divider()

st.subheader(
    "Drowsiness Status Over Time"
)


if (
    "Timestamp" in df.columns
    and
    "Drowsiness_Status" in df.columns
):

    status_data = df[
        [
            "Timestamp",
            "Drowsiness_Status"
        ]
    ].dropna(
        subset=["Timestamp"]
    )

    if not status_data.empty:

        st.dataframe(
            status_data.tail(20),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# RECORDED DATA
# ============================================================

st.divider()

st.subheader(
    "Recorded Data"
)


st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD DATA
# ============================================================

st.divider()

st.subheader(
    "Download Data"
)


# Convert the dataframe back to TAB-separated format
# so the downloaded file maintains the same structure.

download_data = df.to_csv(
    sep="\t",
    index=False
).encode(
    "utf-8"
)


st.download_button(
    label="Download Recorded Data",
    data=download_data,
    file_name="drowsiness_dashboard_data.tsv",
    mime="text/tab-separated-values"
)