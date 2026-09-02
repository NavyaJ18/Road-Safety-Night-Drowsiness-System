import streamlit as st
import pandas as pd
import os

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Drowsiness Detection Dashboard",
    layout="wide"
)

st.title("🚗 Driver Drowsiness Detection Dashboard")

# -------------------------------------------------
# CSV FILE
# -------------------------------------------------

CSV_FILE = "mar_data.csv"

# -------------------------------------------------
# CHECK IF FILE EXISTS
# -------------------------------------------------

if not os.path.exists(CSV_FILE):
    st.error(f"CSV file '{CSV_FILE}' not found.")
    st.info("Make sure mar_data.csv is in the same folder as dashboard.py")
    st.stop()


# -------------------------------------------------
# READ CSV SAFELY
# -------------------------------------------------

try:
    df = pd.read_csv(
        CSV_FILE,
        engine="python",
        on_bad_lines="skip"
    )

except Exception as e:
    st.error(f"Error reading CSV file: {e}")
    st.stop()


# -------------------------------------------------
# DISPLAY CSV INFORMATION
# -------------------------------------------------

st.sidebar.header("Dashboard Controls")

st.sidebar.write("Available Columns:")
st.sidebar.write(list(df.columns))


# -------------------------------------------------
# TIMESTAMP PROCESSING
# -------------------------------------------------

if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )


# -------------------------------------------------
# CONVERT NUMERIC COLUMNS
# -------------------------------------------------

numeric_columns = [
    "EAR",
    "MAR",
    "PERCLOS",
    "PERCLOS (%)"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# -------------------------------------------------
# REMOVE COMPLETELY INVALID ROWS
# -------------------------------------------------

df = df.dropna(how="all")


# -------------------------------------------------
# CHECK DATA
# -------------------------------------------------

if df.empty:
    st.warning("No valid data available in the CSV file.")
    st.stop()


# -------------------------------------------------
# LATEST VALUES
# -------------------------------------------------

latest = df.iloc[-1]


# -------------------------------------------------
# METRICS
# -------------------------------------------------

col1, col2, col3, col4 = st.columns(4)


# EAR

with col1:

    if "EAR" in df.columns:

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


# MAR

with col2:

    if "MAR" in df.columns:

        mar_value = latest["MAR"]

        if pd.notna(mar_value):

            st.metric(
                "MAR",
                f"{mar_value:.2f}"
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


# PERCLOS

with col3:

    if "PERCLOS" in df.columns:

        perclos_value = latest["PERCLOS"]

        if pd.notna(perclos_value):

            st.metric(
                "PERCLOS (%)",
                f"{perclos_value:.2f}"
            )

        else:

            st.metric(
                "PERCLOS (%)",
                "N/A"
            )

    elif "PERCLOS (%)" in df.columns:

        perclos_value = latest["PERCLOS (%)"]

        if pd.notna(perclos_value):

            st.metric(
                "PERCLOS (%)",
                f"{perclos_value:.2f}"
            )

        else:

            st.metric(
                "PERCLOS (%)",
                "N/A"
            )

    else:

        st.metric(
            "PERCLOS (%)",
            "Not Available"
        )


# TOTAL RECORDS

with col4:

    st.metric(
        "Total Records",
        len(df)
    )


# -------------------------------------------------
# THRESHOLD INFORMATION
# -------------------------------------------------

st.divider()

st.subheader("Detection Thresholds")

threshold_col1, threshold_col2, threshold_col3 = st.columns(3)

with threshold_col1:

    st.write("### EAR")
    st.write("Threshold: **0.30**")

    st.write("EAR ≥ 0.30 → Eyes Open")

    st.write("Below threshold → Eye closure detected")


with threshold_col2:

    st.write("### MAR")
    st.write("Yawn Threshold: **20**")

    st.write("MAR > 20 → Yawn Detected")


with threshold_col3:

    st.write("### PERCLOS")

    st.write("Window: **90 seconds**")

    st.write("Threshold: **80%**")


# -------------------------------------------------
# EAR GRAPH
# -------------------------------------------------

st.divider()

st.subheader("EAR Analysis")

if "EAR" in df.columns:

    if "Timestamp" in df.columns:

        chart_data = df[
            ["Timestamp", "EAR"]
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

    st.info("EAR column not found in CSV.")


# -------------------------------------------------
# MAR GRAPH
# -------------------------------------------------

st.divider()

st.subheader("MAR Analysis")

if "MAR" in df.columns:

    if "Timestamp" in df.columns:

        chart_data = df[
            ["Timestamp", "MAR"]
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

    st.info("MAR column not found in CSV.")


# -------------------------------------------------
# PERCLOS GRAPH
# -------------------------------------------------

st.divider()

st.subheader("PERCLOS Analysis")

perclos_column = None

if "PERCLOS" in df.columns:

    perclos_column = "PERCLOS"

elif "PERCLOS (%)" in df.columns:

    perclos_column = "PERCLOS (%)"


if perclos_column is not None:

    if "Timestamp" in df.columns:

        chart_data = df[
            ["Timestamp", perclos_column]
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

    st.info("PERCLOS column not found in CSV.")


# -------------------------------------------------
# DATA TABLE
# -------------------------------------------------

st.divider()

st.subheader("Recorded Data")

st.dataframe(
    df,
    use_container_width=True
)


# -------------------------------------------------
# DOWNLOAD DATA
# -------------------------------------------------

csv_data = df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Recorded Data",
    data=csv_data,
    file_name="drowsiness_dashboard_data.csv",
    mime="text/csv"
)