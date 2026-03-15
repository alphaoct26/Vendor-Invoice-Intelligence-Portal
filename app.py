from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
FREIGHT_MODEL_PATH = (
    BASE_DIR / "frieght_cost_prediction" / "models" / "freight_model.pkl"
)
INVOICE_MODEL_PATH = (
    BASE_DIR / "invoice_flagging" / "models" / "predict_flag_invoice.pkl"
)
INVOICE_SCALER_PATH = BASE_DIR / "invoice_flagging" / "models" / "scaler.pkl"

INVOICE_FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "invoice_freight",
    "total_brands",
    "total_item_quantity",
    "days_po_to_invoice",
    "total_item_dollars",
]


@st.cache_resource
def load_artifacts():
    """Load ML artifacts once per app session."""
    freight_model = joblib.load(FREIGHT_MODEL_PATH)
    invoice_model = joblib.load(INVOICE_MODEL_PATH)
    invoice_scaler = joblib.load(INVOICE_SCALER_PATH)
    return freight_model, invoice_model, invoice_scaler


def predict_freight_cost(freight_model, quantity: float, dollars: float) -> float:
    input_df = pd.DataFrame([{"Quantity": quantity, "Dollars": dollars}])
    prediction = freight_model.predict(input_df)[0]
    return float(prediction)


def prepare_invoice_input(values: dict) -> pd.DataFrame:
    input_df = pd.DataFrame([values])

    for col in INVOICE_FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df[INVOICE_FEATURES]


def predict_invoice_flag(invoice_model, invoice_scaler, invoice_values: dict) -> int:
    input_df = prepare_invoice_input(invoice_values)
    scaled_input = invoice_scaler.transform(input_df)
    prediction = invoice_model.predict(scaled_input)[0]
    return int(prediction)


def render_freight_section(freight_model):
    st.subheader("Freight Cost Prediction")
    st.caption("Estimate expected freight cost from quantity and dollar amount.")

    with st.form("freight_form"):
        quantity = st.number_input("Quantity", min_value=0.0, value=100.0, step=1.0)
        dollars = st.number_input("Dollars", min_value=0.0, value=1000.0, step=10.0)
        submitted = st.form_submit_button("Predict Freight Cost")

    if submitted:
        predicted_cost = predict_freight_cost(freight_model, quantity, dollars)
        st.success(f"Predicted Freight Cost: ${predicted_cost:,.2f}")


def render_invoice_section(invoice_model, invoice_scaler):
    st.subheader("Invoice Manual Approval Flag")
    st.caption("Predict whether an invoice should be flagged for manual review.")

    with st.form("invoice_form"):
        invoice_values = {
            "invoice_quantity": st.number_input(
                "Invoice Quantity", min_value=0.0, value=20.0, step=1.0
            ),
            "invoice_dollars": st.number_input(
                "Invoice Dollars", min_value=0.0, value=1000.0, step=10.0
            ),
            "invoice_freight": st.number_input(
                "Invoice Freight", min_value=0.0, value=45.0, step=1.0
            ),
            "total_brands": st.number_input(
                "Total Brands", min_value=0.0, value=3.0, step=1.0
            ),
            "total_item_quantity": st.number_input(
                "Total Item Quantity", min_value=0.0, value=120.0, step=1.0
            ),
            "days_po_to_invoice": st.number_input(
                "Days PO To Invoice", min_value=0.0, value=7.0, step=1.0
            ),
            "total_item_dollars": st.number_input(
                "Total Item Dollars", min_value=0.0, value=955.0, step=10.0
            ),
        }
        submitted = st.form_submit_button("Predict Invoice Flag")

    if submitted:
        predicted_flag = predict_invoice_flag(
            invoice_model, invoice_scaler, invoice_values
        )
        label = "Manual Review Required" if predicted_flag == 1 else "Auto-Approve"
        st.success(f"Prediction: {label} (Predicted_Flag={predicted_flag})")


def main():
    st.set_page_config(
        page_title="Vendor Invoice Intelligence Portal",
        page_icon="AI",
        layout="wide",
    )

    st.title("Vendor Invoice Intelligence Portal")
    st.markdown("AI-driven freight cost prediction and invoice risk flagging.")
    st.divider()

    st.sidebar.title("Model Selection")
    selected_model = st.sidebar.radio(
        "Choose Prediction Module",
        ["Freight Cost Prediction", "Invoice Manual Approval Flag"],
    )

    st.sidebar.markdown(
        """
        **Business Impact**
        - Improved cost forecasting
        - Reduced invoice anomalies
        - Faster finance operations
        """
    )

    try:
        freight_model, invoice_model, invoice_scaler = load_artifacts()
    except FileNotFoundError as exc:
        st.error(f"Model artifact not found: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load ML artifacts: {exc}")
        st.stop()

    if selected_model == "Freight Cost Prediction":
        render_freight_section(freight_model)
    else:
        render_invoice_section(invoice_model, invoice_scaler)


if __name__ == "__main__":
    main()
