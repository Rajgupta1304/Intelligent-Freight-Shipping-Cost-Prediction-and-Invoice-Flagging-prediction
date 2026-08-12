import streamlit as st


from inference.predict_shipping_cost import predict_freight_cost  # your inference.py file

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Intelligent Freight/Shipping Cost Prediction",
    page_icon="🚚",
    layout="centered"
)

# -------------------------------
# Title & Description
# -------------------------------
st.title("🚚 Intelligent Freight/Shipping Cost Prediction")
st.write(
    "This app predicts the estimated shipping cost based on the "
    "order value (in dollars), using a trained ML model."
)

st.divider()
st.write(
    " Intelligence invoice_flagging_prediction coming soon..."
)
st.divider()
# -------------------------------
# Input Form
# -------------------------------
st.subheader("Enter Shipment Details")

dollars = st.number_input("Order Value ($)", min_value=0.0, value=100.0, step=1.0)

st.divider()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Shipping Cost", use_container_width=True):
    sample_data = {"Dollars": [dollars]}

    try:
        result_df = predict_freight_cost(sample_data)
        predicted_cost = result_df["Predicted_shipping_cost"].iloc[0]
        st.success(f"### 💰 Estimated Shipping Cost: ${predicted_cost:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed.\n\nError: {e}")

st.divider()
st.caption("Built with Streamlit by RajGupta | Model: Machine Learning Regression")