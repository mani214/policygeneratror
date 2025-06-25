import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model and examples
df = pd.read_csv("control_rego_examples.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['control'].tolist(), convert_to_tensor=True)

st.title("AI-Powered Policy Authoring Portal")

# Step 1: User inputs a control
user_control = st.text_area("Paste your control requirement:")

if user_control:
    input_embedding = model.encode([user_control], convert_to_tensor=True)
    scores = cosine_similarity(input_embedding.cpu().numpy(), embeddings.cpu().numpy())[0]
    top_indices = scores.argsort()[-2:][::-1]
    
    st.subheader("🔍 Retrieved Examples")
    for i in top_indices:
        st.markdown(f"**Control:** {df.iloc[i]['control']}")
        st.code(df.iloc[i]['rego'], language='rego')

    # Step 2: Simulated LLM Output
    st.subheader("🤖 Generated Rego Policy")
    default_rego = f"""package generated

# Simulated output for: {user_control}

default allow = false

allow {{
    # Replace with real logic based on control
}}"""
    generated_rego = st.text_area("Edit the generated Rego policy:", value=default_rego, height=200)

    # Step 3: Manual Validation Actions
    st.subheader("🧪 Review & Validate")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Approve & Submit"):
            st.success("Policy approved and saved!")
            # TODO: Save to file/git/DB
    with col2:
        if st.button("✏️ Edit Later"):
            st.info("Policy saved for future editing.")
            # TODO: Save as draft
    with col3:
        if st.button("❌ Reject"):
            st.warning("Policy rejected. No action taken.")

    # Optional reviewer notes
    st.text_area("💬 Reviewer Notes (optional):")
