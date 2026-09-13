import os
import streamlit as st
os.environ["SUPPORT_AGENT_INDEX"] = "demo"

from src.response_generator import run_support_agent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Support Intelligence",
    page_icon="◈",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL BACKGROUND
       ======================================================== */

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 5% 5%,
                rgba(99, 102, 241, 0.18),
                transparent 28%
            ),
            radial-gradient(
                circle at 95% 5%,
                rgba(14, 165, 233, 0.16),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(168, 85, 247, 0.10),
                transparent 32%
            ),
            linear-gradient(
                135deg,
                #f4f5ff 0%,
                #fafbff 50%,
                #f1faff 100%
            );
    }


    [data-testid="stHeader"] {
        background: transparent;
    }


    .block-container {
        max-width: 1180px;
        padding-top: 2.6rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       HIDE STREAMLIT DEFAULT ELEMENTS
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ========================================================
       HEADINGS
       ======================================================== */

    h1 {
        letter-spacing: -1.6px;
        font-weight: 750;
    }

    h2,
    h3 {
        letter-spacing: -0.6px;
    }


    /* ========================================================
       TEXT AREA
       ======================================================== */

    div[data-testid="stTextArea"] textarea {
        background: rgba(255, 255, 255, 0.94) !important;

        border: 1px solid #d8deea;
        border-radius: 14px;

        color: #344054;

        box-shadow:
            0 8px 25px rgba(31, 41, 55, 0.05);

        font-size: 15px;
    }


    div[data-testid="stTextArea"] textarea:focus {
        border-color: #818cf8;

        box-shadow:
            0 0 0 1px #818cf8,
            0 8px 25px rgba(99, 102, 241, 0.10);
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    div.stButton > button {
        width: 100%;
        height: 50px;

        border: none;
        border-radius: 11px;

        background: #ff4b4b;
        color: white;

        font-size: 15px;
        font-weight: 700;

        box-shadow:
            0 8px 20px rgba(255, 75, 75, 0.18);

        transition: all 0.2s ease;
    }


    div.stButton > button:hover {
        background: #e63e3e;
        color: white;

        transform: translateY(-1px);

        box-shadow:
            0 10px 24px rgba(255, 75, 75, 0.23);
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.88);

        border: 1px solid rgba(255, 255, 255, 0.95);

        border-radius: 14px;

        padding: 14px 16px;

        box-shadow:
            0 8px 24px rgba(31, 41, 55, 0.05);
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 13px;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.82);

        border: 1px solid rgba(148, 163, 184, 0.18);

        border-radius: 13px;

        box-shadow:
            0 6px 20px rgba(31, 41, 55, 0.04);
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: rgba(148, 163, 184, 0.22);
    }


    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.title("◈ Support Intelligence")

st.caption(
    "AI-powered customer support triage and response assistant"
)

st.success("● Agent ready")


# ============================================================
# PIPELINE
# ============================================================

st.caption("SUPPORT AGENT PIPELINE")

steps = st.columns(5)

pipeline_data = [
    ("01", "Message", "Customer query"),
    ("02", "Intent", "Classify issue"),
    ("03", "Retrieve", "Find similar cases"),
    ("04", "Draft", "Grounded response"),
    ("05", "Decision", "Auto or human"),
]


for col, (number, title, description) in zip(
    steps,
    pipeline_data,
):

    with col:

        st.metric(
            number,
            title,
        )

        st.caption(description)


st.divider()


# ============================================================
# CUSTOMER MESSAGE
# ============================================================

st.subheader("Customer Message")

customer_message = st.text_area(
    "Customer message",
    placeholder=(
        "Describe the customer's issue...\n\n"
        "Example: My iPhone battery is draining very quickly "
        "after the latest update."
    ),
    height=145,
    label_visibility="collapsed",
)


st.write("")


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "Analyze Customer Message  →"
)


# ============================================================
# RUN SUPPORT AGENT
# ============================================================

if analyze:

    if not customer_message.strip():

        st.warning(
            "Please enter a customer message first."
        )

    else:

        with st.spinner(
            "Running support intelligence..."
        ):

            result = run_support_agent(
                customer_message.strip()
            )


        # ====================================================
        # RESULT VALUES
        # ====================================================

        intent = result.get(
            "intent",
            "general_troubleshooting",
        )


        confidence = float(
            result.get(
                "confidence",
                0.0,
            )
        )


        second_intent = result.get(
            "second_intent",
            "",
        )


        second_score = float(
            result.get(
                "second_score",
                0.0,
            )
        )


        intent_margin = float(
            result.get(
                "intent_margin",
                confidence - second_score,
            )
        )


        response = result.get(
            "response",
            "",
        )


        decision = result.get(
            "decision",
            "ESCALATE_TO_HUMAN",
        )


        reason = result.get(
            "reason",
            "Unable to determine a safe handling decision.",
        )


        retrieved_cases = result.get(
            "retrieved_cases",
            [],
        )


        # ====================================================
        # AGENT DECISION
        # ====================================================

        st.divider()

        st.subheader("Agent Decision")


        if decision == "AUTO_HANDLE":

            st.success(
                "●  AUTO-HANDLE"
            )

        else:

            st.warning(
                "●  ESCALATE TO HUMAN"
            )


        st.caption(reason)


        # ====================================================
        # INTENT ANALYSIS
        # ====================================================

        st.subheader("Intent Analysis")


        display_intent = (
            intent
            .replace("_", " ")
            .title()
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "Predicted Intent",
                display_intent,
            )


        with c2:

            st.metric(
                "Confidence",
                f"{confidence:.3f}",
            )


        with c3:

            st.metric(
                "Intent Margin",
                f"{intent_margin:.3f}",
            )


        # ====================================================
        # SUGGESTED RESPONSE
        # ====================================================

        st.subheader("Suggested Response")


        st.info(
            response
        )


        # ====================================================
        # RETRIEVED EVIDENCE
        # ====================================================

        st.subheader("Retrieved Evidence")


        st.caption(
            "Similar historical customer-support cases "
            "used as grounding evidence."
        )


        if retrieved_cases:

            for i, case in enumerate(
                retrieved_cases[:3],
                start=1,
            ):

                similarity = float(
                    case.get(
                        "similarity",
                        0.0,
                    )
                )


                customer_text = case.get(
                    "customer_text",
                    "",
                )


                support_text = case.get(
                    "support_text",
                    case.get(
                        "historical_response",
                        "",
                    ),
                )


                with st.expander(
                    f"Case {i}  ·  similarity {similarity:.3f}"
                ):

                    st.markdown(
                        "**Customer issue**"
                    )


                    st.write(
                        customer_text
                    )


                    if support_text:

                        st.markdown(
                            "**Historical support response**"
                        )


                        st.write(
                            support_text
                        )


        else:

            st.info(
                "No relevant historical cases were retrieved."
            )


        # ====================================================
        # CLASSIFICATION CONTEXT
        # ====================================================

        if second_intent:

            st.subheader(
                "Classification Context"
            )


            alternative = (
                second_intent
                .replace("_", " ")
                .title()
            )


            c1, c2 = st.columns(2)


            with c1:

                st.metric(
                    "Alternative Intent",
                    alternative,
                )


            with c2:

                st.metric(
                    "Alternative Score",
                    f"{second_score:.3f}",
                )


        # ====================================================
        # FOOTER
        # ====================================================

        st.divider()


        st.caption(
            "Semantic classification  ·  "
            "Historical retrieval  ·  "
            "Grounded response drafting  ·  "
            "Human escalation"
        )