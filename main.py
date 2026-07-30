import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.workflow import app
from app.models.schemas import PatientContext


st.set_page_config(page_title="MediMatch AGI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stApp { background: linear-gradient(145deg, #F3F7FA 0%, #FDFDFD 50%, #F5F9FC 100%) !important; }
    .hero-title { font-size: 2.9rem !important; color: #0F172A; font-weight: 800; margin-bottom: 0.4rem; letter-spacing: -1px; }
    .hero-subtitle { font-size: 1.25rem !important; color: #475569; margin-bottom: 2.5rem; border-bottom: 2px solid #E2E8F0; padding-bottom: 1.2rem; }
    [data-testid="stChatMessage"] { background-color: rgba(255, 255, 255, 0.9) !important; border-radius: 16px !important; border: 1px solid rgba(226, 232, 240, 0.8) !important; padding: 1.2rem !important; margin-bottom: 1.2rem !important; }
    [data-testid="stChatMessage"] a { color: #1E40AF !important; background-color: #EFF6FF; padding: 6px 14px; border-radius: 8px; text-decoration: none !important; font-weight: 600; display: inline-block; }
    .status-card { padding: 14px 18px; border-radius: 10px; font-weight: 700; font-size: 1rem; display: flex; align-items: center; gap: 10px; }
    .status-standby { background-color: #FFFFFF; color: #64748B; border: 1px solid #E2E8F0; }
    .status-inquiring { background-color: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }
    .status-minor { background-color: #F0FDF4; color: #166534; border: 1px solid #BBF7D0; }
    .status-moderate { background-color: #FFF7ED; color: #9A3412; border: 1px solid #FED7AA; }
    .status-urgent { background-color: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
</style>
""", unsafe_allow_html=True)


def get_status_html(status_code: str) -> str:
    if status_code == "INQUIRING":
        return "<div class='status-card status-inquiring'>🔎 <span><strong>Triage:</strong> Gathering Info...</span></div>"
    elif status_code == "MINOR":
        return "<div class='status-card status-minor'>💝 <span><strong>Triage:</strong> Minor (Self-Care)</span></div>"
    elif status_code == "MODERATE":
        return "<div class='status-card status-moderate'>🔚 <span><strong>Triage:</strong> Moderate (GP Advised)</span></div>"
    elif status_code == "URGENT":
        return "<div class='status-card status-urgent'>🔶 <span><strong>Triage:</strong> URGENT (ER / 111)</span></div>"
    else:
        return "<div class='status-card status-standby'>⚕️<span><strong>System:</strong> Standby</span></div>"


# ===========================
# Session Initialization
# ===========================

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(
            content="Kia ora! I am your intelligent triage assistant. Please describe your symptoms or tell me what medical assistance you need today."
        )
    ]

if "current_status" not in st.session_state:
    st.session_state.current_status = "STANDBY"

if "patient_context" not in st.session_state:
    st.session_state.patient_context = PatientContext()


# ===========================
# Sidebar
# ===========================

with st.sidebar:

    st.markdown("### 🩺 MediMatch Engine")
    st.caption("Auckland Node | Powered by LangGraph")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("##### Live Triage State")

    status_placeholder = st.empty()

    status_placeholder.markdown(
        get_status_html(st.session_state.current_status),
        unsafe_allow_html=True,
    )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("##### ⚙️ System Parameters")
    st.caption("• Core Model: Llama-3.3-70b-versatile")
    st.caption("• Knowledge Base: Local FAISS Index")
    st.caption("• Location Context: Auckland, NZ")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.messages = [
            AIMessage(
                content="Kia ora! I am your intelligent triage assistant. Please describe your symptoms or tell me what medical assistance you need today."
            )
        ]
        st.session_state.current_status = "STANDBY"
        st.session_state.patient_context = PatientContext()
        st.rerun()

    # ===== Debug Memory =====
    st.markdown("---")
    st.markdown("### 🧠 Conversation Memory")
    st.json(st.session_state.patient_context.model_dump())


# ===========================
# Main Page
# ===========================

st.markdown(
    '<div class="hero-title">MediMatch Pro Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-subtitle">Intelligent triage and local healthcare routing for Auckland, NZ.</div>',
    unsafe_allow_html=True,
)


# ===========================
# Chat History
# ===========================

for msg in st.session_state.messages:

    avatar = "🩺" if isinstance(msg, AIMessage) else "👤"

    role = "assistant" if isinstance(msg, AIMessage) else "user"

    st.chat_message(role, avatar=avatar).write(msg.content)


# ===========================
# Chat Input
# ===========================

user_input = st.chat_input("E.g., I have a severe headache...")


if user_input:

    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )

    st.chat_message(
        "user",
        avatar="👤",
    ).write(user_input)

    with st.chat_message("assistant", avatar="🩺"):

        with st.spinner("Analyzing presentation..."):

            final_state = app.invoke(
                {
                    "messages": st.session_state.messages,
                    "patient_context": st.session_state.patient_context,
                }
            )

            # Save updated memory
            if "patient_context" in final_state:
                st.session_state.patient_context = final_state["patient_context"]

            decision = final_state.get("decision")

            if decision:
                st.session_state.current_status = decision.status
                status_placeholder.markdown(
                    get_status_html(decision.status),
                    unsafe_allow_html=True,
                )

            ai_reply = final_state["messages"][-1].content

            st.write(ai_reply)

            st.session_state.messages.append(
                AIMessage(content=ai_reply)
            )