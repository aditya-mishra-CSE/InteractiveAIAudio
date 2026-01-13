import uuid
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage

from backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# ================= UTILS =================
def extract_ai_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)

def parse_dialogue(text):
    dialogue = []
    for line in text.splitlines():
        if ":" in line:
            speaker, msg = line.split(":", 1)
            dialogue.append((speaker.strip(), msg.strip()))
    return dialogue

# def speak_dialogue(dialogue):
#     js = """
#     <script>
#     speechSynthesis.cancel();
#     const voices = speechSynthesis.getVoices();

#     function speak(text, voiceIndex) {
#         let u = new SpeechSynthesisUtterance(text);
#         u.voice = voices[voiceIndex] || voices[0];
#         u.rate = 1;
#         speechSynthesis.speak(u);
#     }
#     """

#     for speaker, text in dialogue:
#         voice = 0 if speaker == "Speaker A" else 1
#         js += f"speak({text!r}, {voice});\n"

#     js += "</script>"

#     components.html(js, height=0)

# def speak_dialogue(dialogue):
#     """
#     Real-time TTS for two speakers:
#     Speaker A -> Male
#     Speaker B -> Female
#     Works reliably in Streamlit.
#     """
#     js = """
#     <script>
#     speechSynthesis.cancel();

#     // Ensure voices are loaded
#     function loadVoices() {
#         return new Promise(resolve => {
#             let voices = speechSynthesis.getVoices();
#             if (voices.length > 0) {
#                 resolve(voices);
#             } else {
#                 speechSynthesis.onvoiceschanged = () => {
#                     resolve(speechSynthesis.getVoices());
#                 };
#             }
#         });
#     }

#     async function speakDialogue() {
#         const voices = await loadVoices();

#         // Heuristics to pick male/female voices
#         const maleVoice = voices.find(v =>
#             /male|david|mark|daniel|alex|fred|john/i.test(v.name)
#         ) || voices[0];

#         const femaleVoice = voices.find(v =>
#             /female|zira|susan|samantha|karen|victoria/i.test(v.name)
#         ) || voices[1] || voices[0];

#     """

#     # Add each utterance
#     for speaker, text in dialogue:
#         if speaker == "Speaker A":
#             js += f"""
#             let uA = new SpeechSynthesisUtterance({text!r});
#             uA.voice = maleVoice;
#             uA.rate = 1;
#             speechSynthesis.speak(uA);
#             """
#         else:
#             js += f"""
#             let uB = new SpeechSynthesisUtterance({text!r});
#             uB.voice = femaleVoice;
#             uB.rate = 1;
#             speechSynthesis.speak(uB);
#             """

#     js += """
#     }

#     speakDialogue();
#     </script>
#     """

#     components.html(js, height=0)



def speak_dialogue(dialogue):
    """
    Speaker A -> Indian English (Male)
    Speaker B -> Indian English (Female)
    Deterministic & Streamlit-safe
    """

    js = """
    <script>
    speechSynthesis.cancel();

    function waitForVoices(callback) {
        let voices = speechSynthesis.getVoices();
        if (voices.length) {
            callback(voices);
        } else {
            speechSynthesis.onvoiceschanged = () => {
                callback(speechSynthesis.getVoices());
            };
        }
    }

    waitForVoices(function(voices) {

        // 🇮🇳 Indian English voices
        const indianVoices = voices.filter(v => v.lang === "en-IN");

        // Male Indian voice
        const maleVoice =
            indianVoices.find(v =>
                /male|ravi|amit|arjun|rahul|dev/i.test(v.name)
            ) ||
            indianVoices.find(v => !/female|zira|susan|samantha/i.test(v.name)) ||
            indianVoices[0] ||
            voices[0];

        // Female Indian voice
        const femaleVoice =
            indianVoices.find(v =>
                /female|zira|susan|samantha|kiran|neha|anita/i.test(v.name)
            ) ||
            indianVoices.find(v => v !== maleVoice) ||
            voices[1] ||
            voices[0];
    """

    for speaker, text in dialogue:
        if speaker == "Speaker A":
            js += f"""
            let uA = new SpeechSynthesisUtterance({text!r});
            uA.voice = maleVoice;
            uA.rate = 1;
            speechSynthesis.speak(uA);
            """
        else:
            js += f"""
            let uB = new SpeechSynthesisUtterance({text!r});
            uB.voice = femaleVoice;
            uB.rate = 1;
            speechSynthesis.speak(uB);
            """

    js += """
    });
    </script>
    """

    components.html(js, height=0)

# ================= SESSION =================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "threads" not in st.session_state:
    st.session_state.threads = retrieve_all_threads()

def new_chat():
    tid = str(uuid.uuid4())
    st.session_state.thread_id = tid
    st.session_state.messages = []
    st.session_state.threads.append(tid)

# ================= SIDEBAR =================
st.sidebar.title("📄 LangGraph PDF Chatbot")
st.sidebar.button("➕ New Chat", on_click=new_chat)

uploaded = st.sidebar.file_uploader("Upload PDF", type="pdf")
if uploaded:
    ingest_pdf(uploaded.getvalue(), st.session_state.thread_id, uploaded.name)
    st.sidebar.success("PDF Indexed")

st.sidebar.subheader("Previous Chats")
for tid in st.session_state.threads[::-1]:
    if st.sidebar.button(tid):
        st.session_state.thread_id = tid
        state = chatbot.get_state(
            config={"configurable": {"thread_id": tid}}
        )
        st.session_state.messages = [
            {
                "role": "user" if m.type == "human" else "assistant",
                "content": extract_ai_text(m.content),
            }
            for m in state.values.get("messages", [])
        ]
        st.rerun()

# ================= MAIN =================
st.title("🤖 Two-Speaker Conversational AI")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": st.session_state.thread_id}},
    )

    ai_text = extract_ai_text(response["messages"][-1].content)

    st.session_state.messages.append(
        {"role": "assistant", "content": ai_text}
    )

    with st.chat_message("assistant"):
        st.markdown(ai_text)

    # 🎙 REAL-TIME TWO-VOICE TTS
    dialogue = parse_dialogue(ai_text)
    if dialogue:
        speak_dialogue(dialogue)

    meta = thread_document_metadata(st.session_state.thread_id)
    if meta:
        st.caption(
            f"📄 {meta['filename']} | "
            f"{meta['chunks']} chunks | "
            f"{meta['documents']} pages"
        )
