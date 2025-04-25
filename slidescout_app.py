import streamlit as st
import os
import base64
import shutil
import tempfile
import time
import statistics
from slidescout import crop_slide_images, summarize_presentation_from_images, encode_image_base64, get_slide_prompt, client

st.set_page_config(page_title="SlideScout", layout="centered")
st.title("📊 SlideScout")
st.subheader("AI-powered presentation intelligence")

st.markdown("Upload slide images (JPG/PNG), and get a complete presentation narrative.")

# === Session state init ===
if "has_run" not in st.session_state:
    st.session_state.has_run = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "full_narrative" not in st.session_state:
    st.session_state.full_narrative = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""

# === Step 1: File Upload ===
uploaded_files = st.file_uploader("Upload your slide images:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

run_button = st.button("Run SlideScout")

if run_button and uploaded_files:
    st.session_state.has_run = True
    with st.spinner("Processing - run time depends on number of slides uploaded."):
        start_time = time.time()

        with tempfile.TemporaryDirectory() as temp_raw, tempfile.TemporaryDirectory() as temp_cropped:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_raw, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            crop_slide_images(temp_raw, temp_cropped)

            try:
                st.session_state.full_narrative = ""
                slide_times = []

                image_files = sorted(
                    [f for f in os.listdir(temp_cropped) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                    key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
                )

                story_so_far = None
                progress_bar = st.progress(0, text="Starting analysis...")

                for i, filename in enumerate(image_files):
                    slide_path = os.path.join(temp_cropped, filename)
                    b64_image = encode_image_base64(slide_path)
                    prompt = get_slide_prompt(i + 1, len(image_files), "HCP", story_so_far)

                    t0 = time.time()

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                            ]}
                        ],
                        max_tokens=1000,
                        temperature=0.7
                    )

                    t1 = time.time()
                    slide_time = t1 - t0
                    slide_times.append(slide_time)

                    story_so_far = response.choices[0].message.content.strip()
                    st.session_state.full_narrative += f"\n=== Slide {i+1} ===\n{story_so_far}\n"
                    progress_bar.progress((i + 1) / len(image_files), text=f"Processed slide {i+1} of {len(image_files)}")

                end_time = time.time()
                duration = end_time - start_time
                avg_time = statistics.mean(slide_times)

                st.success(f"Done in {duration:.1f} seconds!")
                st.info(f"Average time per slide: {avg_time:.1f} seconds")

                # === Generate Summary ===
                with st.spinner("Creating overall presentation summary..."):
                    summary_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": (
                                "You are an assistant summarizing a scientific presentation. "
                                "Write a general summary that explains the background, rationale, key results, and conclusions. "
                                "Be clear and concise, and highlight any particularly important findings or data points."
                            )},
                            {"role": "user", "content": st.session_state.full_narrative}
                        ],
                        max_tokens=800,
                        temperature=0.5
                    )
                    st.session_state.summary = summary_response.choices[0].message.content.strip()

                # === Initialize chat with Scout ===
                st.session_state.chat_history = [
                    {"role": "system", "content": (
                        "You are 'Scout', an expert assistant who helps users understand scientific presentations.\n"
                        "ONLY use the information found in the presentation below to answer questions. "
                        "If the answer is not in the content, say that the information is not available.\n\n"
                        f"Presentation content:\n{st.session_state.full_narrative}"
                    )}
                ]

            except Exception as e:
                st.error(f"Error: {e}")

# === Show Presentation Summary ===
if st.session_state.has_run and st.session_state.summary:
    with st.expander("📄 Presentation Summary", expanded=True):
        st.markdown(st.session_state.summary)

# === Chat Section ===
if st.session_state.has_run:
    st.subheader("💬 Chat with Scout")

    user_input = st.text_input("Ask Scout a question about the presentation:")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        try:
            chat_response = client.chat.completions.create(
                model="gpt-4o",
                messages=st.session_state.chat_history,
                max_tokens=700,
                temperature=0.7
            )
            reply = chat_response.choices[0].message.content.strip()
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.markdown(f"**Scout:** {reply}")
        except Exception as e:
            st.error(f"Chat failed: {e}")

    if len(st.session_state.chat_history) > 1:
        with st.expander("🕓 Chat History"):
            for msg in st.session_state.chat_history[1:]:
                speaker = "You" if msg["role"] == "user" else "Scout"
                st.markdown(f"**{speaker}:** {msg['content']}")

    # 📥 FINAL DOWNLOAD BUTTON AT BOTTOM
    st.markdown("---")
    st.markdown("### 📎 Export")
    b64 = base64.b64encode(st.session_state.full_narrative.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="slidescout_summary.txt">📥 Download Presentation Intelligence Capture</a>'
    st.markdown(href, unsafe_allow_html=True)

elif not st.session_state.has_run:
    st.info("Upload images and click 'Run SlideScout' to begin.")
