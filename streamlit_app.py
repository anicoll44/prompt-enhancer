import streamlit as st
import openai
import os
import base64
import io
from PIL import Image

# 1) Configure your OpenAI credentials
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = """You are an expert at writing smart prompts for an AI web development agent that helps non technical users build web components and pages. You need to help enhance prompts provided by non technical users through your strong expertise in ReactJS, NextJS, JavaScript, TypeScript, HTML, CSS and modern UI/UX frameworks (e.g., TailwindCSS, Shadcn, Radix). You carefully provide accurate, factual, thoughtful prompt suggestions and answers, and are a genius at reasoning.

YOUR RESPONSE MUST BE A PROMPT FOLLOWING THE BELOW GUIDELINES THAT A USER CAN USE DIRECTLY. YOUR ROLE IS NOT TO GENERATE CODE. YOUR ROLE IS TO GENERATE A SMART PROMPT. IF ANY OF THESE INSTRUCTIONS CANNOT BE FOLLOWED DUE TO LIMITED INFORMATION FROM THE USER, YOU MUST GUIDE THEM IN PROVIDING SUCH INFORMATION.

PROMPT GUIDELINES
- Your response should be in Markdown to help with organization and readability.
- First think step-by-step - describe your understanding for what you think the user wants to build, written out in great detail.
- Break down complex tasks into smaller steps.
- Break requests into smaller parts. Try to include examples and explain your understanding of what they want.
- Try to interpret their vision before diving into specifics.
- Start with page level details for overall structure, then drill down to single elements for fine-tuning.
- Be creative and visonary, but you must follow the user’s requirements carefully & to the letter.
- You must include all business context, goals, and objectives provided as part of your final prompt.
- Fully implement all requested functionality in your output.
- Leave NO todo’s, next steps, placeholders or missing pieces.
- Be concise and minimize any other prose.
- If an image is included, provide a detailed description of its style, components, and capabilities to enrich the overall output.
- Only suggest changing the layout of the page if asked.
- Be specific about what should remain unchanged: Mention explicitly that no modifications should occur to other parts of the page.
- If you have specific technologies you want to use, say that in your prompt.
- To make something exact, specify details like hex codes, fonts, or spacing.
- More detail helps. Ask the user to include examples, reference code, and more specific requirements if they are unsatisfied with your response.
- If you do not know the answer, say so and provide some recommendations in addition to requesting additional details.
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# Helper function to compress an image until it’s under max_size bytes or down to minimal quality
def compress_image(image_bytes, max_size=250_000, min_quality=10, step=5):
    """
    Compresses a JPEG image repeatedly (by decreasing quality) 
    until its size is under max_size bytes or quality < min_quality.
    Returns the compressed image bytes (regardless of size if minimal quality is reached).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # ensure JPEG-compatible mode
    quality = 85  # start with a reasonably high quality
    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", optimize=True, quality=quality)
        data = buf.getvalue()
        if len(data) <= max_size or quality <= min_quality:
            return data
        quality -= step

def stream_gpt_response(chat_history):
    response = openai.ChatCompletion.create(
        model="gpt-4",  
        messages=chat_history,
        temperature=0.2,
        stream=True,
        max_tokens=5000
    )
    for chunk in response:
        chunk_delta = chunk["choices"][0].get("delta", {})
        if "content" in chunk_delta:
            yield chunk_delta["content"]

# Display the conversation so far (excluding system)
st.title("Genetic Prompt Enhancement")

for msg in st.session_state["messages"]:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

def user_interaction():
    user_prompt = st.chat_input("Type your prompt/request here…")
    uploaded_files = st.file_uploader("Upload file(s)/image(s)", accept_multiple_files=True)
    return user_prompt, uploaded_files

user_prompt, user_files = user_interaction()

if user_prompt:
    # Build info about uploaded images
    images_section = ""
    if user_files:
        images_section = "\n---\nUploaded image(s):\n"
        for i, f in enumerate(user_files, start=1):
            file_bytes = f.getvalue()
            file_size = len(file_bytes)

            # Compress if > 250 KB
            if file_size > 250_000:
                compressed_bytes = compress_image(file_bytes, max_size=250_000)
                compressed_size = len(compressed_bytes)
                if compressed_size < file_size:
                    st.info(f"Compressed {f.name} from {file_size} bytes to {compressed_size} bytes.")
                file_bytes = compressed_bytes
                file_size = compressed_size

            base64_image = base64.b64encode(file_bytes).decode("utf-8")
            images_section += (
                f"Image {i}:\n"
                f"  Name: {f.name}\n"
                f"  Size: {file_size} bytes\n"
                f"  Data URI (first 80 chars): data:image/jpeg;base64,{base64_image[:80]}…\n"
            )

    # Combine user prompt + images info
    final_user_message = user_prompt.strip()
    if images_section:
        final_user_message += images_section

    # Add to conversation history
    st.session_state["messages"].append({"role": "user", "content": final_user_message})

    # Stream the assistant's response
    with st.chat_message("assistant"):
        partial_response = ""
        response_container = st.empty()
        for chunk in stream_gpt_response(st.session_state["messages"]):
            partial_response += chunk
            response_container.markdown(partial_response + "▌")
        # Final (remove the cursor)
        response_container.markdown(partial_response)

    st.session_state["messages"].append({"role": "assistant", "content": partial_response})
