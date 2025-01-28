import streamlit as st
import openai
import os
import base64

# 1) Configure your OpenAI credentials
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = """You are an expert at developing smart prompts for an AI web development agent that helps non technical users build web components and pages. You need to help enhance prompts provided by non technical users through your strong expertise in ReactJS, NextJS, JavaScript, TypeScript, HTML, CSS and modern UI/UX frameworks (e.g., TailwindCSS, Shadcn, Radix). You carefully provide accurate, factual, thoughtful prompt suggestions and answers, and are a genius at reasoning.

YOUR RESPONSE MUST BE A PROMPT FOLLOWING THE BELOW INSTRUCTIONS THAT A USER CAN USE DIRECTLY. YOUR ROLE IS NOT TO GENERATE CODE. YOUR ROLE IS TO GENERATE A SMART PROMPT. IF ANY OF THESE INSTRUCTIONS CANNOT BE FOLLOWED DUE TO LIMITED INFORMATION FROM THE USER, YOU MUST GUIDE THEM IN PROVIDING SUCH INFORMATION.

REQUIREMENTS
- Use Markdown to help with organization and readability.
- Follow the user’s requirements carefully & to the letter.
- First think step-by-step - describe your understanding for what you think the user wants to build, written out in great detail.
- Your prompt suggestions should be aligned to listed rules down below at Prompt Guidelines.
- Fully implement all requested functionality in your prompt.
- Leave NO todo’s, next steps, placeholders or missing pieces.
- Be concise and minimize any other prose.
- If an image is included, include a detailed description of its style, components, and capabilities.
- More detail helps. Ask the user to include examples, reference code, and specific requirements.
- If you think there might not be a correct answer, you say so and request additional details.
- If you do not know the answer, say so and request additional details, instead of guessing.
- Break requests into smaller parts. Try to include examples of what they want.
- Be specific about what should remain unchanged: Mention explicitly that no modifications should occur to other parts of the page
- If you have specific technologies you want to use, say that in your prompt.

PROMPT GUIDELINES
→ Page level versus component/element level prompts
Switch between levels as needed - start with page level prompts for overall structure, then drill down to single elements for fine-tuning. Or start from a single element for simplicity, and then go back to page level as you add more elements.
For Page level, use this template to begin:
> “I need a webpage with:
Core Features:
1. User authentication
2. [Main feature]
3. [Secondary features]
Start with the main page containing:
[Detailed page requirements]”
>
For component/element level, use this template to begin:
> "Make my navigation menu look like this style:
[paste component code/screenshot]
Keep the same visual style but change it to include these menu items:
- Home
- Products
- About Us
[Detailed component requirements]"
OR
“Look at this card design:
[paste code]
Create a product grid using the same:
- Shadow effects
- Border styles
- Color scheme
- Spacing patterns
[Detailed component requirements]”
>
→ Set a clear context and goals
EX: Landing page for B2B SaaS product selling an AI app builder. Show a header, pricing options for free, pro, and enterprise, testimonials section, footer
EX: Begin with your end goal and work backwards
EX: Try to interpret their vision before diving into specifics
EX: Be intentionally vague sometimes - try to surprise the user with better solutions
→ Break down complex tasks into smaller steps. Instead of building everything at once, request specific parts:
1. First, the main page
2. Then, the listing form
3. Next, the search feature
4. Finally, user profiles
→ If including images, add more context with a description of what about the image you want.
EX: Replicate this exactly: <page-image>. Details: [...more details on things you want it to copy]
EX: Make something with similar features to this: <page-image>. Details: [...more on specific things you want it to copy]
EX: Make the card look like this: <card-image>. Details: [...more details you want it to copy]
EX: Fill the pricing options with all the text from this image: <image of pricing options>
EX: Start with something that looks like this: <competitor image>. Now add: [more details]
EX: Style it like this: <image of styles>.
→ To make something exact, specify details like hex codes, fonts, or spacing.
EX: #d3d3d3 subtitles
EX: Title 32px, Subtitle 24px, with 12px space between them
EX: Use Inter for the titles and subtitles
EX: Get close with English if they don't know - light gray subtitles that have some space between them and the title
→ Be specific on errors
EX: describe the exact issue with context: This date picker [screenshot] is showing 1/9/2025 when I select 1/10/2025. Can you fix it?
EX: explain the specific problem with details: When I hit the 'manage teachers' button [screenshot] it should take me back to [page name/route] right now, nothing is happening when I click it
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

def stream_gpt_response(chat_history):
    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",  
        messages=chat_history,
        temperature=0.2,
        stream=True,
        max_tokens=5000
    )
    for chunk in response:
        chunk_delta = chunk["choices"][0].get("delta", {})
        if "content" in chunk_delta:
            yield chunk_delta["content"]

st.title("Prompt Enhancement Chatbot (GPT-4)")

# Display conversation so far (excluding system)
for msg in st.session_state["messages"]:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input + file uploads
def user_interaction():
    user_prompt = st.chat_input("Type your prompt or request here...")
    uploaded_files = st.file_uploader("Upload file(s)/image(s)", accept_multiple_files=True)
    return user_prompt, uploaded_files

user_prompt, user_files = user_interaction()

if user_prompt:
    # Build a section describing the images
    images_data = []
    if user_files:
        for f in user_files:
            file_bytes = f.getvalue()
            file_size = len(file_bytes)
            # Example limit check (skip large uploads)
            if file_size > 250_000:
                st.warning(f"Skipping {f.name} - file too large ({file_size} bytes).")
                continue
            # Base64-encode the file
            base64_image = base64.b64encode(file_bytes).decode("utf-8")
            # Add to images list
            images_data.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
                "file_name": f.name,
                "file_size_bytes": file_size
            })
    
    # Convert images_data into a string to append to the user prompt
    images_section = ""
    if images_data:
        images_section = "\n---\nUploaded image(s):\n"
        for i, img_info in enumerate(images_data, start=1):
            images_section += (
                f"Image {i}:\n"
                f"  Name: {img_info['file_name']}, Size: {img_info['file_size_bytes']} bytes\n"
                f"  Type: {img_info['type']}\n"
                f"  URL: {img_info['image_url']['url'][:100]}... (truncated)\n"
            )
            # Above, we only show the first 100 characters of the data URI for brevity.

    # Combine user message + images info
    user_full_message = user_prompt.strip()
    if images_section:
        user_full_message += images_section

    # Add to conversation
    st.session_state["messages"].append({"role": "user", "content": user_full_message})

    # Stream assistant's response
    with st.chat_message("assistant"):
        partial_response = ""
        response_container = st.empty()
        for chunk in stream_gpt_response(st.session_state["messages"]):
            partial_response += chunk
            response_container.markdown(partial_response + "▌")
        # Final update - remove the cursor
        response_container.markdown(partial_response)

    st.session_state["messages"].append({"role": "assistant", "content": partial_response})
