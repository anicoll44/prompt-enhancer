import streamlit as st
import openai
import os

# -----------------------------------------------------------------
# 1) Configure your OpenAI credentials
# -----------------------------------------------------------------
# If using Streamlit secrets (recommended):
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

openai.api_key = OPENAI_API_KEY

# -----------------------------------------------------------------
# 2) The system prompt (verbatim from your request)
# -----------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert at developing smart prompts for an AI web development agent that helps non technical users build web components and pages. You need to help enhance prompts provided by non technical users through your strong expertise in ReactJS, NextJS, JavaScript, TypeScript, HTML, CSS and modern UI/UX frameworks (e.g., TailwindCSS, Shadcn, Radix). You carefully provide accurate, factual, thoughtful prompt suggestions and answers, and are a genius at reasoning.

YOUR RESPONSE MUST BE A PROMPT FOLLOWING THE BELOW INSTRUCTIONS THAT A USER CAN USE DIRECTLY. IF ANY OF THESE INSTRUCTIONS CANNOT BE FOLLOWED DUE TO LIMITED INFORMATION FROM THE USER, YOU MUST GUIDE THEM IN PROVIDING SUCH INFORMATION.

REQUIREMENTS
- Follow the user’s requirements carefully & to the letter.
- First think step-by-step - describe your understanding for what you think the user wants to build, written out in great detail.
- Your prompt suggestions should be aligned to listed rules down below at Prompt Guidelines.
- Fully implement all requested functionality in your prompt.
- Leave NO todo’s, next steps, placeholders or missing pieces.
- Be concise and minimize any other prose.
- If an image is included, help explain it in detail.
- More detail helps. Include examples, reference code, and specific requirements.
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

# -----------------------------------------------------------------
# 3) Initialize session state for storing the conversation
# -----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# -----------------------------------------------------------------
# 4) Helper function to stream ChatCompletion responses
# -----------------------------------------------------------------
def stream_gpt_response(chat_history):
    """
    Expects chat_history to be a list of {"role":..., "content":...} dicts.
    Yields tokens from OpenAI's ChatCompletion API using stream=True.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",  # or your GPT4o endpoint
        messages=chat_history,
        temperature=0,
        stream=True
    )
    for chunk in response:
        chunk_delta = chunk["choices"][0].get("delta", {})
        if "content" in chunk_delta:
            yield chunk_delta["content"]

# -----------------------------------------------------------------
# 5) Display the existing conversation (excluding the system role)
# -----------------------------------------------------------------
st.title("Genetic Prompt Enhancer")

for msg in st.session_state["messages"]:
    if msg["role"] == "system":
        # Hide system message from chat display
        continue
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------------------------------------------
# 6) Input area for the user's text, plus optional file uploads
# -----------------------------------------------------------------
def user_interaction():
    # Let user type in text
    user_prompt = st.chat_input("Type your prompt or request here...")
    # Allow multi-file upload (images, docs, etc.)
    uploaded_files = st.file_uploader("Upload any related files or images (optional)", 
                                      accept_multiple_files=True, type=None)

    return user_prompt, uploaded_files

user_prompt, user_files = user_interaction()

# -----------------------------------------------------------------
# 7) If the user submitted text, handle it and produce a streamed response
# -----------------------------------------------------------------
if user_prompt:
    # Incorporate file info in the user's message if provided
    file_info_text = ""
    if user_files:
        file_descriptions = []
        for f in user_files:
            content_size = len(f.getvalue())
            file_descriptions.append(f"File name: {f.name}, size: {content_size} bytes")
        file_info_text = "\nAttached file(s):\n" + "\n".join(file_descriptions)

    # Add user's new message to chat history
    user_full_message = user_prompt.strip()
    if file_info_text:
        user_full_message += "\n" + file_info_text

    st.session_state["messages"].append({"role": "user", "content": user_full_message})

    # Prepare the streamed chatbot response
    with st.chat_message("assistant"):
        partial_response = ""
        response_container = st.empty()
        for chunk in stream_gpt_response(st.session_state["messages"]):
            partial_response += chunk
            # Display the growing response with a typing indicator
            response_container.markdown(partial_response + "▌")
        # Final updated response (remove typing cursor)
        response_container.markdown(partial_response)

    # Add the assistant's full response to the chat history
    st.session_state["messages"].append({"role": "assistant", "content": partial_response})
