import io
import os
import base64
import uuid
import json
from datetime import datetime
from PIL import Image
import requests
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import open_clip
import torch
import numpy as np

# ==================== read secrets ====================
try:
    PINECONE_KEY = st.secrets["PINECONE_API_KEY"]
    AZURE_KEY = st.secrets["AZURE_OPENAI_KEY"]
    AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
    AZURE_DEPLOYMENT_VISION = st.secrets["AZURE_DEPLOYMENT_VISION"]
    AZURE_DEPLOYMENT_TEXT = st.secrets["AZURE_DEPLOYMENT_TEXT"]
    AZURE_API_VER = st.secrets["AZURE_API_VERSION"]
except Exception:
    st.error("请检查 .streamlit/secrets.toml 是否正确填写密钥")
    st.stop()


# ==================== Pinecone initialization ====================
@st.cache_resource
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_KEY)
    index_name = "outfitai-closet"
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

index = get_pinecone_index()

# ==================== CLIP model ====================
@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    return model, preprocess, tokenizer

clip_model, clip_preprocess, clip_tokenizer = load_clip()

def get_clip_text_vector(text: str):
    tokens = clip_tokenizer([text])
    with torch.no_grad():
        vec = clip_model.encode_text(tokens)
        vec /= vec.norm(dim=-1, keepdim=True)
    return vec[0].cpu().numpy().tolist()

def get_clip_vector(img):
    img_tensor = clip_preprocess(img).unsqueeze(0)
    with torch.no_grad():
        vec = clip_model.encode_image(img_tensor)
        vec /= vec.norm(dim=-1, keepdim=True)
    return vec[0].cpu().numpy().tolist()


# ==================== GPT-4o Vision analysis ====================
def analyze_with_gpt4o(image_bytes):
    b64 = base64.b64encode(image_bytes).decode()
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT_VISION}/chat/completions?api-version={AZURE_API_VER}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_KEY}
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a world-class fashion stylist. Analyze this SINGLE garment "
                        "and respond ONLY with valid JSON (no extra text). Use this schema:\n\n"
                        "{\n"
                        '  "category": "top/bottom/dress/outerwear/shoes/accessory",\n'
                        '  "main_colors": ["color1", "color2"],\n'
                        '  "style": "e.g., minimal, romantic, edgy, classic",\n'
                        '  "material_guess": "short phrase",\n'
                        '  "season_band": "hot/warm/mild/cool/cold",\n'
                        '  "suitable_occasions": ["casual", "office", "date", "party", "interview", ...],\n'
                        '  "mood": "e.g., powerful, playful, relaxed, elegant",\n'
                        '  "summary": "1–2 sentence human-readable description"\n'
                        "}\n\n"
                        "Important: Reply with JSON only."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                }
            ]
        }],
        "max_tokens": 400,
        "temperature": 0.2
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=90)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    # parse JSON safely
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # fallback: wrap in try/fix or store raw text
        data = {"summary": content}

    return data

def sanitize_meta(v):
    """Pinecone metadata cannot contain None or nested dicts."""
    if v is None:
        return ""  # or "unknown"
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        # Pinecone only allows list of strings
        return [str(x) for x in v if x is not None]
    # fallback: convert anything else to string
    return str(v)


# ==================== save to closet ====================
IMAGES_DIR = "closet_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

def save_to_closet(image_bytes, analysis, filename):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    vector = get_clip_vector(img)
    item_id = str(uuid.uuid4())

    save_name = f"{item_id}.jpg"
    save_path = os.path.join(IMAGES_DIR, save_name)
    img.save(save_path, format="JPEG", quality=90)

    # unpack analysis
    category = sanitize_meta(analysis.get("category"))
    main_colors = sanitize_meta(analysis.get("main_colors"))
    style = sanitize_meta(analysis.get("style"))
    material_guess = sanitize_meta(analysis.get("material_guess"))
    season_band = sanitize_meta(analysis.get("season_band"))
    suitable_occasions = sanitize_meta(analysis.get("suitable_occasions"))
    mood = sanitize_meta(analysis.get("mood"))
    summary = sanitize_meta(analysis.get("summary"))

    meta = {
        "user_id": "scarlett",
        "filename": sanitize_meta(filename),
        "uploaded_at": sanitize_meta(datetime.now().isoformat()),
        "image_path": sanitize_meta(save_path),
        "ai_category": category,
        "ai_colors": main_colors,              # becomes list[str] or ""
        "ai_style": style,
        "ai_material": material_guess,
        "ai_season_band": season_band,
        "ai_occasions": suitable_occasions,    # becomes list[str] or ""
        "ai_mood": mood,
        "ai_summary": summary,
    }

    index.upsert([(item_id, vector, meta)], namespace="scarlett")

    return item_id

# ==================== GPT-4o outfit advice ====================
def gpt_outfit_advice(user_profile, context, closet_items):
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT_TEXT}/chat/completions?api-version={AZURE_API_VER}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_KEY}

    system_msg = (
        "You are an expert fashion stylist.\n"
        "STRICT RULES:\n"
        "- You MUST ONLY recommend items that appear in the provided CLOSET.\n"
        "- DO NOT suggest buying anything.\n"
        "- Return JSON only.\n\n"
        "OUTFIT STRUCTURE RULE:\n"
        "- You must output ONE of these valid structures:\n"
        "  A) top + bottom (+ optional outerwear/accessory)\n"
        "  B) dress (+ optional outerwear/accessory)\n"
        "- Never output two tops unless one is explicitly 'outerwear'.\n"
        "- If you cannot find a required slot, leave it out and add it to missing_gaps (e.g., 'Bottom not available')."
    )


    user_msg = {
        "user_profile": user_profile,
        "scenario": context,
        "closet": closet_items,
        "output_schema": {
            "outfit": [{"closet_id": "...", "role": "top|bottom|dress|outerwear|shoes|accessory", "reason": "..."}],
            "alternatives": [{"outfit": [], "why": "..."}],
            "missing_gaps": ["..."],
            "style_notes": "..."
        }
    }

    payload = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_msg)}
        ],
        "temperature": 0.2,
        "max_tokens": 600,
        # If your Azure deployment supports it, this helps force JSON:
        # "response_format": {"type": "json_object"}
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    text = text.replace("```json", "").replace("```", "").strip()


    # Robust parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "GPT returned invalid JSON", "raw_text": text}


# ==================== Streamlit interface ====================
PAGES = ["Get Outfit", "Upload & Analyze", "My Closet"]

if "page" not in st.session_state:
    st.session_state.page = "Get Outfit"

# Apply navigation change BEFORE the radio widget is created
if "nav_target" in st.session_state:
    st.session_state.page = st.session_state.nav_target
    del st.session_state["nav_target"]

page = st.sidebar.radio(
    "Navigation",
    PAGES,
    index=PAGES.index(st.session_state.page),
    key="page",
)

# -------- Get Outfit (primary page) --------
if page == "Get Outfit":
    st.title("OutfitAI: Your Smart Outfit Assistant")
    st.caption("Upload → AI analyzes → permanently saved in your vector closet")

    st.header("Get an Outfit Recommendation")

    col1, col2 = st.columns(2)
    with col1:
        skin = st.selectbox("Your skin tone", ["Fair", "Light", "Medium", "Tan", "Deep"])
    with col2:
        occasion = st.selectbox("Occasion", ["Casual", "Office", "Date", "Party", "Interview"])

    temp = st.selectbox("Temperature", ["hot", "warm", "mild", "cool", "cold"])
    mood = st.selectbox("Mood", ["playful", "elegant", "casual", "edgy", "classic"])

    if st.button("Suggest Outfit", type="primary"):

        # 1) Build a semantic query from the scenario (this replaces filter-only)
        query_text = f"""
        Occasion: {occasion}.
        Temperature: {temp}.
        Mood: {mood}.
        Skin tone: {skin}.
        """

        qvec = get_clip_text_vector(query_text)

        # (Optional) you can still keep a lightweight filter if you want:
        # pc_filter = {"user_id": "scarlett"}

        with st.spinner("Searching your closet (vector search)..."):
            results = index.query(
                vector=qvec,
                top_k=20,
                namespace="scarlett",          # use namespace if you switched to it
                include_metadata=True
                # filter=pc_filter             # only if you still use metadata filter
            )

            # Pinecone client sometimes returns dict vs object depending on version
            matches = results["matches"] if isinstance(results, dict) else results.matches

        if not matches:
            st.warning("No good matches yet. Try uploading more garments!")
        else:
            # 2) Build closet context GPT-4o is allowed to use
            closet_items = []
            for m in matches[:12]:
                meta = m["metadata"] if isinstance(m, dict) else m.metadata
                closet_items.append({
                    "closet_id": (m["id"] if isinstance(m, dict) else m.id),
                    "category": meta.get("ai_category"),
                    "colors": meta.get("ai_colors"),
                    "style": meta.get("ai_style"),
                    "season_band": meta.get("ai_season_band"),
                    "occasions": meta.get("ai_occasions"),
                    "summary": meta.get("ai_summary"),
                    "image_path": meta.get("image_path"),
                    "filename": meta.get("filename"),
                })
            closet_by_id = {it["closet_id"]: it for it in closet_items}


            # 3) GPT-4o writes outfit ONLY from closet_items
            user_profile = {"skin_tone": skin}
            context = {"occasion": occasion, "temperature": temp, "mood": mood}

            with st.spinner("GPT-4o is composing an outfit (closet-only)..."):
                try:
                    advice = gpt_outfit_advice(user_profile, context, closet_items)
                    st.subheader("Outfit Plan (Closet-only)")
                    st.json(advice)
                except Exception as e:
                    st.error(f"GPT-4o output parsing failed: {e}")

            # 4) Show retrieved candidates visually
            st.subheader("Selected Outfit Pieces")

            selected = advice.get("outfit", [])

            if not selected:
                st.info("No outfit items were selected.")
            else:
                cols = st.columns(4)
                for i, pick in enumerate(selected):
                    cid = pick.get("closet_id")
                    item = closet_by_id.get(cid)

                    with cols[i % 4]:
                        if item and item.get("image_path") and os.path.exists(item["image_path"]):
                            st.image(item["image_path"], use_container_width=True)
                            st.caption(item.get("filename", "—"))
                            st.caption(pick.get("role", ""))
                            st.write(pick.get("reason", ""))
                        else:
                            st.warning("Selected item not found in retrieved closet set.")



    st.markdown("---")
    st.write("Don’t see something you like?")

    colA, colB = st.columns(2)
    with colA:
        if st.button("⬆️ Upload more garments"):
            st.session_state.nav_target = "Upload & Analyze"
            st.rerun()

    with colB:
        if st.button("👗 View my full closet"):
            st.session_state.nav_target = "My Closet"
            st.rerun()


# -------- Upload & Analyze page --------
elif page == "Upload & Analyze":
    st.header("Upload a New Garment")
    uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

    if uploaded:
        bytes_data = uploaded.read()
        st.image(bytes_data, use_container_width=True)

        if st.button("Analyze & Save to Closet", type="primary"):
            with st.spinner("Analyzing with GPT-4o Vision + saving to Pinecone..."):
                try:
                    analysis = analyze_with_gpt4o(bytes_data)
                    save_to_closet(bytes_data, analysis, uploaded.name)
                    st.success("Successfully saved to your closet!")
                    st.subheader("AI Analysis")
                    st.json(analysis)
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {e}")

# -------- My Closet page (REPLACED) --------
elif page == "My Closet":
    st.header("My Closet")

    # 1. Fetch all items
    with st.spinner("Loading from Pinecone..."):
        # Query a dummy vector to retrieve everything
        results = index.query(
            vector=[0.0] * 768,
            top_k=500,
            namespace="scarlett",
            include_metadata=True
        )
        items = results["matches"] if isinstance(results, dict) else results.matches

    if not items:
        st.info("Your closet is empty – upload your first piece!")
    else:
        st.success(f"You have {len(items)} garments in your closet")

        # 2. Display items in a grid
        cols = st.columns(3)  # Changed to 3 columns for better spacing
        for i, item in enumerate(items):
            # Handle different Pinecone return formats safely
            meta = item["metadata"] if isinstance(item, dict) else item.metadata
            item_id = item["id"] if isinstance(item, dict) else item.id

            image_path = meta.get("image_path")

            with cols[i % 3]:
                st.markdown("---")  # separator

                # A. Display Image
                if image_path and os.path.exists(image_path):
                    st.image(image_path, use_container_width=True)
                else:
                    st.error("🚫 Image file missing")

                # B. Display Details (The Fix)
                # We check if summary exists; if not, we build one from tags
                summary = meta.get("ai_summary", "")
                category = meta.get("ai_category", "Unknown")
                style = meta.get("ai_style", "Unknown")

                # If summary is too short or empty, show tags instead
                if len(str(summary)) < 5:
                    display_text = f"**Category:** {category}\n\n**Style:** {style}"
                else:
                    display_text = summary

                st.caption(f"**{meta.get('filename', 'Unknown')}**")

                with st.expander("📝 AI Details"):
                    st.write(display_text)
                    st.caption(f"Occasion: {meta.get('ai_occasions', '—')}")

                # C. Delete Button (The Fix)
                # unique key is needed for every button
                if st.button("🗑️ Delete", key=f"del_{item_id}"):
                    # 1. Delete from Pinecone
                    index.delete(ids=[item_id], namespace="scarlett")

                    # 2. Delete local file
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)

                    st.toast(f"Deleted {meta.get('filename')}")
                    st.rerun()

