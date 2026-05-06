# src/phase1_router.py
# Phase 1: Vector-Based Persona Matching
# Uses sentence-transformers for embeddings and FAISS for vector storage.
# Finds which bots "care" about a given post using cosine similarity.

import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# 1. BOT PERSONAS
# Enriched with topic keywords so the embeddings
# overlap better with real post content.
# ─────────────────────────────────────────

BOT_PERSONAS = [
    {
        "id": "bot_a",
        "name": "Tech Maximalist",
        "persona": (
            "I believe AI and crypto will solve all human problems. "
            "I am highly optimistic about technology, Elon Musk, and space exploration. "
            "I dismiss regulatory concerns. "
            "Topics I care about: artificial intelligence, machine learning, OpenAI, "
            "ChatGPT, large language models, automation, robotics, cryptocurrency, "
            "Bitcoin, blockchain, startups, Silicon Valley, SpaceX, Tesla."
        ),
    },
    {
        "id": "bot_b",
        "name": "Doomer / Skeptic",
        "persona": (
            "I believe late-stage capitalism and tech monopolies are destroying society. "
            "I am highly critical of AI, social media, and billionaires. "
            "I value privacy and nature. "
            "Topics I care about: surveillance, data privacy, Meta, Facebook, Google, "
            "Big Tech, AI risks, automation unemployment, wealth inequality, "
            "climate change, corporate exploitation, regulatory capture."
        ),
    },
    {
        "id": "bot_c",
        "name": "Finance Bro",
        "persona": (
            "I strictly care about markets, interest rates, trading algorithms, "
            "and making money. I speak in finance jargon and view everything "
            "through the lens of ROI. "
            "Topics I care about: stock market, S&P 500, Federal Reserve, interest rates, "
            "rate cuts, inflation, hedge funds, derivatives, earnings reports, "
            "Wall Street, trading, portfolio, bonds, equities, GDP."
        ),
    },
]

# ─────────────────────────────────────────
# 2. LOAD EMBEDDING MODEL
# ─────────────────────────────────────────

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")

# ─────────────────────────────────────────
# 3. BUILD FAISS INDEX
# ─────────────────────────────────────────

def build_persona_index(personas: list) -> tuple:
    """
    Embeds each bot persona and stores vectors in a FAISS index.
    Returns the index and the list of persona metadata.
    """
    texts = [p["persona"] for p in personas]
    embeddings = model.encode(texts, normalize_embeddings=True)  # L2-normalized for cosine sim

    dim = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2

    # IndexFlatIP = Inner Product on normalized vectors = Cosine Similarity
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    print("FAISS index built.")
    print(f"  Stored {index.ntotal} persona vectors (dim={dim})\n")

    return index, personas


# ─────────────────────────────────────────
# 4. ROUTING FUNCTION
# ─────────────────────────────────────────

def route_post_to_bots(
    post_content: str,
    index: faiss.IndexFlatIP,
    personas: list,
    threshold: float = 0.20,
) -> list:
    """
    Given a post, returns all bots whose persona similarity exceeds the threshold.

    NOTE on threshold: The assignment specifies 0.85, which is calibrated for
    OpenAI text-embedding-ada-002 style embeddings that produce very tight,
    high-magnitude similarity scores. sentence-transformers (all-MiniLM-L6-v2)
    compares semantic meaning of full sentences, so cross-domain scores
    (persona description vs. news post) naturally land in the 0.15–0.35 range.
    We use 0.20 here for realistic routing, as explicitly permitted by the
    assignment: "tweak threshold depending on your embedding model."
    """

    # Embed the incoming post
    post_vector = model.encode([post_content], normalize_embeddings=True)
    post_vector = np.array(post_vector, dtype=np.float32)

    # Query FAISS — get similarity scores for ALL bots
    scores, indices = index.search(post_vector, k=len(personas))

    matched_bots = []
    print(f"Post: \"{post_content}\"")
    print(f"Threshold: {threshold}\n")
    print(f"{'Bot':<20} {'ID':<10} {'Score':<10} {'Matched?'}")
    print("-" * 55)

    for score, idx in zip(scores[0], indices[0]):
        bot = personas[idx]
        matched = score >= threshold
        print(f"{bot['name']:<20} {bot['id']:<10} {score:.4f}     {'✅ YES' if matched else '❌ NO'}")
        if matched:
            matched_bots.append({
                "bot_id": bot["id"],
                "bot_name": bot["name"],
                "similarity_score": round(float(score), 4),
            })

    print()
    return matched_bots


# ─────────────────────────────────────────
# 5. TEST RUN
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Build the index once
    index, personas = build_persona_index(BOT_PERSONAS)

    # --- Test Post 1: AI / Tech topic ---
    post1 = "OpenAI just released a new model that might replace junior developers."
    results1 = route_post_to_bots(post1, index, personas)
    print("Matched Bots:", results1)

    print("\n" + "="*55 + "\n")

    # --- Test Post 2: Finance topic ---
    post2 = "Federal Reserve signals three rate cuts in 2025, markets rally hard."
    results2 = route_post_to_bots(post2, index, personas)
    print("Matched Bots:", results2)

    print("\n" + "="*55 + "\n")

    # --- Test Post 3: Anti-tech / Privacy topic ---
    post3 = "Meta is harvesting your data and selling it to insurance companies."
    results3 = route_post_to_bots(post3, index, personas)
    print("Matched Bots:", results3)