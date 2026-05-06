# src/phase2_content_engine.py
# Phase 2: Autonomous Content Engine using LangGraph
# Flow: Decide Search Query → Web Search → Draft Post
# Output: Strict JSON {bot_id, topic, post_content}

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

load_dotenv()

# ─────────────────────────────────────────
# 1. BOT PERSONAS
# ─────────────────────────────────────────

BOT_PERSONAS = {
    "bot_a": {
        "id": "bot_a",
        "name": "Tech Maximalist",
        "system_prompt": (
            "You are Bot A, the Tech Maximalist. You believe AI and crypto will solve "
            "all human problems. You are highly optimistic about technology, Elon Musk, "
            "and space exploration. You dismiss regulatory concerns. You speak with "
            "supreme confidence and use tech buzzwords. You never break character."
        ),
    },
    "bot_b": {
        "id": "bot_b",
        "name": "Doomer / Skeptic",
        "system_prompt": (
            "You are Bot B, the Doomer. You believe late-stage capitalism and tech "
            "monopolies are destroying society. You are highly critical of AI, social "
            "media, and billionaires. You speak with cynicism and urgency. You never "
            "break character."
        ),
    },
    "bot_c": {
        "id": "bot_c",
        "name": "Finance Bro",
        "system_prompt": (
            "You are Bot C, the Finance Bro. You strictly care about markets, interest "
            "rates, trading algorithms, and making money. You speak in finance jargon "
            "and view everything through the lens of ROI. You never break character."
        ),
    },
}

# ─────────────────────────────────────────
# 2. LLM SETUP
# ─────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

# ─────────────────────────────────────────
# 3. MOCK SEARCH TOOL
# ─────────────────────────────────────────

@tool
def mock_searxng_search(query: str) -> str:
    """
    Mock search engine. Returns hardcoded recent headlines
    based on keywords in the query.
    """
    query_lower = query.lower()

    if any(k in query_lower for k in ["crypto", "bitcoin", "blockchain", "btc"]):
        return (
            "HEADLINE: Bitcoin surpasses $100k as spot ETF inflows hit record $2B in a day. "
            "HEADLINE: Ethereum Layer-2 adoption triples as gas fees drop to near zero. "
            "HEADLINE: SEC approves three new crypto ETFs amid growing institutional demand."
        )
    elif any(k in query_lower for k in ["ai", "openai", "llm", "gpt", "artificial intelligence", "model"]):
        return (
            "HEADLINE: OpenAI releases GPT-5 with autonomous agent capabilities. "
            "HEADLINE: Google DeepMind claims new AI model solves graduate-level math. "
            "HEADLINE: 40% of entry-level coding jobs automated by AI tools, study finds."
        )
    elif any(k in query_lower for k in ["market", "fed", "interest rate", "stocks", "wall street", "inflation"]):
        return (
            "HEADLINE: Federal Reserve holds rates steady, signals two cuts by end of 2025. "
            "HEADLINE: S&P 500 hits all-time high as tech earnings beat expectations. "
            "HEADLINE: Inflation cools to 2.1%, closest to Fed target in three years."
        )
    elif any(k in query_lower for k in ["privacy", "meta", "surveillance", "data", "facebook"]):
        return (
            "HEADLINE: Meta fined $1.3B by EU regulators for illegal data transfers. "
            "HEADLINE: New study reveals 87% of apps share location data without consent. "
            "HEADLINE: US Senate debates federal privacy bill amid Big Tech lobbying surge."
        )
    elif any(k in query_lower for k in ["space", "elon", "tesla", "spacex", "mars"]):
        return (
            "HEADLINE: SpaceX Starship completes first successful orbital flight. "
            "HEADLINE: Elon Musk announces Mars colony target date of 2030. "
            "HEADLINE: Tesla Full Self-Driving achieves 99.96% safety record in new study."
        )
    else:
        return (
            "HEADLINE: Global tech investment reaches record $500B in Q1 2025. "
            "HEADLINE: AI adoption accelerates across all major industries worldwide. "
            "HEADLINE: Digital transformation reshapes economy as automation spreads."
        )

# ─────────────────────────────────────────
# 4. LANGGRAPH STATE
# ─────────────────────────────────────────

class BotState(TypedDict):
    bot_id: str
    bot_name: str
    system_prompt: str
    search_query: Optional[str]
    search_results: Optional[str]
    topic: Optional[str]
    post_content: Optional[str]
    final_output: Optional[dict]

# ─────────────────────────────────────────
# 5. LANGGRAPH NODES
# ─────────────────────────────────────────

def node_decide_search(state: BotState) -> BotState:
    """
    Node 1: The LLM decides what topic to post about today
    and formats a search query based on the bot's persona.
    """
    print(f"\n[Node 1] Deciding search query for {state['bot_name']}...")

    messages = [
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=(
            "You are about to make a post on social media. "
            "Based on your persona, decide what topic you want to post about today. "
            "Respond with ONLY a JSON object in this exact format, nothing else:\n"
            '{"topic": "brief topic name", "search_query": "search query string"}'
        )),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    print(f"  Topic: {parsed['topic']}")
    print(f"  Search Query: {parsed['search_query']}")

    return {
        **state,
        "topic": parsed["topic"],
        "search_query": parsed["search_query"],
    }


def node_web_search(state: BotState) -> BotState:
    """
    Node 2: Executes the mock search tool using the query from Node 1.
    """
    print(f"\n[Node 2] Running search: \"{state['search_query']}\"...")

    results = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"  Results: {results[:80]}...")

    return {
        **state,
        "search_results": results,
    }


def node_draft_post(state: BotState) -> BotState:
    """
    Node 3: LLM uses persona + search results to draft a post.
    Output is a strict JSON object.
    """
    print(f"\n[Node 3] Drafting post for {state['bot_name']}...")

    messages = [
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=(
            f"You searched the web and found these headlines:\n{state['search_results']}\n\n"
            f"Now write a highly opinionated social media post about: {state['topic']}\n\n"
            "Rules:\n"
            "- Maximum 280 characters\n"
            "- Stay completely in character\n"
            "- Be provocative and opinionated\n"
            "- Respond with ONLY a JSON object, nothing else:\n"
            '{"bot_id": "...", "topic": "...", "post_content": "..."}'
        )),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)

    # Enforce bot_id from state, not LLM hallucination
    parsed["bot_id"] = state["bot_id"]
    parsed["topic"] = state["topic"]

    # Enforce 280 char limit
    if len(parsed["post_content"]) > 280:
        parsed["post_content"] = parsed["post_content"][:277] + "..."

    print(f"  Post ({len(parsed['post_content'])} chars): {parsed['post_content']}")

    return {
        **state,
        "post_content": parsed["post_content"],
        "final_output": parsed,
    }

# ─────────────────────────────────────────
# 6. BUILD THE GRAPH
# ─────────────────────────────────────────

def build_content_graph() -> StateGraph:
    """
    Assembles the 3-node LangGraph state machine:
    decide_search → web_search → draft_post → END
    """
    graph = StateGraph(BotState)

    graph.add_node("decide_search", node_decide_search)
    graph.add_node("web_search", node_web_search)
    graph.add_node("draft_post", node_draft_post)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search", "draft_post")
    graph.add_edge("draft_post", END)

    return graph.compile()


# ─────────────────────────────────────────
# 7. RUNNER FUNCTION
# ─────────────────────────────────────────

def run_content_engine(bot_id: str) -> dict:
    """
    Runs the full LangGraph pipeline for a given bot.
    Returns the final JSON output.
    """
    if bot_id not in BOT_PERSONAS:
        raise ValueError(f"Unknown bot_id: {bot_id}")

    persona = BOT_PERSONAS[bot_id]

    initial_state: BotState = {
        "bot_id": persona["id"],
        "bot_name": persona["name"],
        "system_prompt": persona["system_prompt"],
        "search_query": None,
        "search_results": None,
        "topic": None,
        "post_content": None,
        "final_output": None,
    }

    graph = build_content_graph()
    final_state = graph.invoke(initial_state)

    return final_state["final_output"]


# ─────────────────────────────────────────
# 8. TEST RUN
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("TEST 1: Bot A — Tech Maximalist")
    print("=" * 55)
    result_a = run_content_engine("bot_a")
    print(f"\nFINAL JSON OUTPUT:\n{json.dumps(result_a, indent=2)}")

    print("\n" + "=" * 55)
    print("TEST 2: Bot C — Finance Bro")
    print("=" * 55)
    result_c = run_content_engine("bot_c")
    print(f"\nFINAL JSON OUTPUT:\n{json.dumps(result_c, indent=2)}")