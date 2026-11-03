import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from google import genai
from google.genai import errors, types

# Topic schedule mapped by Python weekday integer (0 = Monday, 2 = Wednesday, 4 = Friday)
WEEKDAY_TOPICS = {
    0: "Industry Research & Market Trends: Share a data-backed insight or emerging shift relevant to LELNOB's audience, citing recent sources.",
    2: "Thought Leadership & Execution: Break down a practical framework, operational lesson, or strategic perspective for LELNOB.",
    4: "Hypothetical Campaign Concept & Creative Strategy: Pitch a clearly labelled hypothetical campaign idea or breakdown of a recent industry move.",
}

DEFAULT_TOPIC = "General Brand Insight: Share a research-backed perspective relevant to LELNOB's core audience."

# Prioritized fallback list of active Gemini text models
DEFAULT_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
]


def get_ist_now() -> datetime:
    """Return current datetime in India Standard Time (UTC+5:30)."""
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist_tz)


def load_system_prompt() -> str:
    """Load the editorial prompt from prompts/lelnob_post_prompt.txt."""
    prompt_path = Path("prompts/lelnob_post_prompt.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file at {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def extract_grounding_sources(response) -> str:
    """Extract real clickable URLs from Gemini Google Search grounding metadata."""
    sources = []
    try:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            metadata = getattr(candidates[0], "grounding_metadata", None)
            chunks = getattr(metadata, "grounding_chunks", None) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web and getattr(web, "uri", None):
                    title = getattr(web, "title", None) or web.uri
                    entry = f"- [{title}]({web.uri})"
                    if entry not in sources:
                        sources.append(entry)
    except Exception:
        pass
    return "\n".join(sources)


def get_candidate_models(client: genai.Client) -> list[str]:
    """Build a deduplicated list of preset and dynamically discovered models."""
    env_model = (os.environ.get("GEMINI_MODEL") or "").strip()
    candidates = []

    if env_model:
        candidates.append(env_model)

    for m in DEFAULT_FALLBACK_MODELS:
        if m not in candidates:
            candidates.append(m)

    # Dynamically fetch models supported by this specific API key as backup
    try:
        skip_keywords = (
            "tts",
            "live",
            "image",
            "banana",
            "transcribe",
            "embedding",
            "lyria",
            "robotics",
            "omni",
            "gemma",
            "argon",
        )
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            supported = getattr(m, "supported_actions", []) or []
            if (
                name
                and "generateContent" in supported
                and not any(k in name.lower() for k in skip_keywords)
                and name not in candidates
            ):
                candidates.append(name)
    except Exception as e:
        print(f"⚠️ Could not dynamically list models: {e}")

    return candidates


def generate_post_with_fallback(
    client: genai.Client, user_input: str, system_instructions: str
) -> tuple[object, str]:
    """Try models in order; fall back across models and grounding settings if needed."""
    candidate_models = get_candidate_models(client)
    last_error = None

    grounded_config = types.GenerateContentConfig(
        system_instruction=system_instructions,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.7,
    )

    ungrounded_config = types.GenerateContentConfig(
        system_instruction=system_instructions,
        temperature=0.7,
    )

    # Pass 1: Try candidate models WITH Google Search Grounding
    for model_name in candidate_models:
        try:
            print(f"Trying model (with Google Search grounding): {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=user_input,
                config=grounded_config,
            )
            if response and getattr(response, "text", None):
                print(f"✅ Successfully generated post using {model_name}")
                return response, model_name
        except errors.APIError as e:
            print(f"⚠️ {model_name} (grounded) failed [{e.code}]: {e.message}")
            last_error = e
        except Exception as e:
            print(f"⚠️ {model_name} (grounded) unexpected error: {e}")
            last_error = e

    # Pass 2: If grounding tool caused failures across models, retry WITHOUT Google Search tool
    print("🔄 Grounded calls failed. Retrying candidate models without search tool...")
    for model_name in candidate_models:
        try:
            print(f"Trying model (standard generation): {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=user_input,
                config=ungrounded_config,
            )
            if response and getattr(response, "text", None):
                print(f"✅ Successfully generated post using {model_name} (ungrounded)")
                return response, model_name
        except Exception as e:
            print(f"⚠️ {model_name} (ungrounded) failed: {e}")
            last_error = e

    raise RuntimeError(f"All Gemini fallback models failed. Last error: {last_error}")


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

    now_ist = get_ist_now()
    weekday = now_ist.weekday()
    day_name = now_ist.strftime("%A")
    date_str = now_ist.strftime("%Y-%m-%d")

    assigned_topic = WEEKDAY_TOPICS.get(weekday, DEFAULT_TOPIC)
    system_instructions = load_system_prompt()

    user_input = (
        f"Date: {date_str} ({day_name})\n"
        f"Assigned Focus: {assigned_topic}\n\n"
        "Use Google Search to find recent, credible information and statistics to ground the post. "
        "Generate one complete LinkedIn post draft for LELNOB following the editorial guidelines, "
        "followed by a '## Sources & Verification' section listing every factual claim and source."
    )

    client = genai.Client(api_key=api_key)

    response, used_model = generate_post_with_fallback(
        client=client,
        user_input=user_input,
        system_instructions=system_instructions,
    )

    draft_text = (response.text or "").strip()
    grounded_urls = extract_grounding_sources(response)

    if grounded_urls:
        draft_text += f"\n\n### Grounded Web Sources (Auto-Captured)\n{grounded_urls}"

    drafts_dir = Path("drafts")
    drafts_dir.mkdir(parents=True, exist_ok=True)

    output_file = drafts_dir / f"{date_str}-lelnob-linkedin-draft.md"

    header = (
        f"---\n"
        f"brand: LELNOB\n"
        f"date: {date_str}\n"
        f"day: {day_name}\n"
        f"model: {used_model}\n"
        f"status: DRAFT_ONLY_REQUIRES_HUMAN_REVIEW\n"
        f"---\n\n"
    )

    output_file.write_text(header + draft_text + "\n", encoding="utf-8")
    print(f"Saved draft to {output_file} (generated with {used_model})")


if __name__ == "__main__":
    main()
