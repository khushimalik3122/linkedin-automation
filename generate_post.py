import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from google import genai
from google.genai import types

# Topic schedule mapped by Python weekday integer (0 = Monday, 2 = Wednesday, 4 = Friday)
WEEKDAY_TOPICS = {
    0: "Industry Research & Market Trends: Share a data-backed insight or emerging shift relevant to LELNOB's audience, citing recent sources.",
    2: "Thought Leadership & Execution: Break down a practical framework, operational lesson, or strategic perspective for LELNOB.",
    4: "Hypothetical Campaign Concept & Creative Strategy: Pitch a clearly labelled hypothetical campaign idea or breakdown of a recent industry move.",
}

DEFAULT_TOPIC = "General Brand Insight: Share a research-backed perspective relevant to LELNOB's core audience."


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


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

    # Default model is gemini-2.5-flash (free tier friendly + supports Google Search grounding)
    model = os.environ.get("GEMINI_MODEL") or "gemini-3.8-flash"
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

    response = client.models.generate_content(
        model=model,
        contents=user_input,
        config=types.GenerateContentConfig(
            system_instruction=system_instructions,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.7,
        ),
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
        f"model: {model}\n"
        f"status: DRAFT_ONLY_REQUIRES_HUMAN_REVIEW\n"
        f"---\n\n"
    )

    output_file.write_text(header + draft_text + "\n", encoding="utf-8")
    print(f"Saved draft to {output_file}")


if __name__ == "__main__":
    main()
