import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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

# Free-tier friendly Flash & Gemma models ONLY (Pro and 2.5 models excluded to avoid limit:0 and 404s)
FREE_TIER_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
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


def fetch_free_news_context(query: str = "B2B marketing strategy brand growth AI market trends", max_items: int = 6) -> tuple[str, str]:
    """
    Fetch real, current news articles from Google News RSS for free (no API quota used).
    Acts as reliable grounding when Gemini's native google_search tool hits 429 quota limits.
    """
    encoded_q = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_q}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
    context_lines = []
    markdown_links = []

    try:
        req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        items = root.findall(".//item")[:max_items]
        for item in items:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            source_el = item.find("source")
            source_name = source_el.text.strip() if source_el is not None and source_el.text else "News Source"
            if title and link:
                context_lines.append(f"- {title} (Source: {source_name}, Date: {pub_date}) | URL: {link}")
                markdown_links.append(f"- [{title} ({source_name})]({link})")
    except Exception as e:
        print(f"ℹ️ Free RSS news fetch skipped: {e}")

    return "\n".join(context_lines), "\n".join(markdown_links)


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
    """Build a clean list of Free-Tier eligible models (skipping Pro & deprecated 2.5 models)."""
    env_model = (os.environ.get("GEMINI_MODEL") or "").strip()
    candidates = []

    if env_model and "2.5" not in env_model and "pro" not in env_model.lower():
        candidates.append(env_model)

    for m in FREE_TIER_MODELS:
        if m not in candidates:
            candidates.append(m)

    # Dynamically discover any additional Flash or Gemma text models on this API key
    try:
        skip_keywords = (
            "pro",        # Limit: 0 on free tier
            "2.5",        # Deprecated for new users
            "2.0",        # Deprecated
            "1.5",        # Deprecated
            "tts",
            "live",
            "image",
            "banana",
            "transcribe",
            "embedding",
            "lyria",
            "robotics",
            "omni",
            "argon",
            "computer-use",
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
    """
    Attempt generation cleanly without hammering rate limits.
    First tries ungrounded generation (enriched with live RSS feeds) because the
    Free Tier google_search tool quota frequently throws 429 across all models.
    """
    candidate_models = get_candidate_models(client)
    last_error = None

    ungrounded_config = types.GenerateContentConfig(
        system_instruction=system_instructions,
        temperature=0.7,
    )

    grounded_config = types.GenerateContentConfig(
        system_instruction=system_instructions,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.7,
    )

    # Pass 1: Standard generation (uses live RSS articles already injected into user_input)
    # This avoids the strict Free-Tier Google Search tool 429 block.
    for model_name in candidate_models:
        try:
            print(f"Trying model (standard + RSS grounded): {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=user_input,
                config=ungrounded_config,
            )
            if response and getattr(response, "text", None):
                print(f"✅ Successfully generated post using {model_name}")
                return response, model_name
        except errors.APIError as e:
            print(f"⚠️ {model_name} failed [{e.code}]: {e.message}")
            last_error = e
            if e.code == 429:
                time.sleep(3)  # Brief cooldown before trying next model
        except Exception as e:
            print(f"⚠️ {model_name} unexpected error: {e}")
            last_error = e

    # Pass 2: Try with native Google Search tool if Pass 1 failed for non-quota reasons
    for model_name in candidate_models[:4]:
        try:
            print(f"Trying model (with native Google Search tool): {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=user_input,
                config=grounded_config,
            )
            if response and getattr(response, "text", None):
                print(f"✅ Successfully generated post using {model_name} (native search)")
                return response, model_name
        except Exception as e:
            print(f"⚠️ {model_name} (native search) failed: {e}")
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

    # Fetch live Google News RSS articles for free so we never depend on paid Search tool quotas
    rss_context, rss_markdown_links = fetch_free_news_context()

    rss_prompt_block = (
        f"\n\nRecent Live Web Sources (Fetched {date_str}):\n{rss_context}\n"
        if rss_context
        else ""
    )

    user_input = (
        f"Date: {date_str} ({day_name})\n"
        f"Assigned Focus: {assigned_topic}"
        f"{rss_prompt_block}\n"
        "Use the provided recent web sources and credible industry data to ground the post. "
        "Generate one complete LinkedIn post draft for LELNOB following the editorial guidelines, "
        "followed by a '## Sources & Verification' section listing every factual claim and source."
    )

    # Disable internal tenacity 5x retries so a 429 on one model doesn't burn the whole minute's quota
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )

    response, used_model = generate_post_with_fallback(
        client=client,
        user_input=user_input,
        system_instructions=system_instructions,
    )

    draft_text = (response.text or "").strip()
    grounded_urls = extract_grounding_sources(response) or rss_markdown_links

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
