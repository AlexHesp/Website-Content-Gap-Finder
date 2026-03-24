"""
Website Content Gap Finder
===========================
Identifies topics, entities, and concepts that appear across your site content
but don't have dedicated pages — surfacing content gap opportunities.

Usage:
    python content_gap_analyzer.py --input inputs/crawl.csv
    python content_gap_analyzer.py --input inputs/crawl.csv --use-llm

Input CSV format:
    url, content (two columns — content is markdown from Crawl4AI)

Requirements:
    pip install -r requirements.txt
    python -m spacy download en_core_web_md
"""

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import spacy
from thefuzz import fuzz

# =============================================================================
# CONFIGURATION — edit these to customise for your project
# =============================================================================

# Minimum number of pages an entity must appear on to be considered a gap
MIN_PAGE_FREQUENCY = 3

# Minimum fuzzy match score (0-100) to consider a URL as "covering" a topic
FUZZY_MATCH_THRESHOLD = 75

# Seed terms — only entities matching these terms are surfaced as gaps.
# This lets you focus results on your site's topic areas.
# When empty, all entities passing the frequency threshold are included.
# Example: ["react", "nextjs", "typescript", "css", "testing", "api"]
SEED_TERMS = []

# Terms to exclude — these appear everywhere but aren't content gaps.
# Example: ["google", "amazon", "team", "teams", "solution"]
STOPWORD_ENTITIES = []

# Description of your site/business — used in the LLM prompt when --use-llm
# is enabled. The LLM uses this to judge whether a term is a relevant gap.
# Example: "Acme Corp, a B2B SaaS company that provides project management tools"
LLM_PROMPT_CONTEXT = ""  # e.g. "Acme Corp, a B2B SaaS company that provides project management tools"

# Approximate token costs per 1M tokens for gpt-4o-mini
MODEL_PRICING = {"input": 0.08, "output": 0.60}


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

def clean_markdown(text: str) -> str:
    """Strip markdown formatting to get clean text for NLP processing."""
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_entities(nlp, text: str, stopword_entities: set) -> set:
    """
    Extract named entities and key noun phrases from text using spaCy.
    Returns a set of lowercased entity strings.
    """
    clean_text = clean_markdown(text)
    doc = nlp(clean_text)

    entities = set()

    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT", "WORK_OF_ART", "LAW", "EVENT", "GPE", "NORP"}:
            normalised = ent.text.lower().strip()
            if len(normalised) > 2 and normalised not in stopword_entities:
                entities.add(normalised)

    for chunk in doc.noun_chunks:
        words = chunk.text.lower().strip().split()
        if 2 <= len(words) <= 5:
            cleaned_words = []
            skip_pos = {"DET", "PRON", "ADP", "CCONJ", "PUNCT", "NUM"}
            started = False
            for token in chunk:
                if not started and token.pos_ in skip_pos:
                    continue
                started = True
                cleaned_words.append(token.text.lower())

            phrase = " ".join(cleaned_words)
            if len(phrase) > 3 and phrase not in stopword_entities:
                entities.add(phrase)

    return entities


# =============================================================================
# URL / TOPIC MATCHING
# =============================================================================

def extract_slug_topic(url: str) -> str:
    """Extract a readable topic from a URL slug."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return ""
    segments = [s for s in path.split("/") if s]
    if not segments:
        return ""
    slug = segments[-1]
    topic = slug.replace("-", " ").replace("_", " ").lower()
    return topic


def get_existing_topics(urls: list) -> dict:
    """Build a mapping of URL -> primary topic for all existing pages."""
    topics = {}
    for url in urls:
        topic = extract_slug_topic(url)
        if topic:
            topics[url] = topic
    return topics


def entity_has_page(entity: str, existing_topics: dict, threshold: int) -> tuple:
    """
    Check if an entity already has a dedicated page.
    Returns (bool, best_matching_url or None)
    """
    best_score = 0
    best_url = None

    for url, topic in existing_topics.items():
        if entity in topic or topic in entity:
            return True, url

        score = fuzz.token_sort_ratio(entity, topic)
        if score > best_score:
            best_score = score
            best_url = url

    if best_score >= threshold:
        return True, best_url

    return False, None


# =============================================================================
# RELEVANCE FILTERING
# =============================================================================

def is_relevant_to_seed_terms(entity: str, seed_terms: list) -> bool:
    """
    Check if an entity is relevant to the configured seed terms.
    Uses token overlap with seed terms.
    """
    entity_lower = entity.lower()
    entity_tokens = set(entity_lower.split())

    for seed in seed_terms:
        seed_lower = seed.lower()
        seed_tokens = set(seed_lower.split())

        if seed_lower in entity_lower or entity_lower in seed_lower:
            return True

        overlap = entity_tokens & seed_tokens
        meaningful_overlap = {w for w in overlap if len(w) > 2}
        if meaningful_overlap:
            return True

    return False


# =============================================================================
# LLM FILTERING
# =============================================================================

LLM_CONCURRENCY = 20  # Max simultaneous API requests


def build_system_prompt(context: str) -> str:
    """
    Build the static system prompt (cached by OpenAI across requests).
    Everything that doesn't change per-term goes here.
    """
    return f"""You are an SEO strategist evaluating content gaps for {context}.

You will receive a single term/topic extracted from the website's content
that doesn't currently have a dedicated page.

Decide whether this term represents a genuine content gap — a real topic
someone would search for that the site could create a dedicated page about.

Mark relevant=FALSE for any of the following:
- Garbled, truncated, or concatenated text — fragments that aren't coherent
  phrases. Watch for odd spacing around hyphens/ampersands/slashes, or
  text that mashes together unrelated phrases.
- Marketing slogans, CTAs, or sales copy — promotional language rather
  than informational topics (e.g. "get a free quote", "trusted by thousands",
  "get started today")
- Internal/operational references — specific office locations, branch names,
  team names, or company-specific labels
- Generic adjective phrases that describe product traits rather than
  searchable topics (e.g. "easy to use", "great value", "premium quality")
- UI/navigation fragments or spec-table labels — text that looks like it
  was pulled from menus, table headers, or filter labels rather than
  actual content
- Overly vague or broad terms that are too generic to be a standalone page
  (e.g. "locations", "options", "features", "types")
- Pricing/commercial terms — transactional phrases about costs, plans, or
  fees rather than informational topics people search for

Mark relevant=TRUE only if ALL of these are true:
- The term is a coherent, real topic someone would plausibly search for
- The site could write a meaningful, informational standalone page about it
- It relates to the site's products, services, or industry knowledge

Respond with ONLY a JSON object, no other text:
{{"term": "<the term>", "relevant": true/false}}
"""


def estimate_llm_cost(candidate_count: int, context: str) -> dict:
    """
    Estimate token usage and cost for LLM filtering with gpt-4o-mini.
    One API call per term, with the system prompt cached after the first call.
    """
    # System prompt: ~150 tokens (sent every time, cached after first)
    # User message: ~10 tokens per term
    # Output: ~15 tokens per term (small JSON object)
    system_tokens = 150 + len(context.split()) * 2
    input_per_call = system_tokens + 10
    output_per_call = 15

    total_input = input_per_call * candidate_count
    total_output = output_per_call * candidate_count

    # After first request, system prompt is cached at 50% discount
    cached_savings = (system_tokens * (candidate_count - 1) * 0.5) if candidate_count > 1 else 0
    effective_input = total_input - cached_savings

    input_cost = (effective_input / 1_000_000) * MODEL_PRICING["input"]
    output_cost = (total_output / 1_000_000) * MODEL_PRICING["output"]

    return {
        "api_calls": candidate_count,
        "total_input_tokens": int(effective_input),
        "total_output_tokens": total_output,
        "estimated_cost": input_cost + output_cost,
    }


async def _evaluate_term(
    client,
    gap: dict,
    system_msg: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Evaluate a single term via the OpenAI API."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": gap["entity"]},
                ],
                temperature=0,
            )
            result = json.loads(response.choices[0].message.content)
            gap["llm_relevant"] = result.get("relevant", False)
        except Exception:
            gap["llm_relevant"] = None
    return gap


async def _filter_all(candidate_gaps: list, context: str) -> list:
    """Run all LLM evaluations concurrently with a concurrency limit."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()  # Uses OPENAI_API_KEY env var
    system_msg = build_system_prompt(context)
    semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

    total = len(candidate_gaps)
    completed = 0

    async def evaluate_and_report(gap):
        nonlocal completed
        result = await _evaluate_term(client, gap, system_msg, semaphore)
        completed += 1
        if completed % 25 == 0 or completed == total:
            print(f"  Evaluated {completed}/{total} terms...")
        return result

    tasks = [evaluate_and_report(gap) for gap in candidate_gaps]
    return await asyncio.gather(*tasks)


def filter_with_llm(candidate_gaps: list, context: str) -> list:
    """
    Pass candidate gaps through gpt-4o-mini one-by-one (async).
    Returns the gaps list with an added 'llm_relevant' field.
    """
    return asyncio.run(_filter_all(candidate_gaps, context))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Content Gap Analyzer")
    parser.add_argument("--input", "-i", required=True,
                        help="Input CSV (url, content)")
    parser.add_argument("--min-freq", type=int, default=MIN_PAGE_FREQUENCY,
                        help=f"Minimum page frequency (default: {MIN_PAGE_FREQUENCY})")
    parser.add_argument("--include-all", action="store_true",
                        help="Include all entities, not just seed-term-relevant ones")
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable LLM filtering pass with OpenAI (gpt-4o-mini)")
    args = parser.parse_args()

    # Load spaCy model
    print("[1/5] Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        print("  [!] Model not found. Install it with:")
        print("      python -m spacy download en_core_web_md")
        sys.exit(1)

    nlp.max_length = 2_000_000

    # Read input CSV
    print(f"[2/5] Reading input CSV: {args.input}")
    try:
        df = pd.read_csv(args.input, dtype=str).fillna("")
    except Exception as e:
        print(f"  [!] Failed to read CSV: {e}")
        sys.exit(1)

    if "url" not in df.columns or "content" not in df.columns:
        print("  [!] CSV must have 'url' and 'content' columns")
        sys.exit(1)

    total_pages = len(df)
    print(f"  Found {total_pages} pages")

    # Extract entities from all pages
    print(f"[3/5] Extracting entities from {total_pages} pages...")
    entity_pages = defaultdict(set)

    for idx, row in df.iterrows():
        url = row["url"].strip()
        content = row["content"]

        if not content or len(content) < 50:
            continue

        entities = extract_entities(nlp, content, set(STOPWORD_ENTITIES))

        for entity in entities:
            entity_pages[entity].add(url)

        if (idx + 1) % 100 == 0 or idx == total_pages - 1:
            print(f"  Processed {idx + 1}/{total_pages} pages...")

    print(f"  Extracted {len(entity_pages)} unique entities/phrases")

    # Build existing topic map
    print("[4/5] Matching entities against existing pages...")
    all_urls = df["url"].tolist()
    existing_topics = get_existing_topics(all_urls)

    # Find gaps
    gaps = []
    for entity, pages in entity_pages.items():
        page_count = len(pages)

        if page_count < args.min_freq:
            continue

        has_page, matching_url = entity_has_page(
            entity, existing_topics, FUZZY_MATCH_THRESHOLD
        )
        if has_page:
            continue

        if not args.include_all and SEED_TERMS:
            if not is_relevant_to_seed_terms(entity, SEED_TERMS):
                continue

        gaps.append({
            "entity": entity,
            "page_frequency": page_count,
            "appears_on": " | ".join(sorted(pages)[:5]),
            "total_pages_mentioned": page_count,
        })

    print(f"  Found {len(gaps)} potential content gaps")

    # Setup output directory and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(args.input)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Sort by frequency (most mentioned first)
    gaps.sort(key=lambda x: x["page_frequency"], reverse=True)

    # Write pre-LLM results
    base_cols = ["entity", "page_frequency", "appears_on", "total_pages_mentioned"]

    if gaps:
        pre_llm_path = os.path.join(output_dir, f"content_gaps_{timestamp}.csv")
        pre_llm_df = pd.DataFrame(gaps)[base_cols]
        pre_llm_df.to_csv(pre_llm_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"\n[5/5] {len(gaps)} content gaps written to {pre_llm_path}")

        print(f"\n  Top 10 gaps:")
        for i, gap in enumerate(gaps[:10], 1):
            print(f"    {i}. {gap['entity']} "
                  f"(mentioned on {gap['page_frequency']} pages)")
    else:
        print("\n  No content gaps found. Try lowering --min-freq "
              "or using --include-all")

    # LLM filtering
    if args.use_llm and gaps:
        estimate = estimate_llm_cost(
            candidate_count=len(gaps),
            context=LLM_PROMPT_CONTEXT,
        )

        print(f"\n  LLM Filtering Summary:")
        print(f"    Model: gpt-4o-mini")
        print(f"    Candidates to evaluate: {len(gaps)}")
        print(f"    API calls: {estimate['api_calls']}")
        print(f"    Estimated tokens: "
              f"~{estimate['total_input_tokens']:,} input, "
              f"~{estimate['total_output_tokens']:,} output")
        print(f"    Estimated cost: "
              f"~${estimate['estimated_cost']:.4f}")
        print(f"    API key: OPENAI_API_KEY environment variable")

        try:
            answer = input("\n  Proceed with LLM filtering? [Y/n]: ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer.lower() in ("", "y", "yes"):
            print(f"\n  Running LLM filtering with gpt-4o-mini...")
            gaps = filter_with_llm(gaps, context=LLM_PROMPT_CONTEXT)
            llm_approved = [g for g in gaps if g.get("llm_relevant") is True]
            print(f"  {len(llm_approved)} of {len(gaps)} gaps "
                  f"confirmed as relevant by LLM")

            if llm_approved:
                llm_path = os.path.join(
                    output_dir, f"content_gaps_llm_{timestamp}.csv"
                )
                llm_df = pd.DataFrame(llm_approved)
                llm_cols = [c for c in base_cols + ["llm_relevant"] if c in llm_df.columns]
                llm_df[llm_cols].to_csv(llm_path, index=False, quoting=csv.QUOTE_ALL)
                print(f"  LLM-filtered gaps written to {llm_path}")
        else:
            print("  Skipping LLM filtering.")


if __name__ == "__main__":
    main()
