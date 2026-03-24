# Website Content Gap Finder

A Python CLI tool that identifies content gap opportunities for any website. It uses NLP (spaCy) to extract entities and noun phrases from your site's pages, then finds topics that are frequently mentioned but don't have dedicated pages — revealing where you could create new content.

Optionally passes candidates through GPT-4o-mini to filter out noise (garbled text, marketing slogans, generic phrases) and keep only genuine content gaps.

## How it works

1. **Extracts entities** — Uses spaCy NER and noun phrase chunking on each page's content
2. **Counts frequency** — Tracks how many pages mention each entity
3. **Matches against existing pages** — Uses fuzzy matching to check if a topic already has a dedicated URL
4. **Surfaces gaps** — Entities that appear on many pages but have no dedicated page are flagged as content gaps
5. **LLM filtering (optional)** — Sends candidates to GPT-4o-mini to confirm they're real, searchable topics

## Quick start

```bash
# Clone the repo
git clone https://github.com/AlexHesp/Website-Content-Gap-Finder.git
cd Website-Content-Gap-Finder

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

## Input format

Place your crawl CSV in the `inputs/` folder. It needs `url` and `content` columns — the content should be markdown. You can generate this by scraping your site with [Crawl4AI](https://github.com/unclecode/crawl4ai) and exporting each page's URL and markdown content.

| url | content |
|-----|---------|
| https://example.com/services/ | # Our Services\n\nWe provide... |

## Usage

```bash
# Try it with the included sample data
python content_gap_analyzer.py --input inputs/sample_input.csv --min-freq 2

# Run on your own crawl data
python content_gap_analyzer.py --input inputs/crawl.csv

# Lower frequency threshold (useful for smaller sites)
python content_gap_analyzer.py --input inputs/crawl.csv --min-freq 2

# Include all entities (skip seed term filtering)
python content_gap_analyzer.py --input inputs/crawl.csv --include-all

# Enable LLM filtering to remove noise
python content_gap_analyzer.py --input inputs/crawl.csv --use-llm
```

Results are saved to an `outputs/` directory alongside your input file.

## Output

CSV with columns:

- **entity** — the topic/term identified as a gap
- **page_frequency** — number of pages it appears on
- **appears_on** — up to 5 example URLs where it's mentioned
- **total_pages_mentioned** — total count

With `--use-llm`, a second CSV is produced containing only LLM-approved gaps.

## Configuration

Edit the top of `content_gap_analyzer.py` to customise for your site:

| Setting | Default | Description |
|---------|---------|-------------|
| `MIN_PAGE_FREQUENCY` | `3` | Minimum pages an entity must appear on |
| `FUZZY_MATCH_THRESHOLD` | `75` | How closely a URL must match to count as "covered" (0-100) |
| `SEED_TERMS` | `[]` | Focus results on specific topic areas. Empty = include all |
| `STOPWORD_ENTITIES` | `[]` | Generic terms to always exclude |
| `LLM_PROMPT_CONTEXT` | `""` | Description of your site/business for the LLM prompt |

## LLM filtering setup

1. Install the OpenAI package (included in `requirements.txt`)
2. Set your API key:
   ```bash
   export OPENAI_API_KEY=your-key-here
   ```
   Or create a `.env` file (git-ignored):
   ```
   OPENAI_API_KEY=your-key-here
   ```
3. Run with `--use-llm` — the tool will show an estimated cost and ask for confirmation before making API calls

## License

MIT
