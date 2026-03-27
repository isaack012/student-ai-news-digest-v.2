import os
from textwrap import shorten
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import requests
from bs4 import BeautifulSoup

load_dotenv(override=True)

MODEL_NAME = "gpt-4o-mini"
MAX_ARTICLE_CHARS = 12000

def extract_article(url: str) -> Dict[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(paragraphs)

    if not text.strip():
        raise ValueError("Could not extract article text.")

    return {
        "url": url,
        "title": "Extracted Article",
        "text": text[:MAX_ARTICLE_CHARS],
    }
    
def summarize_article(
    client: OpenAI, topic: str, audience: str, article: Dict[str, str]
) -> str:
    """Summarize an article into audience-aware bullet points."""
    prompt = f"""
You are creating a tech and AI news digest about "{topic}".

Audience: {audience}
Article title: {article["title"]}
Article URL: {article["url"]}

Summarize the article into 4-6 concise bullet points.
Focus on the most important ideas, technologies, announcements, and implications.
Write in a clear, engaging tone appropriate for {audience}.
Avoid repeating the article title as a bullet.

Article text:
{article["text"][:MAX_ARTICLE_CHARS]}
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.4,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise and helpful editor who summarizes "
                    "technology news for different audiences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def synthesize_insights(
    client: OpenAI,
    topic: str,
    audience: str,
    article_summaries: List[Dict[str, str]],
) -> str:
    """Compare summaries across sources and identify themes and trends."""
    compiled_summaries = "\n\n".join(
        [
            (
                f"Source {index}:\n"
                f"Title: {item['title']}\n"
                f"URL: {item['url']}\n"
                f"Summary:\n{item['summary']}"
            )
            for index, item in enumerate(article_summaries, start=1)
        ]
    )

    prompt = f"""
Compare these article summaries about "{topic}" for an audience of {audience}.

Organize the response with these exact section headers:
Common Themes
Differences
Key Trends
Why It Matters for {audience}

Keep the analysis practical, specific, and easy to scan.
Use bullets under each section.

Summaries:
{compiled_summaries}
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": (
                    "You compare multiple sources and highlight patterns, "
                    "contrasts, and audience-specific takeaways."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def generate_final_output(
    client: OpenAI,
    topic: str,
    audience: str,
    output_format: str,
    article_summaries: List[Dict[str, str]],
    cross_source_insights: str,
) -> str:
    """Generate the polished final digest in the selected format."""
    compiled_summaries = "\n\n".join(
        [
            (
                f"Title: {item['title']}\n"
                f"URL: {item['url']}\n"
                f"Summary:\n{item['summary']}"
            )
            for item in article_summaries
        ]
    )

    prompt = f"""
Create a polished {output_format} about "{topic}".

Target audience: {audience}

Instructions:
- Make it engaging, informative, and easy to read.
- Tailor examples, language, and framing for {audience}.
- Include a clear headline.
- Reference the major themes and trends from the source material.
- End with a short takeaway or call-to-action relevant to the audience.

Source summaries:
{compiled_summaries}

Cross-source insights:
{cross_source_insights}
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an engaging writer who turns research notes "
                    "into polished audience-aware content."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def validate_inputs(topic: str, urls: List[str]) -> List[str]:
    """Return a list of validation errors."""
    errors: List[str] = []

    if not topic.strip():
        errors.append("Please enter a topic.")

    filled_urls = [url.strip() for url in urls if url.strip()]
    if len(filled_urls) != 3:
        errors.append("Please provide exactly 3 article URLs.")

    return errors


def get_openai_client() -> OpenAI:
    """Create an OpenAI client from environment configuration."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your environment or .env file."
        )
    return OpenAI(api_key=api_key)


def render_article_result(article_result: Dict[str, Any], index: int) -> None:
    """Render a single article result block in the UI."""
    st.subheader(f"Article {index}: {article_result['title']}")
    st.caption(article_result["url"])
    st.markdown(article_result["summary"])


def main() -> None:
    st.set_page_config(
        page_title="Student Tech/AI News Digest",
        page_icon="📰",
        layout="wide",
    )

    st.title("Student Tech/AI News Digest")
    st.write(
        "Enter a topic and three article URLs to create an AI-powered digest "
        "tailored to your audience."
    )

    with st.sidebar:
        st.header("Settings")
        audience = st.selectbox(
            "Audience",
            options=["College Students", "Beginners", "Professionals"],
            index=0,
        )
        output_format = st.selectbox(
            "Output Format",
            options=["Newsletter", "Blog Post", "Social Media Post"],
            index=0,
        )

    topic = st.text_input(
        "Topic",
        placeholder="Example: AI tools for student productivity",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        url_1 = st.text_input("Article URL 1", placeholder="https://example.com/article-1")
    with col2:
        url_2 = st.text_input("Article URL 2", placeholder="https://example.com/article-2")
    with col3:
        url_3 = st.text_input("Article URL 3", placeholder="https://example.com/article-3")

    if st.button("Generate Digest", type="primary", use_container_width=True):
        urls = [url_1, url_2, url_3]
        errors = validate_inputs(topic, urls)
        if errors:
            for error in errors:
                st.error(error)
            return

        try:
            client = get_openai_client()
        except RuntimeError as exc:
            st.error(str(exc))
            return

        article_results: List[Dict[str, Any]] = []
        failed_urls: List[str] = []

        with st.spinner("Extracting and summarizing articles..."):
            for url in urls:
                cleaned_url = url.strip()
                try:
                    article = extract_article(cleaned_url)
                    article["summary"] = summarize_article(client, topic, audience, article)
                    article_results.append(article)
                except Exception as exc:  # noqa: BLE001
                    failed_urls.append(
                        f"{cleaned_url} ({shorten(str(exc), width=140, placeholder='...')})"
                    )

        if failed_urls:
            for failed_url in failed_urls:
                st.warning(f"Could not process: {failed_url}")

        if not article_results:
            st.error("None of the articles could be processed. Please try different URLs.")
            return

        with st.spinner("Comparing sources and generating final digest..."):
            try:
                cross_source_insights = synthesize_insights(
                    client=client,
                    topic=topic,
                    audience=audience,
                    article_summaries=article_results,
                )
                final_output = generate_final_output(
                    client=client,
                    topic=topic,
                    audience=audience,
                    output_format=output_format,
                    article_summaries=article_results,
                    cross_source_insights=cross_source_insights,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"OpenAI generation failed: {exc}")
                return

        st.success("Digest generated successfully.")

        st.header("Article Summaries")
        for index, article_result in enumerate(article_results, start=1):
            render_article_result(article_result, index)
            st.divider()

        st.header("Cross-Source Insights")
        st.markdown(cross_source_insights)

        st.header(f"Final {output_format}")
        st.markdown(final_output)


if __name__ == "__main__":
    main()
