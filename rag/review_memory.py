from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Any

from .config import RAGConfig
from .llm import RAGLLMAgent
from .models import ReviewMemoryCase
from .providers.base import (
    PaperSearchProvider,
    ProviderHTTPError,
    http_status_code,
    is_forbidden_error,
    is_rate_limited_error,
)


OPENREVIEW_LOGIN_URL = "https://api2.openreview.net/login"
OPENREVIEW_NOTES_URL = "https://api2.openreview.net/notes"
OPENREVIEW_SEARCH_URL = "https://api2.openreview.net/notes/search"
OPENREVIEW_FORUM_URL = "https://openreview.net/forum?id={forum}"

OFFICIAL_REVIEW_INVITATION_RE = re.compile(
    r"(?:^|/)-/(?:official[_ -]?review|review)(?:\d+)?$",
    re.IGNORECASE,
)
FINAL_DECISION_INVITATION_RE = re.compile(
    r"(?:^|/)-/(?:decision|paper[_ -]?decision|acceptance[_ -]?decision|final[_ -]?decision)(?:$|/)",
    re.IGNORECASE,
)
NON_REVIEW_INVITATION_RE = re.compile(
    r"(?:^|/)-/[^/]*(?:comment|rebuttal|response|decision|recommendation|meta[_ -]?review)[^/]*(?:$|/)",
    re.IGNORECASE,
)
REVIEW_SIGNAL_KEYS = {
    "confidence",
    "numerical_score",
    "overall_assessment",
    "overall_recommendation",
    "quality",
    "rating",
    "recommendation",
    "reviewer_confidence",
    "score",
    "soundness",
}
REVIEW_BODY_KEYS = {
    "claims_and_evidence",
    "detailed_feedback",
    "detailed_feedback_to_authors",
    "experimental_designs_or_analyses",
    "main_review",
    "methods_and_evaluation_criteria",
    "paper_strengths",
    "paper_summary",
    "paper_weaknesses",
    "reasons_to_accept",
    "reasons_to_reject",
    "review",
    "strength_and_weaknesses",
    "strengths",
    "strengths_and_weaknesses",
    "summary",
    "summary_of_strengths",
    "summary_of_weaknesses",
    "theoretical_claims",
    "weaknesses",
}
REVIEW_TEXT_KEYS = [
    "summary",
    "paper_summary",
    "main_review",
    "review",
    "comment",
    "comments",
    "strengths",
    "paper_strengths",
    "summary_of_strengths",
    "main_strengths",
    "other_strengths",
    "weaknesses",
    "paper_weaknesses",
    "summary_of_weaknesses",
    "main_weaknesses",
    "other_weaknesses",
    "strengths_and_weaknesses",
    "strength_and_weaknesses",
    "other_strengths_and_weaknesses",
    "detailed_feedback",
    "detailed_feedback_to_authors",
    "claims_and_evidence",
    "methods_and_evaluation_criteria",
    "theoretical_claims",
    "experimental_designs_or_analyses",
    "relation_to_broader_scientific_literature",
    "essential_references_not_discussed",
    "reasons_to_accept",
    "reasons_to_reject",
    "questions",
]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if "value" in value:
            return _as_text(value["value"])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "\n".join(_as_text(item) for item in value if _as_text(item))
    return str(value).strip()


def _as_float(value: Any) -> float | None:
    text = _as_text(value)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _content(note: dict[str, Any]) -> dict[str, Any]:
    content = note.get("content") or {}
    normalized = {}
    for key, value in content.items():
        normalized[key] = value.get("value") if isinstance(value, dict) and "value" in value else value
    return normalized


def _note_id(note: dict[str, Any]) -> str:
    return str(note.get("id") or note.get("forum") or "")


def _note_forum(note: dict[str, Any]) -> str:
    return str(note.get("forum") or note.get("id") or "")


def _canonical_field_name(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")


def _invitation_values(note: dict[str, Any]) -> list[str]:
    values = []
    for key in ("invitation", "invitations"):
        value = note.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))
    return values


def _has_invitation(note: dict[str, Any], pattern: re.Pattern[str]) -> bool:
    return any(pattern.search(invitation) for invitation in _invitation_values(note))


def _has_author_signature(note: dict[str, Any]) -> bool:
    signatures = " ".join(str(value) for value in note.get("signatures") or [])
    return bool(re.search(r"author", signatures, re.IGNORECASE))


def _has_reviewer_signature_or_invitation(note: dict[str, Any]) -> bool:
    values = [*(note.get("signatures") or []), *_invitation_values(note)]
    return bool(re.search(r"reviewer|anonreviewer", " ".join(str(value) for value in values), re.IGNORECASE))


def _normalize_title(title: str) -> str:
    normalized = title.casefold()
    normalized = re.sub(r"[*_`#\"'“”‘’]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _title_similarity(a: str, b: str) -> float:
    compact_a = re.sub(r"[^a-z0-9]+", "", a.casefold())
    compact_b = re.sub(r"[^a-z0-9]+", "", b.casefold())
    if compact_a and compact_a == compact_b:
        return 1.0
    stop = {"a", "an", "the", "of", "in", "on", "for", "to", "and", "with", "via", "using", "by"}
    ta = {w for w in _normalize_title(a).split() if w not in stop and len(w) > 2}
    tb = {w for w in _normalize_title(b).split() if w not in stop and len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def extract_openreview_forum_id(url_or_id: str) -> str:
    text = (url_or_id or "").strip()
    if not text:
        return ""
    if "openreview.net" not in text and re.fullmatch(r"[A-Za-z0-9_-]{6,}", text):
        return text
    parsed = urllib.parse.urlparse(text)
    query = urllib.parse.parse_qs(parsed.query)
    if query.get("id"):
        return query["id"][0]
    match = re.search(r"/forum\?id=([^&#]+)", text)
    if match:
        return urllib.parse.unquote(match.group(1))
    return ""


def _is_official_review(note: dict[str, Any]) -> bool:
    if _has_invitation(note, OFFICIAL_REVIEW_INVITATION_RE):
        return True
    if _has_invitation(note, NON_REVIEW_INVITATION_RE) or _has_author_signature(note):
        return False
    content = _content(note)
    keys = {_canonical_field_name(key) for key in content}
    has_review_body = bool(keys & REVIEW_BODY_KEYS)
    has_review_signal = bool(keys & REVIEW_SIGNAL_KEYS)
    if not has_review_body and not has_review_signal:
        return False
    if _is_decision(note):
        return False
    if has_review_body and has_review_signal:
        return True
    return _has_reviewer_signature_or_invitation(note) and has_review_body


def _is_public_note(note: dict[str, Any]) -> bool:
    readers = note.get("readers") or []
    if not readers:
        return True
    return any(str(reader).casefold() == "everyone" for reader in readers)


def _is_decision(note: dict[str, Any]) -> bool:
    if _has_invitation(note, FINAL_DECISION_INVITATION_RE):
        return True
    if _has_invitation(note, OFFICIAL_REVIEW_INVITATION_RE):
        return False
    content = _content(note)
    keys = {_canonical_field_name(key) for key in content}
    return bool(keys & {"decision", "final_decision"})


def _classify_decision(text: str) -> str:
    lower = text.lower()
    if any(word in lower for word in ("accept", "poster", "spotlight", "oral")):
        return "accept"
    if "reject" in lower:
        return "reject"
    return "unknown"


def _decision_from_notes(notes: list[dict[str, Any]]) -> str:
    for note in notes:
        if not _is_decision(note):
            continue
        content = _content(note)
        text = " ".join(_as_text(content.get(key)) for key in content)
        decision = _classify_decision(text)
        if decision != "unknown":
            return decision
    return "unknown"


def _rating(content: dict[str, Any]) -> float | None:
    for key in (
        "rating",
        "Rating",
        "recommendation",
        "Recommendation",
        "overall_recommendation",
        "Overall Recommendation",
        "overall_assessment",
        "Overall Assessment",
    ):
        value = _as_float(content.get(key))
        if value is not None:
            return value
    return None


def _confidence(content: dict[str, Any]) -> float | None:
    for key in ("confidence", "Confidence"):
        value = _as_float(content.get(key))
        if value is not None:
            return value
    return None


def _review_text(content: dict[str, Any], max_chars: int = 2500) -> str:
    ignored = {
        "rating",
        "confidence",
        "recommendation",
        "overall_recommendation",
        "overall_assessment",
        "ethics_review",
    }
    parts: list[str] = []
    seen: set[str] = set()
    for preferred_key in REVIEW_TEXT_KEYS:
        for key, value in content.items():
            canonical = _canonical_field_name(key)
            if canonical != preferred_key or canonical in seen:
                continue
            text = _as_text(value)
            if text:
                parts.append(f"{key}: {text}")
            seen.add(canonical)
            break
    for key, value in content.items():
        canonical = _canonical_field_name(key)
        if canonical in seen or canonical in ignored:
            continue
        text = _as_text(value)
        if len(text) >= 30:
            parts.append(f"{key}: {text}")
    return "\n\n".join(parts)[:max_chars]


def _score_range(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [review["rating"] for review in reviews if review.get("rating") is not None]
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": round(sum(scores) / len(scores), 3),
    }


class OpenReviewMemoryProvider(PaperSearchProvider):
    name = "OpenReview Review Memory"

    def __init__(self, cache_dir: str = "data/rag_cache", timeout: int = 20):
        super().__init__(cache_dir, timeout=timeout)
        self._openreview_auth_headers: dict[str, str] | None = None
        self._openreview_auth_attempted = False
        self._openreview_auth_warning = ""

    def _should_cache(self, data: Any) -> bool:
        # searchUnavailable is a transient backend hiccup on OpenReview's end, not
        # a real "no results" answer; caching it would make our own retry loop
        # in _lookup_forum_by_title just replay the same failure from disk.
        if isinstance(data, dict) and data.get("searchUnavailable"):
            return False
        return True

    def _has_openreview_auth_config(self) -> bool:
        return bool(
            os.environ.get("OPENREVIEW_ACCESS_TOKEN")
            or os.environ.get("OPENREVIEW_TOKEN")
            or (os.environ.get("OPENREVIEW_USERNAME") and os.environ.get("OPENREVIEW_PASSWORD"))
        )

    def _headers_from_token(self, token: str) -> dict[str, str]:
        token = token.strip()
        return {
            "Authorization": f"Bearer {token}",
            "Cookie": f"openreview.accessToken={token}",
        }

    def _login_with_env_credentials(self) -> str:
        username = os.environ.get("OPENREVIEW_USERNAME") or ""
        password = os.environ.get("OPENREVIEW_PASSWORD") or ""
        payload = json.dumps({"id": username, "password": password}).encode("utf-8")
        request = urllib.request.Request(
            OPENREVIEW_LOGIN_URL,
            data=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": self.default_headers["User-Agent"],
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            reason = getattr(exc, "reason", "") or getattr(exc, "msg", "")
            raise ProviderHTTPError(OPENREVIEW_LOGIN_URL, exc.code, reason) from exc
        token = str(data.get("token") or "").strip()
        if not token:
            raise RuntimeError("OpenReview login response did not include an access token")
        return token

    def _auth_headers(self) -> dict[str, str]:
        existing_headers = getattr(self, "_openreview_auth_headers", None)
        if existing_headers is not None:
            return existing_headers

        token = os.environ.get("OPENREVIEW_ACCESS_TOKEN") or os.environ.get("OPENREVIEW_TOKEN")
        if token:
            self._openreview_auth_headers = self._headers_from_token(token)
            return self._openreview_auth_headers

        if not (os.environ.get("OPENREVIEW_USERNAME") and os.environ.get("OPENREVIEW_PASSWORD")):
            self._openreview_auth_headers = {}
            return self._openreview_auth_headers

        if getattr(self, "_openreview_auth_attempted", False):
            return self._openreview_auth_headers or {}
        self._openreview_auth_attempted = True
        try:
            self._openreview_auth_headers = self._headers_from_token(self._login_with_env_credentials())
        except Exception as exc:
            self._openreview_auth_headers = {}
            self._openreview_auth_warning = f"OpenReview credential login failed; using guest access: {exc}"
        return self._openreview_auth_headers

    def _json_endpoint(self, base_url: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode(params)
        headers = {
            "Accept": "application/json",
            "User-Agent": "paper-reviewer-rag/1.0",
            **self._auth_headers(),
        }
        return self._json_get(f"{base_url}?{query}", headers=headers)

    def _lookup_forum_by_title(self, title: str, max_attempts: int = 3) -> str:
        if not title:
            return ""
        search_term = _normalize_title(title)
        params = {"term": search_term, "limit": 50, "source": "forum"}
        data = self._json_endpoint(OPENREVIEW_SEARCH_URL, params)
        # OpenReview's search backend intermittently returns a 200 with
        # searchUnavailable=True (not an HTTP error, so the generic retry in
        # base.py never sees it); re-issuing the same request moments later
        # reliably succeeds, so retry a few times before giving up.
        attempt = 1
        while data.get("searchUnavailable") and attempt < max_attempts:
            time.sleep(2.0 * attempt)
            data = self._json_endpoint(OPENREVIEW_SEARCH_URL, params)
            attempt += 1
        for note in data.get("notes", []):
            note_title = _as_text(_content(note).get("title"))
            if note_title and _title_similarity(title, note_title) >= 0.75:
                return _note_forum(note)
        return ""

    def _fetch_forum_notes(self, forum: str) -> list[dict[str, Any]]:
        notes: list[dict[str, Any]] = []
        seen: set[str] = set()
        for params in ({"forum": forum, "limit": 1000}, {"replyto": forum, "limit": 1000}):
            try:
                data = self._json_endpoint(OPENREVIEW_NOTES_URL, params)
            except Exception as exc:
                if "replyto" in params and http_status_code(exc) == 400:
                    continue
                raise
            for note in data.get("notes", []):
                key = _note_id(note) or str(id(note))
                if key in seen:
                    continue
                seen.add(key)
                notes.append(note)
        return notes

    def find_review_memory_case(
        self,
        ranked_candidates: list[dict[str, Any]],
        max_reviews: int = 4,
    ) -> tuple[ReviewMemoryCase | None, list[str], list[str]]:
        warnings: list[str] = []
        attempted: list[str] = []
        for candidate in ranked_candidates:
            source_paper_id = str(candidate.get("paper_id") or "")
            title = str(candidate.get("title") or "")
            attempted.append(source_paper_id or title)
            forum = ""
            for value in (
                candidate.get("url"),
                candidate.get("openreview_url"),
                (candidate.get("source_ids") or {}).get("OpenReview"),
            ):
                forum = extract_openreview_forum_id(str(value or ""))
                if forum:
                    break
            try:
                if not forum:
                    forum = self._lookup_forum_by_title(title)
                if not forum:
                    continue
                notes = self._fetch_forum_notes(forum)
            except Exception as exc:
                auth_warning = getattr(self, "_openreview_auth_warning", "")
                if auth_warning and auth_warning not in warnings:
                    warnings.append(auth_warning)
                if is_forbidden_error(exc):
                    if self._has_openreview_auth_config():
                        warnings.append(
                            "OpenReview review-memory lookup unavailable after authentication (HTTP 403); "
                            "skipped remaining candidates."
                        )
                    else:
                        warnings.append(
                            "OpenReview review-memory lookup requires OpenReview authentication or browser challenge "
                            "(HTTP 403); skipped remaining candidates."
                        )
                    break
                if is_rate_limited_error(exc):
                    warnings.append(
                        "OpenReview review-memory lookup rate-limited (HTTP 429); "
                        "skipped remaining candidates."
                    )
                    break
                warnings.append(f"OpenReview review-memory lookup failed for {source_paper_id or title}: {exc}")
                continue

            public_notes = [note for note in notes if _is_public_note(note)]
            official_reviews = [note for note in public_notes if _is_official_review(note)]
            if not official_reviews:
                continue
            official_reviews.sort(key=lambda note: note.get("cdate") or note.get("tcdate") or 0)
            reviews = []
            for index, note in enumerate(official_reviews[:max_reviews]):
                content = _content(note)
                review_text = _review_text(content)
                if not review_text:
                    continue
                signatures = note.get("signatures") or []
                reviewer = signatures[0].split("/")[-1] if signatures else f"Reviewer_{index + 1}"
                reviews.append({
                    "review_id": _note_id(note) or hashlib.sha1(review_text.encode("utf-8")).hexdigest()[:12],
                    "reviewer_id": reviewer.replace("AnonReviewer", "Reviewer"),
                    "rating": _rating(content),
                    "confidence": _confidence(content),
                    "recommendation": _as_text(content.get("recommendation") or content.get("Recommendation")),
                    "text": review_text,
                })
            if not reviews:
                continue
            return (
                ReviewMemoryCase(
                    source_paper_id=source_paper_id,
                    source_rank=int(candidate.get("rank") or 0),
                    title=title,
                    year=candidate.get("year"),
                    openreview_forum_id=forum,
                    openreview_url=OPENREVIEW_FORUM_URL.format(forum=forum),
                    decision=_decision_from_notes(public_notes),
                    score_range=_score_range(reviews),
                    reviews=reviews,
                ),
                warnings,
                attempted,
            )
        return None, warnings, attempted


def _ranked_candidates_from_package(package: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = {paper.get("paper_id"): paper for paper in package.get("paper_metadata", [])}
    ranked = []
    for item in package.get("reranking_results", []):
        paper = metadata.get(item.get("paper_id"))
        if not paper:
            continue
        merged = dict(paper)
        merged["rank"] = item.get("rank")
        merged["relevance_score"] = item.get("relevance_score")
        ranked.append(merged)
    ranked.sort(key=lambda item: item.get("rank") or 10_000)
    return ranked


def _review_memory_backend_unavailable(warnings: list[str]) -> bool:
    return any(
        "OpenReview review-memory lookup unavailable" in warning
        or "OpenReview review-memory lookup requires OpenReview authentication" in warning
        or "OpenReview review-memory lookup rate-limited" in warning
        for warning in warnings
    )


def _fallback_summary(case: ReviewMemoryCase) -> dict[str, Any]:
    decision = case.decision if case.decision != "unknown" else "decision unavailable"
    score = case.score_range
    score_text = (
        f"{score.get('min')} to {score.get('max')} with mean {score.get('mean')}"
        if score.get("count")
        else "unavailable"
    )
    return {
        "summary": (
            f"OpenReview memory for [{case.source_paper_id}] shows {len(case.reviews)} official reviews, "
            f"{decision}, and score range {score_text}."
        ),
        "common_strengths": [],
        "common_weaknesses": [],
        "score_range": score,
        "decision_pattern": decision,
        "calibration_notes": [
            "Use this as auxiliary calibration context only; it is not direct evidence about the target paper."
        ],
        "caveats": ["LLM summarization failed or was unavailable, so only structured fields are reported."],
    }


def _summarize_case_with_llm(case: ReviewMemoryCase, llm_agent: RAGLLMAgent, warnings: list[str]) -> dict[str, Any]:
    system_prompt = (
        "You summarize public peer-review records into structured review-memory patterns. "
        "Use only the supplied OpenReview records. Do not copy long review text. "
        "Do not invent paper metadata, scores, decisions, or review claims. Return only valid JSON."
    )
    user_prompt = f"""
The following public OpenReview records come from one related paper selected from related-work RAG.
Summarize how that related paper was evaluated. This is auxiliary calibration context for reviewer agents,
not direct evidence for judging the target paper.

Return JSON with:
{{
  "summary": "short paragraph with [source_paper_id] reference",
  "common_strengths": ["pattern-level strength"],
  "common_weaknesses": ["pattern-level weakness"],
  "score_range": {{"min": 0, "max": 0, "mean": 0, "count": 0}},
  "decision_pattern": "accept/reject/unknown plus brief context",
  "calibration_notes": ["how reviewers should use this pattern cautiously"],
  "caveats": ["similarity/reuse caveats"]
}}

OpenReview review-memory case:
{json.dumps(case.to_dict(), ensure_ascii=False)}
""".strip()
    try:
        data = llm_agent.complete_json(system_prompt, user_prompt)
    except Exception as exc:
        warnings.append(f"Review-memory summarization LLM failed; using structured fallback: {exc}")
        return _fallback_summary(case)
    summary = {
        "summary": str(data.get("summary", "")).strip(),
        "common_strengths": [str(item).strip() for item in data.get("common_strengths", []) if str(item).strip()],
        "common_weaknesses": [str(item).strip() for item in data.get("common_weaknesses", []) if str(item).strip()],
        "score_range": data.get("score_range") if isinstance(data.get("score_range"), dict) else case.score_range,
        "decision_pattern": str(data.get("decision_pattern", "")).strip() or case.decision,
        "calibration_notes": [str(item).strip() for item in data.get("calibration_notes", []) if str(item).strip()],
        "caveats": [str(item).strip() for item in data.get("caveats", []) if str(item).strip()],
    }
    if not summary["summary"]:
        summary = _fallback_summary(case)
    return summary


def build_review_memory_from_package(
    package: dict[str, Any],
    llm_agent: RAGLLMAgent,
    config: RAGConfig | None = None,
    provider: OpenReviewMemoryProvider | None = None,
) -> dict[str, Any]:
    config = config or RAGConfig()
    if not config.enable_review_memory_rag:
        return {
            "status": "disabled",
            "summary": {},
            "selected_case": None,
            "attempted_source_paper_ids": [],
            "warnings": [],
        }
    ranked_candidates = _ranked_candidates_from_package(package)
    if not ranked_candidates:
        return {
            "status": "not_found",
            "summary": {},
            "selected_case": None,
            "attempted_source_paper_ids": [],
            "warnings": ["No related-work reranking candidates were available for review-memory lookup."],
        }

    memory_provider = provider or OpenReviewMemoryProvider(config.rag_cache_dir)
    case, warnings, attempted = memory_provider.find_review_memory_case(
        ranked_candidates,
        max_reviews=config.review_memory_max_reviews,
    )
    if case is None:
        backend_unavailable = _review_memory_backend_unavailable(warnings)
        return {
            "status": "unavailable" if backend_unavailable else "not_found",
            "summary": {},
            "selected_case": None,
            "attempted_source_paper_ids": attempted,
            "warnings": warnings + [
                "Review-memory lookup was unavailable; no public review-memory evidence was added."
                if backend_unavailable
                else "No OpenReview public official reviews found for the ranked related-work candidates."
            ],
        }

    summary = _summarize_case_with_llm(case, llm_agent, warnings)
    return {
        "status": "used",
        "summary": summary,
        "selected_case": case.to_dict(),
        "attempted_source_paper_ids": attempted,
        "warnings": warnings,
    }
