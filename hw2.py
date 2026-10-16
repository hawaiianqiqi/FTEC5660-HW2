#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


# ---------------------------------------------------------------------------
# Student code. Everything from here to ``build_agent`` may be edited freely.
# ---------------------------------------------------------------------------

import os

from langchain_deepseek import ChatDeepSeek

try:  # LangChain 1.x tool-calling agent factory.
    from langchain.agents import create_agent
except Exception:  # pragma: no cover - older/newer LangChain layouts
    create_agent = None

MAX_CONCURRENCY = 3        # the MCP server is shared: at most ~3 CVs in flight
MAX_PROFILES = 12          # upper bound on profile lookups per CV
PROFILE_BATCH = 6          # look profiles up in batches, stop early on a strong hit
MAX_JUDGE_PROFILES = 8     # candidates shown to the identity judge at once
MAX_CV_CHARS = 6000        # CV text handed to the model
MCP_ATTEMPTS = 4           # retries per MCP tool call
MCP_CONNECT_ATTEMPTS = 5   # retries for the one-shot MCP handshake

DEBUG = os.environ.get("HW2_DEBUG") == "1"


def _log(*parts: Any) -> None:
    if DEBUG:
        print("[hw2]", *parts, flush=True)


# Robustness shim.  The provided ``load_mcp_tools`` calls
# ``MultiServerMCPClient.get_tools()`` exactly once, so a single transient disconnect from the
# shared ngrok endpoint raises out of ``run()``, no results.csv is written, and the run scores
# zero.  I do not edit that provided function; I only wrap the library method it calls with a
# bounded exponential-backoff retry.  This can turn a crash into a retry and can never change the
# outcome of a successful connection.
try:
    import langchain_mcp_adapters.client as _mcp_client

    _ORIGINAL_GET_TOOLS = _mcp_client.MultiServerMCPClient.get_tools
    if not getattr(_ORIGINAL_GET_TOOLS, "_hw2_retrying", False):

        async def _retrying_get_tools(self: Any, *args: Any, **kwargs: Any) -> Any:
            delay = 1.0
            for attempt in range(MCP_CONNECT_ATTEMPTS):
                try:
                    return await _ORIGINAL_GET_TOOLS(self, *args, **kwargs)
                except Exception as exc:
                    _log("mcp handshake failed", attempt, exc)
                    if attempt == MCP_CONNECT_ATTEMPTS - 1:
                        raise
                    await asyncio.sleep(delay)
                    delay *= 2

        _retrying_get_tools._hw2_retrying = True  # type: ignore[attr-defined]
        _mcp_client.MultiServerMCPClient.get_tools = _retrying_get_tools
except Exception as exc:  # pragma: no cover - the shim must never break the import
    _log("could not install the MCP retry shim", exc)


EXTRACT_SYSTEM = """You extract structured claims from a CV.
Reply with a single JSON object and nothing else:
{"name": str, "city": str, "country": str, "headline": str, "field": str,
 "jobs": [{"company": str, "title": str, "start_year": int|null,
           "end_year": int|null, "is_current": bool}],
 "education": [{"school": str, "degree": str, "field": str, "grad_year": int|null}],
 "skills": [str]}
Rules:
- "city" and "country" are the candidate's own location: "London, UK" -> city "London", country "UK";
  "Central, Hong Kong" -> city "Central", country "Hong Kong".
- Ignore the hometown part (anything after "|").
- "field" is the job family word taken from the headline:
  "Logistics Professional" -> "Logistics", "AI Professional" -> "AI", "Design Professional" -> "Design".
- Copy "name" exactly as printed.
- "end_year" is null and "is_current" true for a job that says "Present".
- Keep the degree wording used in the CV ("Master of Business Administration", "BSc", "PhD").
- Copy skills exactly as printed. Use [] when a section is missing."""


IDENTITY_SYSTEM = """You decide which social profile belongs to the person who wrote a CV.
You receive the claims extracted from the CV and several candidate profiles that share the name.
Reply with a single JSON object and nothing else:
{"matched_person_id": int|null, "confidence": "high"|"medium"|"low", "reason": str}
Rules:
- Prefer the candidate that agrees on the largest number of independent attributes, in this order of
  importance: employers, schools, field/industry, city.
- Do NOT reject a candidate because one detail differs. A CV may contain a single deliberately false
  claim, so pick the candidate that agrees on the most attributes even if one of them disagrees.
- Wording differences are not disagreements ("Bachelor of Science" == "BSc",
  "UI/UX Design" == "UI/UX", "The University of Hong Kong" == "HKU",
  title "Senior Engineer" == title "Engineer" with seniority "senior").
- If no candidate shares any employer or school, still return the closest candidate by city and field
  and set "confidence" to "low". Use null only when the candidate list is empty."""


JUDGE_SYSTEM = """You verify a CV against the social profiles of the person who wrote it.
You receive the CV claims, the chosen LinkedIn profile, and optionally a Facebook profile.
Reply with a single JSON object and nothing else:
{"discrepancies": [{"field": str, "cv_value": str, "profile_value": str, "detail": str}],
 "notes": str}
A discrepancy is a CV claim that contradicts the profile. Only these fields can be discrepant:
- name
- city / country
- jobs: company, title, seniority, start year, end year
- education: school, degree, field, graduation year
- skills
Never report job descriptions, the headline, the hometown, or the profile summary/bio.
The following are NOT discrepancies:
- wording only: "Bachelor of Science" vs "BSc", "Master of Business Administration" vs "MBA",
  "UI/UX Design" vs "UI/UX", "Microsoft PowerPoint" vs "PowerPoint", "PolyU" vs "The Hong Kong
  Polytechnic University", "Standard Chartered" vs "Standard Chartered Bank";
- a title that only differs by a seniority word that agrees with the profile's seniority:
  "Senior Engineer" for title "Engineer" with seniority "senior" is fine,
  "Senior Scientist" for title "Scientist" with seniority "senior" is fine,
  but "Senior Analyst" for title "Analyst" with seniority "junior" IS a discrepancy;
- the CV listing FEWER skills than the profile (missing skills are fine);
- a CV skill that the profile also has under a different spelling;
- a year that agrees, or an ongoing job that agrees.
A CV skill that the profile does NOT have at all IS a discrepancy.
A shifted employment or graduation year IS a discrepancy.
An upgraded degree, a fake school or employer, a wrong location, an inflated title and a made-up
skill are all discrepancies.
Report every genuine discrepancy and only genuine discrepancies. Return [] when the CV agrees."""


def _model() -> Any:
    """The one model allowed by the homework."""
    return ChatDeepSeek(model=MODEL_NAME, temperature=0)


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _extract_json(text: Any) -> Any:
    """Parse the first JSON object/array found in *text* (tolerates code fences)."""
    if not isinstance(text, str):
        text = str(getattr(text, "content", text))
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    return None


def _parse_tool_output(raw: Any) -> Any:
    """MCP tools return ``[{"type": "text", "text": "<JSON>"}]``; parse that JSON."""
    if isinstance(raw, dict):
        if "content" in raw and isinstance(raw["content"], list):
            return _parse_tool_output(raw["content"])
        return raw
    if isinstance(raw, list):
        if not raw:
            return []
        if all(isinstance(b, dict) and b.get("type") == "text" for b in raw):
            return _extract_json("\n".join(b.get("text", "") for b in raw))
        if all(isinstance(b, dict) for b in raw):
            return raw
        return None
    if isinstance(raw, str):
        return _extract_json(raw)
    return None


async def _call_tool(
    tool_map: dict[str, Any], name: str, args: dict[str, Any], attempts: int = MCP_ATTEMPTS
) -> Any:
    """Call one MCP tool, retrying transient failures (the server is shared and flaky)."""
    tool = tool_map.get(name)
    if tool is None:
        return None
    clean = {k: v for k, v in args.items() if v not in (None, "", [])}
    for attempt in range(attempts):
        try:
            raw = await tool.ainvoke(clean)
        except Exception as exc:  # a single failed lookup must never kill the run
            _log("tool error", name, clean, "attempt", attempt, exc)
            await asyncio.sleep(0.7 * (attempt + 1))
            continue
        parsed = _parse_tool_output(raw)
        if parsed is not None:
            return parsed
        _log("empty tool output", name, clean, "attempt", attempt)
        await asyncio.sleep(0.5 * (attempt + 1))
    return None


async def _llm_json(model: Any, system: str, user: str, attempts: int = 3) -> Any:
    """Ask the model for JSON and parse it, retrying on a malformed reply."""
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = user
    for attempt in range(attempts):
        try:
            message = await model.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=prompt)]
            )
        except Exception as exc:
            _log("model error", exc)
            await asyncio.sleep(1 + attempt)
            continue
        data = _extract_json(message)
        if data is not None:
            return data
        prompt = user + "\n\nReply with JSON only, no prose and no code fences."
    return None


def _claims_from(data: Any) -> dict[str, Any]:
    """Normalise the extraction result into a predictable dict."""
    data = data if isinstance(data, dict) else {}

    def num(value: Any) -> int | None:
        match = re.search(r"\d{4}", str(value)) if value is not None else None
        return int(match.group()) if match else None

    jobs = []
    for job in data.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        jobs.append(
            {
                "company": str(job.get("company") or "").strip(),
                "title": str(job.get("title") or "").strip(),
                "start_year": num(job.get("start_year")),
                "end_year": num(job.get("end_year")),
                "is_current": bool(job.get("is_current")),
            }
        )

    education = []
    for edu in data.get("education") or []:
        if not isinstance(edu, dict):
            continue
        education.append(
            {
                "school": str(edu.get("school") or "").strip(),
                "degree": str(edu.get("degree") or "").strip(),
                "field": str(edu.get("field") or "").strip(),
                "grad_year": num(edu.get("grad_year")),
            }
        )

    skills = [str(s).strip() for s in (data.get("skills") or []) if str(s).strip()]

    return {
        "name": str(data.get("name") or "").strip(),
        "city": str(data.get("city") or "").strip(),
        "country": str(data.get("country") or "").strip(),
        "headline": str(data.get("headline") or "").strip(),
        "field": str(data.get("field") or "").strip(),
        "jobs": jobs,
        "education": education,
        "skills": skills,
    }


def _regex_claims(text: str) -> dict[str, Any]:
    """Cheap deterministic fallback when the extraction call fails."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    claims = _claims_from({})
    if lines:
        claims["name"] = lines[0]
    if len(lines) > 1:
        claims["headline"] = lines[1]
        claims["field"] = re.sub(
            r"\s*(professional|specialist|expert|consultant)\s*$", "", lines[1], flags=re.I
        ).strip()
    for line in lines:
        if "|" in line:
            head = line.split("|")[0].strip()
            if "," in head:
                city, country = head.split(",", 1)
                claims["city"], claims["country"] = city.strip(), country.strip()
            break
    for match in re.finditer(
        r"^(.+?),\s*([^,\n]+?)\s{2,}(\d{4})\s*[–-]\s*(Present|\d{4})", text, re.M
    ):
        title, company, start, end = match.groups()
        claims["jobs"].append(
            {
                "company": company.strip(),
                "title": title.strip(),
                "start_year": int(start),
                "end_year": None if end == "Present" else int(end),
                "is_current": end == "Present",
            }
        )
    school = re.search(r"(?:Graduated\s+(\d{4})\s*\n\s*)(.+)", text)
    if school:
        claims["education"].append(
            {"school": school.group(2).strip(), "degree": "", "field": "", "grad_year": int(school.group(1))}
        )
    return claims


def _profile_digest(profile: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields that can carry a discrepancy."""
    return {
        "id": profile.get("id"),
        "name": profile.get("name"),
        "headline": profile.get("headline"),
        "city": profile.get("city"),
        "country": profile.get("country"),
        "industry": profile.get("industry"),
        "status": profile.get("status"),
        "years_experience": profile.get("years_experience"),
        "experience": [
            {
                "company": e.get("company"),
                "title": e.get("title"),
                "seniority": e.get("seniority"),
                "start_year": e.get("start_year"),
                "end_year": e.get("end_year"),
                "is_current": e.get("is_current"),
            }
            for e in (profile.get("experience") or [])
            if isinstance(e, dict)
        ],
        "education": [
            {
                "school": e.get("school"),
                "degree": e.get("degree"),
                "field": e.get("field"),
                "end_year": e.get("end_year"),
            }
            for e in (profile.get("education") or [])
            if isinstance(e, dict)
        ],
        "skills": [s.get("name") for s in (profile.get("skills") or []) if isinstance(s, dict)],
    }


def _rank_candidates(claims: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order search hits so the most plausible namesakes are fetched first."""
    city = _norm(claims.get("city"))
    headline = _norm(claims.get("headline"))
    field = _norm(claims.get("field"))

    def score(candidate: dict[str, Any]) -> tuple[float, int]:
        location = _norm(candidate.get("location") or candidate.get("city"))
        points = 0.0
        if city and city in location:
            points += 2.0
        if headline and _norm(candidate.get("headline")) == headline:
            points += 1.0
        if field and field == _norm(candidate.get("industry")):
            points += 1.0
        if _norm(candidate.get("name")) == _norm(claims.get("name")):
            points += 0.5
        return (-points, int(candidate.get("id") or 0))

    return sorted(candidates, key=score)


async def _search_candidates(
    tool_map: dict[str, Any], claims: dict[str, Any]
) -> list[dict[str, Any]]:
    """Find namesakes, narrowing on city and industry first (the search matches on substrings)."""
    name, city, field = claims.get("name"), claims.get("city"), claims.get("field")
    attempts = [
        {"q": name, "location": city, "industry": field, "limit": 20},
        {"q": name, "location": city, "limit": 20},
        {"q": name, "industry": field, "limit": 20},
        {"q": name, "limit": 20},
    ]
    tried: set[tuple] = set()
    for args in attempts:
        args = {k: v for k, v in args.items() if v}
        if len(args) < 2 or tuple(sorted(args.items())) in tried:
            continue
        tried.add(tuple(sorted(args.items())))
        hits = await _call_tool(tool_map, "search_linkedin_people", args)
        if isinstance(hits, list) and hits:
            _log("search", args, "->", len(hits), "hits")
            return [h for h in hits if isinstance(h, dict) and h.get("id") is not None]
    return []


def _overlap(claims: dict[str, Any], profile: dict[str, Any]) -> tuple[set[str], set[str]]:
    companies = {_norm(j["company"]) for j in claims.get("jobs") or [] if j.get("company")}
    schools = {_norm(e["school"]) for e in claims.get("education") or [] if e.get("school")}
    profile_companies = {_norm(e.get("company")) for e in profile.get("experience") or []}
    profile_schools = {_norm(e.get("school")) for e in profile.get("education") or []}
    profile_companies.discard("")
    profile_schools.discard("")
    return companies & profile_companies, schools & profile_schools


async def _collect_profiles(
    tool_map: dict[str, Any], claims: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fetch profiles in batches, stopping as soon as an employer or school agrees."""
    ranked = _rank_candidates(claims, candidates)
    profiles: list[dict[str, Any]] = []
    for start in range(0, min(len(ranked), MAX_PROFILES), PROFILE_BATCH):
        for candidate in ranked[start : start + PROFILE_BATCH]:
            profile = await _call_tool(
                tool_map, "get_linkedin_profile", {"person_id": candidate.get("id")}
            )
            if isinstance(profile, dict) and "error" not in profile:
                profiles.append(profile)
        if any(any(_overlap(claims, p)) for p in profiles):
            return profiles
    return profiles


async def _agent_probe(agent: Any, claims: dict[str, Any]) -> list[dict[str, Any]]:
    """Fallback: let the tool-calling agent hunt for the person when plain search finds nobody."""
    investigator = agent.get("investigator") if isinstance(agent, dict) else None
    if investigator is None:
        return []
    prompt = (
        f"Find the LinkedIn profile of {claims.get('name')!r}, who works in "
        f"{claims.get('field')!r} and lives in {claims.get('city')!r}.\n"
        f"CV employers: {[j['company'] for j in claims.get('jobs') or []]}\n"
        f"CV schools: {[e['school'] for e in claims.get('education') or []]}\n"
        "Use the search tools, trying different spellings, cities and industries. Then reply with a "
        'JSON list of candidate ids only, e.g. {"ids": [1, 2]}.'
    )
    try:
        result = await investigator.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": 16},
        )
        messages = result.get("messages") if isinstance(result, dict) else None
        final = messages[-1] if messages else result
        data = _extract_json(getattr(final, "content", final))
    except Exception as exc:
        _log("agent probe failed", exc)
        return []
    ids: list[int] = []
    if isinstance(data, dict):
        ids = [int(i) for i in (data.get("ids") or []) if str(i).isdigit()]
    elif isinstance(data, list):
        ids = [int(i) for i in data if str(i).isdigit()]
    candidates = []
    for person_id in ids[:MAX_PROFILES]:
        profile = await _call_tool(agent["tool_map"], "get_linkedin_profile", {"person_id": person_id})
        if isinstance(profile, dict) and "error" not in profile:
            candidates.append(profile)
    return candidates


async def _identify(agent: Any, claims: dict[str, Any], profiles: list[dict[str, Any]]) -> Any:
    """Which candidate is the CV's author?

    A shared employer or school is treated as a hard identity anchor: it is far stronger
    evidence than the model's overall impression, so it settles the choice whenever exactly
    one candidate carries the strongest anchor. The model decides the remaining cases (and
    breaks ties between equally anchored candidates).
    """
    anchors: dict[Any, int] = {}
    for profile in profiles:
        companies, schools = _overlap(claims, profile)
        anchors[profile.get("id")] = len(companies) + len(schools)
    best_anchor = max(anchors.values()) if anchors else 0
    anchored = [pid for pid, count in anchors.items() if count == best_anchor and count > 0]

    shown = [
        {k: v for k, v in _profile_digest(p).items() if k != "status"}
        for p in profiles[:MAX_JUDGE_PROFILES]
    ]
    user = (
        "CV claims:\n" + json.dumps(claims, ensure_ascii=False) + "\n\n"
        "Candidate profiles:\n" + json.dumps(shown, ensure_ascii=False)
    )
    verdict = await _llm_json(agent["model"], IDENTITY_SYSTEM, user)
    model_id = None
    if isinstance(verdict, dict):
        try:
            raw_id = verdict.get("matched_person_id")
            model_id = int(raw_id) if raw_id is not None else None
        except Exception:
            model_id = None
        _log("identity model ->", model_id, verdict.get("confidence"), verdict.get("reason"))
    else:
        _log("identity model -> no JSON")

    if len(anchored) == 1:
        chosen = anchored[0]
    elif model_id in anchored:
        chosen = model_id
    elif model_id is not None and anchors.get(model_id, 0) > 0:
        chosen = model_id
    elif anchored:
        chosen = anchored[0]
    else:
        chosen = model_id
    _log("identity anchors", anchors, "-> chosen", chosen)
    return chosen


def _save_cv_text(name: str, text: str) -> None:
    """Dump the markdown the model sees; handy for tuning Task 2's attack surface."""
    if not DEBUG:
        return
    try:
        out = Path("_debug_cv_text")
        out.mkdir(exist_ok=True)
        (out / f"{Path(name).stem}.md").write_text(text, encoding="utf-8")
    except Exception:
        pass


async def _verify(agent: Any, text: str) -> float:
    """Full pipeline for one CV: extract -> resolve identity -> compare fields -> score."""
    text = text[:MAX_CV_CHARS]
    model = agent["model"]

    data = await _llm_json(
        model, EXTRACT_SYSTEM, "CV text:\n\n" + text
    )
    claims = _claims_from(data) if isinstance(data, dict) else _regex_claims(text)
    if not claims.get("name"):
        claims = _regex_claims(text)
    _log("claims", json.dumps(claims, ensure_ascii=False)[:400])

    candidates = await _search_candidates(agent["tool_map"], claims)
    profiles = await _collect_profiles(agent["tool_map"], claims, candidates) if candidates else []
    if not profiles:
        profiles = await _agent_probe(agent, claims)
    if not profiles:
        _log("no profile found for", claims.get("name"))
        return 0.2

    person_id = await _identify(agent, claims, profiles)
    chosen = next((p for p in profiles if p.get("id") == person_id), None)
    if chosen is None:
        chosen = profiles[0]
    elif person_id != profiles[0].get("id"):
        # keep the chosen profile first so the comparison sees it as the person
        profiles = [chosen] + [p for p in profiles if p is not chosen]

    facebook = await _call_tool(
        agent["tool_map"], "get_facebook_profile", {"user_id": chosen.get("id")}
    )
    facebook_digest = None
    if isinstance(facebook, dict) and "error" not in facebook:
        facebook_digest = {
            "display_name": facebook.get("display_name"),
            "original_name": facebook.get("original_name"),
            "city": facebook.get("city"),
            "country": facebook.get("country"),
            "education": facebook.get("education"),
            "current_job": facebook.get("current_job"),
            "current_company": facebook.get("current_company"),
        }

    user = (
        "CV claims:\n" + json.dumps(claims, ensure_ascii=False) + "\n\n"
        "LinkedIn profile of the person (primary source):\n"
        + json.dumps(_profile_digest(chosen), ensure_ascii=False)
    )
    if facebook_digest:
        user += "\n\nFacebook profile of the same person (secondary, agrees with LinkedIn):\n" + json.dumps(
            facebook_digest, ensure_ascii=False
        )

    verdict = await _llm_json(model, JUDGE_SYSTEM, user)
    if not isinstance(verdict, dict):
        return 0.5
    discrepancies = [d for d in (verdict.get("discrepancies") or []) if isinstance(d, dict)]
    _log("verdict", json.dumps(verdict, ensure_ascii=False)[:600])
    if not discrepancies:
        return 0.9
    return 0.1


def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once.

    ``tools`` are the six SocialGraph MCP tools (Facebook + LinkedIn search and
    profile lookup), already wrapped as LangChain tools. You may add your own
    local tools as well.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env.
    """
    model = _model()
    tool_map = {getattr(tool, "name", str(tool)): tool for tool in tools}

    investigator = None
    if create_agent is not None and tools:
        try:
            investigator = create_agent(
                model=model,
                tools=tools,
                system_prompt=(
                    "You are a KYC researcher. Use the SocialGraph tools to look up a person on the "
                    "mock LinkedIn and Facebook networks. Names are heavily duplicated, so narrow "
                    "searches with the city and industry, and try several spellings before giving up."
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive
            _log("create_agent failed", exc)

    return {"model": model, "tools": tools, "tool_map": tool_map, "investigator": investigator}


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV.

    ``cvs`` maps each file name to its text, e.g. ``{"CV_1.pdf": "...", ...}``.
    Return a float in [0, 1] for every file name: higher means the CV is more
    likely consistent with the candidate's LinkedIn/Facebook data. A score
    above 0.5 counts as "valid", 0.5 or below counts as "has discrepancy".

        {"CV_1.pdf": 0.9, "CV_4.pdf": 0.1, ...}

    Catch errors per CV (e.g. a failed API call) and still return a score for
    it: an exception here means no results.csv, which scores zero.

    MCP tools are async, so call your agent with ``await agent.ainvoke(...)``.
    You may verify CVs in parallel (e.g. ``asyncio.gather``), but keep at most
    about 3 CVs in flight (e.g. with ``asyncio.Semaphore(3)``): the MCP server is
    shared by the whole class.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def one(name: str, text: str) -> tuple[str, float]:
        async with semaphore:
            _save_cv_text(name, text)
            for attempt in range(2):
                try:
                    score = await _verify(agent, text)
                    break
                except Exception as exc:
                    _log("verify failed", name, "attempt", attempt, exc)
                    await asyncio.sleep(1 + attempt)
            else:
                score = 0.5
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                score = 0.5
            score = float(score)
            if math.isnan(score) or not 0.0 <= score <= 1.0:
                score = 0.5
            _log("score", name, score)
            return name, score

    pairs = await asyncio.gather(*(one(name, text) for name, text in cvs.items()))
    return dict(pairs)


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
