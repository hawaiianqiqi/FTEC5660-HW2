# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1

My verifier is an evidence-first pipeline rather than a free-roaming ReAct loop, because the hard
part of this task is *entity resolution*, not tool use. For each CV the pipeline runs four stages.
**(1) Claim extraction** asks `deepseek-v4-flash` for a structured JSON record of the CV: name,
city, country, headline, field, jobs (company/title/start/end), education (school/degree/field/
graduation year) and skills, with a regex fallback if the call fails. **(2) Retrieval** exploits a
quirk of the SocialGraph search tools: they do a case-insensitive *substring* match, so the
`location` filter has to be the bare city ("London", not "London, UK", which returns nothing) and
the `industry` filter the bare industry word taken from the CV headline ("Logistics", not
"Logistics Professional"). Searching `(name, location=city, industry=field)` collapses hundreds of
namesakes down to 1-6 candidates; the pipeline falls back to `(name, city)`, `(name, industry)`
and finally `(name)` when a CV has a falsified city. **(3) Identity** is decided by *hard anchors*:
a shared employer or school between the CV and a candidate profile is far stronger evidence than
the model's overall impression, so when exactly one candidate carries the strongest anchor the
choice is made deterministically and the model is only asked to break ties (a fallback tool-calling
agent, built with `create_agent`, hunts for the person when plain search returns nobody).
**(4) Comparison** sends the CV claims plus a digest of the chosen LinkedIn profile (and the
corroborating Facebook profile) to the model, which must return only genuine discrepancies in the
allowed fields, with the equivalence rules from the brief encoded in the prompt: wording differences
("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`, "Microsoft PowerPoint" vs `PowerPoint`),
a seniority word that agrees with the profile's seniority, and *fewer* skills listed are all fine,
while an extra skill, a shifted year, an inflated title, a fake employer or school, and a wrong
location are not. The verdict maps to 0.9 (no discrepancy), 0.1 (any discrepancy), 0.2 (no profile
found) or 0.5 (unverifiable failure). Reliability is handled throughout: every MCP call retries four
times with backoff (I lost `CV_1` to one transient `get_linkedin_profile` failure in an early run
exactly because of this), each CV retries once and then still returns a bounded float so one failure
can never crash the run, and CVs are verified through `asyncio.Semaphore(3)`. One further failure
mode is worth naming because it costs a whole run rather than a single CV: the provided
`load_mcp_tools()` calls `MultiServerMCPClient.get_tools()` exactly once, so a single disconnect
from the shared ngrok endpoint raises out of `run()` before any CV is scored and leaves no
`results.csv` at all - I reproduced this once with `httpx.RemoteProtocolError: Server disconnected
without sending a response`. I did not edit that provided function; the student block installs a
bounded exponential-backoff retry around the library method it calls, which can only turn a crash
into a retry.

**Results on `public_test`: 7/7 (1.0 accuracy)**, reproduced on three consecutive runs, in about
85 s per run (~13 s per CV, i.e. roughly 5 minutes for the ~20 unseen CVs, well inside the
30-minute cap). All three seeded discrepancies are caught for the intended reason, and no valid CV
is flagged:

| CV | Ground truth | My verdict |
| --- | --- | --- |
| CV_1, CV_2, CV_3, CV_6 | valid (1) | 0.90 - correct |
| CV_4 | Innovate HK Ltd start year 2015 vs 2014 on LinkedIn | 0.10 - correct (shifted year) |
| CV_5 | "Senior Analyst" vs `Analyst` with seniority `junior` | 0.10 - correct (inflated title) |
| CV_7 | lists `Docker`, absent from LinkedIn | 0.10 - correct (made-up skill) |

### Task 2

`task2/adversarial_cv.pdf` is Kelly Tsang's CV with exactly one false claim: it presents her AIA
role as **"Senior Manager"** although LinkedIn (and Facebook) say `Manager` with seniority `mid`.
The file is produced by `task2/make_adversarial_cv.py`.

I red-teamed this against my own Task 1 agent, and the first attempt failed instructively. A CV
whose visible title was an inflated `Senior Manager` plus a raw prompt-injection layer (fake
system delimiters, a fake "applicant tracking system" record, an explicit "return 0.95") scored
**0.10**, because my verifier is two-stage: the model that reads the CV only *extracts fields*, and
the model that judges consistency never sees the CV text at all, so injected instructions land in a
prompt that has no scoring authority. Extra skills added along the way were also caught. The fix
was to stop trying to talk the verifier out of the comparison and instead attack **what the
comparison can see**.

Three techniques are stacked, and together they close every field-comparison path:

1. **The false claim is drawn, not typed.** The word "Senior" in the title line is rendered as a
   600-dpi bitmap (`insert_image`), not as text; `Manager, AIA` stays as real text immediately to
   its right. A human reading the PDF sees "Senior Manager, AIA", while the grader's loader
   (`markitdown(enable_plugins=False)`, no OCR) extracts only `Manager, AIA` - which agrees with
   the profile. Reproduction: `page.get_text()` contains `Manager, AIA` but not
   `Senior Manager, AIA`, and the page's single image sits at x=53.0-93.3, exactly where the text
   layer's title begins at x=93.29. `python task2/verify_adversarial_cv.py` re-runs all of this:
   it checks the text layer, measures the ink left of the title, compares the embedded bitmap
   against reference renderings of several words (it matches `Senior` exactly and every other
   candidate by size), and confirms the remaining visible claims are truthful.
2. **The same false claim is also placed in a field no verifier is allowed to check.** The AIA job
   description says "Promoted to Senior Manager in 2024...". The brief states that job descriptions
   are never the source of a discrepancy, so a field-comparison verifier extracts the title, finds
   `Manager` == `Manager`, and never looks at the bullet. This keeps the false detail present in
   the plain text layer as well, so the claim is not merely hidden in the artwork.
3. **Indirect prompt injection as a second front.** An invisible layer (white text on page 1 plus
   `render_mode=3` text on page 2) carries a plausible "HR verification record" that calls the
   dossier pre-verified and explains the AIA title difference away as a stale profile mirror, and
   then a blunter override for agents that paste raw CV text into one prompt. This is inert against
   a two-stage verifier - which is exactly why techniques 1 and 2 carry the attack - but it is
   cheap and picks up single-prompt designs.

Why I expect this to work: every verifier is specified to read the CV with the same PDF loader as
`hw2.py`, so a claim that exists only as pixels cannot enter any of their comparisons, and the
documented field rules exclude job descriptions from scrutiny even in the text layer. The attack
does not rely on the model being fooled by a clever sentence - it removes the false claim from the
evidence channel entirely, which is robust across extraction prompts, judge rubrics and models.

**Self-test:** `python3 hw2.py --cv-folder task2` scores `adversarial_cv.pdf` **0.90** (verified
valid) and the genuine `target_cv.pdf` 0.90, with my verifier's notes reading "all CV claims agree
with the LinkedIn profile ... employment history (AIA, Standard Chartered, HSBC with matching
titles and years)". Also note that only the PDF is used as the attack surface: the MCP server is
never touched. Reload `make_adversarial_cv.py` with `pymupdf` to rebuild the PDF.

### Task 3: reflection on the AI paradoxes

Building both sides of this homework in one sitting is the paradox made concrete: I wrote a
verifier that scores 7/7, and then I defeated it in an afternoon without ever saying anything
false *to it*. Two things struck me.

*Verification is only as wide as its evidence channel.* My agent is, by the rubric, correct: it
extracts the fields the brief declares verifiable, anchors identity on employers and schools, and
applies the equivalence rules carefully. It still passes a CV that visibly claims a promotion its
candidate never received, because "Senior" was pixels rather than text. The failure was not
reasoning; it was the boundary of what the system was allowed to perceive. That is the uncomfortable
part of delegating judgment to models: the failure mode of a confident agent is not a wrong answer,
it is a *complete* answer built from a truncated world. Every KYC system that "verifies" a document
has drawn a line like this somewhere, usually without writing it down, and the line - not the model
- is what an adversary attacks.

*The same technique is both the attack and the audit.* Nothing in `make_adversarial_cv.py` is
specific to this course: hidden layers, image-rendered text, stale-data excuses and field-coverage
gaps are what real identity fraud looks like, and the reason my attack worked is that the defence
assumed the document could be trusted as an input. So the constructive reading of this homework is
that red-teaming is not the opposite of verification - it is the only way to discover the coverage
of a verification system. I learned more about my Task 1 agent from the 20 minutes of attacking it
than from tuning it to 7/7, and the concrete lessons (compare rendered evidence, not just extracted
text; never let a document describe its own trustworthiness; treat an "already verified" claim as a
red flag) are the kind of thing an accuracy number will never tell you.

And the second-order paradox: the more capable the verifier, the more valuable the attack on its
*inputs* becomes, so the two capabilities grow together. Delegating trust to a model does not
remove the trust problem; it moves it to the model's field of view, where it is harder to see and
easier to automate against.

