"""Build a neofetch-style SVG card for a GitHub profile README.

Pulls live stats from the GitHub API and writes assets/card-dark.svg and assets/card-light.svg.
Run locally from the repo root with:  GH_TOKEN=... python card/generate.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import requests
from dateutil.relativedelta import relativedelta

# --------------------------------------------------------------------------- #
# Edit this block. Everything below it is plumbing.
# --------------------------------------------------------------------------- #
USERNAME = os.environ.get("GH_USER", "Falli007")
HEADER = "fredrick@falli007"
UPTIME_SINCE = date(2018, 1, 1)  # TODO: birthday, or the date you started coding

SECTIONS: list[list[tuple[str, str]]] = [
    [
        ("OS", "macOS, Linux, Windows"),
        ("Uptime", "{uptime}"),
        ("Host", "Fa Grace Ltd"),
        ("Kernel", "Data Scientist, AI & Data Engineer"),
        ("IDE", "VS Code, Jupyter, Databricks"),
    ],
    [
        ("Languages.Programming", "Python, SQL, SPL"),
        ("Languages.Data", "Snowflake, dbt, PySpark, Delta"),
        ("Languages.Real", "English"),
    ],
    [
        ("Stack.AI", "RAG, LLMs, LangGraph, FastAPI"),
        ("Stack.ML", "NLP, Risk Analytics, MLflow"),
        ("Stack.Cloud", "Azure, AWS, Docker, CI/CD"),
    ],
]
CONTACT: list[tuple[str, str]] = [
    ("Portfolio", "fredrickalli.com"),
    ("LinkedIn", "fredrick-a-ab81922"),
    ("GitHub", "Falli007"),
]
HERE = Path(__file__).resolve().parent
PORTRAIT_FILE = HERE / "portrait.json"  # coloured portrait from portrait.py; preferred if present
ASCII_FILE = HERE / "ascii.txt"  # plain fallback from ascii.py
CACHE_FILE = HERE / "cache" / "loc.json"
OUT_DIR = HERE.parent / "assets"
# --------------------------------------------------------------------------- #

API = "https://api.github.com"
ROW_WIDTH = 60  # characters per info row, including dot leaders
FONT_SIZE = 16
LINE_HEIGHT = 20
CHAR_WIDTH = 9.65  # ConsolasFallback advance at 16px; tweak if columns drift

THEMES = {
    "dark": dict(bg="#161b22", text="#c9d1d9", key="#ffa657", value="#a5d6ff",
                 dots="#616e7f", add="#3fb950", sub="#f85149"),
    "light": dict(bg="#f6f8fa", text="#24292f", key="#953800", value="#0a3069",
                  dots="#c2cfde", add="#1a7f37", sub="#cf222e"),
}


@dataclass
class Stats:
    repos: int
    contributed: int
    stars: int
    followers: int
    commits: int
    loc_add: int
    loc_del: int

    @property
    def loc_net(self) -> int:
        return self.loc_add - self.loc_del


class GitHub:
    def __init__(self, token: str) -> None:
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"bearer {token}",
                               "Accept": "application/vnd.github+json"})

    def graphql(self, query: str, **variables) -> dict:
        r = self.s.post(f"{API}/graphql", json={"query": query, "variables": variables}, timeout=30)
        r.raise_for_status()
        body = r.json()
        if "errors" in body:
            raise RuntimeError(body["errors"])
        return body["data"]

    def contributor_stats(self, repo: str, retries: int = 6) -> list[dict]:
        """GitHub computes these lazily and returns 202 until ready."""
        for attempt in range(retries):
            r = self.s.get(f"{API}/repos/{repo}/stats/contributors", timeout=30)
            if r.status_code == 200:
                return r.json() or []
            if r.status_code == 204:  # empty repo
                return []
            time.sleep(2 * (attempt + 1))
        print(f"warn: stats for {repo} not ready, using cache if any")
        return []


def fetch_stats(gh: GitHub, user: str) -> Stats:
    data = gh.graphql(
        """query($login: String!) {
          user(login: $login) {
            createdAt
            followers { totalCount }
            repositoriesContributedTo(contributionTypes: [COMMIT, PULL_REQUEST, REPOSITORY]) { totalCount }
            repositories(first: 100, ownerAffiliations: OWNER, isFork: false) {
              totalCount
              nodes { nameWithOwner stargazerCount pushedAt }
            }
          }
        }""",
        login=user,
    )["user"]
    repos = data["repositories"]["nodes"]

    commits = 0
    start = datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    while start < now:  # contributionsCollection spans at most one year per call
        end = min(start + relativedelta(years=1), now)
        commits += gh.graphql(
            """query($login: String!, $from: DateTime!, $to: DateTime!) {
              user(login: $login) {
                contributionsCollection(from: $from, to: $to) {
                  totalCommitContributions restrictedContributionsCount
                }
              }
            }""",
            login=user, **{"from": start.isoformat(), "to": end.isoformat()},
        )["user"]["contributionsCollection"]["totalCommitContributions"]
        start = end

    loc_add, loc_del = lines_of_code(gh, user, repos)
    return Stats(
        repos=data["repositories"]["totalCount"],
        contributed=data["repositoriesContributedTo"]["totalCount"],
        stars=sum(r["stargazerCount"] for r in repos),
        followers=data["followers"]["totalCount"],
        commits=commits,
        loc_add=loc_add,
        loc_del=loc_del,
    )


def lines_of_code(gh: GitHub, user: str, repos: list[dict]) -> tuple[int, int]:
    """Sum the user's additions/deletions per repo, skipping repos unchanged since last run."""
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    for repo in repos:
        name, pushed = repo["nameWithOwner"], repo["pushedAt"]
        if cache.get(name, {}).get("pushed") == pushed:
            continue
        mine = next((c for c in gh.contributor_stats(name)
                     if c.get("author") and c["author"]["login"].lower() == user.lower()), None)
        if mine is None and name in cache:
            continue  # stats not ready; keep the old figure rather than zero it
        weeks = mine["weeks"] if mine else []
        cache[name] = {"pushed": pushed,
                       "add": sum(w["a"] for w in weeks),
                       "del": sum(w["d"] for w in weeks)}
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return sum(v["add"] for v in cache.values()), sum(v["del"] for v in cache.values())


def uptime(since: date) -> str:
    d = relativedelta(date.today(), since)
    plural = lambda n, unit: f"{n} {unit}{'' if n == 1 else 's'}"
    return ", ".join([plural(d.years, "year"), plural(d.months, "month"), plural(d.days, "day")])


# ------------------------------- SVG rendering ----------------------------- #

def t(cls: str, text: str) -> str:
    return f'<tspan class="{cls}">{escape(text)}</tspan>'


def row(key: str, value: str, width: int = ROW_WIDTH) -> str:
    """'. Key: ......... value' with dots filling to a fixed width."""
    dots = max(1, width - len(key) - len(value) - 5)
    return f'{t("cc", ". ")}{t("key", key)}:{t("cc", " " + "." * dots + " ")}{t("value", value)}'


def heading(label: str, width: int = ROW_WIDTH) -> str:
    return t("text", f"- {label} ") + t("cc", "-" * (width - len(label) - 3))


def split_row(k1: str, v1: str, k2: str, v2: str) -> str:
    """Two stats on one line, separated by a pipe, like the original."""
    left_w = 32
    right_w = ROW_WIDTH - left_w - 3
    left = row(k1, v1, left_w)
    d = max(1, right_w - len(k2) - len(v2) - 3)
    right = f'{t("key", k2)}:{t("cc", " " + "." * d + " ")}{t("value", v2)}'
    return f'{left}{t("cc", " | ")}{right}'


def build_lines(s: Stats) -> list[str]:
    lines = [t("text", HEADER + " ") + t("cc", "-" * (ROW_WIDTH - len(HEADER) - 1))]
    for i, section in enumerate(SECTIONS):
        if i:
            lines.append(t("cc", ". "))
        lines += [row(k, v.format(uptime=uptime(UPTIME_SINCE))) for k, v in section]
    lines += [""] + [heading("Contact")] + [row(k, v) for k, v in CONTACT]
    lines += [""] + [heading("GitHub Stats")]
    lines.append(split_row("Repos", f"{s.repos} {{Contributed: {s.contributed}}}", "Stars", f"{s.stars:,}"))
    lines.append(split_row("Commits", f"{s.commits:,}", "Followers", f"{s.followers:,}"))
    loc = (f'{t("cc", ". ")}{t("key", "Lines of Code on GitHub")}:{t("cc", " ")}'
           f'{t("value", f"{s.loc_net:,}")} ( {t("add", f"{s.loc_add:,}++")}, '
           f'{t("sub", f"{s.loc_del:,}--")} )')
    lines.append(loc)
    return lines


def portrait_svg(rows: list[list[list[str]]], line_h: float, x0: float, y0: float) -> str:
    """One <text> per row; each colour run is a tspan so the photo's colours come through."""
    out = []
    for i, runs in enumerate(rows):
        spans = "".join(
            escape(txt) if not col else f'<tspan fill="{col}">{escape(txt)}</tspan>' for txt, col in runs)
        out.append(f'<text x="{x0:.1f}" y="{y0 + i * line_h:.1f}">{spans}</text>')
    return "\n".join(out)


def render(s: Stats, theme: dict[str, str]) -> str:
    info = build_lines(s)
    info_h = len(info) * LINE_HEIGHT
    pad = 15

    if PORTRAIT_FILE.exists():
        rows = json.loads(PORTRAIT_FILE.read_text())
        cols = max(sum(len(t) for t, _ in r) for r in rows)
        line_h = info_h / len(rows)  # scale the portrait to the height of the info column
        art_font = line_h * 0.83  # keeps a monospace glyph about half as wide as it is tall
        art_w = cols * art_font * 0.62 + 2 * pad
        art = (f'<rect x="{pad}" y="{pad}" width="{art_w:.0f}" height="{info_h + 2 * pad - 10:.0f}" '
               f'rx="10" fill="#0d1117"/>\n'
               f'<g xml:space="preserve" font-size="{art_font:.2f}px">\n'
               f'{portrait_svg(rows, line_h, 2 * pad, pad + line_h + 4)}\n</g>')
        info_x = art_w + 3 * pad
    elif ASCII_FILE.exists():
        lines = ASCII_FILE.read_text().rstrip("\n").splitlines()
        art = (f'<text xml:space="preserve" fill="{theme["text"]}">' + "".join(
            f'<tspan x="{pad}" y="{30 + i * LINE_HEIGHT}">{escape(l)}</tspan>' for i, l in enumerate(lines))
            + "</text>")
        info_x = pad + (max(map(len, lines)) + 3) * CHAR_WIDTH
    else:
        art, info_x = "", pad

    width = int(info_x + (ROW_WIDTH + 2) * CHAR_WIDTH + 20)  # slack for wider fallback fonts
    height = 30 + info_h + 10
    info_svg = "\n".join(
        f'<tspan x="{info_x:.0f}" y="{30 + i * LINE_HEIGHT}">{l}</tspan>' for i, l in enumerate(info))
    art_svg = art

    return f"""<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" font-family="ConsolasFallback,Consolas,'DejaVu Sans Mono',monospace" width="{width}px" height="{height}px" font-size="{FONT_SIZE}px">
<style>
@font-face {{ src: local('Consolas'), local('Consolas Bold'); font-family: 'ConsolasFallback'; font-display: swap; -webkit-size-adjust: 109%; size-adjust: 109%; }}
.key {{fill: {theme['key']};}} .value {{fill: {theme['value']};}} .cc {{fill: {theme['dots']};}}
.add {{fill: {theme['add']};}} .sub {{fill: {theme['sub']};}} .text, text, tspan {{white-space: pre;}}
</style>
<rect width="{width}px" height="{height}px" fill="{theme['bg']}" rx="15"/>
{art_svg}
<text xml:space="preserve" fill="{theme['text']}">
{info_svg}
</text>
</svg>
"""


def main() -> None:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("Set GH_TOKEN to a GitHub token")
    stats = fetch_stats(GitHub(token), USERNAME)
    for name, theme in THEMES.items():
        (OUT_DIR / f"card-{name}.svg").write_text(render(stats, theme), encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
