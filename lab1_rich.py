from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from lab1_metadata import as_dict, as_list, text

RICH_CAT = ["graph_company", "graph_person", "profile_company"]
COUNTERS = ["commentsCount", "repostsCount", "likeCount", "empathyCount", "praiseCount"]
IDS = re.compile(r"(?:ORG|PERSON|AUTHOR)_\d+")


def identifiers(value):
    return sorted(set(IDS.findall(str(value))))


def numeric(value):
    return float(value) if isinstance(value, (int, float)) else 0.0


def post_values(post):
    images = [value for value in as_list(post.get("image")) if isinstance(value, dict)]
    widths = np.array([numeric(value.get("width")) for value in images])
    heights = np.array([numeric(value.get("height")) for value in images])
    ratios = widths / np.maximum(heights, 1)
    shared = as_dict(post.get("resharedPost"))
    companies = identifiers(post.get("companyMentions", []))
    people = identifiers(post.get("mentions", []))
    companies += identifiers(post.get("company", {}))
    companies = sorted(set(companies))
    values = {
        "actual_image_count": len(images),
        "actual_image_log_area": (
            float(np.log1p((widths * heights).max())) if len(images) else 0
        ),
        "actual_image_aspect": float(np.median(ratios)) if len(images) else 0,
        "actual_image_portrait": float(np.mean(ratios < 0.9)) if len(images) else 0,
        "actual_image_square": (
            float(np.mean((ratios >= 0.9) & (ratios <= 1.1))) if len(images) else 0
        ),
        "actual_image_landscape": float(np.mean(ratios > 1.1)) if len(images) else 0,
        "reshared_image_count": len(as_list(shared.get("image"))),
        "reshared_video_count": len(as_list(shared.get("video"))),
        "entity_chars": len(text(as_dict(post.get("entity")).get("title")))
        + len(text(as_dict(post.get("entity")).get("subtitle"))),
        "document_title_chars": len(text(as_dict(post.get("document")).get("title"))),
        "graph_company_count": len(companies),
        "graph_person_count": len(people),
        "graph_company": companies[0] if companies else "unknown",
        "graph_person": people[0] if people else "unknown",
        "graph_text": " ".join(companies + people) or "NO_MENTIONS",
    }
    return values, companies, people


def causal_rich(posts, owner):
    posts = [post for post in posts if as_dict(post.get("author")).get("code") == owner]
    posts = sorted(posts, key=lambda post: numeric(post.get("postedDateTimestamp")))
    rows = []
    reactions = []
    counters = []
    times = []
    groups = defaultdict(list)
    mention_history = defaultdict(list)
    for post in posts:
        values, companies, people = post_values(post)
        current_time = numeric(post.get("postedDateTimestamp")) / 86400000
        history = np.asarray(reactions, dtype=float)
        level = np.median(history) if len(history) else np.nan
        content = text(post.get("contentType")) or "unknown"
        formats = [
            content,
            str(bool(post.get("resharedPost"))),
            str(values["actual_image_count"] > 0),
        ]
        for kind, group in enumerate(formats):
            past = np.asarray(groups[(kind, group)], dtype=int)
            values[f"format{kind}_history_count"] = len(past)
            values[f"format{kind}_history_delta"] = (
                float(history[past].mean() - level) if len(past) else np.nan
            )
            values[f"format{kind}_history_spread"] = (
                float(history[past].std()) if len(past) else np.nan
            )
        for kind, mentions in [("company", companies), ("person", people)]:
            past = sorted(
                set(index for code in mentions for index in mention_history[code])
            )
            values[f"graph_{kind}_seen_fraction"] = (
                np.mean([bool(mention_history[code]) for code in mentions])
                if mentions
                else 0
            )
            values[f"graph_{kind}_history_count"] = len(past)
            values[f"graph_{kind}_history_delta"] = (
                float(history[past].mean() - level) if past else np.nan
            )
        if len(history):
            past_times = np.asarray(times)
            for days in [7, 30, 90, 365]:
                recent = history[past_times >= current_time - days]
                values[f"past{days}d_count"] = len(recent)
                values[f"past{days}d_delta"] = (
                    float(recent.mean() - level) if len(recent) else np.nan
                )
            for count in [5, 20, 100000]:
                past = history[-count:]
                for quantile in [0.1, 0.25, 0.75, 0.9]:
                    values[f"past{count}_quantile{int(quantile * 100)}"] = float(
                        np.quantile(past, quantile) - level
                    )
                values[f"past{count}_below"] = float(
                    np.mean(past - level < -0.5493061443340549)
                )
                values[f"past{count}_above"] = float(
                    np.mean(past - level > 1.157452788691043)
                )
                previous = np.asarray(counters[-count:])
                for index, name in enumerate(COUNTERS):
                    values[f"past{count}_{name}"] = float(
                        np.mean(np.log1p(previous[:, index]))
                    )
            previous = np.asarray(counters)
            for index, name in enumerate(COUNTERS):
                values[f"lag_{name}"] = float(np.log1p(previous[-1, index]))
            values["recent_reaction_trend"] = (
                float(np.polyfit(np.arange(min(10, len(history))), history[-10:], 1)[0])
                if len(history) > 1
                else 0
            )
            values["history_zero_fraction"] = float(np.mean(history == 0))
            values["history_unique_fraction"] = len(set(history)) / len(history)
        rows.append({"author": owner, "post": post["urn"], **values})
        for kind, group in enumerate(formats):
            groups[(kind, group)].append(len(reactions))
        for code in companies + people:
            mention_history[code].append(len(reactions))
        reactions.append(
            float(np.log1p(max(0, numeric(post.get("totalReactionCount")))))
        )
        counters.append([max(0, numeric(post.get(name))) for name in COUNTERS])
        times.append(current_time)
    return rows


def rich_table(table, data):
    data = Path(data)
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for directory in ["authors", "posts"]:
        for path in sorted((data / directory).glob("*.json")):
            stat = path.stat()
            digest.update(
                f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            )
    cache = Path(__file__).parent / ".cache" / f"rich-{digest.hexdigest()[:20]}.joblib"
    if cache.exists():
        extra, names = joblib.load(cache)
    else:
        rows = []
        for path in sorted((data / "posts").glob("AUTHOR_*.json")):
            feed = json.loads(path.read_text())
            rows.extend(causal_rich(feed["posts"], feed["author"]))
        extra = pd.DataFrame(rows)
        profiles = []
        for path in sorted((data / "authors").glob("AUTHOR_*.json")):
            profile = json.loads(path.read_text())
            positions = as_list(profile.get("fullPositions")) or as_list(
                profile.get("position")
            )
            company = (
                identifiers(as_dict(positions[0]).get("companyName", ""))
                if positions
                else []
            )
            profiles.append(
                {
                    "author": profile.get("author", path.stem),
                    "profile_company": company[0] if company else "unknown",
                    "profile_certifications": len(
                        as_list(profile.get("certifications"))
                    ),
                    "profile_courses": len(as_list(profile.get("courses"))),
                    "profile_creator": int(bool(profile.get("isCreator"))),
                    "profile_position_description_chars": sum(
                        len(text(as_dict(position).get("description")))
                        for position in positions
                    ),
                    "profile_skill_endorsements": float(
                        np.log1p(
                            sum(
                                numeric(as_dict(skill).get("endorsementsCount"))
                                for skill in as_list(profile.get("skills"))
                            )
                        )
                    ),
                }
            )
        extra = extra.merge(pd.DataFrame(profiles), on="author", validate="many_to_one")
        names = [
            name
            for name in extra
            if name not in ["author", "post", "graph_text"] + RICH_CAT
        ]
        joblib.dump((extra, names), cache, compress=3)
    result = table.merge(
        extra, on=["author", "post"], how="left", validate="one_to_one"
    )
    for column in RICH_CAT:
        result[column] = result[column].fillna("unknown").astype(str)
    return result, names
