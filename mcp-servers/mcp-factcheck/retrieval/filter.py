"""Source filtering — ported from InTruth's BLOCKED_DOMAINS + date filter.

Drops low-credibility / partisan / state-media sources and sources dated >1 year after
the event (prevents future-knowledge contamination of historical claims).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from urllib.parse import urlparse

from .base import OrganicResult

# Verbatim port of InTruth's BLOCKED_DOMAINS (service-worker.js lines 61-106)
BLOCKED_DOMAINS = {
    # social media
    "reddit.com", "facebook.com", "twitter.com", "x.com", "tiktok.com", "instagram.com",
    "pinterest.com", "quora.com", "yelp.com", "tripadvisor.com", "youtube.com",
    # partisan left
    "democrats.org", "dnc.org", "afscme.org", "americanprogress.org", "dailykos.com",
    "mediamatters.org", "motherjones.com", "moveon.org", "huffpost.com", "thenation.com",
    "jacobinmag.com", "truthout.org", "commondreams.org", "alternet.org", "rawstory.com",
    "palmerreport.com", "occupydemocrats.com", "crooksandliars.com", "bipartisanreport.com",
    "wonkette.com",
    # partisan right
    "republicans.org", "gop.com", "ntu.org", "heritage.org", "breitbart.com", "newsmax.com",
    "thefederalist.com", "nationalreview.com", "dailywire.com", "townhall.com",
    "westernjournal.com", "lifesitenews.com", "oann.com", "theblaze.com",
    "americanthinker.com", "dailycaller.com", "thepostmillennial.com", "redstate.com",
    "pjmedia.com", "bizpacreview.com", "americangreatness.com", "gellerreport.com",
    "thepoliticalinsider.com", "twitchy.com", "wnd.com",
    # government / state media
    "gov.il", "rt.com", "sputniknews.com", "xinhua.net", "globaltimes.cn", "presstv.ir",
    "almayadeen.net", "idf.il", "cgtn.com", "tass.com", "mintpressnews.com",
    "thegrayzone.com", "strategic-culture.org", "southfront.org", "veteranstoday.com",
    # conspiracy / low credibility
    "infowars.com", "naturalnews.com", "zerohedge.com", "thegatewaypundit.com",
    "beforeitsnews.com", "activistpost.com", "newspunch.com", "neonnettle.com",
    "worldtruth.tv", "mercola.com", "greenmedinfo.com", "childrenshealthdefense.org",
    # advocacy organizations
    "ajc.org", "cair.com", "adl.org", "aclu.org",
    "democrats-appropriations.house.gov", "waysandmeans.house.gov",
    # misc low quality
    "bostonkravmaga.com", "israelpolicyforum.org",
    # PDF hosting / document repositories
    "dokumen.pub", "dokumen.tips", "slideshare.net", "pdfdrive.com", "pdfcoffee.com", "issuu.com",
}


def domain_of(url: str) -> str:
    """Extract registrable domain from a URL (rough — strips subdomains)."""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return ""
    # crude: take last two labels (handles co.uk poorly but sufficient for filtering)
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def is_blocked(url: str) -> bool:
    d = domain_of(url)
    if not d:
        return False
    return any(blocked in d or d in blocked for blocked in BLOCKED_DOMAINS)


def filter_results(
    results: list[OrganicResult],
    event_date: str | None = None,
    max_results: int = 4,
    block_partisan: bool = True,
) -> list[OrganicResult]:
    """Apply InTruth's filters: blocklist + date + cap."""
    one_year_after = None
    if event_date:
        try:
            one_year_after = datetime.fromisoformat(event_date[:10]) + timedelta(days=365)
        except ValueError:
            pass

    out = []
    for r in results:
        if block_partisan and is_blocked(r.url):
            continue
        if one_year_after and r.date:
            try:
                src_date = datetime.fromisoformat(r.date[:10])
                if src_date > one_year_after:
                    continue
            except ValueError:
                pass
        out.append(r)
        if len(out) >= max_results:
            break
    return out
