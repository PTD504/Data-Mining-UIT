import json, re, time, csv, math, itertools, pathlib, logging, sys
from typing import Dict, List
import requests, pandas as pd
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

BASE          = "https://api.coursera.org/api/"
FIELDS_COURSE  = "name,slug,workload,duration,primaryLanguages,partnerIds,instructorIds"
BATCH          = 500                 
CHECKPOINT_EVERY = 5_0          
OUT_DIR        = pathlib.Path("data").resolve()
OUT_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
s      = requests.Session()

edge_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0"
s.headers.update({
    "Accept": "application/json, text/plain, */*",
    "User-Agent": edge_user_agent
})

def chunks(lst, n):
    it = iter(lst)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch: break
        yield batch

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def get_json(url:str, params:Dict=None, method:str="GET") -> Dict:
    """Request wrapper with retries & 4xx handling"""
    r = s.get(url, params=params, timeout=30) if method=="GET" else s.post(url, json=params, timeout=30)
    if r.status_code==414:                     
        logging.warning("414 at %s – replace with POST", url.split("?")[0])
        r = s.post(url.split("?")[0], json=params, timeout=30)
    r.raise_for_status()
    return r.json()

def save_checkpoint_json(df: pd.DataFrame, idx: int):
    f = OUT_DIR / f"coursera_courses_ckpt_{idx}.json"
    df.to_json(f, orient="records", lines=True, force_ascii=False)
    logging.info("Finished %s (%d rows)", f.name, len(df))

def crawl_course_headers() -> List[Dict]:
    start, got, headers = 0, 1, []
    pbar = tqdm(desc="Fetching headers", unit="course")
    while got:
        url = f"{BASE}courses.v1"
        params = {"start":start, "limit":BATCH, "fields":FIELDS_COURSE}
        data = get_json(url, params)
        got  = len(data.get("elements",[]))
        headers.extend(data["elements"])
        pbar.update(got)
        start += got
    pbar.close()
    logging.info("Total number of courses: %d", len(headers))
    return headers

def enrich_partners(df:pd.DataFrame):
    partner_map = {}
    all_ids = sorted({pid for ids in df.partnerIds for pid in ids})
    for chunk_ids in chunks(all_ids, 50):          
        ids_str = ",".join(map(str, chunk_ids))
        url = f"{BASE}partners.v1"
        partners = get_json(url, params={"ids":ids_str})["elements"]
        partner_map.update({p["id"]:p["name"] for p in partners})
    df["organization"] = df.partnerIds.apply(lambda x:", ".join(partner_map.get(pid,"") for pid in x))
    return df.drop(columns=["partnerIds"])

def enrich_instructors(df:pd.DataFrame):
    inst_map={}
    all_ids = sorted({iid for ids in df.instructorIds for iid in ids})
    for chunk_ids in chunks(all_ids, 50):
        url = f"{BASE}instructors.v1"
        insts = get_json(url, params={"ids":",".join(map(str, chunk_ids))})["elements"]
        inst_map.update({i["id"]:i.get("fullName") or i.get("name","") for i in insts})
    df["instructors"] = df.instructorIds.apply(lambda x:", ".join(inst_map.get(iid,"") for iid in x))
    return df.drop(columns=["instructorIds"])

LEVEL_RE   = re.compile(r"(Beginner|Intermediate|Advanced) level", re.I)
HOUR_RE    = re.compile(r"Approx\.\s+([\d\.]+\s+\w+)", re.I)
SKILL_RE   = re.compile(r'("skills"\s*:\s*\[(.*?)\])|("What you\'ll learn"\s*:\s*\[(.*?)\])', re.S)
ENROLL_RE  = re.compile(r'([\d,]+)\s+already enrolled', re.I)
PRICE_RE   = re.compile(r'\$[\d\.]+|Enroll for Free')
PHOTO_RE   = re.compile(r'"imageUrl"\s*:\s*"([^"]+)"')  
RATING_RE  = re.compile(r'aria-label="([\d\.]+)\s+stars"')  
REVIEWS_RE = re.compile(r'\(([\d,]+)\s+reviews\)')  
ENROLLED_RE = re.compile(r'([\d,]+)\s+learners') 

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
def parse_course_page(slug:str)->Dict:
    url = f"https://www.coursera.org/learn/{slug}?__amp=1"  
    html = s.get(url, timeout=20).text
    level       = LEVEL_RE.search(html)
    duration    = HOUR_RE.search(html)
    skills_blk  = SKILL_RE.search(html)
    enroll      = ENROLL_RE.search(html)
    price       = PRICE_RE.search(html)
    photo_url   = PHOTO_RE.search(html)
    rating      = RATING_RE.search(html)
    reviews     = REVIEWS_RE.search(html)
    enrolled    = ENROLLED_RE.search(html)
    
    return {
        "level"     : level.group(1).lower() if level else "",
        "est_hours" : duration.group(1) if duration else "",
        "skills"    : ", ".join(json.loads("["+skills_blk.group(2)+"]")) if skills_blk else "",
        "enrolled"  : int(enroll.group(1).replace(",","")) if enroll else None,
        "price"     : ("free" if price and "free" in price.group(0).lower() else price.group(0) if price else ""),
        "photo_url" : photo_url.group(1) if photo_url else "",
        "rating"    : float(rating.group(1)) if rating else None,
        "reviews"   : int(reviews.group(1).replace(",", "")) if reviews else None,
        "total_enrolled" : int(enrolled.group(1).replace(",", "")) if enrolled else None
    }


def main():
    headers = crawl_course_headers()
    df = pd.DataFrame(headers)
    df = enrich_partners(df)
    df = enrich_instructors(df)

    extra_cols = {k: [] for k in ["level", "est_hours", "skills", "enrolled", "price", "photo_url", "rating", "reviews", "total_enrolled"]}

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Parsing pages"):
        try:
            info = parse_course_page(row.slug)
        except Exception as e:
            logging.warning("❗ %s (%s)", e, row.slug)
            info = {k: "" for k in extra_cols}

        for k, v in info.items():
            extra_cols[k].append(v)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            tmp = pd.concat(
                [
                    df.iloc[: i + 1].reset_index(drop=True),
                    pd.DataFrame({k: extra_cols[k][: i + 1] for k in extra_cols}),
                ],
                axis=1,
            )
            save_checkpoint_json(tmp, i + 1)

    df_final = pd.concat([df, pd.DataFrame(extra_cols)], axis=1).fillna("")
    df_final.to_csv("coursera_courses_full.csv", index=False)
    logging.info("Finish: coursera_courses_full.csv (%d rows)", len(df_final))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Pausing…")
