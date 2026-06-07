"""
Patch MediaCrawler base_config.py for a specific event + platform.

Usage:
    .venv/bin/python scripts/set_crawl.py <event_id> <platform>

Platform: wb | dy | zhihu | bili

Example:
    .venv/bin/python scripts/set_crawl.py wang_xing_rescue_001 wb
"""
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MEDIACRAWLER_CONFIG = BASE_DIR.parent / "MediaCrawler" / "config" / "base_config.py"
CRAWL_DATA_ROOT = BASE_DIR.parent / "MediaCrawler" / "data" / "crawled"

# ── Event registry ─────────────────────────────────────────────────────────────
EVENTS = {
    "wang_xing_rescue_001": {
        "keywords": "王星被救,缅北诈骗,王星缅甸,电信诈骗东南亚",
        "start": "2025-01-01",
        "end":   "2025-01-20",
    },
    "china_astock_rally_001": {
        "keywords": "A股暴涨,牛市2024,央行降准,股市行情2024",
        "start": "2024-09-24",
        "end":   "2024-10-15",
    },
    "nongfu_spring_boycott_001": {
        "keywords": "农夫山泉抵制,钟睒睒,娃哈哈宗庆后,矿泉水国货",
        "start": "2024-02-01",
        "end":   "2024-04-30",
    },
    "korea_martial_law_001": {
        "keywords": "韩国戒严,尹锡悦戒严,韩国国会,韩国政变",
        "start": "2024-12-03",
        "end":   "2024-12-20",
    },
    "hunan_flood_2024_001": {
        "keywords": "湖南洪灾,洞庭湖决口,湖南防汛,华容溃堤",
        "start": "2024-06-15",
        "end":   "2024-07-10",
    },
    "china_retirement_reform_001": {
        "keywords": "延迟退休,退休年龄改革,法定退休年龄,养老金政策",
        "start": "2024-09-01",
        "end":   "2024-10-31",
    },
    "xizang_earthquake_2025_001": {
        "keywords": "西藏地震,定日地震,西藏救援,地震捐款2025",
        "start": "2025-01-07",
        "end":   "2025-01-20",
    },
    "manus_ai_launch_001": {
        "keywords": "Manus AI,Manus人工智能,Manus体验,Manus智能体",
        "start": "2025-03-06",
        "end":   "2025-03-31",
    },
    "pan_zhanle_100m_001": {
        "keywords": "潘展乐,世界纪录,巴黎奥运会游泳,兴奋剂质疑",
        "start": "2024-07-27",
        "end":   "2024-08-15",
    },
    "hua_chenyu_custody_001": {
        "keywords": "华晨宇张碧晨,华晨宇抚养权,华晨宇女儿",
        "start": "2024-03-01",
        "end":   "2024-06-30",
    },
    "eu_china_ev_tariffs_001": {
        "keywords": "欧盟电动车关税,中欧贸易争端,比亚迪欧洲,新能源汽车出海",
        "start": "2024-06-01",
        "end":   "2024-09-30",
    },
    "ukraine_kursk_incursion_001": {
        "keywords": "乌克兰库尔斯克,乌军进攻俄领土,俄乌最新战况,乌克兰反攻",
        "start": "2024-08-06",
        "end":   "2024-08-31",
    },
    "guo_meimei_release_001": {
        "keywords": "郭美美,郭美美出狱,红十字会,网红出狱",
        "start": "2024-04-01",
        "end":   "2024-06-30",
    },
}

PLATFORM_MAP = {
    "wb":    "wb",
    "dy":    "dy",
    "zhihu": "zhihu",
    "bili":  "bili",
}


def patch(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if n == 0:
        print(f"  WARN: pattern not found: {pattern!r}")
        return
    path.write_text(new_text, encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    event_id, platform = sys.argv[1], sys.argv[2]

    if event_id not in EVENTS:
        print(f"Unknown event: {event_id}")
        print(f"Known events: {', '.join(EVENTS)}")
        sys.exit(1)

    if platform not in PLATFORM_MAP:
        print(f"Unknown platform: {platform}. Choose from: {', '.join(PLATFORM_MAP)}")
        sys.exit(1)

    cfg = EVENTS[event_id]
    mc_platform = PLATFORM_MAP[platform]
    save_path = str(CRAWL_DATA_ROOT / event_id)

    if not MEDIACRAWLER_CONFIG.exists():
        print(f"MediaCrawler config not found: {MEDIACRAWLER_CONFIG}")
        sys.exit(1)

    print(f"Patching {MEDIACRAWLER_CONFIG.name} for {event_id} / {platform}")

    patch(MEDIACRAWLER_CONFIG,
          r'^(PLATFORM\s*=\s*).*$',
          f'\\g<1>"{mc_platform}"')
    patch(MEDIACRAWLER_CONFIG,
          r'^(KEYWORDS\s*=\s*).*$',
          f'\\g<1>"{cfg["keywords"]}"')
    patch(MEDIACRAWLER_CONFIG,
          r'^(SAVE_DATA_PATH\s*=\s*).*$',
          f'\\g<1>"{save_path}"')

    print(f"  PLATFORM      = {mc_platform!r}")
    print(f"  KEYWORDS      = {cfg['keywords']!r}")
    print(f"  SAVE_DATA_PATH = {save_path!r}")
    print()
    print(f"Date range for convert step:  --start {cfg['start']} --end {cfg['end']}")
    if platform == "bili":
        print(f"  ⚠️  Bilibili: also set START_DAY={cfg['start']} END_DAY={cfg['end']} manually in base_config.py")
    print()
    print("Next: cd ../MediaCrawler && .venv/bin/python main.py")


if __name__ == "__main__":
    main()
