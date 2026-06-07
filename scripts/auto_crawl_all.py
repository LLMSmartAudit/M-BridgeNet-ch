"""
Master orchestration script: crawl all expansion events × 4 platforms.
Run from M-BridgeNet root:
    .venv/bin/python scripts/auto_crawl_all.py [--start-event N] [--platform wb|dy|zhihu|bili]

Logs everything to logs/auto_crawl_YYYYMMDD_HHMMSS.log
On completion or Ctrl-C, prints a summary of succeeded/failed/skipped crawls.

Phase 0d (2026-05-23): 13 expansion events (11 viable, 2 skipped <50 posts)
Phase 1  (2026-05-23): 10 new test events — 5 domestic (wuyifan, huawei_mate60,
         shanghai_lockdown, henan_bank, li_jiaqi_huaxizi) + 5 international
         (spy_balloon, saudi_iran_mediation, prigozhin_mutiny,
          india_chandrayaan3, us_svb_collapse)
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MC_DIR   = BASE_DIR.parent / "MediaCrawler"
MC_CFG   = MC_DIR / "config" / "base_config.py"
MC_VENV  = MC_DIR / ".venv" / "bin" / "python"
CRAWL_ROOT = MC_DIR / "data" / "crawled"
LOG_DIR  = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Event registry ─────────────────────────────────────────────────────────────
# Tuples: (event_id, keywords, start, end[, dy_keywords])
# dy_keywords: Douyin-specific keywords replacing blocked terms (政变/戒严/暴涨/抵制/进攻 etc.)
# If omitted, the default keywords are used for all platforms.
EVENTS = [
    ("wang_xing_rescue_001",
     "王星被救,缅北诈骗,王星缅甸,电信诈骗东南亚",
     "2025-01-01", "2025-01-20"),
    ("china_astock_rally_001",
     "A股暴涨,牛市2024,央行降准,股市行情2024",
     "2024-09-24", "2024-10-15",
     "A股,大盘涨了,股市行情,炒股2024"),
    ("nongfu_spring_boycott_001",
     "农夫山泉抵制,钟睒睒,娃哈哈宗庆后,矿泉水国货",
     "2024-02-01", "2024-04-30",
     "农夫山泉,钟睒睒,娃哈哈,宗庆后"),
    ("korea_martial_law_001",
     "韩国戒严,尹锡悦戒严,韩国国会,韩国政变",
     "2024-12-03", "2024-12-20",
     "尹锡悦,韩国总统,韩国局势,韩国国会"),
    ("hunan_flood_2024_001",
     "湖南洪灾,洞庭湖决口,湖南防汛,华容溃堤",
     "2024-06-15", "2024-07-10"),
    ("china_retirement_reform_001",
     "延迟退休,退休年龄改革,法定退休年龄,养老金政策",
     "2024-09-01", "2024-10-31",
     "延迟退休,退休改革,退休年龄新规,养老金"),
    ("xizang_earthquake_2025_001",
     "西藏地震,定日地震,西藏救援,地震捐款2025",
     "2025-01-07", "2025-01-20",
     "定日地震,西藏地震,地震救援,藏区地震"),
    ("manus_ai_launch_001",
     "Manus AI,Manus人工智能,Manus体验,Manus智能体",
     "2025-03-06", "2025-03-31",
     "Manus,Manus教程,AI智能体,Manus演示"),
    ("pan_zhanle_100m_001",
     "潘展乐,世界纪录,巴黎奥运会游泳,兴奋剂质疑",
     "2024-07-27", "2024-08-15",
     "潘展乐,游泳世界纪录,奥运游泳冠军,巴黎奥运游泳"),
    ("hua_chenyu_custody_001",
     "华晨宇张碧晨,华晨宇抚养权,华晨宇女儿",
     "2024-03-01", "2024-06-30",
     "华晨宇,张碧晨,华晨宇近况,华晨宇演唱会"),
    ("eu_china_ev_tariffs_001",
     "欧盟电动车关税,中欧贸易争端,比亚迪欧洲,新能源汽车出海",
     "2024-06-01", "2024-09-30",
     "比亚迪,新能源汽车出海,电动车欧洲,汽车出口"),
    ("ukraine_kursk_incursion_001",
     "乌克兰库尔斯克,乌军进攻俄领土,俄乌最新战况,乌克兰反攻",
     "2024-08-06", "2024-08-31",
     "俄乌战争,乌克兰,库尔斯克,俄乌局势"),
    ("guo_meimei_release_001",
     "郭美美,郭美美出狱,红十字会,网红出狱",
     "2024-04-01", "2024-06-30",
     "郭美美,郭美美近况,郭美美直播,郭美美最新"),
    # ── Phase 1 expansion (10 new events, 2026-05-23) — 5 domestic + 5 international ──
    # Domestic
    ("wuyifan_trial_001",
     "吴亦凡判刑,吴亦凡案宣判,吴亦凡13年,吴亦凡强奸罪",
     "2023-07-01", "2023-09-30",
     "吴亦凡,吴亦凡近况,明星案件,娱乐圈事件"),
    ("huawei_mate60_launch_001",
     "华为Mate60,麒麟9000s,华为遥遥领先,华为芯片突破",
     "2023-08-01", "2023-11-30",
     "华为Mate60,华为手机,麒麟芯片,华为新机"),
    ("shanghai_lockdown_2022_001",
     "上海封控,上海解封,上海疫情物资,上海隔离",
     "2022-04-01", "2022-06-30",
     "上海疫情,上海核酸,上海隔离,上海生活物资"),
    ("henan_bank_crisis_001",
     "河南村镇银行,取款难,郑州维权,银行存款冻结",
     "2022-06-01", "2022-10-31",
     "银行取款,理财暴雷,河南银行,储蓄安全"),
    ("li_jiaqi_huaxizi_001",
     "李佳琦翻车,花西子事件,李佳琦道歉,国货品牌",
     "2023-09-01", "2023-11-30",
     "李佳琦,花西子,直播带货,国货"),
    # International
    ("china_spy_balloon_001",
     "中国气球,侦察气球,美国击落气球,中美气球事件",
     "2023-02-01", "2023-04-30",
     "气球,低空飞行器,无人气球,中美关系"),
    ("saudi_iran_china_mediation_001",
     "中国斡旋沙特伊朗,沙伊复交,北京协议沙伊,中东外交",
     "2023-03-01", "2023-06-30",
     "沙特伊朗,中国外交,中东和平,沙伊关系"),
    ("prigozhin_mutiny_001",
     "普里戈津兵变,瓦格纳叛乱,俄罗斯政变,普里戈津死亡",
     "2023-06-01", "2023-09-30",
     "普里戈津,瓦格纳,俄罗斯局势,俄军事件"),
    ("india_chandrayaan3_001",
     "印度月球探测器,月船三号登月,印度航天,印度登月成功",
     "2023-07-01", "2023-09-30",
     "印度月球,印度航天,登月成功,太空探索"),
    ("us_svb_collapse_001",
     "硅谷银行暴雷,SVB倒闭,美国银行危机,硅谷银行挤兑",
     "2023-03-01", "2023-05-31",
     "银行倒闭,美国金融,硅谷银行,金融危机"),
]

PLATFORMS = ["wb", "dy", "zhihu", "bili"]
# Config PLATFORM value → (config string, output directory name)
PLATFORM_MC  = {"wb": "wb",    "dy": "dy",     "zhihu": "zhihu", "bili": "bili"}
PLATFORM_DIR = {"wb": "weibo", "dy": "douyin", "zhihu": "zhihu", "bili": "bili"}

TIMEOUT_SEC = 45 * 60  # 45 minutes per crawl

# ── Config patcher ─────────────────────────────────────────────────────────────

def patch_cfg(platform: str, keywords: str, save_path: str,
              start: str = "", end: str = "") -> None:
    text = MC_CFG.read_text(encoding="utf-8")

    def sub(pat: str, repl: str) -> None:
        nonlocal text
        text, n = re.subn(pat, repl, text, count=1, flags=re.MULTILINE)
        if n == 0:
            print(f"  WARN: pattern not found: {pat!r}")

    sub(r'^(PLATFORM\s*=\s*).*$',      f'\\g<1>"{PLATFORM_MC[platform]}"')
    sub(r'^(KEYWORDS\s*=\s*).*$',      f'\\g<1>"{keywords}"')
    sub(r'^(SAVE_DATA_PATH\s*=\s*).*$', f'\\g<1>"{save_path}"')

    # Bilibili: also patch START_DAY / END_DAY if present
    if platform == "bili" and start and end:
        for tag, val in [("START_DAY", start), ("END_DAY", end)]:
            if tag in text:
                text, n = re.subn(
                    rf'^({tag}\s*=\s*).*$', f'\\g<1>"{val}"',
                    text, count=1, flags=re.MULTILINE)

    MC_CFG.write_text(text, encoding="utf-8")


# ── Crawl runner ───────────────────────────────────────────────────────────────

def run_crawl(event_id: str, platform: str, log_fh) -> str:
    """Returns 'ok', 'timeout', or 'error:<msg>'."""
    save_path = str(CRAWL_ROOT / event_id)
    log_fh.write(f"\n{'='*60}\n")
    log_fh.write(f"START  {datetime.now():%H:%M:%S}  {event_id} / {platform}\n")
    log_fh.flush()

    # Remove ALL stale SingletonLock files (any platform) before each run
    for lock in (MC_DIR / "browser_data").glob("*/SingletonLock"):
        lock.unlink()
        log_fh.write(f"  Removed stale lock: {lock}\n")

    try:
        result = subprocess.run(
            [str(MC_VENV), "main.py"],
            cwd=str(MC_DIR),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
        )
        log_fh.write(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
        if result.stderr:
            log_fh.write("\n--- STDERR ---\n")
            log_fh.write(result.stderr[-3000:])
        log_fh.flush()

        if result.returncode != 0:
            return f"error:rc={result.returncode}"

        # Verify output files exist (output goes to {event}/weibo/jsonl/, douyin/jsonl/, etc.)
        out_dir = Path(save_path) / PLATFORM_DIR[platform] / "jsonl"
        files = list(out_dir.glob("*.jsonl")) if out_dir.exists() else []
        if not files:
            return "error:no_output_files"

        total_lines = sum(sum(1 for _ in open(f, encoding="utf-8", errors="ignore"))
                          for f in files)
        log_fh.write(f"\n  Output: {len(files)} file(s), {total_lines} lines\n")
        return "ok"

    except subprocess.TimeoutExpired:
        log_fh.write(f"\n  TIMEOUT after {TIMEOUT_SEC}s\n")
        return "timeout"
    except Exception as e:
        log_fh.write(f"\n  EXCEPTION: {e}\n")
        return f"error:{e}"


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-event", type=int, default=0,
                        help="Resume from event index (0-based)")
    parser.add_argument("--platform", choices=PLATFORMS, default=None,
                        help="Run only this platform (default: all)")
    args = parser.parse_args()

    platforms = [args.platform] if args.platform else PLATFORMS
    events = EVENTS[args.start_event:]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"auto_crawl_{timestamp}.log"

    results: dict[tuple, str] = {}

    print(f"Logging to {log_path}")
    print(f"Events: {len(events)}, Platforms: {platforms}")
    print(f"Total crawls: {len(events) * len(platforms)}")

    with open(log_path, "w", encoding="utf-8") as log_fh:
        log_fh.write(f"auto_crawl_all.py  started {datetime.now()}\n")
        log_fh.write(f"Events: {[e[0] for e in events]}\n")
        log_fh.write(f"Platforms: {platforms}\n\n")

        try:
            for ev_idx, ev_tuple in enumerate(events):
                event_id, keywords, start, end = ev_tuple[:4]
                dy_keywords = ev_tuple[4] if len(ev_tuple) >= 5 else keywords
                for platform in platforms:
                    key = (event_id, platform)
                    t0 = time.time()
                    print(f"[{ev_idx+args.start_event+1}/{len(EVENTS)}] {event_id}/{platform} ...",
                          end=" ", flush=True)

                    kw = dy_keywords if platform == "dy" else keywords
                    patch_cfg(platform, kw,
                              str(CRAWL_ROOT / event_id), start, end)

                    status = run_crawl(event_id, platform, log_fh)
                    elapsed = time.time() - t0

                    results[key] = status
                    icon = "✓" if status == "ok" else "✗"
                    print(f"{icon} {status}  ({elapsed:.0f}s)")
                    log_fh.write(f"  STATUS: {status}  elapsed={elapsed:.0f}s\n")
                    log_fh.flush()

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            log_fh.write("\nInterrupted by user.\n")

    # ── Summary ────────────────────────────────────────────────────────────────
    ok      = [(k, v) for k, v in results.items() if v == "ok"]
    failed  = [(k, v) for k, v in results.items() if v != "ok"]

    print(f"\n{'='*60}")
    print(f"DONE  {len(ok)}/{len(results)} succeeded")
    if failed:
        print("FAILED:")
        for (eid, plat), status in failed:
            print(f"  {eid}/{plat}: {status}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
