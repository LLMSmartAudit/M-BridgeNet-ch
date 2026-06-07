"""Phase 0b/0c — Event registry for test-set expansion (15 → ~32 events).

Each entry defines the crawl parameters and post-processing config for a
new held-out test event. Used by:
  scripts/phase0b_setup_crawl.py   — patches MediaCrawler config per platform
  scripts/phase0b_process.py       — convert → validate → annotate → prepare → copy

Priority groups (A=highest, F=lowest):
  A — Social justice / cyberbullying (strong Weibo→Zhihu arc)
  B — Livestream / e-commerce controversies
  C — Consumer culture / viral trends
  D — Spectacle / disaster / cultural events
  E — Youth employment / social anxiety
  F — International / political
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class EventConfig:
    event_id: str              # unique slug, e.g. "dongfang_selection_001"
    name_zh: str               # 中文描述 (for logging / crawl guide)
    keywords: str              # comma-separated MediaCrawler KEYWORDS
    start: str                 # YYYY-MM-DD  (inclusive)
    end: str                   # YYYY-MM-DD  (inclusive)
    platforms: list[str]       # subset of ["weibo","bili","zhihu","douyin"]
    tau: float = 0.82          # cosine threshold for llm_annotate.py --tau
    max_k: int = 1000          # llm_annotate.py --max-k
    group: str = "A"           # priority group
    notes: str = ""            # human notes


# ---------------------------------------------------------------------------
# The 15 priority events for Phase 0b
# ---------------------------------------------------------------------------

PHASE0B_EVENTS: list[EventConfig] = [

    # ── Group A — Social justice / cyberbullying ───────────────────────────
    EventConfig(
        event_id  = "zheng_linghua_001",
        name_zh   = "郑灵华粉发女孩网络暴力",
        keywords  = "郑灵华,粉发女孩,网络暴力",
        start     = "2022-07-01",
        end       = "2023-02-28",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.80,
        group     = "A",
        notes     = "Weibo-primary cyberbullying → Zhihu policy analysis; victim died Jan 2023",
    ),
    EventConfig(
        event_id  = "chengdu_49_school_001",
        name_zh   = "成都49中学生坠楼事件",
        keywords  = "成都49中,坠楼事件,学生坠亡",
        start     = "2021-05-09",
        end       = "2021-08-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.80,
        group     = "A",
        notes     = "Fact-contested incident; Weibo outrage → Zhihu forensic analysis",
    ),
    EventConfig(
        event_id  = "yuhuaying_trafficking_001",
        name_zh   = "余华英拐卖儿童死刑案",
        keywords  = "余华英,拐卖儿童,死刑判决",
        start     = "2023-11-01",
        end       = "2024-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.80,
        group     = "A",
        notes     = "Legal case; multi-wave; strong Weibo→Bili documentary arc",
    ),
    EventConfig(
        event_id  = "pinduoduo_overwork_001",
        name_zh   = "拼多多员工过劳猝死",
        keywords  = "拼多多员工猝死,拼多多加班,过劳死",
        start     = "2023-01-01",
        end       = "2024-01-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "A",
        notes     = "Labor rights; Weibo outrage → Zhihu legal analysis",
    ),

    # ── Group B — Livestream / e-commerce controversies ───────────────────
    EventConfig(
        event_id  = "dongfang_selection_001",
        name_zh   = "新东方东方甄选直播出圈",
        keywords  = "东方甄选,新东方直播,董宇辉带货",
        start     = "2022-06-01",
        end       = "2022-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "B",
        notes     = "Douyin-origin viral; education nostalgia → capitalism critique arc",
    ),
    EventConfig(
        event_id  = "xiaoyangshu_scandal_001",
        name_zh   = "小杨哥三只羊带货翻车",
        keywords  = "小杨哥,三只羊月饼,带货翻车",
        start     = "2024-08-01",
        end       = "2024-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "B",
        notes     = "Weibo-origin exposure → Zhihu consumer rights → Bili investigative",
    ),
    EventConfig(
        event_id  = "luo_yonghao_comeback_001",
        name_zh   = "罗永浩抖音直播还债",
        keywords  = "罗永浩直播,罗永浩还债,老罗带货",
        start     = "2020-04-01",
        end       = "2021-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.80,
        group     = "B",
        notes     = "Personal redemption; Weibo emotional vs Zhihu entrepreneurship vs Douyin live",
    ),
    EventConfig(
        event_id  = "xibei_precooked_001",
        name_zh   = "西贝莜面村预制菜半成品争议",
        keywords  = "西贝莜面村,预制菜,半成品",
        start     = "2021-02-01",
        end       = "2021-09-30",
        platforms = ["weibo", "bili", "zhihu"],
        tau       = 0.82,
        group     = "B",
        notes     = "Specific incident (CEO interview revelation); skip douyin — platform too early",
    ),

    # ── Group C — Consumer culture / viral trends ──────────────────────────
    EventConfig(
        event_id  = "xiaomi_su7_launch_001",
        name_zh   = "小米SU7汽车发布雷军出圈",
        keywords  = "小米汽车,小米SU7,雷军造车",
        start     = "2024-03-01",
        end       = "2024-08-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "C",
        notes     = "Single-date event (2024-03-28 launch); Weibo frenzy → Zhihu specs → Bili test drive",
    ),
    EventConfig(
        event_id  = "kemu3_dance_001",
        name_zh   = "科目三舞蹈爆火出圈",
        keywords  = "科目三舞,广西神曲,科目三舞蹈",
        start     = "2023-09-01",
        end       = "2024-01-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "C",
        notes     = "Douyin-first viral; 4-platform spread; entertainment/meme content",
    ),

    # ── Group D — Spectacle / disaster / cultural events ──────────────────
    EventConfig(
        event_id  = "beijing_flood_2023_001",
        name_zh   = "京津冀特大暴雨洪涝灾害",
        keywords  = "北京暴雨,涿州洪灾,河北洪水",
        start     = "2023-07-29",
        end       = "2023-10-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.80,
        group     = "D",
        notes     = "Crisis event; timeline-driven bridging; rescue narratives cross-platform",
    ),
    EventConfig(
        event_id  = "wuhan_univ_cherry_001",
        name_zh   = "武汉大学樱花季人流管控争议",
        keywords  = "武汉大学樱花,武大樱花,赏樱人潮",
        start     = "2023-03-01",
        end       = "2023-05-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "D",
        notes     = "Seasonal event (March only); Bili/Douyin travel video → Weibo reaction",
    ),
    EventConfig(
        event_id  = "drone_show_event_001",
        name_zh   = "无人机编队表演失控或爆火事件",
        keywords  = "无人机表演,无人机编队,无人机秀失控",
        start     = "2023-01-01",
        end       = "2024-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "D",
        notes     = "Visual-first (Bili/Douyin) → Weibo reaction; validate data volume first",
    ),

    # ── Group E — Youth employment / social anxiety ────────────────────────
    EventConfig(
        event_id  = "hetongshu_plagiarism_001",
        name_zh   = "何同学苹果广告抄袭争议",
        keywords  = "何同学,苹果广告抄袭,学术不端",
        start     = "2023-10-01",
        end       = "2024-03-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "E",
        notes     = "Tech-influencer scandal; Bili-origin verification arc",
    ),

    # ── Group F — International / political ────────────────────────────────
    EventConfig(
        event_id  = "xinjiang_cotton_001",
        name_zh   = "新疆棉花H&M抵制运动",
        keywords  = "新疆棉花,HM抵制,棉花声明",
        start     = "2021-03-01",
        end       = "2021-07-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.80,
        group     = "F",
        notes     = "Boycott event; strong Weibo patriotism vs Zhihu nuanced analysis",
    ),
]

# ---------------------------------------------------------------------------
# Phase 0c — 8 new events: military/national security + 2025 hot topics
# ---------------------------------------------------------------------------
#
# Coverage gaps addressed:
#   G — Military / national security (0 events in test_real → 2)
#   H — 2025 viral topics (deepseek/zhang_xuefeng only → +4 more)
#   H — Douyin-origin fix, public health, Weibo-dominant events
#
# Note on military crawl (Group G): Weibo likely has thin/censored content
# for military corruption topics; Zhihu analysis + Bili commentary are the
# main bridge sources. If Weibo returns <50 posts after crawl, drop Weibo
# from platforms and re-run with zhihu+bili only.

PHASE0C_EVENTS: list[EventConfig] = [

    # ── Group G — Military / national security ─────────────────────────────
    EventConfig(
        event_id  = "pla_rocket_force_purge_001",
        name_zh   = "火箭军腐败案系列审判",
        keywords  = "火箭军腐败,李尚福,军队反腐",
        start     = "2023-07-01",
        end       = "2025-03-31",
        platforms = ["weibo", "bili", "zhihu"],
        tau       = 0.78,
        max_k     = 800,
        group     = "G",
        notes     = (
            "PLA Rocket Force purge (Li Yuchao, Li Shangfu dismissed 2023; trials 2024-2025). "
            "Weibo breaking news → Zhihu political analysis → Bili commentary. "
            "Skip Douyin — military corruption topic blocked. "
            "If Weibo <50 posts, drop Weibo and run bili+zhihu only."
        ),
    ),
    EventConfig(
        event_id  = "taiwan_strait_drill_2025_001",
        name_zh   = "联合利剑-2024B台海军演",
        keywords  = "联合利剑2024,东部战区军演,台海演习",
        start     = "2024-10-01",
        end       = "2025-03-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.80,
        group     = "G",
        notes     = (
            "联合利剑-2024B exercise (Oct 14-17 2024) + subsequent coverage into 2025. "
            "Multi-platform: Weibo live reactions, Bili military analysis videos, "
            "Zhihu geopolitical discussion, Douyin short clips of drill footage. "
            "Strong Weibo×Zhihu bridges (official statements → analytical responses)."
        ),
    ),

    # ── Group H — 2025 hot topics ──────────────────────────────────────────
    EventConfig(
        event_id  = "nezha2_movie_001",
        name_zh   = "哪吒之魔童闹海票房破纪录",
        keywords  = "哪吒2,哪吒之魔童闹海,国漫崛起",
        start     = "2025-01-20",
        end       = "2025-04-30",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "H",
        notes     = (
            "Released 2025-01-29; broke records as highest-grossing Chinese animated film. "
            "Bili (in-depth reviews + production analysis) → Weibo (fan reactions, box office) "
            "→ Zhihu (industry/cultural analysis) bridges; national pride narrative diverges "
            "across platforms. 4-platform event. High engagement, very crawlable."
        ),
    ),
    EventConfig(
        event_id  = "xiaomi_su7_accident_001",
        name_zh   = "小米SU7连环事故安全争议",
        keywords  = "小米SU7事故,小米汽车碰撞,小米汽车安全",
        start     = "2025-03-01",
        end       = "2025-05-15",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "H",
        notes     = (
            "Multiple SU7 accidents in March 2025 triggered safety controversy. "
            "Weibo-origin (victims posting first) → Zhihu technical analysis → "
            "Bili investigation videos → Douyin reaction clips. "
            "Rare Weibo-origin event; fixes Douyin underrepresentation. "
            "High bridge density expected (single product, specific safety incidents)."
        ),
    ),
    EventConfig(
        event_id  = "trump_tariff_2025_001",
        name_zh   = "特朗普对华关税战2025",
        keywords  = "特朗普关税,中美关税,对等关税",
        start     = "2025-01-20",
        end       = "2025-05-15",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "H",
        notes     = (
            "Trump 2.0 tariff escalation: 10% (Feb), 54% (Apr), 145% (May) on China imports. "
            "Distinct from us_china_reckoning_001 (retrospective) — this is live policy reaction. "
            "Weibo (emotional/patriotic reactions) × Zhihu (economic analysis) × "
            "Bili (commentary) bridges. Very high volume, good bridge density expected."
        ),
    ),
    EventConfig(
        event_id  = "gaokao_reform_2025_001",
        name_zh   = "2025高考改革新政策争议",
        keywords  = "高考改革,新高考政策,2025高考",
        start     = "2024-12-01",
        end       = "2025-06-10",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.80,
        group     = "H",
        notes     = (
            "2025 新高考 policy changes (subject selection reform, scoring adjustments). "
            "Strong Weibo (students/parents emotional reactions) × Zhihu (expert/policy "
            "analysis) bridge pattern. Bili: teachers explaining reforms. "
            "tau=0.80 (slightly lower) — education posts are semantically diverse."
        ),
    ),
    EventConfig(
        event_id  = "ai_deepfake_scam_001",
        name_zh   = "AI换脸诈骗曝光与反诈宣传",
        keywords  = "AI换脸诈骗,深度伪造诈骗,AI诈骗防范",
        start     = "2024-01-01",
        end       = "2025-05-15",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "H",
        notes     = (
            "Douyin-origin event: viral exposé videos of AI deepfake scams spreading "
            "to Weibo (victim testimonials) → Zhihu (technical analysis) → Bili (tutorials). "
            "Fixes Douyin-as-origin underrepresentation in test_real. "
            "Multi-wave: several high-profile cases in 2024-2025."
        ),
    ),
    EventConfig(
        event_id  = "covid_jn1_wave_001",
        name_zh   = "2025年初新冠JN.1感染潮",
        keywords  = "新冠感染2025,JN.1,新冠变异,新冠复阳",
        start     = "2024-12-01",
        end       = "2025-03-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.80,
        group     = "H",
        notes     = (
            "JN.1 sub-variant wave peaked China Jan-Mar 2025. "
            "Weibo (patient reports, fear/symptom sharing) → Zhihu (medical analysis, "
            "vaccine discussion) → Bili (science explainers) bridges. "
            "Fills public health gap in test_real. tau=0.80 (patient posts vary in language)."
        ),
    ),

    # ── Stage-2 expansion (37 new events, 2026-06; WIDE window → burst-onset bounds later) ──
    EventConfig(
        event_id  = "fujian_carrier_trial_001",
        name_zh   = "福建舰",
        keywords  = "福建舰,003航母,海试,电磁弹射",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2024-05-01..2024-06-30",
    ),
    EventConfig(
        event_id  = "sixthgen_fighter_2024_001",
        name_zh   = "六代机",
        keywords  = "六代机,歼36,成飞,沈飞,银杏叶",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2024-12-01..2025-01-15",
    ),
    EventConfig(
        event_id  = "zhuhai_airshow_2024_001",
        name_zh   = "珠海航展",
        keywords  = "珠海航展,中国航展,歼35,运20",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2024-11-01..2024-11-30",
    ),
    EventConfig(
        event_id  = "joint_sword_2024b_001",
        name_zh   = "联合利剑2024B",
        keywords  = "联合利剑2024B,环台军演,解放军,台海",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2024-10-01..2024-10-31",
    ),
    EventConfig(
        event_id  = "taiwan_election_2024_001",
        name_zh   = "台湾大选",
        keywords  = "台湾大选,赖清德,民进党,2024台湾",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2024-01-01..2024-02-15",
    ),
    EventConfig(
        event_id  = "scs_philippines_2024_001",
        name_zh   = "仁爱礁",
        keywords  = "仁爱礁,中菲对峙,南海,海警船",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2024-06-01..2024-07-31",
    ),
    EventConfig(
        event_id  = "lebanon_pager_2024_001",
        name_zh   = "黎巴嫩",
        keywords  = "黎巴嫩,寻呼机爆炸,真主党,BP机",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2024-09-01..2024-09-30",
    ),
    EventConfig(
        event_id  = "syria_assad_fall_001",
        name_zh   = "叙利亚",
        keywords  = "叙利亚,阿萨德,政权垮台,大马士革",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2024-12-01..2024-12-31",
    ),
    EventConfig(
        event_id  = "nk_troops_russia_001",
        name_zh   = "朝鲜出兵",
        keywords  = "朝鲜出兵,俄罗斯,库尔斯克,朝鲜士兵",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2024-10-15..2024-12-15",
    ),
    EventConfig(
        event_id  = "houthi_redsea_001",
        name_zh   = "胡塞武装",
        keywords  = "胡塞武装,红海,商船袭击,航运",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2024-01-01..2024-02-29",
    ),
    EventConfig(
        event_id  = "dji_us_ban_001",
        name_zh   = "大疆",
        keywords  = "大疆,美国禁令,无人机,制裁",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2024-09-01..2024-10-31",
    ),
    EventConfig(
        event_id  = "nvidia_h20_china_001",
        name_zh   = "英伟达",
        keywords  = "英伟达,H20,芯片,出口管制",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2025-04-01..2025-05-31",
    ),
    EventConfig(
        event_id  = "taiwan_lai_twostate_001",
        name_zh   = "赖清德",
        keywords  = "赖清德,双十演讲,两国论,国庆",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2024-10-01..2024-10-31",
    ),
    EventConfig(
        event_id  = "philippines_scs_collision_001",
        name_zh   = "中菲",
        keywords  = "中菲,船只碰撞,南海,黄岩岛",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (military); orig window 2024-08-01..2024-09-30",
    ),
    EventConfig(
        event_id  = "unitree_gala_2025_001",
        name_zh   = "宇树机器人",
        keywords  = "宇树机器人,春晚,扭秧歌,人形机器人",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (tech_ai); orig window 2025-01-20..2025-02-28",
    ),
    EventConfig(
        event_id  = "doubao_launch_001",
        name_zh   = "豆包",
        keywords  = "豆包,字节跳动,AI大模型,豆包APP",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (tech_ai); orig window 2024-05-01..2024-06-30",
    ),
    EventConfig(
        event_id  = "openai_o1_001",
        name_zh   = "OpenAI",
        keywords  = "OpenAI,o1,草莓模型,推理模型",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (tech_ai); orig window 2024-09-01..2024-10-15",
    ),
    EventConfig(
        event_id  = "robotaxi_apollo_2024_001",
        name_zh   = "萝卜快跑",
        keywords  = "萝卜快跑,无人出租车,百度,自动驾驶",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (tech_ai); orig window 2024-07-01..2024-08-15",
    ),
    EventConfig(
        event_id  = "kimi_longcontext_001",
        name_zh   = "Kimi",
        keywords  = "Kimi,月之暗面,长文本,大模型",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (tech_ai); orig window 2024-03-01..2024-04-15",
    ),
    EventConfig(
        event_id  = "nvidia_market_cap_001",
        name_zh   = "英伟达",
        keywords  = "英伟达,市值,黄仁勋,AI芯片",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (tech_ai); orig window 2024-06-01..2024-07-15",
    ),
    EventConfig(
        event_id  = "gansu_quake_2023_001",
        name_zh   = "甘肃地震",
        keywords  = "甘肃地震,积石山,地震救援",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (disaster); orig window 2023-12-15..2024-01-15",
    ),
    EventConfig(
        event_id  = "zhuozhou_flood_2023_001",
        name_zh   = "涿州洪水",
        keywords  = "涿州洪水,河北洪水,泄洪",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (disaster); orig window 2023-08-01..2023-08-31",
    ),
    EventConfig(
        event_id  = "qiqihar_collapse_2023_001",
        name_zh   = "齐齐哈尔",
        keywords  = "齐齐哈尔,体育馆坍塌,中学",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (disaster); orig window 2023-07-20..2023-08-10",
    ),
    EventConfig(
        event_id  = "meida_highway_2024_001",
        name_zh   = "梅大高速",
        keywords  = "梅大高速,塌方,广东,高速路面",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (disaster); orig window 2024-05-01..2024-05-31",
    ),
    EventConfig(
        event_id  = "zhuhai_car_attack_2024_001",
        name_zh   = "珠海",
        keywords  = "珠海,驾车撞人,体育中心,珠海航空城",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (disaster); orig window 2024-11-01..2024-11-30",
    ),
    EventConfig(
        event_id  = "yinchuan_explosion_2023_001",
        name_zh   = "银川",
        keywords  = "银川,烧烤店爆炸,燃气爆炸",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (disaster); orig window 2023-06-20..2023-07-10",
    ),
    EventConfig(
        event_id  = "jiangping_math_2024_001",
        name_zh   = "姜萍",
        keywords  = "姜萍,阿里数学竞赛,中专生,涟水",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (social); orig window 2024-06-01..2024-07-31",
    ),
    EventConfig(
        event_id  = "qinlang_hoax_2024_001",
        name_zh   = "秦朗",
        keywords  = "秦朗,巴黎丢作业,寒假作业,摆拍",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (social); orig window 2024-02-01..2024-04-30",
    ),
    EventConfig(
        event_id  = "shanghai_halloween_2023_001",
        name_zh   = "上海万圣节",
        keywords  = "上海万圣节,cosplay,巨鹿路",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (social); orig window 2023-10-25..2023-11-15",
    ),
    EventConfig(
        event_id  = "wuhan_mother_2023_001",
        name_zh   = "武汉",
        keywords  = "武汉,校内碾压,母亲坠楼,小学生",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (social); orig window 2023-06-01..2023-06-30",
    ),
    EventConfig(
        event_id  = "zhongzhi_collapse_2023_001",
        name_zh   = "中植系",
        keywords  = "中植系,暴雷,理财,财富",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (social); orig window 2023-11-01..2023-12-31",
    ),
    EventConfig(
        event_id  = "evergrande_liquidation_001",
        name_zh   = "恒大",
        keywords  = "恒大,清盘,许家印,退市",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (social); orig window 2024-01-15..2024-02-15",
    ),
    EventConfig(
        event_id  = "geng_academic_fraud_001",
        name_zh   = "耿同学",
        keywords  = "耿同学,耿同学讲故事,学术打假,论文造假,同济王平",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (social); orig window 2026-04-01..2026-05-31",
    ),
    EventConfig(
        event_id  = "quan_hongchan_2024_001",
        name_zh   = "全红婵",
        keywords  = "全红婵,跳水,巴黎奥运,水花消失术",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (sports_ent); orig window 2024-08-01..2024-08-31",
    ),
    EventConfig(
        event_id  = "zheng_qinwen_olympics_001",
        name_zh   = "郑钦文",
        keywords  = "郑钦文,网球,奥运金牌,巴黎",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (sports_ent); orig window 2024-08-01..2024-08-31",
    ),
    EventConfig(
        event_id  = "fanzhendong_wtt_2024_001",
        name_zh   = "樊振东",
        keywords  = "樊振东,WTT,退出,世界排名",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (sports_ent); orig window 2024-12-01..2024-12-31",
    ),
    EventConfig(
        event_id  = "heigui_dlc_001",
        name_zh   = "黑神话",
        keywords  = "黑神话,DLC,游戏科学",
        start     = "2020-01-01",
        end       = "2026-12-31",
        platforms = ["weibo", "bili", "zhihu", "douyin"],
        tau       = 0.82,
        group     = "S2",
        notes     = "stage2 (sports_ent); orig window 2025-01-01..2025-12-31",
    ),
]

# Combined registry (Phase 0b + 0c)
PHASE0B_EVENTS = PHASE0B_EVENTS + PHASE0C_EVENTS  # type: ignore[assignment]

# Quick lookup by event_id
EVENT_MAP: dict[str, EventConfig] = {e.event_id: e for e in PHASE0B_EVENTS}


if __name__ == "__main__":
    print(f"Phase 0b event registry: {len(PHASE0B_EVENTS)} events")
    for ev in PHASE0B_EVENTS:
        print(f"  [{ev.group}] {ev.event_id:40s} {ev.start} → {ev.end}  "
              f"platforms={ev.platforms}  tau={ev.tau}")
