"""Generate synthetic CPHot-format event data for M-BridgeNet development.

Creates realistic military/social event posts across 4 Chinese platforms
(weibo / zhihu / bilibili / douyin) with lifecycle volume curves, seeded
bridge pairs, and pre-computed scored_pairs for MLP training.

Usage:
    python scripts/generate_synthetic_data.py \\
        --events 20 --output data/cphot/processed --seed 42

Output layout:
    <output>/train/event_000.json … event_015.json
    <output>/test/event_016.json  … event_019.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# ── Constants ─────────────────────────────────────────────────────────────────

PLATFORMS = ["weibo", "zhihu", "bilibili", "douyin"]
TOTAL_HOURS = 72
PHASE_WINDOWS = {"emergence": 6, "diffusion": 24, "peak": 48, "decline": 72}

# ── Topic bank (10 military / social event templates) ────────────────────────

TOPICS: List[Dict] = [
    {
        "id": "topic_01",
        "name": "南海军事演习",
        "details": [
            "多艘驱逐舰和护卫舰参与，",
            "歼-20隐身战机首次公开亮相演习，",
            "演习涵盖实弹射击和联合作战科目，",
            "海陆空三军协同参演，",
            "演习区域覆盖南海重要战略水域，",
        ],
        "platforms": {
            "weibo": [
                "【突发】解放军南海军事演习正式开始！{detail}多型战机低空掠过，场面震撼！#南海演习# #解放军#",
                "刚刚！南海军演消息传来，{detail}网友：霸气！转发支持！ #军事 #南海",
                "南海演习实况：{detail}现场画面曝光，战机编队令人振奋✈️🌊 #解放军 #南海演习",
            ],
            "zhihu": [
                "如何看待此次解放军南海军事演习？{detail}从战略层面分析，此次演习释放了哪些信号？",
                "解放军南海演习深度解析：{detail}这次演习的规模和意义远超以往，以下是我的分析……",
                "军事专家视角：南海演习{detail}背后的战略逻辑是什么？",
            ],
            "bilibili": [
                "【军事纪录】南海演习全程记录！{detail}高清画面带你感受解放军实力【强烈推荐】",
                "UP主亲历南海演习区域！{detail}战机轰鸣声震天，这就是中国军人的底气！",
                "【独家】南海军演幕后故事：{detail}那些镜头背后的英雄们",
            ],
            "douyin": [
                "南海演习来了！{detail}#解放军 #南海 #军事 #强国",
                "震撼！{detail}这就是中国力量💪 #南海演习 #军事 #爱国",
                "{detail}南海演习实况，转发让更多人看到！#军事新闻",
            ],
        },
    },
    {
        "id": "topic_02",
        "name": "边境巡逻英雄事迹",
        "details": [
            "在海拔5000米以上高原坚守的",
            "连续15年未回家过年的老兵",
            "用脚步丈量每一寸边境线的",
            "在零下40度严寒中坚持巡逻的",
            "以雪山为伴、以风雪为友的",
        ],
        "platforms": {
            "weibo": [
                "【致敬】边防战士{detail}零下40度坚守边境，这就是最可爱的人！#戍边英雄# #边防战士#",
                "泪目！{detail}边防战士的故事让无数网友动容，转发致敬英雄！#边防 #军人",
                "边境线上的守护者：{detail}他们用青春守护万家灯火 #戍边 #边防战士",
            ],
            "zhihu": [
                "边防战士{detail}的事迹令人动容，我们应该如何理解这种奉献精神？",
                "从{detail}边防巡逻事迹看军人职业精神：牺牲与荣誉的深层含义",
                "边防战士{detail}长期驻守高原的生理和心理挑战，军方如何保障健康？",
            ],
            "bilibili": [
                "【感人】边防战士{detail}高原巡逻纪实，看完泪流满面【致敬英雄】",
                "跟随边防战士{detail}体验一次高原巡逻，零下40度的坚守",
                "【军旅vlog】边防战士{detail}的一天，平凡中的伟大",
            ],
            "douyin": [
                "致敬！边防战士{detail}#戍边英雄 #边防 #军人 #感动",
                "{detail}边防巡逻，这才是真正的英雄💪 #边防战士 #致敬",
                "泪目了，{detail}#边防 #军人精神 #感动中国",
            ],
        },
    },
    {
        "id": "topic_03",
        "name": "新型战机首飞",
        "details": [
            "采用先进隐身技术的",
            "具备超音速巡航能力的",
            "搭载国产新型发动机的",
            "集成人工智能辅助系统的",
            "突破多项关键技术的",
        ],
        "platforms": {
            "weibo": [
                "【重磅】国产新型战机{detail}首飞成功！中国航空工业再创辉煌！#国产战机# #航空工业#",
                "历史性时刻！{detail}新型战机腾空而起，为中国骄傲！#战机首飞 #国产装备",
                "刚刚传来消息：{detail}国产新型战机首飞成功，网友沸腾了！#战机 #国防科技",
            ],
            "zhihu": [
                "国产新型战机{detail}首飞成功意味着什么？对我国空军战略能力有何影响？",
                "深度分析：{detail}新型战机的技术特点及其在现代空战中的定位",
                "从{detail}新型战机首飞看中国航空工业的技术突破之路",
            ],
            "bilibili": [
                "【重磅】国产新型战机{detail}首飞全程记录！航空工业再创奇迹【强烈推荐】",
                "历史时刻！{detail}战机首飞现场画面，感受中国航空力量的崛起",
                "【军事科普】新型战机{detail}技术详解，为什么说这是重大突破？",
            ],
            "douyin": [
                "国产战机{detail}首飞成功！#国产战机 #航空工业 #中国制造 #强国",
                "震撼！{detail}这就是中国力量！#战机 #国防 #航空",
                "{detail}战机首飞，历史见证！#国产 #军事 #科技强国",
            ],
        },
    },
    {
        "id": "topic_04",
        "name": "军民融合重大项目",
        "details": [
            "涉及航天推进技术民用化的",
            "将军用通信技术转化为5G应用的",
            "把先进材料技术引入民用制造的",
            "推动无人机技术广泛应用的",
            "实现军用雷达技术民用转化的",
        ],
        "platforms": {
            "weibo": [
                "【政策】军民融合{detail}重大项目正式启动，国防科技惠及民生！#军民融合# #科技兴国#",
                "重大消息！{detail}军民融合项目落地，多项军工技术将走进日常生活 #国防科技",
                "军民融合新进展：{detail}这些曾经的军事技术即将走入寻常百姓家 #科技 #军民融合",
            ],
            "zhihu": [
                "此次军民融合{detail}项目对国内相关产业链有何深远影响？",
                "军民融合{detail}：国防技术民用化的路径分析与前景展望",
                "如何评价{detail}军民融合项目在推动经济转型中的战略意义？",
            ],
            "bilibili": [
                "【科普】军民融合{detail}项目解析：哪些军事黑科技将走进我们生活？",
                "深度解读{detail}军民融合：军工技术如何改变普通人的生活",
                "【财经军事】{detail}军民融合大项目启动，这些行业将迎来重大机遇",
            ],
            "douyin": [
                "军民融合{detail}启动！#军民融合 #科技 #国防 #创新",
                "{detail}军工技术走进生活，厉害了我的国！#科技强国 #军民融合",
                "重磅！{detail}#军民融合 #国防科技 #科技兴国",
            ],
        },
    },
    {
        "id": "topic_05",
        "name": "国防预算新政策",
        "details": [
            "同比增长7.2%，",
            "创五年来最高增幅，",
            "重点支持新型武器装备研发，",
            "信息化作战能力建设获专项资金，",
            "海军和空军现代化建设获重点倾斜，",
        ],
        "platforms": {
            "weibo": [
                "【重磅】新年度国防预算{detail}正式公布！军事现代化建设再提速！#国防预算# #强军#",
                "国防预算{detail}增幅超预期，网友：该花的钱一分不能省！#国防 #军费",
                "刚刚：国防预算{detail}方案出炉，这些领域将获重点投入 #国防建设 #军事现代化",
            ],
            "zhihu": [
                "如何解读今年国防预算{detail}的增幅？这背后释放了哪些战略信号？",
                "国防预算{detail}重点投向分析：武器装备现代化为何成为优先方向？",
                "从国际比较视角看：{detail}我国国防预算规模是否合理？",
            ],
            "bilibili": [
                "【军事财经】国防预算{detail}深度解读：钱花在哪了？值不值？",
                "国防预算{detail}发布！这些武器装备项目将获大额资金支持",
                "【数据解析】{detail}国防预算背后的战略逻辑，普通人看懂这篇就够了",
            ],
            "douyin": [
                "国防预算{detail}出炉！#国防预算 #强军 #军事现代化",
                "{detail}国防建设加速！#国防 #军事 #强国梦",
                "重磅！国防预算{detail}#国防建设 #军费 #现代化",
            ],
        },
    },
    {
        "id": "topic_06",
        "name": "反恐联合演习",
        "details": [
            "在西部山地地形中展开，",
            "涵盖城市反恐和野外作战科目，",
            "动用直升机和无人机协同作战，",
            "模拟真实恐怖袭击场景，",
            "特种作战与信息战结合，",
        ],
        "platforms": {
            "weibo": [
                "【实况】多国联合反恐演习{detail}正式开始！特种部队展示硬核实力！#反恐演习# #特种部队#",
                "震撼！联合反恐演习{detail}现场视频曝光，特战队员行动干净利落 #反恐 #特种部队",
                "反恐演习{detail}激战正酣，这些画面让人热血沸腾！#联合反恐 #特战",
            ],
            "zhihu": [
                "此次多国联合反恐演习{detail}有何特殊意义？传递了哪些反恐合作信号？",
                "从专业角度解读{detail}联合反恐演习：战术创新与协同作战能力评估",
                "联合反恐演习{detail}背后：国际反恐合作机制的现状与前景",
            ],
            "bilibili": [
                "【军事】联合反恐演习{detail}全程记录！特种部队硬核表现【震撼】",
                "反恐演习{detail}现场：特战队员的每一个动作都是教科书级别的",
                "【专业解说】联合反恐演习{detail}战术动作详解，这才是真正的精英部队",
            ],
            "douyin": [
                "反恐演习{detail}来了！#反恐 #特种部队 #军事 #硬核",
                "震撼！{detail}联合反恐演习，这就是中国力量！#反恐 #特战",
                "{detail}特种部队反恐演习，太帅了！#特种兵 #反恐 #军事",
            ],
        },
    },
    {
        "id": "topic_07",
        "name": "海军新型舰艇服役",
        "details": [
            "配备先进防空反导系统的",
            "具备远洋作战能力的新型",
            "搭载舰载直升机的大型",
            "集成最新电子战系统的",
            "排水量超过万吨的新型",
        ],
        "platforms": {
            "weibo": [
                "【重磅】国产新型舰艇{detail}正式服役！中国海军实力再上新台阶！#中国海军# #国产舰艇#",
                "历史时刻！{detail}新型战舰入列海军，扬我国威！#海军 #国产装备",
                "中国海军新成员：{detail}这艘舰艇的服役有多重要？#海军现代化 #国产舰艇",
            ],
            "zhihu": [
                "国产新型舰艇{detail}正式服役的战略意义：对我国海洋权益维护有何影响？",
                "深度解析{detail}新型舰艇技术特点：与世界先进水平相比如何？",
                "从{detail}新型战舰入列看中国海军现代化建设的整体布局",
            ],
            "bilibili": [
                "【海军】国产新型舰艇{detail}服役全程记录！中国海军实力展示【震撼】",
                "新型舰艇{detail}服役！带你深入了解这艘战舰的强大之处",
                "【军事科普】{detail}新型战舰技术解密，为什么说这是划时代的装备？",
            ],
            "douyin": [
                "新型战舰{detail}服役！#中国海军 #国产舰艇 #海洋强国",
                "震撼！{detail}海军新成员，为中国喝彩！#海军 #国产 #强军",
                "{detail}战舰入列，厉害了！#海军 #军事 #国产装备",
            ],
        },
    },
    {
        "id": "topic_08",
        "name": "军事改革新举措",
        "details": [
            "聚焦联合作战指挥体系的",
            "推进军种协同作战能力的",
            "完善战区联合作战机制的",
            "提升信息化作战水平的",
            "优化军事力量结构的",
        ],
        "platforms": {
            "weibo": [
                "【重要】军事改革{detail}新政策正式出台！联合作战能力建设大幅提升！#军事改革# #强军#",
                "深化军事改革：{detail}这些重大变化将全面提升我军战斗力 #军改 #联合作战",
                "军改新举措{detail}发布，军队现代化建设进入新阶段 #军事改革 #国防",
            ],
            "zhihu": [
                "此次军事改革{detail}的核心逻辑是什么？对提升战斗力有何实质性作用？",
                "深度解读：{detail}军事改革背后的战略考量与现实挑战",
                "从{detail}军事改革看中国军队现代化转型的关键突破点",
            ],
            "bilibili": [
                "【军事】深化军事改革{detail}解读！这些变化将如何改变中国军队？",
                "军事改革{detail}大变化：普通士兵会受到哪些影响？",
                "【深度】{detail}军事改革背后：联合作战为何成为核心关键词？",
            ],
            "douyin": [
                "军事改革{detail}来了！#军事改革 #联合作战 #强军",
                "{detail}军改新举措，中国军队越来越强！#军改 #军事 #强军",
                "重磅！{detail}#军事改革 #现代化 #联合作战",
            ],
        },
    },
    {
        "id": "topic_09",
        "name": "航天军事技术突破",
        "details": [
            "新型高分辨率侦察卫星",
            "具备快速响应能力的军用卫星",
            "全天候全天时侦察卫星",
            "新型导航增强卫星",
            "军民两用遥感卫星",
        ],
        "platforms": {
            "weibo": [
                "【突破】军事航天新进展：{detail}成功发射！太空军事力量再增强！#航天 #军事卫星#",
                "重大消息！{detail}军事航天技术突破，中国太空实力持续提升 #航天军事 #科技",
                "航天军事新成就：{detail}这次发射的意义为何如此重大？#军事航天 #太空",
            ],
            "zhihu": [
                "此次军事航天{detail}技术突破对我国太空战略能力有何实质提升？",
                "深度分析：{detail}军事卫星的战略价值与现代战争中的关键作用",
                "从{detail}航天技术突破看中国军事现代化的太空维度",
            ],
            "bilibili": [
                "【航天军事】重大突破！{detail}成功发射，中国太空实力飞跃【深度解析】",
                "{detail}军事航天新成就！这项技术为何如此关键？",
                "【科普】{detail}军事卫星详解：在现代战争中发挥什么作用？",
            ],
            "douyin": [
                "航天突破！{detail}#航天 #军事 #太空 #科技强国",
                "震撼！{detail}中国航天军事力量腾飞！#航天 #卫星 #军事",
                "{detail}太空新成就！#航天军事 #科技 #强国",
            ],
        },
    },
    {
        "id": "topic_10",
        "name": "军人荣誉保障新政",
        "details": [
            "涵盖住房、就业、医疗全方位的",
            "建立军人荣誉积分制度的",
            "提高伤亡抚恤标准的",
            "完善退役军人再就业帮扶的",
            "设立军人荣誉日的",
        ],
        "platforms": {
            "weibo": [
                "【好消息】军人荣誉保障{detail}新政来了！让军人成为全社会尊崇的职业！#军人荣誉# #强军#",
                "重磅！{detail}军人保障新政出台，退役士兵待遇大幅提升！#退役军人 #军人待遇",
                "军人荣誉新保障：{detail}这些实实在在的改变让军人更有尊严！#军人 #荣誉",
            ],
            "zhihu": [
                "新出台的军人荣誉保障{detail}政策从哪些方面真正提升了军人地位？",
                "军人荣誉保障{detail}新政解读：制度设计如何确保军人受到社会尊崇？",
                "从{detail}军人保障政策看如何构建真正尊崇军人的社会环境？",
            ],
            "bilibili": [
                "【政策解读】军人荣誉保障{detail}新政！这些变化和军人息息相关【必看】",
                "退役军人注意！{detail}保障新政出台，你的权益有了更有力的保障",
                "【深度】军人荣誉{detail}保障新政全解析：让当兵成为光荣的事",
            ],
            "douyin": [
                "军人荣誉{detail}新政！#军人荣誉 #退役军人 #尊崇军人",
                "好政策！{detail}军人待遇提升！#军人 #荣誉 #退役军人",
                "{detail}军人保障新政来了！#军人 #政策 #强军",
            ],
        },
    },
]


# ── Lifecycle helpers ─────────────────────────────────────────────────────────

def _lifecycle_volumes(total_hours: int, peak_frac: float, rng: random.Random) -> List[float]:
    """Bell-shaped volume curve with Gaussian noise."""
    peak_h = int(total_hours * peak_frac)
    sigma = total_hours * 0.18
    vols = []
    for h in range(total_hours):
        v = 200.0 * math.exp(-0.5 * ((h - peak_h) / sigma) ** 2)
        v += rng.gauss(0, v * 0.10 + 1.0)
        vols.append(max(0.0, round(v, 2)))
    return vols


def _hour_to_phase(hour: float, peak_h: int) -> str:
    """Map an event-relative hour to a lifecycle phase label."""
    if hour <= peak_h * 0.15:
        return "emergence"
    elif hour <= peak_h * 0.50:
        return "diffusion"
    elif hour <= peak_h * 1.20:
        return "peak"
    else:
        return "decline"


# ── Text helpers ──────────────────────────────────────────────────────────────

def _render(template: str, detail: str) -> str:
    return template.replace("{detail}", detail)


def _bridge_text(topic: dict, platform: str, detail: str, rng: random.Random) -> str:
    """Post that shares core vocabulary with its paired post — high s1."""
    tpl = rng.choice(topic["platforms"][platform])
    return _render(tpl, detail)


def _background_text(topic: dict, platform: str, rng: random.Random) -> str:
    """Standalone background post — uses a different detail to lower overlap."""
    tpl = rng.choice(topic["platforms"][platform])
    detail = rng.choice(topic["details"])
    return _render(tpl, detail)


# ── Signal helpers ────────────────────────────────────────────────────────────

def _s2(delta_t_hours: float, phase: str) -> float:
    w = PHASE_WINDOWS.get(phase, 72)
    return float(np.clip(1.0 - delta_t_hours / w, 0.0, 1.0))


def _s3(platform_a: str, platform_b: str,
        migration_counts: Dict[Tuple[str, str], int]) -> float:
    total = sum(migration_counts.values()) or 1
    freq = migration_counts.get((platform_a, platform_b), 0)
    return float(np.clip(1.0 - freq / total, 0.0, 1.0))


# ── Core event generator ──────────────────────────────────────────────────────

def generate_event(
    topic: dict,
    event_id: str,
    n_posts_total: int,
    n_bridge_seeds: int,
    rng: random.Random,
) -> dict:
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=rng.randint(0, 300)
    )
    peak_frac = rng.uniform(0.30, 0.60)
    peak_h = int(TOTAL_HOURS * peak_frac)
    hourly_volumes = _lifecycle_volumes(TOTAL_HOURS, peak_frac, rng)

    posts: List[dict] = []
    bridge_pair_ids: List[Tuple[str, str]] = []
    bridge_post_ids: set = set()
    counter = [0]

    def new_post(platform: str, text: str, hour: float) -> dict:
        pid = f"{event_id}_p{counter[0]:03d}"
        counter[0] += 1
        ts = base_time + timedelta(hours=hour, minutes=rng.randint(0, 59))
        return {
            "post_id": pid,
            "text": text,
            "account_id": f"acc_{pid}",
            "platform": platform,
            "event_id": event_id,
            "timestamp": ts.isoformat(),
        }

    # ── 1. Seeded bridge pairs ────────────────────────────────────────────────
    # Cap seeds so we don't overflow the post budget
    n_bridge_seeds = min(n_bridge_seeds, n_posts_total // 3)

    for _ in range(n_bridge_seeds):
        plat_a, plat_b = rng.sample(PLATFORMS, 2)
        detail = rng.choice(topic["details"])

        hour_a = rng.uniform(0, TOTAL_HOURS - 12)
        phase_a = _hour_to_phase(hour_a, peak_h)
        window = PHASE_WINDOWS[phase_a]
        delta = rng.uniform(1.0, min(8.0, window * 0.8))
        hour_b = min(hour_a + delta, TOTAL_HOURS - 1)

        pa = new_post(plat_a, _bridge_text(topic, plat_a, detail, rng), hour_a)
        pb = new_post(plat_b, _bridge_text(topic, plat_b, detail, rng), hour_b)
        posts.extend([pa, pb])
        bridge_pair_ids.append((pa["post_id"], pb["post_id"]))
        bridge_post_ids.update([pa["post_id"], pb["post_id"]])

    # ── 2. Background posts ───────────────────────────────────────────────────
    n_bg = max(0, n_posts_total - len(posts))
    for _ in range(n_bg):
        platform = rng.choice(PLATFORMS)
        hour = rng.uniform(0, TOTAL_HOURS - 1)
        posts.append(new_post(platform, _background_text(topic, platform, rng), hour))

    posts.sort(key=lambda p: p["timestamp"])
    pid_to_post = {p["post_id"]: p for p in posts}

    # ── 3. TF-IDF cosine similarity matrix ───────────────────────────────────
    texts = [p["text"] for p in posts]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)
    tfidf = vec.fit_transform(texts)
    sim_mat = sk_cosine(tfidf).astype(float)
    pid_to_idx = {p["post_id"]: i for i, p in enumerate(posts)}

    # ── 4. Migration counts (for s3) ─────────────────────────────────────────
    migration_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for pa_id, pb_id in bridge_pair_ids:
        migration_counts[
            (pid_to_post[pa_id]["platform"], pid_to_post[pb_id]["platform"])
        ] += 1
    # Add small background counts so s3 is non-trivial
    for pa_id in [p["post_id"] for p in posts]:
        for pb_id in [p["post_id"] for p in posts]:
            if pa_id != pb_id:
                pa_p = pid_to_post[pa_id]["platform"]
                pb_p = pid_to_post[pb_id]["platform"]
                if pa_p != pb_p:
                    migration_counts[(pa_p, pb_p)] += 0  # ensure key exists

    # ── 5. Betweenness proxy (s4) ─────────────────────────────────────────────
    bridge_degree: Dict[str, int] = defaultdict(int)
    for pa_id, pb_id in bridge_pair_ids:
        bridge_degree[pa_id] += 1
        bridge_degree[pb_id] += 1
    max_deg = max(bridge_degree.values(), default=1)
    s4_map = {pid: deg / max_deg for pid, deg in bridge_degree.items()}

    # ── 6. Scored pairs ───────────────────────────────────────────────────────
    bridge_set = set(bridge_pair_ids)

    all_cross: List[Tuple[str, str]] = []
    for pa in posts:
        for pb in posts:
            if pa["post_id"] == pb["post_id"]:
                continue
            if pa["platform"] == pb["platform"]:
                continue
            ts_a = datetime.fromisoformat(pa["timestamp"])
            ts_b = datetime.fromisoformat(pb["timestamp"])
            delta_h = (ts_b - ts_a).total_seconds() / 3600
            if 0 < delta_h <= TOTAL_HOURS:
                all_cross.append((pa["post_id"], pb["post_id"]))

    positives = [(a, b) for a, b in all_cross if (a, b) in bridge_set]
    negatives = [(a, b) for a, b in all_cross if (a, b) not in bridge_set]
    n_neg = min(len(negatives), max(len(positives) * 3, 20))
    sampled_neg = rng.sample(negatives, n_neg) if negatives else []

    scored_pairs = []
    for pa_id, pb_id in positives + sampled_neg:
        pa_post = pid_to_post[pa_id]
        pb_post = pid_to_post[pb_id]
        ts_a = datetime.fromisoformat(pa_post["timestamp"])
        ts_b = datetime.fromisoformat(pb_post["timestamp"])
        delta_h = (ts_b - ts_a).total_seconds() / 3600
        hour_offset = (ts_a - base_time).total_seconds() / 3600
        phase = _hour_to_phase(hour_offset, peak_h)

        s1 = float(sim_mat[pid_to_idx[pa_id], pid_to_idx[pb_id]])
        s2 = _s2(delta_h, phase)
        s3 = _s3(pa_post["platform"], pb_post["platform"], migration_counts)
        s4 = s4_map.get(pa_id, rng.uniform(0.0, 0.05))
        label = 1 if (pa_id, pb_id) in bridge_set else 0

        scored_pairs.append({
            "post_a_id": pa_id,
            "post_b_id": pb_id,
            "s1": round(s1, 4),
            "s2": round(s2, 4),
            "s3": round(s3, 4),
            "s4": round(s4, 4),
            "phase": phase,
            "label": label,
        })

    return {
        "event_id": event_id,
        "topic": topic["name"],
        "posts": posts,
        "bridge_pairs": [[a, b] for a, b in bridge_pair_ids],
        "hourly_volumes": hourly_volumes,
        "scored_pairs": scored_pairs,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CPHot-format events for M-BridgeNet"
    )
    parser.add_argument("--events",    type=int,   default=20,
                        help="Total events to generate (default: 20)")
    parser.add_argument("--output",    default="data/cphot/processed",
                        help="Root output directory (creates train/ and test/)")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--test-frac", type=float, default=0.20,
                        help="Fraction reserved for test split (default: 0.20)")
    parser.add_argument("--posts",     type=int,   default=50,
                        help="Approx posts per event (default: 50)")
    parser.add_argument("--bridges",   type=int,   default=8,
                        help="Bridge seeds per event (default: 8)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    out = Path(args.output)
    train_dir = out / "train"
    test_dir  = out / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    n_test  = max(1, int(args.events * args.test_frac))
    n_train = args.events - n_test

    # Cycle through topics in shuffled order
    topic_pool = TOPICS * (args.events // len(TOPICS) + 1)
    rng.shuffle(topic_pool)

    print(f"Generating {args.events} events  "
          f"(train={n_train}, test={n_test})  seed={args.seed}\n")

    for i in range(args.events):
        topic = topic_pool[i]
        event_id = f"event_{i:03d}"
        n_posts   = args.posts   + rng.randint(-10, 10)
        n_bridges = args.bridges + rng.randint(-2,   2)

        event = generate_event(topic, event_id, n_posts, n_bridges, rng)

        split_dir = train_dir if i < n_train else test_dir
        out_path  = split_dir / f"{event_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False, indent=2)

        n_pos = sum(1 for sp in event["scored_pairs"] if sp["label"] == 1)
        n_neg = sum(1 for sp in event["scored_pairs"] if sp["label"] == 0)
        print(
            f"  [{i+1:>2}/{args.events}] {event_id}  topic={topic['name']:<12}  "
            f"posts={len(event['posts'])}  bridges={len(event['bridge_pairs'])}  "
            f"scored={n_pos}pos/{n_neg}neg  → {out_path}"
        )

    print(f"\nDone.")
    print(f"  Train → {train_dir}  ({n_train} events)")
    print(f"  Test  → {test_dir}   ({n_test} events)")
    print(f"\nNext steps:")
    print(f"  python scripts/train.py    --data {train_dir} --output checkpoints/mlp_fold{{fold}}.pt")
    print(f"  python scripts/evaluate.py --data {test_dir}  --checkpoint checkpoints/mlp_fold1.pt")


if __name__ == "__main__":
    main()
