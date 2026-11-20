"""数据生成：模拟校园快递驿站的工单文本、用户行为、日到件量与政策知识库。

设计原则
--------
1. 真实感：模板 + 槽位 + 口语噪声，长度与渠道分布接近真实工单；
2. 可失败：每类都包含"边界样本"（如"签收了但没拿到，是不是丢了"），
   保证分类器不会轻易达到 100%，为失败案例分析提供真实素材；
3. 合规：全部为程序合成数据，不含任何真实学生个人信息。
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from common import DATA_RAW, KNOWLEDGE, SEED, ensure_dirs

rng = random.Random(SEED)
np_rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ 槽位词表
PLACES = ["菜鸟驿站", "校园驿站", "西区快递柜", "东区驿站", "南门快递点", "三号柜", "快递超市"]
GOODS = ["耳机", "充电宝", "球鞋", "化妆品", "键盘", "显示器", "行李箱", "零食", "书", "羽绒服"]
PROVINCES = ["广东", "浙江", "江苏", "四川", "北京", "上海", "河南", "湖北"]
PREFIX = ["", "", "", "你好，", "您好，", "请问", "麻烦问一下，", "在吗？", "客服在吗？"]
SUFFIX = ["", "", "", "谢谢", "麻烦了", "急，在线等", "！", "😭", "麻烦尽快回复"]
HOURS = ["十", "十一", "九", "十二", "八"]

LABELS = {
    "pickup": "取件问题",
    "send": "寄件问题",
    "return": "退换货",
    "lost": "丢件破损",
    "compensate": "赔付投诉",
    "other": "其他咨询",
}

# ------------------------------------------------------------------ 工单模板
TEMPLATES = {
    "pickup": [
        "取件码{tcode}在{place}的柜子上输进去没反应，柜门打不开",
        "收到短信说包裹到{place}了，我过去找了二十分钟也没找到",
        "我是{building}的，能不能把包裹放到{place}，晚上{hour}点下课后才方便取",
        "取件码过期了还能取吗？小程序显示已失效",
        "包裹太大放不进柜子，{place}的人让我明天再来",
        "帮我查一下尾号{phone4}的件到了没，一直没收到通知",
        "代取需要什么手续？我室友拿我手机去可以吗",
        "{place}柜子满了，我的件被放到旁边超市，但是没人通知我",
        "晚上{hour}点以后还能取件吗？{place}几点关门",
        "取件短信没收到，只看到{place}有一条取件提醒",
        "两个件的取件码一样，是不是系统出错了",
        "我已经到{place}了，扫码一直提示「暂无可取包裹」",
    ],
    "send": [
        "寄到{province}首重多少钱？大概{weight}公斤",
        "约的{hour}点上门取件，师傅一直没来，电话也打不通",
        "易碎品你们提供包装吗？{place}有没有纸箱卖",
        "充电宝能寄到{province}吗？会不会被拦",
        "小程序下单一直提示「{err}」，下不了单",
        "寄件可以开发票吗？公司报销要用",
        "想改一下收件地址，已经从{place}寄出去了还能改吗",
        "寄到{province}一般几天能到？比较急",
        "保价怎么收费？{value}元的东西保价要多少钱",
        "寄件上门要不要另外加钱？学校里面能进来吗",
        "下单的时候选错网点了，能改到{place}吗",
    ],
    "return": [
        "淘宝退货，{place}能不能直接寄回？需要退货单号吗",
        "退货上门取件约的{hour}点，快递员说不在这片服务范围",
        "退款一直显示待揽收，我都寄出去三天了",
        "换货运费谁出？商家让我先垫付",
        "退货地址填错了，件已经到{place}了怎么办",
        "七天无理由退货，包装拆了还能退吗",
        "退货的件显示签收了，商家说没收到，让我找快递",
        "拼多多退货能不能在{place}直接扫码寄回",
    ],
    "lost": [
        "我的件在{place}放了三天，今天去取发现不见了",
        "包裹外包装破了，里面的{goods}碎了",
        "{goods}少了一件，箱子完整但胶带被撕开过",
        "显示已签收，但是我没收到，是不是被人错拿了",
        "快递员说放{place}了，能查监控吗",
        "件丢了，里面是{goods}，价值{value}元",
        "取件的时候发现柜子是空的，但是取件码显示已取件",
        "包裹被雨淋湿了，里面的{goods}进水了",
    ],
    "compensate": [
        "件丢了一个星期了，赔偿标准是多少？",
        "延误了五天，能不能赔付",
        "{place}的工作人员态度很差，我要投诉",
        "保价{value}元的件坏了，只赔{small}元不合理",
        "投诉了三次没人回复，工单号{tcode}",
        "理赔材料提交了一周还没结果",
        "你们承诺24小时回复，现在三天了没人联系我",
        "赔付的钱打到哪里？多久到账",
    ],
    "other": [
        "{place}几点开门？周末营业吗",
        "我想加盟你们驿站，需要什么条件",
        "会员每个月多少钱？有什么优惠",
        "发票怎么开？电子发票在哪里下载",
        "你们招学生兼职吗？",
        "包裹可以放几天？超过几天会退回",
        "{place}能不能代收京东和顺丰的件",
        "校园里还有别的网点吗？{place}太远了",
    ],
}

# 边界样本：故意跨类，用来制造"合理"的误分类
HARD_TEMPLATES = [
    ("显示签收了但我没拿到，是不是丢了，能赔吗", "lost"),
    ("找不到包裹，柜子空着，能不能直接赔钱", "lost"),
    ("取件码失效，重新发一条要收费吗", "pickup"),
    ("件放了两天没取，被退回了，退款找谁", "return"),
    ("快递员把我件放错柜子了，取不出来", "pickup"),
    ("投诉之后说赔我{small}元，我不接受", "compensate"),
    ("寄件下单失败，提示{err}，钱扣了", "send"),
    ("退货寄回后商家说少件，是不是路上丢了", "lost"),
    ("超时保管费一天多少钱？我放了四天", "pickup"),
    ("保价的件延误了，保价费能退吗", "compensate"),
]

ERRS = ["系统繁忙", "网络异常", "该地址不支持配送", "手机号格式错误", "网点已满"]
BUILDINGS = ["1号楼", "3号楼", "5号楼", "7号楼", "研究生公寓", "南区宿舍"]

# 长文本：多句复合诉求，更接近真实工单，也用于检验模型对长度的鲁棒性
LONG_TEMPLATES = [
    ("你好，我上周三在淘宝买了个耳机，物流显示已经放到{place}了，但是到现在都没收到取件短信，小程序里也查不到这个件，麻烦帮我查一下是不是丢件了，急用", "lost"),
    ("客服你好，我昨天在{place}寄了一个件到{province}，当时说两天能到，现在物流一直停在「已揽收」，对方催得很急，能不能帮我催一下，如果要赔付怎么算", "compensate"),
    ("我想问一下退货的事情，我在拼多多买的东西不合适要退，商家说可以直接在驿站扫码寄回，但是到{place}之后工作人员说要自己下单，到底应该怎么操作，运费谁出", "return"),
    ("下午去{place}取件，柜子显示已打开但是里面是空的，旁边还有两个同学也说找不到件，是不是系统串码了，我这个件里面有{goods}，价值{value}元，怎么处理", "lost"),
    ("我是大三学生，平时课多，晚上{hour}点才下课，能不能申请把包裹固定放在{place}，或者开通代取服务，每个月多少钱，怎么办理", "other"),
    ("寄件下单的时候一直提示{err}，换了两个手机都不行，钱倒是扣了，但是订单里没有记录，这个要怎么退款，还是说会重新下单", "send"),
    ("取件码收到了但是输进去提示已失效，打客服电话一直占线，我在{place}等了半个小时了，能不能现场帮我开一下柜子，里面是急用的{goods}", "pickup"),
    ("保价{value}元的件到的时候外包装是好的，但是打开发现{goods}裂了，拍照上传了理赔一直没通过，说我没有当场验货，这个规则合理吗，能不能申诉", "compensate"),
]


def _fill(tpl: str) -> str:
    return tpl.format(
        tcode=f"{rng.randint(10000000, 99999999)}",
        place=rng.choice(PLACES),
        building=rng.choice(BUILDINGS),
        hour=rng.choice(HOURS),
        phone4=f"{rng.randint(1000, 9999)}",
        province=rng.choice(PROVINCES),
        weight=rng.choice(["0.5", "1", "2", "3", "5"]),
        value=rng.choice(["200", "500", "800", "1200", "2000", "3000"]),
        small=rng.choice(["20", "30", "50", "80"]),
        goods=rng.choice(GOODS),
        err=rng.choice(ERRS),
    )


def _noise(text: str) -> str:
    """口语噪声：前后缀、重复标点、轻微错别字。"""
    s = rng.choice(PREFIX) + text + rng.choice(SUFFIX)
    if rng.random() < 0.08:
        s = s.replace("的", "得", 1)
    if rng.random() < 0.05:
        s = s.replace("已经", "以经")
    if rng.random() < 0.06:
        s = s + rng.choice(["。。。", "!!", "？？"])
    return s


CONNECTORS = ["，", "。", "，另外", "，还有就是", "，顺便问一下", "；"]


def build_tickets(n: int = 500) -> pd.DataFrame:
    """生成工单文本。

    标注约定：多句复合工单以「主诉」为标签（第一句的意图），第二句视为附带诉求。
    这正是真实工单中最容易误判的情形，也是有价值失败案例的来源。
    """
    weights = {"pickup": 0.28, "send": 0.17, "return": 0.13, "lost": 0.15, "compensate": 0.14, "other": 0.13}
    labels = list(weights)
    probs = np.array([weights[k] for k in labels])
    probs = probs / probs.sum()

    rows, seen = [], set()
    hard_n = 0
    while len(rows) < n:
        r = rng.random()
        # 8% 长文本、10% 跨类边界模板、其余按类别采样
        if r < 0.08:
            tpl, lab = rng.choice(LONG_TEMPLATES)
            parts = [_fill(tpl)]
            is_hard = 1
        elif r < 0.18:
            tpl, lab = rng.choice(HARD_TEMPLATES)
            parts = [_fill(tpl)]
            is_hard = 1
        else:
            lab = rng.choices(labels, weights=probs)[0]
            parts = [_fill(rng.choice(TEMPLATES[lab]))]
            r2 = rng.random()
            if r2 < 0.32:          # 跨类多意图：主诉 + 另一类诉求
                sec = rng.choice([x for x in labels if x != lab])
                parts.append(_fill(rng.choice(TEMPLATES[sec])))
                is_hard = 1
            elif r2 < 0.48:        # 同类补充说明
                parts.append(_fill(rng.choice(TEMPLATES[lab])))
                is_hard = 0
            else:
                is_hard = 0

        text = "".join(p + rng.choice(CONNECTORS) for p in parts[:-1]) + parts[-1]
        text = _noise(text)
        if is_hard:
            hard_n += 1
        if text in seen:
            continue
        seen.add(text)
        rows.append(
            {
                "ticket_id": f"T{len(rows) + 1:04d}",
                "user_id": f"U{rng.randint(1, 400):03d}",
                "channel": rng.choices(
                    ["小程序", "在线客服", "微信群", "电话转写"], weights=[0.45, 0.3, 0.15, 0.1]
                )[0],
                "created_at": (
                    datetime(2026, 9, 1) + timedelta(minutes=rng.randint(0, 60 * 24 * 60))
                ).strftime("%Y-%m-%d %H:%M"),
                "text": text,
                "label": lab,
                "label_name": LABELS[lab],
                "is_hard": is_hard,
            }
        )

    df = pd.DataFrame(rows)
    df["text_len"] = df["text"].str.len()
    print(f"[tickets] {len(df)} 条，多意图/边界样本 {hard_n} 条（{hard_n / n:.0%}），类别分布：\n{df['label_name'].value_counts()}")
    return df


# ------------------------------------------------------------------ 用户行为
SEGMENTS = {
    "高频夜间取件族": dict(weekly=(5.0, 1.2), dwell=(6, 2), night=(0.75, 0.08), weekend=(0.3, 0.08),
                     proxy=(0.05, 0.03), large=(0.1, 0.05), complaint=(0.2, 0.4), weight=(0.8, 0.3)),
    "周末集中取件族": dict(weekly=(2.5, 0.8), dwell=(30, 8), night=(0.15, 0.06), weekend=(0.8, 0.08),
                     proxy=(0.1, 0.05), large=(0.2, 0.08), complaint=(0.4, 0.6), weight=(1.5, 0.6)),
    "代取依赖族": dict(weekly=(3.5, 1.0), dwell=(20, 6), night=(0.35, 0.1), weekend=(0.45, 0.1),
                   proxy=(0.75, 0.1), large=(0.35, 0.1), complaint=(0.3, 0.5), weight=(1.8, 0.7)),
    "低频长滞留族": dict(weekly=(1.0, 0.5), dwell=(70, 15), night=(0.2, 0.08), weekend=(0.4, 0.1),
                    proxy=(0.25, 0.1), large=(0.15, 0.06), complaint=(2.2, 0.9), weight=(1.2, 0.5)),
}


def build_users(n: int = 400) -> pd.DataFrame:
    rows = []
    per = n // len(SEGMENTS)
    idx = 0
    for seg, p in SEGMENTS.items():
        for _ in range(per):
            idx += 1
            rows.append(
                {
                    "user_id": f"U{idx:03d}",
                    "weekly_pickups": max(0.0, np_rng.normal(*p["weekly"])),
                    "avg_dwell_hours": max(0.5, np_rng.normal(*p["dwell"])),
                    "night_ratio": float(np.clip(np_rng.normal(*p["night"]), 0, 1)),
                    "weekend_ratio": float(np.clip(np_rng.normal(*p["weekend"]), 0, 1)),
                    "proxy_ratio": float(np.clip(np_rng.normal(*p["proxy"]), 0, 1)),
                    "large_item_ratio": float(np.clip(np_rng.normal(*p["large"]), 0, 1)),
                    "complaint_cnt": int(max(0, np_rng.normal(*p["complaint"]))),
                    "avg_weight_kg": max(0.1, np_rng.normal(*p["weight"])),
                    "true_segment": seg,
                }
            )
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"[users] {len(df)} 人，真实分群：\n{df['true_segment'].value_counts()}")
    return df


# ------------------------------------------------------------------ 日到件量
def _promo_index(d: datetime) -> float:
    md = (d.month, d.day)
    if d.month == 11 and 1 <= d.day <= 15:
        return 1.0
    if d.month == 6 and 1 <= d.day <= 20:
        return 0.65
    if d.month == 12 and 8 <= d.day <= 14:
        return 0.5
    if (d.month == 9 and 1 <= d.day <= 10) or (d.month == 2 and 20 <= d.day <= 28):
        return 0.55
    if d.month == 1 and 10 <= d.day <= 25:
        return 0.35
    return 0.0


HOLIDAYS = {
    (1, 1), (2, 17), (2, 18), (2, 19), (4, 4), (4, 5), (4, 6),
    (5, 1), (5, 2), (5, 3), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7),
}


def _vacation_factor(d: datetime) -> float:
    """校历效应：寒暑假学生离校，驿站到件量明显下降。

    注意：该因素在数据中存在，但建模特征里**故意不提供校历字段**，
    用来暴露「特征缺失导致的系统性偏差」这一真实工程问题（见失败案例分析）。
    """
    if d.month == 7 or (d.month == 8 and d.day <= 25):
        return 0.58
    if d.month == 2 and 1 <= d.day <= 20:
        return 0.55
    return 1.0


def build_volume(days: int = 480, start: str = "2025-06-01") -> pd.DataFrame:
    d0 = datetime.fromisoformat(start)
    rows = []
    hist: list[float] = []
    for i in range(days):
        d = d0 + timedelta(days=i)
        promo = _promo_index(d)
        vac = _vacation_factor(d)
        temp = 16 + 11 * np.sin(2 * np.pi * (d.timetuple().tm_yday - 100) / 365)
        rain = float(np_rng.exponential(3.5)) if np_rng.random() < 0.28 else 0.0
        dow = d.weekday()
        dow_eff = {0: 35, 1: 10, 2: 0, 3: 5, 4: 28, 5: 48, 6: 22}[dow]
        lag1 = hist[-1] if hist else 320.0
        lag7 = hist[-7] if len(hist) >= 7 else 320.0
        v = (
            300 * (1 + 0.55 * promo) * vac
            + dow_eff
            + 0.30 * (lag1 - 320 * vac)
            + 0.10 * (lag7 - 320 * vac)
            + 0.8 * rain
            + 1.2 * temp
            + np_rng.normal(0, 22)
        )
        v = float(max(120, v))
        hist.append(v)
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "volume": round(v),
                "dow": dow,
                "is_weekend": int(dow >= 5),
                "is_holiday": int((d.month, d.day) in HOLIDAYS),
                "promo_index": promo,
                "is_vacation": int(vac < 1.0),
                "temp_c": round(float(temp), 1),
                "rain_mm": round(rain, 1),
                "month": d.month,
            }
        )
        prev = v
    df = pd.DataFrame(rows)
    df["lag1_volume"] = df["volume"].shift(1)
    df["lag7_volume"] = df["volume"].shift(7)
    df["roll7_mean"] = df["volume"].shift(1).rolling(7).mean()
    df = df.dropna().reset_index(drop=True)
    print(f"[volume] {len(df)} 天，均值 {df['volume'].mean():.1f}，最大 {df['volume'].max()}")
    return df


# ------------------------------------------------------------------ 知识库
FAQ = [
    ("取件码", ["取件码在哪看", "取件码丢了", "没收到取件码"], "取件码会在包裹入库后 5 分钟内通过短信和小程序推送；也可在「我的包裹」中查看。若未收到，请确认手机号是否与下单一致，或到驿站柜台凭手机号后四位查询。", "取件"),
    ("取件有效期", ["包裹能放几天", "超时怎么办"], "包裹免费保管 3 天，第 4 天起收取超时保管费 1 元/天，单件最高 5 元；超过 7 天未取将退回发件方。", "取件"),
    ("超时费", ["超时费怎么算", "保管费多少钱"], "超时保管费从入库第 4 天开始计收，1 元/天，单件累计不超过 5 元，取件时一并支付。", "取件"),
    ("代取", ["让同学代取", "代取需要什么"], "代取需提供取件码，并在小程序内将包裹「授权代取」给指定手机号；代取人凭授权码取件。大件或保价件需出示双方手机号。", "取件"),
    ("营业时间", ["驿站几点开门", "周末营业吗"], "校园驿站营业时间为 8:00–22:00，周末及节假日不休；寒暑假期间调整为 9:00–18:00。", "其他"),
    ("寄件价格", ["寄快递多少钱", "首重怎么算"], "省内首重 8 元/公斤，续重 2 元/公斤；跨省首重 12 元/公斤，续重 5 元/公斤。校园网点寄件享 9 折。", "寄件"),
    ("上门取件", ["上门取件时间", "师傅不来"], "上门取件时间为 9:00–20:00，下单后 2 小时内上门；超时未上门可在小程序点击「催单」，或拨打网点电话。", "寄件"),
    ("禁寄品", ["充电宝能寄吗", "什么不能寄"], "禁寄品包括易燃易爆品、管制刀具、活体动物、锂电池（充电宝需符合额定能量 ≤100Wh 且有 3C 标识，仅陆运）。具体以网点安检为准。", "寄件"),
    ("保价", ["保价怎么收费", "要不要保价"], "保价费为声明价值的 0.5%，最低 1 元；保价件需当场验视并拍照。未保价件按实际损失赔偿，最高不超过运费的 7 倍。", "寄件"),
    ("改地址", ["能改地址吗", "地址填错了"], "包裹未揽收前可在小程序免费修改地址；已揽收需联系客服申请中转，可能产生转寄费；已到达派送网点则无法修改。", "寄件"),
    ("退货", ["退货怎么寄", "退货单号"], "支持在驿站扫码寄回，需提供平台退货单号（如淘宝「寄回」二维码）。退货件按平台协议计费，多数场景免运费。", "退换"),
    ("退款进度", ["退款没到账", "显示待揽收"], "退货件揽收后 24 小时内更新物流；商家签收后 1–3 个工作日退款。若物流停滞超过 48 小时，请联系客服核查。", "退换"),
    ("丢件", ["件丢了怎么办", "包裹不见了"], "确认丢件后请在 48 小时内提交工单，提供订单号、取件码与物品价值证明；核实后 3 个工作日内启动赔付。", "赔付"),
    ("破损", ["包裹破了", "东西碎了"], "签收前发现破损可拒收并拍照；签收后发现破损，请在 24 小时内上传外包装与内件照片申请理赔。", "赔付"),
    ("赔偿标准", ["赔多少钱", "赔付标准"], "保价件按声明价值赔付；未保价件按实际损失赔付，最高为运费 7 倍；生鲜、易碎品建议保价。", "赔付"),
    ("投诉", ["怎么投诉", "投诉没人管"], "可在小程序「投诉建议」提交，或拨打网点电话；一般 24 小时内响应，48 小时内给出处理方案。", "赔付"),
    ("发票", ["怎么开发票", "电子发票"], "寄件发票在小程序「我的-发票」申请，电子普票 3 个工作日内开具；月结客户可申请专票。", "其他"),
    ("加盟", ["怎么加盟", "开驿站条件"], "校园加盟需具备 20㎡ 以上场地、消防合规、每日 ≥300 件处理能力；可联系校园运营中心提交申请。", "其他"),
    ("会员", ["会员多少钱", "有什么优惠"], "校园会员 9.9 元/月，含每月 3 次免费超时保管、寄件 8.5 折、优先取件通道。", "其他"),
    ("放错柜", ["放错柜子了", "取不出来"], "若包裹被放入错误柜机，请提供取件码与柜机编号，工作人员将在 30 分钟内协调开柜或转柜。", "取件"),
]


def build_faq() -> list[dict]:
    return [
        {
            "id": f"FAQ{i + 1:02d}",
            "title": t,
            "questions": qs,
            "answer": a,
            "category": c,
        }
        for i, (t, qs, a, c) in enumerate(FAQ)
    ]


def inject_collection_defects(tickets: pd.DataFrame, users: pd.DataFrame, volume: pd.DataFrame):
    """模拟真实采集缺陷：空文本、超短、重复上报、乱码、异常数值、系统缺日。

    这些"脏数据"用于验证清洗流程，清洗前后统计见 data/processed/cleaning_stats.json。
    """
    rnd = random.Random(SEED + 7)
    base = tickets.drop(columns=["text_len"])

    empty = base.sample(8, random_state=SEED).assign(text="")
    short = base.sample(6, random_state=SEED + 1).assign(
        text=lambda d: [rnd.choice(["在吗", "取件", "?", "1", "急"]) for _ in range(len(d))]
    )
    dup = base.sample(10, random_state=SEED + 2)
    garbled = base.sample(4, random_state=SEED + 3).assign(
        text=lambda d: [rnd.choice(["□□□□", "。。。", "asdf", "取件码％＃＠"]) for _ in range(len(d))]
    )
    dirty = pd.concat([empty, short, dup, garbled], ignore_index=True)
    dirty["ticket_id"] = [f"TD{i + 1:03d}" for i in range(len(dirty))]
    out = pd.concat([base, dirty], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    u = users.copy()
    idx = rnd.sample(range(len(u)), 5)
    u.loc[idx[0], "avg_dwell_hours"] = -3.0
    u.loc[idx[1], "night_ratio"] = 1.4
    u.loc[idx[2], "weekly_pickups"] = 60.0
    u.loc[idx[3], "proxy_ratio"] = -0.2
    u.loc[idx[4], "avg_weight_kg"] = 25.0

    v = volume.copy()
    drop_idx = rnd.sample(range(len(v)), 3)
    v = v.drop(index=drop_idx).reset_index(drop=True)

    print(f"[dirty] 注入缺陷：空文本 8、超短 6、重复 10、乱码 4；用户异常 5；缺失日期 3")
    return out, u, v


def main() -> None:
    ensure_dirs()
    tickets = build_tickets(500)
    users = build_users(400)
    volume = build_volume(480)
    faq = build_faq()
    tickets, users, volume = inject_collection_defects(tickets, users, volume)

    tickets.to_csv(DATA_RAW / "tickets.csv", index=False, encoding="utf-8-sig")
    users.to_csv(DATA_RAW / "users.csv", index=False, encoding="utf-8-sig")
    volume.to_csv(DATA_RAW / "daily_volume.csv", index=False, encoding="utf-8-sig")
    (KNOWLEDGE / "faq.json").write_text(json.dumps(faq, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[faq] {len(faq)} 条知识库条目")
    print(f"[raw] 工单 {len(tickets)}、用户 {len(users)}、日到件量 {len(volume)}")
    print("原始数据已写入 data/raw/")


if __name__ == "__main__":
    main()
