"""ShopCare 美妆业务 19 个子意图的单一注册表。

识别器、Embedding、Pattern、Schema 和路由迁移时均从这里读取；本文件只描述业务，
不包含模型调用或工具执行逻辑。
"""

from __future__ import annotations

from typing import Any, Dict, List


COMMON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent_id": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
        "slots": {"type": "object"},
    },
    "required": ["intent_id", "confidence", "reasoning"],
    "additionalProperties": False,
}


def _intent(
    intent_id: str,
    name_zh: str,
    agent: str,
    description: str,
    embedding_examples: List[str],
    exact_patterns: List[str],
    generic_patterns: List[str],
    distinction: str,
    boundary: str,
) -> Dict[str, Any]:
    return {
        "intent_id": intent_id,
        "name_zh": name_zh,
        "agent": agent,
        "schema": COMMON_SCHEMA,
        "description": description,
        "embedding_examples": embedding_examples,
        "exact_patterns": exact_patterns,
        "generic_patterns": generic_patterns,
        "distinction": distinction,
        "boundary": boundary,
    }


BEAUTY_INTENT_REGISTRY: List[Dict[str, Any]] = [
    _intent("ingredient_fit", "成分与肤质适配", "consult", "咨询产品成分、肤质适配和使用风险。", ["我是敏感肌能用吗", "油皮适合这款精华吗", "里面有没有酒精和香精"], ["敏感肌", "油皮", "成分", "酒精", "香精"], ["适合", "能不能用", "担心", "咨询"], "与使用方法的区别是关注能否适配，而不是如何使用。", "涉及疾病、用药或明确不良反应时转入 adverse_reaction。"),
    _intent("shade_match", "色号与搭配推荐", "consult", "根据肤色、妆效和搭配需求提供色号建议。", ["黄皮适合哪个粉底色号", "想要自然妆感选什么色", "这支口红和什么颜色搭配"], ["色号", "黄皮", "冷皮", "暖皮", "搭配"], ["推荐", "选哪个", "适合什么颜色"], "与商品信息的区别是需要个性化选择，而不是查询规格。", "不能承诺绝对显白或替代线下试色。"),
    _intent("usage_howto", "使用方法", "consult", "回答产品用量、顺序、频次和注意事项。", ["这个面霜怎么用", "精华应该在水前还是水后", "面膜多久用一次"], ["怎么用", "使用方法", "用量", "顺序", "多久用一次"], ["如何", "步骤", "频次", "方法"], "与成分适配的区别是用户已经决定使用，只询问操作方式。", "涉及用后发红、刺痛或起疹时转入 adverse_reaction。"),
    _intent("product_info", "商品信息咨询", "consult", "查询商品规格、容量、功效定位和适用部位。", ["这款精华有多少毫升", "产品主要是做什么的", "套装里面有几件"], ["多少毫升", "规格", "容量", "功效", "套装"], ["是什么", "有哪些", "介绍一下", "信息"], "与色号推荐的区别是返回固定商品资料，不做个性化推荐。", "不对图片宣传语之外的功效作额外推断。"),
    _intent("promo_rules", "活动与订单规则咨询", "consult", "解释优惠券、满减、赠品、活动有效期和适用条件。", ["这个活动什么时候结束", "满多少可以送赠品", "优惠券为什么不能用"], ["活动", "优惠券", "满减", "赠品", "有效期"], ["规则", "为什么不能", "怎么参加", "有没有优惠"], "与退货退款的区别是尚未发生售后问题，重点是活动条件。", "具体活动以当前配置规则为准，不能自行承诺补发优惠。"),
    _intent("pregnancy_safety", "孕期与特殊人群适用", "consult", "回答孕期、哺乳期及特殊人群的资料性适用问题。", ["孕妇可以用这款吗", "哺乳期能不能使用", "敏感人群需要注意什么"], ["孕妇", "孕期", "哺乳期", "特殊人群"], ["能不能用", "安全吗", "注意什么"], "与成分适配的区别是明确涉及特殊人群阶段，而不是普通肤质。", "不作医疗安全结论，必要时建议咨询医生。"),
    _intent("order_status", "订单进度与物流查询", "order", "查询订单创建、付款、备货、发货状态及正常物流位置。", ["我的订单处理到哪了", "订单什么时候发货", "物流到哪里了"], ["订单状态", "订单进度", "什么时候发货", "备货", "物流到哪里", "物流在哪", "快递到哪"], ["查询", "查一下", "到哪了", "物流", "快递", "进度"], "普通询问订单或物流节点归此意图；出现长时间无更新、卡住或要求催查时才转物流停滞。", "需要订单号或可唯一匹配的订单信息；查询物流位置时允许调用物流查询工具。"),
    _intent("logistics_stalled", "物流停滞与催查", "order", "处理物流长时间不更新、停滞、催查和异常跟进。", ["物流三天没动了", "快递一直停在中转站", "帮我催查物流"], ["物流停滞", "物流不更新", "没更新", "催查", "一直没动", "卡在"], ["卡住", "几天没变化", "一直不动", "催一下", "延误"], "与普通订单/物流查询的区别是已经明确表达超时、停滞或催查需求。", "只有物流单号和状态时不能编造具体轨迹。"),
    _intent("signed_not_received", "显示签收但未收到", "order", "处理物流显示签收但消费者实际未收到货的情况。", ["物流显示签收但我没收到", "快递说已签收在哪里", "订单签收了但是没有拿到"], ["已签收未收到", "显示签收", "签收但没收到"], ["没收到", "找不到包裹", "快递不见了"], "与物流停滞的区别是系统已经显示签收。", "需要核实代收人、地址和签收时间，不能直接认定丢件。"),
    _intent("intercept_change_address", "拦截与改址", "order", "处理已发货订单的拦截、改地址和配送信息变更。", ["已经发货了还能改地址吗", "请帮我拦截这个包裹", "快递能不能改到另一个地址"], ["拦截", "改地址", "修改收货地址", "改址"], ["能不能改", "换地址", "不要送到这里"], "与退款的区别是目标仍是履约，不是取消交易。", "新地址和拦截结果必须由人工或物流系统确认。"),
    _intent("missing_item", "漏发与补发", "order", "处理订单中商品、赠品或配件漏发并发起补发核实。", ["我的订单少了一件", "赠品没有一起发来", "漏发了帮我补发"], ["漏发", "少发", "缺少商品", "补发", "赠品没收到"], ["少了", "没有收到", "缺件", "补寄"], "与错发的区别是应发商品缺失，而不是收到错误商品。", "需核对订单明细和活动赠品记录。"),
    _intent("wrong_item", "错发与换货", "order", "仅处理收到的商品、规格或色号与订单不一致的情况；单独提出换货但没有说明商品不符时不属于此意图。", ["收到的色号发错了", "我买的是一号却发成二号", "商品发错了想换货"], ["错发", "发错", "色号发错", "商品发错", "规格不符"], ["不一样", "拿错了"], "与漏发的区别是包裹中有商品但内容不匹配。", "必须有收到商品与订单不一致的证据；仅说‘只换货’时先澄清换货原因，不得推断错发。"),
    _intent("damaged_on_arrival", "到手破损", "order", "处理商品或外包装在签收时破损、渗漏、污染等问题。", ["收到时瓶子破了", "快递把包装压坏了", "产品漏液了怎么处理"], ["破损", "压坏", "漏液", "破裂", "到手坏了"], ["坏了", "损坏", "包装问题", "渗漏"], "与不良反应的区别是商品或包装损坏，不是使用后的身体反应。", "需要记录破损范围和订单状态，处理方案由人工确认。"),
    _intent("return_refund", "退货退款申请", "after_sales", "处理消费者明确提出且未被否定的退货或退款申请。", ["我想退货退款", "用了不合适可以退吗", "怎么申请退款"], ["退货退款", "申请退款", "退货", "退款申请"], ["不想要了", "退掉", "退钱", "售后"], "与退款到账异常的区别是尚未确认退款是否已发起或到账。", "需核对订单、退货原因和拆封使用状态；明确说‘不要退款、只换货’时不得归为退款申请。"),
    _intent("adverse_reaction", "不良反应与使用不适", "after_sales", "处理使用产品后出现红肿、刺痛、瘙痒等不适并登记风险。", ["用了之后脸上发红起疹", "涂完很刺痛怎么办", "使用后过敏了想退货"], ["过敏", "发红", "起疹", "刺痛", "瘙痒", "肿"], ["不舒服", "不适", "用后", "症状"], "与成分适配的区别是已经实际使用并出现身体反应。", "不能诊断或判断严重程度；就医、呼吸困难等情况优先人工处理。"),
    _intent("suspected_abnormal_refund", "疑似异常退款", "after_sales", "识别退款行为中的重复申请、异常频次或需要人工核验的情况。", ["同一订单重复申请退款", "退款记录好像不对", "为什么这笔退款被标记异常"], ["异常退款", "重复退款", "退款异常", "风控核验"], ["不对", "重复", "被拦截", "核实"], "与普通退款申请的区别是已有异常信号或需要风控核验。", "AI 不下欺诈结论，只记录风险并请求人工审核。"),
    _intent("price_protection", "价保补差", "after_sales", "处理活动降价后的价保资格和补差申请。", ["刚买就降价了可以退差价吗", "申请价保补差", "活动价和我付款价不一样"], ["价保", "保价", "补差", "差价"], ["降价", "便宜了", "退差价"], "与退款的区别是订单仍保留，只申请价格差额。", "差额和时间窗口必须按活动规则计算。"),
    _intent("refund_not_received", "退款到账异常", "after_sales", "处理已显示退款但消费者未到账、到账金额不符或状态异常。", ["退款显示成功但钱没到账", "退款金额少了", "退款一直不到账"], ["退款未到账", "退款到账异常", "退款金额不对", "退款成功未收到"], ["没收到钱", "到账", "钱去哪了", "退款进度"], "与退货退款申请的区别是退款动作已经产生或显示处理中。", "不能承诺具体到账时间，需核对退款记录和支付渠道。"),
    _intent("shipping_fee_reimbursement", "运费报销", "after_sales", "处理符合规则的退货运费、补寄运费或售后运费报销。", ["退货运费谁来承担", "售后运费可以报销吗", "补发的快递费怎么处理"], ["运费报销", "退货运费", "运费承担", "运费补偿"], ["快递费", "邮费", "报销", "承担"], "与退款的区别是申请对象为物流费用，不是商品货款。", "金额和凭证由人工按规则审核。"),
]


BEAUTY_INTENT_BY_ID = {item["intent_id"]: item for item in BEAUTY_INTENT_REGISTRY}


def build_embedding_templates() -> Dict[str, tuple[str, ...]]:
    """从注册表生成 Embedding 示例，不在识别器中重复维护。"""
    return {
        item["intent_id"]: tuple(item["embedding_examples"])
        for item in BEAUTY_INTENT_REGISTRY
    }


def build_pattern_maps() -> Dict[str, Dict[str, tuple[str, ...]]]:
    """从注册表生成精确与通用 Pattern 规则。"""
    return {
        item["intent_id"]: {
            "exact": tuple(item["exact_patterns"]),
            "generic": tuple(item["generic_patterns"]),
        }
        for item in BEAUTY_INTENT_REGISTRY
    }


def build_llm_intent_schema() -> Dict[str, Any]:
    """从注册表生成 LLM 可调用的意图枚举 Schema。"""
    return {
        "type": "string",
        "enum": [item["intent_id"] for item in BEAUTY_INTENT_REGISTRY],
        "description": "美妆客服 19 个子意图 ID，必须从注册表枚举中选择。",
    }


EMBEDDING_TEMPLATES = build_embedding_templates()
PATTERN_MAPS = build_pattern_maps()
LLM_INTENT_SCHEMA = build_llm_intent_schema()


def validate_registry() -> None:
    if len(BEAUTY_INTENT_REGISTRY) != 19:
        raise ValueError(f"美妆意图注册表应为 19 条，当前为 {len(BEAUTY_INTENT_REGISTRY)} 条")
    if len(BEAUTY_INTENT_BY_ID) != len(BEAUTY_INTENT_REGISTRY):
        raise ValueError("美妆意图 ID 必须唯一")
    valid_agents = {"consult", "order", "after_sales"}
    if any(item["agent"] not in valid_agents for item in BEAUTY_INTENT_REGISTRY):
        raise ValueError("存在未注册的 Agent")


validate_registry()

