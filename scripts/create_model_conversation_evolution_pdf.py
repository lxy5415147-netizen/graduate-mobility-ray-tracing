# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "2026-06-05_model_conversation_evolution.pdf"


def register_fonts() -> tuple[str, str]:
    regular = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
    bold = Path(r"C:\Windows\Fonts\simhei.ttf")
    if not regular.exists():
        regular = Path(r"C:\Windows\Fonts\Deng.ttf")
    if not bold.exists():
        bold = regular
    pdfmetrics.registerFont(TTFont("CN-Regular", str(regular)))
    pdfmetrics.registerFont(TTFont("CN-Bold", str(bold)))
    return "CN-Regular", "CN-Bold"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(esc(text), style)


def section(story: list, title: str, styles: dict[str, ParagraphStyle]) -> None:
    story.append(Spacer(1, 0.24 * cm))
    story.append(para(title, styles["h2"]))
    story.append(Spacer(1, 0.08 * cm))


def bullet(story: list, items: list[str], styles: dict[str, ParagraphStyle]) -> None:
    for item in items:
        story.append(para("• " + item, styles["body"]))


def add_turn(
    story: list,
    idx: int,
    user_text: str,
    answer_summary: list[str],
    key_turn: str,
    styles: dict[str, ParagraphStyle],
) -> None:
    story.append(Spacer(1, 0.16 * cm))
    story.append(para(f"第 {idx} 轮：{key_turn}", styles["h3"]))
    story.append(para("用户原问：", styles["label"]))
    story.append(para(user_text, styles["quote"]))
    story.append(para("回答压缩：", styles["label"]))
    bullet(story, answer_summary, styles)


def build() -> None:
    font, bold = register_fonts()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=bold,
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName=font,
            fontSize=9.5,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#111827"),
            spaceBefore=4,
            spaceAfter=3,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base["Heading3"],
            fontName=bold,
            fontSize=10.8,
            leading=15,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=5,
            spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=font,
            fontSize=8.9,
            leading=14,
            textColor=colors.HexColor("#1f2937"),
            alignment=TA_LEFT,
            spaceAfter=2.5,
        ),
        "quote": ParagraphStyle(
            "quote",
            parent=base["BodyText"],
            fontName=font,
            fontSize=8.4,
            leading=13.2,
            leftIndent=0.28 * cm,
            rightIndent=0.12 * cm,
            borderWidth=0.4,
            borderColor=colors.HexColor("#d1d5db"),
            borderPadding=5,
            backColor=colors.HexColor("#f9fafb"),
            textColor=colors.HexColor("#111827"),
            spaceAfter=4,
        ),
        "label": ParagraphStyle(
            "label",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=8.7,
            leading=12,
            textColor=colors.HexColor("#374151"),
            spaceBefore=2,
            spaceAfter=2,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=font,
            fontSize=8.1,
            leading=12.5,
            textColor=colors.HexColor("#1f2937"),
        ),
    }

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=1.55 * cm,
        rightMargin=1.55 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
        title="模型讨论全过程与思维进阶总结",
        author="Codex and lxy5415147-netizen",
    )

    story: list = []
    story.append(para("模型讨论全过程与思维进阶总结", styles["title"]))
    story.append(para("从 twod-answer 观测复现，到 O-U 条件介质响应，再到 A 类折射/散射光路｜已剔除 GitHub 操作段落", styles["subtitle"]))

    section(story, "0. 总体脉络", styles)
    bullet(
        story,
        [
            "讨论起点：用户已经完成 twod-answer，能复现所需 A/B/C/D/E 表格；核心问题转向预测建模。",
            "第一阶段：明确 ABC 是外地入射光束的介质响应，DE 是本地生内部留存/逃逸模块；二者同属主模型框架但分母不同。",
            "第二阶段：从普通 logit 走向“光学结构化 softmax 响应函数”，强调光线追踪是过程结构，logit/softmax 是估计概率响应的工具。",
            "第三阶段：讨论三维/高维人才光场，厘清响应函数与真实折射光路的关系：响应函数负责分解流量，三维光场主要服务于折射后的方向、距离、高度跃迁与散射。",
            "当前前沿：A 类折射部分成为下一步重点。可先做直接折射/散射核 P(D|O,U,refraction)，再做分层折射模型，最后再考虑递归/树状多跳传播。",
        ],
        styles,
    )

    section(story, "1. 关键转折点", styles)
    rows = [
        ["阶段", "转折点", "模型含义"],
        ["观测复现 → 预测", "twod-answer 已完成，下一步要从表格复现转向预测模型。", "建立可计算的响应层，而不是只统计真实比例。"],
        ["三层模型 → 当前主模型", "先做 O-U 条件介质模型，同时加入 DE；暂缓全局平均和 Ucity-only。", "主模型框架含 ABC 外地响应与 DE 本地响应两个子模块。"],
        ["普通 logit → 光学结构", "用户质疑“只是套壳”，于是将变量按折射/反射/吸收机制进入不同响应势。", "softmax 不再只是黑箱分类，而是守恒式通道分解。"],
        ["RGB/光谱讨论 → 暂缓人才光谱", "用户指出 RGB 与反射/吸收/折射不同，人才光谱与 O-U 路径记忆可能重复。", "第一版不做人光谱；后续再把高校类型、专业结构等作为光束成分。"],
        ["响应层 → 折射层", "用户意识到三维光场主要应在折射之后发挥作用。", "响应函数决定多少进入 A；折射/散射核决定 A 流向哪里。"],
        ["省会中转/分层 → 直接核", "用户提出省域、省会和层级金字塔；最后回到直接折射/散射核也值得做。", "先做 P(D|O,U,T) 的 hit model，再做层级版本作为增强。"],
    ]
    tbl = Table([[para(c, styles["small"]) for c in row] for row in rows], colWidths=[3.2 * cm, 6.2 * cm, 7.2 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(tbl)
    story.append(PageBreak())

    section(story, "2. 逐轮对话记录：用户原问与回答压缩", styles)

    turns = [
        (
            "我现在做好了twod-answer这个功能也就是根据数据复现了我需要的一些表格，现在要预测了，要做模型了。第一个问题：ABC比例的预测是根据模型本身来的，思想是用光学模型，有没有比较好的方法；第二个问题：DE也是预测的，也是用光学知识，你不能直接观测，大概这两种模型不大一样，但都是运用相关光学知识。有什么方法吗，你先和我讨论，然后不要改代码，并且分析一下我这个项目里面的代码都是干什么的",
            ["梳理项目：twod-answer 是观测答案表，run_calibrate_2d 是原校准预测管线。", "提出三层结构：入射光束生成、介质响应、折射/散射核。", "明确 ABC 与 DE 机制不同：ABC 是外地入射响应；DE 是本地留存/逃逸。"],
            "从观测复现走向预测建模",
        ),
        (
            "我同意你说的三层模型，有几个问题：1我现在想先做O-U 条件介质模型，其他两个先不做；Attract_D的问题：不能用D，城市属性我当前数据里没有吗，我可以再上传几个文件，你想办法看看数据里有没有属性",
            ["检查主表字段，发现 A_、B_、C_ 三组城市属性。", "确认 ABC 阶段不能用 C_ 去向城市属性；可用 A_ 来源属性、B_ 高校城市属性、O-U 路径关系。", "发现 A_Level2/B_Level2 是区域字段，不是城市等级；真正等级是老家所在城市等级、高校城市等级。"],
            "确认不使用 Dcity 属性，挖掘现有 O/U 属性",
        ),
        (
            "我的意思是想先做O-U 条件介质模型，当然要加上DE，其他两个先不做指的是基线模型：全局平均响应和Ucity 介质响应模型不做",
            ["纠正理解：暂时不做全局平均和 Ucity-only，但主模型必须包含 DE。", "定义第一阶段为 O-U 条件介质模型 + 本地 D/E 模块。", "ABC 用外地生 O!=U；DE 用本地生 O=U。"],
            "主模型包含 ABC 与 DE 两个子模块",
        ),
        (
            "我不做单条记录，可以用“本地入射规模”作为输入，建模可以先按图片这样。我现在想问你个问题，就是你将要怎么修改和运行我的代码，是你给出代码文件我来运行吗，用pycharm吗，我现在用的是anaconda，可以用GitHub吗\n\n图片内容：ABC：多项 logit / 多分类模型；DE：二项 logit / 二分类模型",
            ["确认不按单条记录建模，而按聚合光束建模。", "ABC 使用多项 logit/多分类模型；DE 使用二项 logit/二分类模型。", "建议新增独立脚本，不破坏旧代码；用 Anaconda/PyCharm 均可运行。"],
            "确定聚合光束建模与代码执行方式",
        ),
        (
            "然后有个问题我想起来了，那个DE的分母你用的是外地入射流吗，不应该是本地流吗",
            ["确认 DE 分母必须是本地流 F_local=D+E。", "ABC 分母是 F_external=A+B+C。", "这是模型物理含义的关键修正：不同光束池不能混分母。"],
            "DE 分母问题被明确修正",
        ),
        (
            "然后有个问题我想起来了，那个DE的分母你用的是外地入射流吗，不应该是本地流吗",
            ["用户重复强调同一问题。", "再次确认：DE 绝不能用外地入射流作分母。", "后续代码中 DE 输出使用 F_local。"],
            "重复确认 DE 分母",
        ),
        (
            "好的，那你现在在现有项目里新增/修改 Python 文件并允许",
            ["新增 run_ou_condition_response.py。", "实现 ABC 按 O-U 聚合、DE 按 U 聚合的条件响应模型。", "使用 Anaconda Python 跑通小样本与全量。"],
            "实现第一版普通 O-U 条件响应模型",
        ),
        (
            "我还准备顺手把“入射规模”作为特征加进去：ABC 用 F_external，DE 用 F_local，统一成 log_input_flow，这正好符合你说的本地入射规模可以作为输入。这句话什么意思，你干了什么",
            ["解释 log_input_flow 是光束强度特征。", "ABC 中 log_input_flow=log(1+F_external)，DE 中 log_input_flow=log(1+F_local)。", "强调只放入分母规模，不把 A/B/C 或 D/E 答案比例当特征。"],
            "解释入射强度特征",
        ),
        (
            "好的，有两个问题，跑代码的时候主要用到的是CPU对吗，GPU加入进来可以跑的更快吗；现在的结果怎么看",
            ["说明 sklearn/logit 主要用 CPU，GPU 暂时意义不大。", "普通条件 logit 结果：ABC 总体比例贴近，dominant accuracy 约 0.85；DE 总体可用但 D 预测偏高。", "指出普通 logit 拟合强，但解释性不如结构化模型。"],
            "评估普通条件 logit 结果与计算资源",
        ),
        (
            "这版没用D对不对，那你是用了什么样的思想来做这个模型，我感觉结果还可以，具体算法是什么？",
            ["确认没有使用 Dcity/C_ 属性。", "解释当前模型是 P(A/B/C|O,U) 与 P(D/E|U)。", "算法本质是聚合光束上的多项/二项逻辑回归，通过 softmax/sigmoid 将线性分数转成概率。"],
            "确认无 D 泄漏并解释算法",
        ),
        (
            "logit是什么鬼解释清楚算法，它属于研究人流的经典算法吗还是什么经典算法？这样做的话感觉只是外面套了个光线跟踪的壳子，里面还是别的内容。你和我好好探讨一下",
            ["解释 logit 是经典离散选择/概率分配模型，在迁移、交通、目的地选择中常见。", "承认 logit 本身不是光线追踪，创新不能只放在算法名上。", "提出关键定位：光线追踪是过程结构；logit 是估计介质响应函数的工具。"],
            "面对“套壳”质疑，重新定位模型创新",
        ),
        (
            "我想知道现在的这个逻辑回归是怎么估计概率的，我趋向于更“光学”的算法，比如说替换逻辑回归里面的一些东西，加入光学属性这样，但是我也知道这个可能要调很多次，我们先讨论讨论，不行就直接用逻辑回归",
            ["解释多项 logit 的 score 与 softmax 概率估计。", "提出三档方案：保留 logit 但换光学特征；半物理响应函数；显式光学参数模型。", "建议走第二档：保留 softmax 归一化，但按反射/吸收/折射机制设计响应势。"],
            "从普通 logit 转向光学结构化响应函数",
        ),
        (
            "我觉得先走第二档吧，第三档先不做。但是有个问题吧，难道这样之前没有人做过吗，因为很好想的是“折射”与一些因素有关，“反射”与另一些因素相关，就算不是套了光学模型的外壳，应该也会有人想到吧",
            ["承认不同结果由不同因素解释并不新。", "强调创新要落在 O-U-D 二阶段路径、条件介质响应、光束守恒分解、出射追踪与散射画像的组合上。", "建议把模型称为光学结构化概率响应模型，而不是宣称全新物理定律。"],
            "澄清真正创新点不在“变量分配”本身",
        ),
        (
            "那你先改logit结构吧，我们来讨论讨论：现有数据字段有的应该是可以一个多用的，比如等级差既可以用在折射也可以用在反射吸收，有的则具有倾向性，比如城市开放度",
            ["实现 run_ou_optical_response.py。", "将变量分成通用入射条件与倾向性机制变量。", "折射、反射、吸收分别有自己的 feature set，再通过 softmax 归一化。"],
            "实现光学结构版响应函数",
        ),
        (
            "等会吧，我说一下我的看法，我觉得RGB三通道还是和光的吸收反射折射有点区别的，一个是材质响应一个是能量守恒。我的意思是RGB通道响应有点和人流聚集的光（不同波长）有些类似（还是说和OU属性进入反射折射吸收类似？），刚好不同城市来源的人具有不同“波长”，可以体现路径记忆，但是如果再单开一个人才光谱属性再加上OU都考虑的路径记忆是不是重复了，毕竟人才光谱属性也就是由OU属性来的？我现在逻辑有点乱了，主要是人才光谱和OU属性的区分问题？",
            ["澄清 RGB/波长属于光束自身属性，不等于反射/吸收/折射响应。", "区分四层：O 光源属性、U 介质属性、O-U 入射条件、人才光谱属性。", "建议第一版暂不做人才光谱，避免与 O-U 路径记忆重复；后续可用高校类型、专业结构、学校层级扩展。"],
            "人才光谱概念被后置",
        ),
        (
            "这样吧，第一版先不做这个人才光谱，可以放到后续实验里面再做。先说你刚刚说的那些变量怎么分配的那个问题，我刚刚突然想到了一些东西：光入射，然后介质响应，这里响应有几个方法，最直观的应该是什么反射定律，斯涅尔定律，但是城市没有界面人流没有法线，怎么模拟？不好模拟，所以想出别的方法。1是逻辑回归，本质上是概率对不对：但是为什么要用概率呢，我刚刚突然想到量子力学：光是波函数，有概率密度，所以我想模拟这个啊，还能用什么方法还是就用变量分配进逻辑回归？2是我想把它们抽象到高维空间，一个抽象的人流空间，多维属性，具体的我写了一个文档你看看（但是应该还不成熟）。你给我梳理一下这两种思路吧，并且提出一些建议，能不能用，怎么才能用的好",
            ["读取实验设计 PDF，梳理三维人才光场与城市小球材质响应。", "建议量子/概率思路可作为启发，但不宜宣称人才流服从量子力学。", "高维人流空间更适合作为理论底座：城市是高维介质节点，O-U 是高维入射向量，softmax 是概率介质响应的实现。"],
            "从概率响应走向高维人才光场",
        ),
        (
            "可以，你现在记得把我要做的对比实验记下来，我先把设计的有理的主模型做完了，最后再做对比实验。（比如那个什么人才光谱啊，散射啊之类的）\n由于聚合人才光束包含个体异质性，其在高校城市界面处并不对应唯一确定出射路径，而是表现为反射、吸收和透射三种响应状态上的概率分布。这句话说的可以，就这样。\n你现在做主模型吧，有什么不确定的地方吗可以和我交流，确认后再写代码也不迟",
            ["记录实验路线：主模型优先，对比与扩展后置。", "确定关键表述：聚合人才光束包含异质性，因此以概率分布表现响应状态。", "提出主模型确认点：A 类命名、DE 纳入、是否使用 O_code/U_code、是否生成设计表。"],
            "正式确定主模型优先级与核心表述",
        ),
        (
            "1就叫折射吧，没必要分那么细。2就是同一个主模型框架，两个子模块。3用光学结构版，不要用 O_code/U_code。4第二套结构。5可以生成一个主模型说明表。然后我的要求是你可以在coding这个文件夹内新建一个完整的主模型文件夹然后进行实验（感觉这样干净一点），为此你需要我做什么？比如帮你把主表和城市地理坐标表复制过来？还是说你认为在原来的文件夹中继续做，毕竟有一个代码是分析了坐标匹配的。",
            ["新建 main_ou_medium_response 文件夹，形成干净主模型。", "不复制 500MB 主表，通过参数引用旧数据路径；坐标表暂时不用。", "输出主模型预测表、指标表、模型设计表与系数表。"],
            "主模型独立文件夹与代码落地",
        ),
        (
            "结果怎么样呢",
            ["总结主模型结果：ABC 总体比例贴近，单条 O-U 主导准确率约 0.723，三类 MAE 约 7.6–8.2 个百分点。", "DE 子模块较稳，D/E MAE 约 3.3 个百分点。", "判断：主模型可用，解释性强于普通 logit，但 ABC 拟合能力有提升空间。"],
            "评估主模型当前表现",
        ),
        (
            "我现在想跟你讨论这个三维光场啊，既然响应函数本质上还是逻辑回归，感觉和我建立这个三维的也没关系啊，我二维四维好像都可以，因为响应函数是数学方法，无法符合物理逻辑。还是说我的折射之后才是要考虑“真实光路”？，那么响应函数在里面又是个什么作用？我折射的之后怎么做？",
            ["澄清响应函数与三维光场的分工：响应函数决定 FOU 分成多少反射/吸收/折射，三维光场主要服务于折射后的出射路径。", "提出流程：O->U 入射，U 处分解，A 类进入三维光场折射/散射核。", "折射后计算 U->D 距离、O->U 与 U->D 转折角、z_D-z_U 高度跃迁、目的地熵。"],
            "三维光场的位置被重新摆正",
        ),
        (
            "对的你说的对，关键点一是介质响应函数，现在我觉得我们做的还可以了；接下来主要考虑折射这部分，也就是整个模型中最重要的看起来最能体现我们思想的部分了。你现在给出的方法仍然是折射/散射核数学方法，我在想我们干脆从本质上来思考吧：一束光的传播的几个属性是什么（我这边想到了光束大小，光能量，光传播方向，光在物体间传播距离），光是怎么传播的，这和光线追踪有什么联系？你能想到一些灵感或启发吗",
            ["从光束本体重新拆解：起点、方向、强度、传播距离、衰减、介质交互、分裂、散射、命中位置。", "映射到人才流：Ocity、O->U 方向、FOU、距离衰减、响应分裂、A 类散射、Dcity 命中。", "提出折射部分五要素：方向惯性、距离衰减、高度跃迁、目标截面、介质散射。"],
            "从折射核公式退回到光传播本质",
        ),
        (
            "我提一个问题：比如重力模型辐射模型也是由物理规律启发而来的，它们真的是像物理规律那样推到模型从而物理可解释吗？还是说它们内部也是这种回归函数，概率函数？",
            ["解释重力模型与辐射模型也是物理启发而非物理定律原封不动迁移。", "重力模型常被写成 log-linear 或 PPML；辐射模型有机会选择假设但也不是物理粒子辐射。", "为当前模型正名：光学提供结构语言，参数估计和概率函数并不矛盾。"],
            "用重力/辐射模型类比确认物理启发模型定位",
        ),
        (
            "你说的参数可以估计这个问题，因为我在想参数容易受到大背景（各种形势变化而变化或者其他影响）。前面做的我觉得可以了，现在主要是折射那块。A的比例判断出来后，大方向是不同（注意这里的不同的定义）的光束折射到不同的地方。我想用的有两个：光传播有距离衰减和散射。\n然后我突然有了一个想法：刚刚我们一直把各个地区当作同等地位，现在灵光一闪：不同的光首先按省来分类折射到省会城市（过渡，中间过程），然后根据散射（这里有一些因素但我还没想好，有什么字段可以提供？）再到其他地方，不过有个问题：为什么要按省分类？没有特别大的道理，万一到某个省会城市之后映射到其他省的别的城市那我不知道怎么修正了。这个你可以想想，有价值吗？我能想到的价值就是比如研究一些省界对人口流动的影响有点用。然后如果按照这个思路，联系到我的三维分层：我想不按省分类，而是单纯看经济社会（或其他/综合）地位，把Z分成几个阶层。然后再把整个三维空间分为两部分，一部分是入射层，包含发出光线的；一部分是折射层，包含接受光线的，然后折射层那边像一个金字塔（或菱形体）一样顶端城市少，底层（或中层）城市多，先折射到的地方是第一层顶端，被顶端吸收和散射，散射到第二层第三层，再进行吸收和散射，以此类推。我这只是描述一个场景，具体实现需要用代码逻辑实现，有点像递归吧或者是树的结构。\n然后你凭感觉觉得第二种分类法比第一种人人平等法会不会好一点》值得去实现吗？",
            ["讨论省会中转：有价值但不适合作主机制，除非主题是省域边界/省会中转。", "更推荐社会经济高度分层：先折射到高度层，再层内散射命中城市。", "建议路线：人人平等目的地核作 baseline；分层折射核作主增强；递归树/多跳传播作后续理论扩展。"],
            "从省域中转转向三维层级折射",
        ),
        (
            "我说的第一种方法是你给我提的这种，感觉或许也能做？相比分类有什么价值，可以做做试试吗\n\n截图内容：折射之后做 P(D | O,U, refraction)，把 A 类折射流分配给候选 Dcity；F_OUD_hat = FT(O,U) * P(D | O,U,T)；候选 Dcity 满足 D != O、D != U；score(D)=目的地吸引势 - U-D 距离衰减 - 转折角惩罚 + 高度跃迁项 + 区域/城市群通道项。",
            ["确认直接折射/散射核值得做，而且应作为 A 类出射模块 v1。", "说明它的价值：直接预测 Dcity，像光追里的 ray hit；可加入目标截面、距离衰减、方向惯性、高度跃迁、区域通道。", "建议顺序：先做直接核，再做分层核，最后再比较二者。"],
            "当前下一步：直接折射/散射核",
        ),
    ]

    for i, (q, ans, key) in enumerate(turns, start=1):
        add_turn(story, i, q, ans, key, styles)
        if i in {7, 14, 20}:
            story.append(PageBreak())

    story.append(PageBreak())
    section(story, "3. 当前形成的模型体系", styles)
    bullet(
        story,
        [
            "已完成主模型：高维人才光场中的 O-U 条件概率介质响应模型。",
            "主模型包含两个子模块：ABC 外地入射响应；DE 本地逃逸/留存响应。",
            "响应层的任务：决定一束 O-U 光束有多少比例反射、吸收、折射；它不是折射后的真实光路几何。",
            "三维/高维光场的重点位置：A 类折射之后，用于定义 U->D 的方向、距离、高度跃迁、目标截面与散射。",
            "当前下一步优先方向：直接折射/散射核 P(D|O,U,refraction)，先将 A 类折射流分配给候选 Dcity。",
        ],
        styles,
    )

    section(story, "4. 当前代码与结果快照", styles)
    bullet(
        story,
        [
            "主模型代码已落地：main_ou_medium_response/run_main_model.py，并已复制到正式项目 graduate-mobility-ray-tracing。",
            "主模型不使用 O_code/U_code 作为类别记忆，不使用 C_ 去向城市属性。",
            "ABC 外地 O-U 光束：78,767 条；外地入射总流量：8,415,336。",
            "DE 本地 Ucity 光束：324 个；本地总流量：1,609,039。",
            "ABC 总体比例复现较好：折射、反射、吸收预测份额与观测份额接近；局部 O-U 光束主导响应准确率约 0.723。",
            "DE 模块较稳：本地逃逸/留存比例 MAE 约 3.3 个百分点。",
        ],
        styles,
    )

    section(story, "5. 下一步实验建议", styles)
    bullet(
        story,
        [
            "A 类折射光路描述：先利用真实 Dcity 计算转折角、U-D 距离、高度跃迁、向前/侧向/后向折射、向上/平层/向下折射。",
            "直接折射/散射核 v1：P(D|O,U,T)，候选 Dcity 满足 D!=O 且 D!=U；score(D) 使用目标截面、距离衰减、转折角、高度跃迁和区域通道。",
            "分层折射核 v2：先预测高度层 layer，再在层内选择 Dcity；用于检验三维社会高度场是否比人人平等候选竞争更好。",
            "省域/省会中转作为专题实验，不作为主机制；省界更适合作为传播阻尼或通道项。",
            "后续扩展：人才光谱、背景光、发射瓣、递归/多跳层级散射。",
        ],
        styles,
    )

    section(story, "6. 收束判断", styles)
    story.append(
        para(
            "这轮讨论的最大推进，是把模型从“光学语言包装的概率分类”推进到“响应层 + 折射层”的过程系统。"
            "响应函数的职责已经清楚：它是介质界面处的流量分解器。三维光场的价值也被重新定位：它不是为了让响应函数更物理，"
            "而是为了在 A 类折射之后定义真实出射路径、距离衰减、方向转折、高度跃迁、目标截面与散射结构。"
            "因此，下一步最有价值的工作不是继续纠结响应层是否像物理定律，而是开始构造 A 类折射/散射核，并用真实 O-U-D 数据检验折射光路画像。",
            styles["body"],
        )
    )

    doc.build(story)


if __name__ == "__main__":
    build()
    print(OUT)
