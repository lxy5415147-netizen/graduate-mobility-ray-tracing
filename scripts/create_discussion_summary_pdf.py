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
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "2026-06-03_model_discussion_summary.pdf"


def register_fonts() -> tuple[str, str]:
    regular = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
    bold = Path(r"C:\Windows\Fonts\simhei.ttf")
    if not regular.exists():
        regular = Path(r"C:\Windows\Fonts\simsun.ttc")
    if not bold.exists():
        bold = regular
    pdfmetrics.registerFont(TTFont("CN-Regular", str(regular)))
    pdfmetrics.registerFont(TTFont("CN-Bold", str(bold)))
    return "CN-Regular", "CN-Bold"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def section(title: str, styles: dict[str, ParagraphStyle], story: list) -> None:
    story.append(Spacer(1, 0.25 * cm))
    story.append(p(title, styles["h2"]))
    story.append(Spacer(1, 0.08 * cm))


def bullet(items: list[str], styles: dict[str, ParagraphStyle], story: list) -> None:
    for item in items:
        story.append(p("• " + item, styles["body"]))


def build() -> None:
    font, bold_font = register_fonts()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=bold_font,
            fontSize=19,
            leading=25,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName=font,
            fontSize=10.5,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=bold_font,
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#111827"),
            spaceBefore=6,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9.5,
            leading=15,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=font,
            fontSize=8.5,
            leading=13,
            textColor=colors.HexColor("#374151"),
        ),
    }

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=1.65 * cm,
        leftMargin=1.65 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.45 * cm,
        title="模型讨论与主模型阶段总结",
        author="Codex and lxy5415147-netizen",
    )

    story: list = []
    story.append(p("模型讨论与主模型阶段总结", styles["title"]))
    story.append(p("2026-06-03｜graduate-mobility-ray-tracing 项目", styles["subtitle"]))

    section("一、今天讨论的核心问题", styles, story)
    bullet(
        [
            "用户已完成 twod-answer，即根据真实 O-U-D 数据复现 A/B/C/D/E 表格；下一步从观测复现转向预测建模。",
            "最初讨论了三层模型：入射光束生成、介质响应、折射/散射目的地核。当前阶段决定优先完成“有路径记忆的 O-U 条件介质响应主模型”。",
            "明确 ABC 与 DE 的分母不同：ABC 使用外地入射流 F_external=A+B+C；DE 使用本地流 F_local=D+E，不能混用外地入射流作 DE 分母。",
            "围绕光学类比进行了边界澄清：光线追踪提供过程框架和守恒式分解，logit/softmax 是估计概率响应的工具，而不是声称人类迁移严格服从物理光学定律。",
        ],
        styles,
        story,
    )

    section("二、遇到的主要概念问题与处理", styles, story)
    data = [
        ["问题", "讨论后的处理"],
        [
            "ABC 与 DE 是否属于同一个模型？",
            "属于同一个主模型框架，但分成两个子模块：外地入射 ABC 响应；本地 DE 留存/逃逸响应。",
        ],
        [
            "Dcity / C_ 属性能不能用于 ABC 预测？",
            "当前响应层不能用。ABC 预测时 Dcity 尚未被假设已知，因此不使用任何 C_ 去向城市属性。",
        ],
        [
            "为什么用概率？",
            "聚合人才光束包含个体异质性，在高校城市界面处不对应唯一确定出射路径，而表现为反射、吸收、折射三种状态的概率分布。",
        ],
        [
            "RGB / 人才光谱是否纳入第一版？",
            "暂不纳入。人才光谱可作为后续扩展实验；第一版先使用 O/U 属性与 O-U 路径关系表达路径记忆。",
        ],
        [
            "是否使用 O_code / U_code？",
            "主模型不使用城市代码作为类别记忆，避免变成城市 ID 拟合；路径记忆由距离、同省/同区域、等级差、属性差和势差表示。",
        ],
    ]
    table = Table([[p(str(c), styles["small"]) for c in row] for row in data], colWidths=[5.1 * cm, 11.3 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)

    section("三、当前主模型变成什么样", styles, story)
    story.append(
        p(
            "当前主模型定义为：<b>高维人才光场中的 O-U 条件概率介质响应模型</b>。城市被视为具有属性的介质节点，O-U 光束由来源城市属性、"
            "高校城市属性和二者之间的路径关系共同刻画；高校城市界面处通过概率响应函数将同一束流分解为若干响应通道。",
            styles["body"],
        )
    )
    bullet(
        [
            "外地生响应：F_OU = F_refraction + F_reflection + F_absorption。",
            "本地生响应：F_local = F_escape + F_retention。",
            "概率形式：P_refraction + P_reflection + P_absorption = 1；P_escape + P_retention = 1。",
            "计算形式：每个响应通道有自己的响应势 S，再通过 softmax 转成概率，保证同一束流的守恒式分解。",
        ],
        styles,
        story,
    )

    section("四、变量如何进入响应通道", styles, story)
    data2 = [
        ["响应通道", "进入变量/机制项", "解释"],
        [
            "折射 refraction",
            "U_gateway_openness、potential_gradient、等级差、中心性差、第三产业差、距离、跨区域、入射强度",
            "高校城市作为门户或跳板，将外地人才导向第三城市。",
        ],
        [
            "反射 reflection",
            "O_return_pull、同省、同区域、距离、U-O 房价压力、等级差、GDP 梯度、U 吸收不足、入射强度",
            "来源地牵引、区域记忆或高校城市吸收不足使人才回到家乡。",
        ],
        [
            "吸收 absorption",
            "U_absorption_capacity、potential_gradient、U 等级、GDP 梯度、人口梯度、房价压力、同省、入射强度",
            "高校城市凭借经济、产业、教育和中心性将外地人才留在本地。",
        ],
        [
            "本地逃逸 local_escape",
            "U 房价、AQI、门户开放度、U 等级、入射强度",
            "本地生从高校城市离开到其他城市。",
        ],
        [
            "本地留存 local_retention",
            "U_absorption_capacity、U GDP、第三产业、U 等级、医疗、教育、房价、入射强度",
            "本地生毕业后继续留在高校城市。",
        ],
    ]
    table2 = Table([[p(str(c), styles["small"]) for c in row] for row in data2], colWidths=[3.4 * cm, 7.3 * cm, 5.7 * cm])
    table2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table2)

    section("五、当前代码与实验结果", styles, story)
    bullet(
        [
            "正式 GitHub 项目路径：coding/github-repos/graduate-mobility-ray-tracing。",
            "主模型脚本：run_main_model.py；说明文件：README.md；忽略规则：.gitignore。",
            "主模型输出目录：main_ou_medium_response/outputs/main_model（结果不上传 GitHub）。",
            "全量样本：外地 ABC O-U 光束 78,767 条，外地入射总流量 8,415,336；本地 DE Ucity 光束 324 个，本地流量 1,609,039。",
            "ABC 总体比例贴近观测：折射观测 0.2714 / 预测 0.2709；反射观测 0.3200 / 预测 0.3182；吸收观测 0.4086 / 预测 0.4109。",
            "ABC 单条光束层面主导类型准确率约 0.723，三类比例 MAE 约 7.6 至 8.2 个百分点。",
            "DE 表现较稳：本地逃逸观测 0.1582 / 预测 0.1531；本地留存观测 0.8418 / 预测 0.8469；D/E 比例 MAE 约 3.3 个百分点。",
        ],
        styles,
        story,
    )

    section("六、后续实验路线", styles, story)
    bullet(
        [
            "主模型优先：继续完善当前 O-U 条件概率介质响应模型，调整响应函数与变量分配。",
            "对比实验后置但必须做：全局平均响应模型、Ucity-only 介质模型，用来检验路径记忆是否有效。",
            "人才光谱后续再做：高校档次、学校类型、专业结构等可作为“人才光谱属性”，当前第一版暂不引入。",
            "散射/目的地熵后续再做：在 A 类折射流之后追踪 Dcity，计算方向偏转、目的地熵、聚焦型/发散型介质。",
            "背景光后续再做：加入全国平均目的地吸引力 B(D)，检验宏观环境项对模型稳定性的影响。",
            "发射瓣后续再做：从给定 F_OU 扩展到模拟 P(U|O)，解释来源城市如何向不同高校城市发射人才光束。",
        ],
        styles,
        story,
    )

    section("七、GitHub 使用过程与后续工作方式", styles, story)
    bullet(
        [
            "已创建空仓库：lxy5415147-netizen/graduate-mobility-ray-tracing。",
            "使用 GitHub Desktop 克隆空仓库到 coding/github-repos/graduate-mobility-ray-tracing。",
            "只上传主模型代码与 README，不上传数据和 outputs 结果。当前 .gitignore 已排除 outputs、CSV、缓存、虚拟环境和 IDE 文件。",
            "以后正式开发都放在 github-repos/graduate-mobility-ray-tracing 里；旧的 talent_ray_2d_calibration 保留作数据、历史代码和坐标匹配参考。",
            "日常流程：修改代码 → 运行验证 → GitHub Desktop 中写 Summary → Commit to main → Push origin。",
            "建议 commit 信息按实验阶段写，例如 Add Ucity-only comparison model、Add scattering entropy experiment、Refactor medium response features。",
        ],
        styles,
        story,
    )

    section("八、当前阶段结论", styles, story)
    story.append(
        p(
            "第一版主模型已经从“观测复现”推进到“可运行的 O-U 条件概率介质响应模型”。它没有使用 Dcity 去向属性，也没有依赖 O_code/U_code 记忆，"
            "而是用高维城市属性、路径关系和响应通道结构来表达路径记忆与介质响应。当前结果显示总体比例复现较好，DE 子模块较稳，ABC 局部光束预测仍有提升空间。"
            "下一步应围绕响应函数结构、变量分配和后续对比实验继续迭代。",
            styles["body"],
        )
    )

    doc.build(story)


if __name__ == "__main__":
    build()
    print(OUT)
