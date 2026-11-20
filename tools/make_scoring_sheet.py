"""把 qa_eval.csv 转成可直接打分的 Excel 工作表。

原始 CSV 列太多、回答列极宽，在 Excel 里几乎没法看；这里只保留
"题号 / 问题 / 策略 / 回答 + 三列分数"，并加评分锚点与自动统计。

用法：python tools/make_scoring_sheet.py
输出：outputs/metrics/qa_scoring.xlsx
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "metrics" / "qa_eval.csv"
OUT = ROOT / "outputs" / "metrics" / "qa_scoring.xlsx"

ORDER = ["零样本", "少样本", "思维链"]
HEADERS = ["题号", "问题", "策略", "回答（评分对象）", "准确性(1-5)", "完整性(1-5)", "可读性(1-5)"]
WIDTHS = [6, 34, 9, 82, 14, 14, 14]

HEAD_FILL = PatternFill("solid", fgColor="24435C")
GROUP_FILL = PatternFill("solid", fgColor="F2F6FB")
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def build_sheet(wb: Workbook, df: pd.DataFrame) -> None:
    ws = wb.active
    ws.title = "评分表"
    ws.append(HEADERS)
    for c in range(1, len(HEADERS) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True, color="FFFFFF", size=11)
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
        ws.column_dimensions[get_column_letter(c)].width = WIDTHS[c - 1]
    ws.row_dimensions[1].height = 30

    df = df.copy()
    df["_o"] = df["strategy"].map({s: i for i, s in enumerate(ORDER)})
    df = df.sort_values(["qid", "_o"])

    row = 2
    for n, (qid, grp) in enumerate(df.groupby("qid", sort=True)):
        start = row
        for _, r in grp.iterrows():
            ws.cell(row=row, column=1, value=int(qid)).alignment = Alignment(horizontal="center", vertical="top")
            ws.cell(row=row, column=2, value=r["question"]).alignment = Alignment(vertical="top", wrap_text=True)
            ws.cell(row=row, column=3, value=r["strategy"]).alignment = Alignment(horizontal="center", vertical="top")
            ws.cell(row=row, column=4, value=r["answer"]).alignment = Alignment(vertical="top", wrap_text=True)
            for c in range(1, 8):
                ws.cell(row=row, column=c).border = BORDER
                if n % 2 == 1:
                    ws.cell(row=row, column=c).fill = GROUP_FILL
            ws.row_dimensions[row].height = 78
            row += 1
        # 同一题的三行合并问题列，视觉上更像一张评分卡
        ws.merge_cells(start_row=start, start_column=2, end_row=row - 1, end_column=2)

    ws.freeze_panes = "A2"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.auto_filter.ref = f"A1:G{row - 1}"

    dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True,
                        showErrorMessage=True, errorTitle="只接受 1–5", error="请输入 1 到 5 的整数")
    ws.add_data_validation(dv)
    dv.add(f"E2:G{row - 1}")

    # 分数上色：1–2 红、3 黄、4–5 绿；空单元格不上色
    rng = f"E2:G{row - 1}"
    ws.conditional_formatting.add(rng, FormulaRule(formula=['AND($E2<>"",$E2<=2)'],
                                                   fill=PatternFill("solid", fgColor="FFC7CE")))
    ws.conditional_formatting.add(rng, FormulaRule(formula=['AND($E2<>"",$E2=3)'],
                                                   fill=PatternFill("solid", fgColor="FFEB9C")))
    ws.conditional_formatting.add(rng, FormulaRule(formula=['AND($E2<>"",$E2>=4)'],
                                                   fill=PatternFill("solid", fgColor="C6EFCE")))


def build_rubric(wb: Workbook) -> None:
    ws = wb.create_sheet("评分标准")
    ws.column_dimensions["A"].width = 12
    for col in "BCD":
        ws.column_dimensions[col].width = 46

    ws["A1"] = "评分锚点（1 / 3 / 5 分）"
    ws["A1"].font = Font(bold=True, size=13)

    rows = [
        ("维度", "1 分", "3 分", "5 分"),
        ("准确性", "关键数字或政策说错，会误导用户", "要点基本正确，但有个别表述不严谨", "数字与条款完全正确，且标明依据"),
        ("完整性", "只答了一半，缺少关键条件", "覆盖主要条款，遗漏次要条件", "覆盖全部适用条款，该说的条件都说了"),
        ("可读性", "结构混乱、口语化严重、无法直接使用", "能读懂，但需要用户自己再整理", "可直接发给用户：分点、简洁、无冗余"),
    ]
    for r in rows:
        ws.append(r)
    for c in ws[2]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = HEAD_FILL
        c.alignment = Alignment(horizontal="center", vertical="center")

    ws.append([])
    ws.append(["评分须知"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=12)
    notes = [
        "1. 只评 D 列「回答」；F 列的模型推理过程（reasoning）不计入分数。",
        "2. 每题三行是同一问题的三种策略，建议先横向比较再打分，避免分数漂移。",
        "3. 同分时以「准确性」优先；数字错但写得漂亮，不应高于 3 分。",
        "4. 满分 5 分不要求完美，只要可直接发给用户即可。",
        "5. 填完后到「统计」页看均值；也可把表发回，我汇总进 summary.md。",
    ]
    for t in notes:
        ws.append([t])
        ws.cell(row=ws.max_row, column=1).alignment = Alignment(wrap_text=False)


def build_stats(wb: Workbook, n_rows: int) -> None:
    ws = wb.create_sheet("统计")
    ws.column_dimensions["A"].width = 14
    for col in "BCDE":
        ws.column_dimensions[col].width = 14
    ws.append(["策略", "准确性均值", "完整性均值", "可读性均值", "综合均值"])
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = HEAD_FILL
        c.alignment = Alignment(horizontal="center")
    for i, s in enumerate(ORDER):
        r = i + 2
        ws.cell(row=r, column=1, value=s)
        for col, letter in ((2, "E"), (3, "F"), (4, "G")):
            ws.cell(row=r, column=col,
                    value=f'=IFERROR(AVERAGEIF(评分表!$C$2:$C${n_rows},$A{r},评分表!{letter}$2:{letter}${n_rows}),"")')
        ws.cell(row=r, column=5, value=f'=IFERROR(AVERAGE(B{r}:D{r}),"")')
    ws.append([])
    ws.append(["已填行数", f'=COUNT(评分表!E2:E{n_rows})'])
    ws.append(["总行数", n_rows - 1])


def main() -> None:
    df = pd.read_csv(SRC)
    wb = Workbook()
    build_sheet(wb, df)
    build_rubric(wb)
    build_stats(wb, len(df) + 1)
    wb.save(OUT)
    print(f"[ok] {OUT} | {len(df)} 行 × 3 列分数")

    # 回读校验
    from openpyxl import load_workbook
    wb2 = load_workbook(OUT)
    ws = wb2["评分表"]
    print("工作表:", wb2.sheetnames, "| 行数:", ws.max_row, "| 列数:", ws.max_column)
    print("首行:", [ws.cell(row=1, column=c).value for c in range(1, 8)])
    print("首条回答前 40 字:", str(ws.cell(row=2, column=4).value)[:40])


if __name__ == "__main__":
    main()
