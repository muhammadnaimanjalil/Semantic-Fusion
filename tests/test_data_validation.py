from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
from openpyxl import Workbook

from data_validation import inspect_excel_sheet, validate_dataframe


def test_dataframe_validation_distinguishes_errors_and_warnings():
    frame = pd.DataFrame(
        {
            "feature": [0.0, np.inf, 2.0],
            "text": ["valid", "#NAME?", None],
            "constant": ["x", "x", "x"],
        }
    )
    report = validate_dataframe(frame)
    error_codes = {issue.code for issue in report.errors}
    warning_codes = {issue.code for issue in report.warnings}

    assert {"infinite_values", "spreadsheet_errors"} <= error_codes
    assert {"zero_values", "missing_values", "constant_columns"} <= warning_codes
    assert not report.can_proceed


def test_excel_formula_scan_blocks_formula_cells():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Data"
    sheet.append(["text", "score"])
    sheet.append(["valid", 1])
    sheet.append(["=- review text interpreted as a formula", 0])
    buffer = BytesIO()
    workbook.save(buffer)

    report = inspect_excel_sheet(buffer.getvalue(), "Data")

    assert {issue.code for issue in report.errors} == {"excel_formulas"}
    assert report.errors[0].examples == ("A3",)
