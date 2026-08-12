"""
merge_fluo_suite_sources.py

Combine the two fluo_suite spreadsheet updates into a single reference
workbook for jump-operator construction:

  - circ_sim/data/fluo_suite/ORCA_NMR_summary_Jcorrected.xlsx
        -> corrected 'J-couplings (1J & 2J)' sheet + 'Corrections log'
  - circ_sim/data/fluo_suite/ORCA_NMR_summary - full shielding.xlsx
        -> 'Shielding tensors (full 3x3)' sheet

'Overview', 'Atomic coordinates', 'F neighbours (<5A)', and the narrow
per-isotope isotropic/anisotropy shielding sheets are identical across
both source files (verified by direct comparison), so they are carried
over unchanged from the J-corrected file.

Output: circ_sim/data/fluo_suite/ORCA_NMR_summary_merged.xlsx
"""

import os
import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'fluo_suite'
)
DATA_DIR = os.path.normpath(DATA_DIR)

PATH_JCORR = os.path.join(DATA_DIR, 'ORCA_NMR_summary_Jcorrected.xlsx')
PATH_FULLSHIELD = os.path.join(DATA_DIR, 'ORCA_NMR_summary - full shielding.xlsx')
PATH_OUT = os.path.join(DATA_DIR, 'ORCA_NMR_summary_merged.xlsx')

HEADER_FONT = Font(bold=True)
BAND_FILL = PatternFill(start_color='FFEAF1F8', end_color='FFEAF1F8', fill_type='solid')


def copy_sheet(src_ws, dst_wb, title, band_from_row=None):
    """Copy all cell values from src_ws into a new sheet in dst_wb, with
    bold header rows (rows whose first cell is a string and not a data row)
    and optional zebra banding from `band_from_row` onward."""
    dst_ws = dst_wb.create_sheet(title=title)
    for row in src_ws.iter_rows(values_only=True):
        dst_ws.append(row)

    # Bold any leading title/description/header rows (heuristic: rows before
    # the first fully-populated data row, detected via band_from_row).
    if band_from_row is not None:
        for r in range(1, band_from_row):
            for c in range(1, src_ws.max_column + 1):
                dst_ws.cell(row=r, column=c).font = HEADER_FONT
        for r in range(band_from_row, dst_ws.max_row + 1):
            if (r - band_from_row) % 2 == 1:
                for c in range(1, src_ws.max_column + 1):
                    dst_ws.cell(row=r, column=c).fill = BAND_FILL

    for c in range(1, src_ws.max_column + 1):
        col_letter = get_column_letter(c)
        max_len = max(
            (len(str(cell.value)) for cell in dst_ws[col_letter] if cell.value is not None),
            default=8,
        )
        dst_ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

    dst_ws.freeze_panes = f'A{band_from_row}' if band_from_row else None
    return dst_ws


def main():
    wb_jcorr = openpyxl.load_workbook(PATH_JCORR, data_only=True)
    wb_fullshield = openpyxl.load_workbook(PATH_FULLSHIELD, data_only=True)

    out = openpyxl.Workbook()
    out.remove(out.active)

    # --- Sheets carried over unchanged (identical in both sources) ---
    copy_sheet(wb_jcorr['Overview'], out, 'Overview', band_from_row=5)
    copy_sheet(wb_jcorr['1H shieldings'], out, '1H shieldings', band_from_row=5)
    copy_sheet(wb_jcorr['13C shieldings'], out, '13C shieldings', band_from_row=5)
    copy_sheet(wb_jcorr['19F shieldings'], out, '19F shieldings', band_from_row=5)

    # --- Full 3x3 Zeeman/shielding tensors (from the full-shielding file) ---
    copy_sheet(
        wb_fullshield['Shielding tensors (full 3x3)'], out,
        'Shielding tensors (full 3x3)', band_from_row=5,
    )

    # --- Corrected J-couplings + provenance (from the J-corrected file) ---
    copy_sheet(wb_jcorr['J-couplings (1J & 2J)'], out, 'J-couplings (1J & 2J)', band_from_row=5)
    copy_sheet(wb_jcorr['Corrections log'], out, 'Corrections log', band_from_row=5)

    copy_sheet(wb_jcorr['F neighbours (<5A)'], out, 'F neighbours (<5A)', band_from_row=5)
    copy_sheet(wb_jcorr['Atomic coordinates'], out, 'Atomic coordinates', band_from_row=5)

    # --- Provenance / usage notes ---
    notes = out.create_sheet('Merge notes', 0)
    notes_rows = [
        ('Merged fluo_suite reference workbook — provenance and usage notes',),
        (),
        ('Purpose:',
         'Single reference workbook for downstream jump-operator construction '
         '(cf. circ_sim/scripts/linblad_dyn/gemcitabine_jump_operators.py), combining '
         'the corrected J-coupling data and the full 3x3 shielding tensors that were '
         'previously only available in two separate spreadsheet updates.'),
        (),
        ('Sources merged:',),
        ('  - ORCA_NMR_summary_Jcorrected.xlsx',
         "-> 'J-couplings (1J & 2J)' sheet (Recommended J iso column) and 'Corrections log'"),
        ('  - ORCA_NMR_summary - full shielding.xlsx',
         "-> 'Shielding tensors (full 3x3)' sheet"),
        ('  - Overview / Atomic coordinates / F neighbours / narrow isotropic shielding sheets',
         'identical in both sources (verified by direct row-by-row comparison); carried over '
         'from the J-corrected file.'),
        (),
        ('Usage notes for jump-operator construction:',),
        ('1. J-couplings:',
         "Use the 'Recommended J iso (Hz)' column, not the raw 'nJ iso (Hz)' ORCA column — "
         "see 'Corrections log' for which coupling classes were replaced (1J(C-F), CF2 2J(F-F)) "
         "vs retained as-is."),
        ('2. Still missing:',
         "The J-coupling sheet remains capped at 1-bond/2-bond pairs only (see the sheet "
         "description). Longer-range couplings (3J, 4J vicinal/W-couplings), which are "
         "non-negligible in the original gemcitabine_jump_operators.py reference system "
         "(values up to ~10 Hz), are not present in this dataset for any molecule."),
        ('3. Shielding tensor symmetry:',
         "The 'Shielding tensors (full 3x3)' sheet stores the raw ORCA lab-frame GIAO tensor, "
         "which is generally NOT symmetric (sXY != sYX etc.). Do NOT symmetrize it before use. "
         "Per liouville_hilbert_basis.tex (Sec. 'Zeeman anisotropy dephasing', eqs. sigma_{l,m}) "
         "and its implementation in linblad_dyn/utils/linblad_utils.py "
         "(_sigma_lm/build_QZ_ops/build_jump_operators), the antisymmetric part of the tensor "
         "generates a physical rank-1 (l=1) dephasing channel (9 additional jump operators, "
         "scale sqrt(2 tau_c/3)) that is dropped only if the tensor happens to be symmetric "
         "(as it is for gemcitabine — that is a property of that molecule's tensor, not an "
         "assumption to impose generally). Pass the raw, non-symmetric sXX..sZZ block directly "
         "to build_jump_operators()."),
        ('4. Frame consistency:',
         "The shielding tensor axes and the 'Atomic coordinates' sheet are in the same "
         "ORCA-native molecular frame for a given (System, Atom Index), so a_2m (dipolar, from "
         "coordinates) and sigma_2m (CSA, from the shielding tensor) can be combined directly "
         "without any additional rotation, matching the Q_CSA + Q_dip construction."),
        (),
        ('Not yet addressed / open items:',),
        ('  - Long-range (3J+) J-couplings — see note 2.',),
        ('  - eta / PAS confidence tiering from ORCA_NMR_summary_corrected_v2.xlsx was NOT '
         'merged here', 'this workbook uses the full non-symmetric tensor directly instead, '
         'which supersedes the eta-based reconstruction (no information loss).'),
    ]
    for row in notes_rows:
        notes.append(row)
    notes['A1'].font = Font(bold=True, size=13)
    for r in (3, 5, 10):
        notes.cell(row=r, column=1).font = HEADER_FONT
    notes.column_dimensions['A'].width = 45
    notes.column_dimensions['B'].width = 110

    out.save(PATH_OUT)
    print(f'Saved merged workbook -> {PATH_OUT}')
    print(f'Sheets: {out.sheetnames}')


if __name__ == '__main__':
    main()
