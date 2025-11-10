# Synapse Labeller (PySide6)

A fast keyboard-driven GUI to label synapse images stored in a directory, with optional batch selection using a master image and per-pixel source mapping.

This app was designed for very rapid annotation via hotkeys and streamlined navigation.

## Key Features

- Directory-based workflow; one row per source-ID (derived from image stem)
- Auto-create and sync CSV (`synapse_labels.csv`) with image set
- Custom table headers via `table_headers.json`
- Per-directory hotkeys via `.synapse_hotkeys.json` (JSON-only)
- Global and per-column hotkeys; batch application supported
- Batch mode with master image, polygon (lasso) selection, and overlay highlighting
- Brightness control in batch mode (vmin/vmax sliders)
- Quick value entry (Ctrl+Shift+V), Notes (Ctrl+N), Soma-depth (Ctrl+D)
- Add columns dynamically (Ctrl+Shift+C)
- Toggle table view, jump to ID, filter unlabeled, save at any time

---

## Installation

Requirements (Python 3.8+ recommended):

- PySide6
- numpy

```
pip install PySide6 numpy
```

No other packages are required.

---

## Launching

- From the project root (or anywhere), run:

```
python synapse_roladex/synapse_labeller.py
```

- Optional: pass the image directory as the first CLI argument to auto-load:

```
python synapse_roladex/synapse_labeller.py "/path/to/your/images"
```

If a directory is not passed, use “Open Directory…” in the toolbar.

---

## Directory Layout and Files

In an image directory, the following files are used/created:

- Images: one PNG per source, e.g. `12345.png`. The stem (e.g. `12345`) is the source-ID row key.
- synapse_labels.csv: the labels table. Auto-created on first open; synced to your images thereafter.
- table_headers.json: list of column names. Auto-created on first open if missing (see defaults below).
- .synapse_hotkeys.json: per-directory hotkey configuration. Auto-created on first open with defaults.
- Optional for batch mode:
  - master_image.png: a grayscale reference image with H×W matching the label map.
  - source_location_key.npz (preferred) with arrays:
    - label_map: int array (H, W), each pixel is an index into id_list; -1 means unused
    - id_list: array/list of strings; `id_list[i]` is the source-ID name for label `i`
  - Fallback: `label_map.npy` + `source_ids.txt` (one ID per line)

CSV sync behavior:
- New images (new stems) are added as new rows (with empty values)
- Rows whose images disappeared are removed
- Extra CSV columns are preserved; unknown columns in CSV are appended to the model

---

## Table Columns (Headers)

Headers are defined in `table_headers.json`. On first open, the file is created with the defaults:

- `source-ID` (always first and derived from filename; not editable)
- `synapse-type`
- `soma-ID`
- `soma-depth`
- `dend-type`
- `dend-ID`
- `notes`

You can edit this file to customize columns before opening the directory in the app, or use the in-app “Add Column” (Ctrl+Shift+C) to append a new column. New columns are added for all existing rows with empty values and saved to CSV.

---

## Hotkeys Overview

Hotkeys are defined per-directory in `.synapse_hotkeys.json` and are JSON-only. Two types:

- Global hotkeys (highest priority): a single key sets a specific column/value regardless of active column
- Per-column hotkeys: a single key sets a value for the active column

On first open (if missing), `.synapse_hotkeys.json` is created with a set of global defaults, for example:

```json
{
  "active_column": "synapse-type",
  "auto_advance": true,
  "global": {
    "p": { "column": "synapse-type", "value": "spine" },
    "s": { "column": "synapse-type", "value": "shaft" },
    "c": { "column": "synapse-type", "value": "soma" },
    "x": { "column": "synapse-type", "value": "axon" },
    "1": { "column": "soma-ID", "value": "soma1" },
    "2": { "column": "soma-ID", "value": "soma2" },
    "3": { "column": "soma-ID", "value": "soma3" },
    "4": { "column": "soma-ID", "value": "soma4" },
    "a": { "column": "dend-type", "value": "apical" },
    "b": { "column": "dend-type", "value": "basal" }
  },
  "columns": {}
}
```

You can edit this file directly or use “Edit Hotkeys…” (Ctrl+H) to manage both global and per-column mappings.

- Global mappings are applied first
- If no global mapping matches, per-column mapping for the active column is used
- Single-character keys only

Tip: you can add more keys (e.g., `u` → `synapse-type: unclear`, `8`/`9`/`q` for special `soma-ID`s, etc.).

Reload hotkeys without reopening via “Reload Hotkeys” (Ctrl+R).

---

## Batch Mode (Master Image + Lasso)

Batch mode lets you select multiple source-IDs by lassoing a region on a master image.

- Load the master/key via the toolbar (“Load Master/Key”) if `master_image.png` and the key (`source_location_key.npz` or `label_map.npy` + `source_ids.txt`) exist
- Toggle “Batch Mode” (Ctrl+B) to switch the central view to the master image
- Draw a polygon by clicking/dragging; the cursor becomes a crosshair while lassoing
- Double-click or press Enter to finalize the lasso; Esc cancels
- Selected IDs are highlighted in red; the toolbar shows “Sel: N”
- Any hotkey/value operation applies to all selected IDs if batch mode is active

Brightness/contrast (windowing):
- Two sliders (Min/Max) adjust vmin/vmax (0–255); useful to improve visibility for dim images

---

## Normal (Single-Image) Mode

- Central panel shows the current synapse image from the directory
- Use the right docked table to view/edit values; double-click a row to load that image

---

## Toolbar and Controls

- Open Directory… (or pass on CLI)
- Prev / Next (Left/Right Arrow)
- Save (Ctrl+S)
- Toggle Table (Ctrl+T)
- Edit Hotkeys… (Ctrl+H)
- Shortcut Help (F1)
- Reload Hotkeys (Ctrl+R)
- Add Column (Ctrl+Shift+C)
- Batch Mode (Ctrl+B)
- Load Master/Key
- Jump: type a source-ID and press Enter
- Column: active column selector (used by per-column hotkeys)
- Auto-advance: move to next row after a value is assigned (single mode)
- Unlabeled only: navigation skips to rows with any missing value
- Sel: current selection count (batch mode)
- Min/Max sliders (batch): windowing controls (vmin/vmax)

---

## Keyboard Reference

Navigation:
- Left Arrow: previous row
- Right Arrow: next row
- Ctrl+T: toggle table panel
- Jump box: type ID and Enter to jump

Hotkeys and value entry:
- Single-character key: assign via global mapping (if any), else assign value for active column via per-column mapping
- Backspace/Delete: clear value for the active column (single row or batch selection)
- Ctrl+Shift+V: “Enter value” dialog → choose column (dropdown), then enter text value; applies to selection or current row
- Ctrl+N: enter notes text; applies to selection or current row (column: `notes`)
- Ctrl+D: enter numeric soma depth; applies to selection or current row (column: `soma-depth`)
- n: enter dend-ID text; creates the `dend-ID` column if missing, then applies to selection or current row (reserved; not used for mapping)

Batch mode:
- Ctrl+B: toggle batch mode
- In batch: click/drag to draw polygon; double-click or Enter to finalize; Esc to cancel
- Min/Max sliders: adjust visualization windowing

Hotkeys management:
- Ctrl+H: edit hotkeys (global and per-column)
- Ctrl+R: reload hotkeys from disk

Saving:
- Ctrl+S: save CSV immediately

---

## Workflow Tips

- Use global hotkeys for the most common assignments, so you don’t have to change the active column
- Use Auto-advance in single mode to annotate quickly in sequence
- Use “Unlabeled only” to navigate only through items needing input
- Add ad-hoc columns with “Add Column” if new annotation categories are needed

---

## Troubleshooting

- “Failed to open directory”: ensure the path exists and contains `.png` images
- Batch won’t load: ensure `master_image.png` exists and dimensions match the `label_map` (H×W)
- “label_map contains indices not covered by id_list length”: your `id_list` is shorter than the maximum index in `label_map`
- No hotkeys seem to work: check `.synapse_hotkeys.json`; ensure keys are single characters and columns match header names (case-sensitive)
- Global key conflicts: remember global mappings take precedence over per-column mappings

---

## Data Persistence Details

- CSV (`synapse_labels.csv`): all values saved as strings; empty means unset
- Headers (`table_headers.json`): the column list; `source-ID` is always first and read-only
- Hotkeys (`.synapse_hotkeys.json`): JSON with `global`, `columns`, `active_column`, and `auto_advance`

You can edit these files directly if the app is closed. Use “Reload Hotkeys” to refresh mappings while the app is open.

---

## Developer Notes

- UI: PySide6 (Qt) with a central stacked widget (image viewer vs batch master)
- Model: `LabelTableModel` wraps the CSV-backed `DataStore`
- Batch: `BatchGraphicsView` handles lasso drawing, selection rasterization, overlay composition, and windowing
- Windowing is applied on a grayscale copy of the master image; overlay is composited separately

---

## License

Internal research tool; adapt as needed.
