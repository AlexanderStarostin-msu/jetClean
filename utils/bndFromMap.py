#!/usr/bin/env python3

# Requirements:
# - Without --centralize, read 0/mapOfBnd. It maps patch names to field names
#   and OpenFOAM boundary-condition entries, for example:
#       inlet:
#         U:
#           type fixedValue;
#           value uniform (1 0 0);
# - Update conventional OpenFOAM field files in 0/, changing only the
#   specified patch/field boundary conditions and preserving all other data.
# - Create a missing field file using the built-in field-name dimensions table,
#   a conventional FoamFile header, and a zero internalField matching its type.
#   Unknown field names cause an error rather than guessed dimensions.
# - With --centralize, collect boundary conditions from field files in 0/ and
#   create 0/mapOfBnd. If mapOfBnd already exists, stop without changing it.
# - Entries absent from mapOfBnd are preserved; removal is not supported and
#   will produce a warning.
# - Report updated fields and newly created field files. Add one UTC timestamp
#   comment per updated field file immediately after "boundaryField {".

# Usage
#   Run from the OpenFOAM case directory:
#   ./bndFromMap.py
#   ./bndFromMap.py --centralize

# Keep top section with requirements and usage, so that it can be extracted by the documentation generator.

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys
import tempfile


@dataclass(frozen=True)
class FieldSpec:
    foam_class: str
    dimensions: str
    zero: str


# Common OpenFOAM field names; add entries here when a case uses other fields.
FIELD_SPECS = {
    "U": FieldSpec("volVectorField", "[0 1 -1 0 0 0 0]", "(0 0 0)"),
    "UMean": FieldSpec("volVectorField", "[0 1 -1 0 0 0 0]", "(0 0 0)"),
    "UPrime2Mean": FieldSpec("volSymmTensorField", "[0 2 -2 0 0 0 0]", "(0 0 0 0 0 0)"),
    "p": FieldSpec("volScalarField", "[0 2 -2 0 0 0 0]", "0"),
    "p_rgh": FieldSpec("volScalarField", "[0 2 -2 0 0 0 0]", "0"),
    "pMean": FieldSpec("volScalarField", "[0 2 -2 0 0 0 0]", "0"),
    "T": FieldSpec("volScalarField", "[0 0 0 1 0 0 0]", "0"),
    "rho": FieldSpec("volScalarField", "[1 -3 0 0 0 0 0]", "0"),
    "nu": FieldSpec("volScalarField", "[0 2 -1 0 0 0 0]", "0"),
    "nut": FieldSpec("volScalarField", "[0 2 -1 0 0 0 0]", "0"),
    "alphat": FieldSpec("volScalarField", "[0 2 -1 0 0 0 0]", "0"),
    "k": FieldSpec("volScalarField", "[0 2 -2 0 0 0 0]", "0"),
    "epsilon": FieldSpec("volScalarField", "[0 2 -3 0 0 0 0]", "0"),
    "omega": FieldSpec("volScalarField", "[0 0 -1 0 0 0 0]", "0"),
    "alpha": FieldSpec("volScalarField", "[0 0 0 0 0 0 0]", "0"),
    "phi": FieldSpec("surfaceScalarField", "[0 3 -1 0 0 0 0]", "0"),
    "R": FieldSpec("volSymmTensorField", "[0 2 -2 0 0 0 0]", "(0 0 0 0 0 0)"),
}


@dataclass
class Entry:
    key: str
    start: int
    end: int
    opening: int | None = None
    closing: int | None = None
    children: dict[str, "Entry"] = field(default_factory=dict)


TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|[{};]|[^\s{};]+',
    re.DOTALL,
)


def tokenize(text):
    tokens = []
    for match in TOKEN_RE.finditer(text):
        value = match.group()
        if value.startswith("//") or value.startswith("/*"):
            continue
        tokens.append((value, match.start(), match.end()))
    return tokens


def parse_dictionary(tokens, open_index):
    if tokens[open_index][0] != "{":
        raise ValueError("Expected an opening brace in OpenFOAM dictionary")
    entries = {}
    index = open_index + 1
    while index < len(tokens) and tokens[index][0] != "}":
        key, start, _ = tokens[index]
        if key in ("{", "}", ";"):
            raise ValueError(f"Unexpected token {key!r} in OpenFOAM dictionary")
        index += 1
        if index >= len(tokens):
            break
        if tokens[index][0] == "{":
            child, index = parse_dictionary(tokens, index)
            entry = Entry(key, start, tokens[index - 1][2], child.opening, child.closing, child.children)
        else:
            while index < len(tokens) and tokens[index][0] not in (";", "}"):
                index += 1
            if index >= len(tokens) or tokens[index][0] != ";":
                raise ValueError(f"Missing semicolon after {key!r} in OpenFOAM dictionary")
            entry = Entry(key, start, tokens[index][2])
            index += 1
        if key in entries:
            raise ValueError(f"Duplicate dictionary entry {key!r}")
        entries[key] = entry
    if index >= len(tokens) or tokens[index][0] != "}":
        raise ValueError("Unclosed brace in OpenFOAM dictionary")
    opening = tokens[open_index][1]
    closing = tokens[index][1]
    return Entry("", opening, tokens[index][2], opening, closing, entries), index + 1


def dictionary_entry(text, key):
    tokens = tokenize(text)
    for index, (value, _, _) in enumerate(tokens[:-1]):
        if value == key and tokens[index + 1][0] == "{":
            return parse_dictionary(tokens, index + 1)[0]
    raise ValueError(f"Missing {key} dictionary")


def read_field_contents(path):
    raw = path.read_bytes()
    header = re.search(rb"\bFoamFile\s*\{(?P<body>.*?)\}", raw, re.DOTALL)
    if header is None:
        return None, raw.decode("utf-8")

    format_match = re.search(rb"\bformat\s+(ascii|binary)\s*;", header["body"])
    if format_match is None or format_match.group(1) == b"ascii":
        return None, raw.decode("utf-8")

    candidates = list(
        re.finditer(rb"\bboundaryField\b(?=\s*\{)", raw[header.end():])
    )
    for candidate in reversed(candidates):
        start = header.end() + candidate.start()
        try:
            text = raw[start:].decode("utf-8")
            boundary = dictionary_entry(text, "boundaryField")
        except (UnicodeDecodeError, ValueError):
            continue
        if re.fullmatch(r"boundaryField\s*", text[:boundary.opening]):
            return raw[:start], text

    raise ValueError(f"{path}: could not read boundaryField from binary field")


def parse_map(path):

    # Read raw bytes first to safely handle binary files without decoding crashes
    try:
        raw_bytes = path.read_bytes()
    except Exception:
        return []

    # Decode using surrogateescape or ignore so we don't crash on binary internalField chunks
    text = raw_bytes.decode("utf-8", errors="ignore")

    lines = text.splitlines()
    #lines = path.read_text(encoding="utf-8").splitlines()
    result = {}
    current_patches = None
    current_field = None
    field_indent = None
    value_lines = []

    def finish_field():
        if current_patches is not None and current_field is not None:
            value = "\n".join(value_lines).strip()
            if not value:
                patches = ", ".join(current_patches)
                raise ValueError(f"Boundary condition for {patches}/{current_field} is empty")
            for patch_name in current_patches:
                result[patch_name][current_field] = value

    for line_number, line in enumerate(lines, 1):
        if "\t" in line:
            raise ValueError(f"{path}:{line_number}: tabs are not supported for indentation")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0 and stripped.endswith(":"):
            finish_field()
            patch_names = [name.strip() for name in stripped[:-1].split(",")]
            if (
                any(
                    not re.fullmatch(r"[A-Za-z0-9_.+-]+", name)
                    or name in (".", "..")
                    for name in patch_names
                )
                or len(set(patch_names)) != len(patch_names)
                or any(name in result for name in patch_names)
            ):
                raise ValueError(f"{path}:{line_number}: empty, invalid, or duplicate patch name")
            current_patches = patch_names
            for patch_name in current_patches:
                result[patch_name] = {}
            current_field = None
            field_indent = None
            continue
        if current_patches is None:
            raise ValueError(f"{path}:{line_number}: expected a patch name")
        match = re.match(r"^(\s+)([^:\s]+)\s*:(?:\s*(.*))?$", line)
        if match:
            indent = len(match.group(1))
            if field_indent is None:
                field_indent = indent
            if indent != field_indent:
                raise ValueError(f"{path}:{line_number}: inconsistent field indentation")
            finish_field()
            current_field = match.group(2)
            if not re.fullmatch(r"[A-Za-z0-9_.+-]+", current_field) or current_field in (".", ".."):
                raise ValueError(f"{path}:{line_number}: invalid field name {current_field!r}")
            if any(current_field in result[patch_name] for patch_name in current_patches):
                raise ValueError(f"{path}:{line_number}: duplicate field {current_field!r}")
            value_lines = [match.group(3)] if match.group(3) else []
        elif current_field is not None and indent > field_indent:
            value_lines.append(line[field_indent + 1:])
        else:
            raise ValueError(f"{path}:{line_number}: expected a field name and boundary condition")
    finish_field()
    if not result:
        raise ValueError(f"{path} contains no patch entries")
    for patch, fields in result.items():
        if not fields:
            raise ValueError(f"Patch {patch!r} has no field entries")
    return result


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def format_boundary_entry(patch_name, boundary_condition, indent):
    body = boundary_condition.strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1].strip()
    body_lines = body.splitlines()
    nonempty = [line for line in body_lines if line.strip()]
    body_indent = min(
        (len(line) - len(line.lstrip()) for line in nonempty),
        default=0,
    )
    lines = [
        f"{' ' * indent}{patch_name}",
        f"{' ' * indent}{{",
    ]
    lines.extend(
        f"{' ' * (indent + 4)}{line[body_indent:].rstrip()}"
        for line in nonempty
    )
    lines.append(f"{' ' * indent}}}")
    return "\n".join(lines)


def new_field_text(field_name, spec):
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "| =========                 |                                                 |\n"
        "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {spec.foam_class};\n"
        '    location    "0";\n'
        f"    object      {field_name};\n"
        "}\n"
        f"dimensions      {spec.dimensions};\n"
        f"internalField   uniform {spec.zero};\n"
        "boundaryField\n"
        "{\n"
        "}\n"
    )


def update_field_file(path, field_name, patch_conditions):
    created = not path.exists()
    spec = FIELD_SPECS.get(field_name)
    if created and spec is None:
        raise ValueError(
            f"Cannot create {path.name}: no built-in dimensions/type mapping for field {field_name!r}"
        )
    if created:
        prefix, text = None, new_field_text(field_name, spec)
    else:
        prefix, text = read_field_contents(path)
    text = re.sub(
        r"(?m)^[ \t]*// Updated by bndFromMap\.py at [^\n]*(?:\n|$)",
        "",
        text,
    )
    boundary = dictionary_entry(text, "boundaryField")
    edits = []

    for patch_name, condition in patch_conditions.items():
        patch = boundary.children.get(patch_name)
        if patch is None:
            insertion = f"{format_boundary_entry(patch_name, condition, 4)}\n"
            edits.append((boundary.closing, boundary.closing, "\n" + insertion))
            continue

        line_start = text.rfind("\n", 0, patch.start) + 1
        edits.append(
            (
                line_start,
                patch.end,
                format_boundary_entry(patch_name, condition, 4),
            )
        )

    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    boundary = dictionary_entry(text, "boundaryField")
    after_opening = boundary.opening + 1
    remainder = text[after_opening:].lstrip("\r\n")
    text = (
        text[:after_opening]
        + f"\n    // Updated by bndFromMap.py at {timestamp()} (UTC)\n"
        + remainder
    )
    return (prefix + text.encode("utf-8") if prefix is not None else text), created


def atomic_write(path, text):
    existing_mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    temporary_path = None
    try:
        options = {"dir": path.parent, "delete": False}
        if isinstance(text, bytes):
            temporary = tempfile.NamedTemporaryFile(mode="wb", **options)
        else:
            temporary = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", **options
            )
        with temporary:
            temporary.write(text)
            temporary_path = Path(temporary.name)
        temporary_path.chmod(existing_mode)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def centralize(case_dir):
    map_path = case_dir / "mapOfBnd"
    if map_path.exists():
        raise ValueError(f"{map_path} already exists; refusing to overwrite it")

    aggregated = defaultdict(dict)
    for field_path in sorted(case_dir.iterdir()):
        if not field_path.is_file() or field_path.name == "mapOfBnd":
            continue
        raw_bytes = field_path.read_bytes()
        if b"boundaryField" not in raw_bytes:
            continue
        _, search_text = read_field_contents(field_path)
        boundary = dictionary_entry(search_text, "boundaryField")

        for patch_name, patch in boundary.children.items():
            condition = search_text[patch.opening + 1:patch.closing].strip()
            if condition:
                if field_path.name in aggregated[patch_name]:
                    raise ValueError(f"Duplicate field {field_path.name!r} for patch {patch_name!r}")
                aggregated[patch_name][field_path.name] = condition
                
    if not aggregated:
        raise ValueError(f"No boundaryField patch conditions found in {case_dir}")

    lines = []
    for patch_name in sorted(aggregated):
        lines.append(f"{patch_name}:")
        for field_name in sorted(aggregated[patch_name]):
            lines.append(f"    {field_name}:")
            lines.extend(
                f"        {line}" for line in aggregated[patch_name][field_name].splitlines()
            )
    atomic_write(map_path, "\n".join(lines) + "\n")
    print(f"Created {map_path}")

def apply_map(case_dir):
    source = case_dir / "mapOfBnd"
    if not source.is_file():
        raise ValueError(f"Required boundary map does not exist: {source}")
    mapping = parse_map(source)
    by_field = defaultdict(dict)
    for patch_name, fields in mapping.items():
        for field_name, condition in fields.items():
            by_field[field_name][patch_name] = condition

    prepared = []
    for field_name, patch_conditions in sorted(by_field.items()):
        path = case_dir / field_name
        updated_text, created = update_field_file(path, field_name, patch_conditions)
        prepared.append((path, updated_text, created, tuple(patch_conditions)))

    for path, contents, created, patches in prepared:
        atomic_write(path, contents)
        if created:
            print(
                f"Created field file {path} with boundary condition for "
                f"patches {', '.join(patches)}"
            )
        else:
            for patch_name in patches:
                print(f"Updated {path.name} boundary condition for patch {patch_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage OpenFOAM boundary conditions from mapOfBnd."
    )
    parser.add_argument(
        "--centralize",
        action="store_true",
        help="create 0/mapOfBnd from field files; refuse to overwrite an existing map",
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=Path.cwd(),
        help="OpenFOAM case directory (default: current directory)",
    )
    args = parser.parse_args()
    case_dir = args.case.resolve() / "0"
    print(
        "Warning: entries absent from mapOfBnd are preserved; removal is not supported.",
        file=sys.stderr,
    )
    try:
        if not case_dir.is_dir():
            raise ValueError(f"OpenFOAM time directory does not exist: {case_dir}")
        if args.centralize:
            centralize(case_dir)
        else:
            apply_map(case_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
