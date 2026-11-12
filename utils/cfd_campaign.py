#!/usr/bin/env python3
# ================================================================
# CFD Campaign Generator for OpenFOAM
# ---------------------------------------------------------------
# Requirements:
#   Python 3.8+
#   pip install jinja2
#
# Usaage:
#   ./cfd_campaign.py campaign.cfdcmp
#   ./cfd_campaign.py campaign.cfdcmp --test
#
#
# Input:
#   - JSON campaign file describing solvers, templates, parameters
#   - Template folders containing OpenFOAM case files with Jinja tags
#
# Output:
#   - A folder containing fully rendered OpenFOAM cases
#   - A CSV file listing all generated cases and their parameters
#   - A log file describing parameter replacements in each file
#
# Features:
#   ✓ Jinja2 template rendering for files that contain declared campaign
#     placeholder tokens such as {{ varParameterName }}
#   ✓ Do not parse unrelated text files that contain brace-like syntax
#     but no campaign placeholders, such as OpenFOAM dynamicCode sources
#   ✓ Place results of redering in the folder named after the campaign JSON filename, unless overridden by --output-root
#   ✓ Template-variable discovery for warnings inspects actual Jinja
#     placeholder usage rather than arbitrary var* text
#   ✓ If the campaign JSON file does not exist, fail with a clear error
#     message listing the requested path and available .cfdcmp files
# Parameter Space Requirements
#   1. Independent parameters:
#      - Parameters without "conditions" and without "paired".
#      - Vary independently using Cartesian product.
#   2. Conditional parameters:
#      - Parameter has "conditions": {paramName: value, ...}.
#      - If all conditions are satisfied: vary over all values in "values".
#      - If conditions are NOT satisfied: parameter is still present but assigned the FIRST value.
#   3. Paired parameters (master–slave):
#      - A parameter with "paired": "<master>" is a slave.
#      - The referenced parameter is the master.
#      - The master does NOT need to declare pairing.
#      - All slaves paired to the same master vary in lockstep (zip-style).
#      - All slaves must have the same number of values as the master.
#   - If lengths differ → raise a comprehensive error.
#   4. Checks for deterministic and readable parameter input:
#      - Warn if a parameter's values contain duplicates.
#      - Warn if a parameter's values are not sorted in either ascending or descending order.
#   ✓ One folder per OpenFOAM case named
#   ✓ Folder name = case number only with postfixes defined below
#   ✓ In the CSV summary, the first column must always be the case number
#     All other parameter columns follow after the case number
#     CSV fieldnames are ordered: case, solver, then all parameters alphabetically
#   ✓ Warn if no template folder is found for a given template name
#   ✓ Warn if specified parameters are not used in the template
#   ✓ Parameter A appears in templates as varA
#   ✓ Warn if template contains var* tokens not corresponding to any parameter
#   ✓ Log what parameters were replaced in which files and how many times
#   ✓ Optional test mode: generate only one case per template
#   ✓ If a solver entry in the campaign JSON has attribute "addPostfix",
#     its value is appended to the rendered case folder name.
#   ✓ If a parameter entry in the campaign JSON has attribute "addPostfix",
#     the rendered case folder name receives a postfix consisting of:
#         <parameter_postfix>_<parameter_value>
#     appended directly to the case name.
#   ✓ Final case folder name format:
#       <case_number><solver_postfix><parameter_postfix>_<value>...
#   ✓ No folder hierarchy is created; postfixes are concatenated directly.
#   ✓ Copy to the folder the batch file to run all generated cases serially templates/runAllSerial.sh
#   ✓ Copy to the folder the batch file to run all generated cases with the cluster scheduler placed in templates/qFoamCluster.sh
#   ✓ After the rendering move files from the folder templates to the folder with the rendered cases
#
# # ================================================================

import os
import json
import shutil
import csv
import re
import warnings
from itertools import product
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


# ================================================================
# Load campaign JSON
# ================================================================
def validate_parameter_values(parameters):
    """Warn for duplicate or unsorted parameter value lists."""
    for p in parameters:
        name = p.get("name", "<unnamed>")
        values = p.get("values", [])

        if not isinstance(values, list):
            continue

        if len(values) != len(set(values)):
            warnings.warn(
                f"Parameter '{name}' has duplicate values: {values}",
                stacklevel=2,
            )

        if len(values) > 1:
            try:
                is_ascending = values == sorted(values)
                is_descending = values == sorted(values, reverse=True)
            except TypeError:
                continue

            if not (is_ascending or is_descending):
                warnings.warn(
                    f"Parameter '{name}' values are not sorted in ascending or descending order: {values}",
                    stacklevel=2,
                )

def load_campaign(json_path):
    json_file = Path(json_path)
    if not json_file.is_file():
        available = sorted(Path.cwd().glob("*.cfdcmp"))
        available_names = ", ".join(path.name for path in available) or "none found"
        raise FileNotFoundError(
            f"Campaign file not found: {json_path}. Available .cfdcmp files in "
            f"{Path.cwd()}: {available_names}"
        )

    with open(json_file, "r") as f:
        data = json.load(f)

    campaign = data["campaign"]
    parameters = campaign.get("parameters", [])
    seen = {}
    for p in parameters:
        name = p.get("name")
        if name is None:
            continue
        if name in seen:
            raise ValueError(
                f"\n============================================================\n"
                f" CFD Campaign Error: Duplicate parameter name '{name}'\n"
                f"------------------------------------------------------------\n"
                f" Parameter names must be unique within a campaign file.\n"
                f" The same parameter appears more than once in the input JSON.\n"
                f" Remove duplicate entries or merge them into a single logical parameter.\n"
                f"============================================================\n"
            )
        seen[name] = p

    validate_parameter_values(parameters)
    return campaign


# ================================================================
# Build parameter combinations with conditional logic
# ================================================================
def build_parameter_space(parameters):
    # Categorize parameters
    conditional = []
    independent = []
    paired_slaves = {}   # slaveName -> masterName
    masters = {}         # masterName -> list of slave parameter dicts

    # First pass: classify parameters
    for p in parameters:
        if "conditions" in p:
            conditional.append(p)
        elif "paired" in p:
            master = p["paired"]
            paired_slaves[p["name"]] = master
            masters.setdefault(master, []).append(p)
        else:
            independent.append(p)

    # Remove master parameters from independent list
    independent = [p for p in independent if p["name"] not in masters]

    # Identify master parameters
    master_params = {p["name"]: p for p in parameters if p["name"] in masters}

    # ============================================================
    # 1. Independent parameters → Cartesian product
    # ============================================================
    combos = [{}]
    for p in independent:
        new = []
        for c in combos:
            for v in p["values"]:
                cc = dict(c)
                cc[p["name"]] = v
                new.append(cc)
        combos = new

    # ============================================================
    # 2. Paired parameters → zip-style variation
    # ============================================================
    for master_name, slaves in masters.items():

        if master_name not in master_params:
            raise ValueError(
                f"\n============================================================\n"
                f" CFD Campaign Error: Master parameter '{master_name}' not found.\n"
                f"------------------------------------------------------------\n"
                f" Slaves referencing this master: {[s['name'] for s in slaves]}\n"
                f"============================================================\n"
            )

        master = master_params[master_name]
        mvals = master["values"]

        # Check all slaves have same length
        for s in slaves:
            if len(s["values"]) != len(mvals):
                raise ValueError(
                    f"\n============================================================\n"
                    f" CFD Campaign Error: Paired Parameter Length Mismatch\n"
                    f"------------------------------------------------------------\n"
                    f" Master parameter:        {master_name}\n"
                    f" Slave parameter:         {s['name']}\n"
                    f" Master value count:      {len(mvals)}\n"
                    f" Slave value count:       {len(s['values'])}\n"
                    f"\n"
                    f" Paired parameters must have equal length because they vary\n"
                    f" in lockstep (zip-style): (master[i], slave[i]).\n"
                    f"\n"
                    f" Actual values:\n"
                    f"   {master_name}: {mvals}\n"
                    f"   {s['name']}: {s['values']}\n"
                    f"\n"
                    f" Fix:\n"
                    f"   - Add/remove values so both lists have equal length, OR\n"
                    f"   - Decide whether this parameter should really be paired.\n"
                    f"============================================================\n"
                )

        # Apply zip-style pairing
        new = []
        for c in combos:
            for i in range(len(mvals)):
                cc = dict(c)
                cc[master_name] = mvals[i]
                for s in slaves:
                    cc[s["name"]] = s["values"][i]
                new.append(cc)
        combos = new

    # ============================================================
    # 3. Conditional parameters → branch or fallback
    # ============================================================
    for p in conditional:
        name = p["name"]
        values = p["values"]
        cond = p["conditions"]

        new = []
        for c in combos:
            cond_ok = all(c.get(k) == v for k, v in cond.items())
            if cond_ok:
                # branch normally
                for v in values:
                    cc = dict(c)
                    cc[name] = v
                    new.append(cc)
            else:
                # fallback to first value
                cc = dict(c)
                cc[name] = values[0]
                new.append(cc)
        combos = new

    return combos


# ================================================================
# Copy template folder
# ================================================================
def copy_template(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


# ================================================================
# Scan template for var* usage
# ================================================================
def scan_template_vars(template_dir):
    var_tokens = set()
    pattern = re.compile(r"\{\{\s*(var[A-Za-z0-9_]+)\s*\}\}")

    for root, dirs, files in os.walk(template_dir):
        for fname in files:
            fpath = Path(root) / fname
            try:
                text = fpath.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            var_tokens.update(pattern.findall(text))

    return var_tokens


def ordered_unique_parameter_names(parameters):
    """Return parameter names in first-seen order while collapsing duplicates."""
    seen = set()
    ordered = []
    for p in parameters:
        name = p["name"]
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def is_parameter_active(parameter, combo):
    """Return True when a conditional parameter is actually active for this combo."""
    conditions = parameter.get("conditions", {})
    return all(combo.get(name) == value for name, value in conditions.items())


# ================================================================
# Render all files in case folder using Jinja2 + replacement logging
# ================================================================
def render_case_with_jinja(case_dir, context, log_file):
    env = Environment(loader=FileSystemLoader(str(case_dir)), autoescape=False)

    for root, dirs, files in os.walk(case_dir):
        for fname in files:
            fpath = Path(root) / fname

            try:
                original_text = fpath.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            keys_in_file = [key for key in context if key in original_text]
            if not keys_in_file:
                continue

            template = env.from_string(original_text)
            rendered = template.render(**context)

            # Count replacements
            for key in keys_in_file:
                value = context[key]
                count = original_text.count(key)
                if count > 0:
                    log_file.write(
                        f"File {fpath}: replaced {key} -> {value} ({count} times)\n"
                    )

            fpath.write_text(rendered, encoding="utf-8")


# ================================================================
# Main campaign driver
# ================================================================
def run_campaign(json_path, templates_root, output_root, test_mode=False):
    campaign = load_campaign(json_path)
    solvers = campaign["solvers"]
    parameters = campaign["parameters"]

    param_space = build_parameter_space(parameters)
    param_names = [p["name"] for p in parameters]

    os.makedirs(output_root, exist_ok=True)

    csv_rows = []
    case_id = 0

    log_path = Path(output_root) / "campaign_log.txt"
    log_file = open(log_path, "w")

    for solver_entry in solvers:
        solver_name = solver_entry["solver"]
        template_name = solver_entry["template"]

        template_dir = Path(templates_root) / template_name

        if not template_dir.exists():
            print(f"WARNING: Template folder not found: {template_dir}. Skipping all cases for solver '{solver_name}'.")
            continue

        template_vars = scan_template_vars(template_dir)
        expected_vars = {f"var{name}" for name in param_names}

        unused_params = expected_vars - template_vars
        if unused_params:
            print(f"WARNING: Parameters not used in template '{template_name}': {', '.join(sorted(unused_params))}")

        unknown_vars = {v for v in template_vars if v not in expected_vars}
        if unknown_vars:
            print(f"WARNING: Template '{template_name}' contains unknown var* tokens: {', '.join(sorted(unknown_vars))}")

        # TEST MODE: only one case per template
        combos_to_run = param_space if not test_mode else [param_space[0]]

        for combo in combos_to_run:
            case_id += 1

            case_name = f"{case_id:04d}"
            case_dir = Path(output_root) / case_name

            # Solver postfix
            solver_postfix = solver_entry.get("addPostfix")
            if solver_postfix:
                case_name += f"_{solver_postfix}"

            # Parameter postfixes
            for p in parameters:
                pname = p["name"]
                if pname not in combo:
                    continue

                param_postfix = p.get("addPostfix")
                if param_postfix and is_parameter_active(p, combo):
                    pvalue = combo[pname]
                    case_name += f"_{param_postfix}_{pvalue}"

            # Final folder path
            case_dir = Path(output_root) / case_name

            print(f"Creating case: {case_dir}")

            copy_template(template_dir, case_dir)

            context = {f"var{k}": v for k, v in combo.items()}
           # context["solver"] = solver_name

            render_case_with_jinja(case_dir, context, log_file)

            row = {"case": case_id, "solver": solver_name}
            row.update(combo)
            csv_rows.append(row)

    log_file.close()

    if csv_rows:
        csv_path = Path(output_root) / "campaign_cases.csv"

        # Collect all keys across all rows
        all_keys = {key for row in csv_rows for key in row.keys()}

        # Force CSV column order:
        # 1) case
        # 2) solver
        # 3) all other parameters alphabetically
        fieldnames = ["case", "solver"] + sorted(k for k in all_keys if k not in ("case", "solver"))

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)

        print(f"\nCampaign complete. CSV summary written to: {csv_path}")
        print(f"Log written to: {log_path}")
    else:
        print("\nNo cases generated.")

# ================================================================
# Copy batch files to run all generated cases
# ================================================================
    required_batch = Path(templates_root) / "runAllSerial.sh"
    if not required_batch.is_file():
        raise FileNotFoundError(
            f"Required batch script template not found: {required_batch}"
        )

    for batch_name in ("runAllSerial.sh", "qFoamCluster.sh"):
        batch_template = Path(templates_root) / batch_name
        if not batch_template.is_file():
            print(f"WARNING: Optional batch script template not found: {batch_template}. Skipping copy.")
            continue

        batch_file = Path(output_root) / batch_template.name
        shutil.copy2(batch_template, batch_file)
        batch_file.chmod(0o755)


# ================================================================
# CLI entry point
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare OpenFOAM CFD campaign from JSON using Jinja templates."
    )
    parser.add_argument("json", help="Path to campaign JSON file")
    parser.add_argument(
        "--templates-root",
        default="TEMPLATES",
        help="Root folder containing solver templates"
    )
    parser.add_argument(
        "--output-root",
        default="cases",
        help="Output folder for generated OpenFOAM cases"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable test mode: generate only one case per template"
    )

    args = parser.parse_args()

    # If user did not specify --output-root, use campaign JSON filename
    if args.output_root == "cases":
        campaign_base = Path(args.json).stem
        args.output_root = campaign_base

    run_campaign(args.json, args.templates_root, args.output_root, args.test)
