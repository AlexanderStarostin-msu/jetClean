#!/usr/bin/env python3

# collectMCR.py
#
# Collect case metrics from immediate subfolders of a given cases directory.
# Metric extraction rules are read from a config file with columns:
# token, path, keyword, mode, default.
#
# Supported extraction modes include:
# LastInColumn, FromLastString, AfterKeyword, and WordAfterKeyword.
#
#Metric Collection Requirements
#------------------------------
#
#1. Config File Format
#   Each line defines one metric extraction rule:
#     token, path, keyword, mode, default
#   - token: name of the metric
#   - path: relative path inside each case folder
#   - keyword: string to search for
#   - mode: extraction mode
#   - default: fallback value if extraction fails
#
#2. Supported Extraction Modes
#
#   2.1 FirstInColumn
#       - Read the file as a table.
#       - Determine the column index from the last header line starting with '#'.
#       - Extract the first numeric value in that column after the header.
#
#   2.2 LastInColumn
#       - Read the file as a table.
#       - Determine the column index from the last header line starting with '#'.
#       - Extract the last numeric value in that column.
#
#   2.3 FromLastString
#       - Scan the file from bottom to top.
#       - Search for the keyword (including quoted variants).
#       - Extract the first number appearing *after* the keyword.
#       - For large OpenFOAM logs (log.*Foam), scan only the last 100 lines.
#
#       FOAMSOLVER postfix rule:
#       - If the filename ends with the postfix "FOAMSOLVER":
#           * Treat FOAMSOLVER as a placeholder for the actual solver name.
#           * Replace the postfix FOAMSOLVER with each of:
#               slurryFoam, buoyantFoam, pimpleFoam, isoFoam
#           * If a file with the replaced postfix exists in the same directory:
#               use that file instead.
#           * This replacement must occur BEFORE checking file existence.
#
#   2.4 AfterKeyword
#       - Scan file top-to-bottom.
#       - When the keyword is found, extract the number immediately after it.
#
#   2.5 WordAfterKeyword
#       - Scan file top-to-bottom.
#       - Extract the next word after the keyword.
#       - If numeric → convert to float.
#       - If not numeric → return the raw string.
#       - Values from this mode must NOT be formatted.
#
#3. Formatting Rules
#   - For all modes except WordAfterKeyword:
#       format numeric values using 4 significant digits.
#   - WordAfterKeyword values remain unformatted.
#
#4. Folder Processing
#   - Only immediate subfolders of the cases root are processed.
#   - Each subfolder is treated as one CFD case.
#
#5. Output
#   - Print collected metrics.
#   - Save CSV summary:
#       Summary_<cases_root>_<YYYYMMDD>.csv
#   - First row contains solver name and version extracted from log.<solver>.#
#
# ######################



import re
import os
import shlex
from pathlib import Path

#from summaryCrossplot import generate_crossplot_from_csv

def format_collected_value(value, decimals: int = 4, enabled: bool = True):
    """Format numeric values with fixed significant digits when enabled."""
    if enabled and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{decimals}g}"
    return value


def resolve_script_local_path(filename: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, os.path.basename(filename))


def iter_lines_reverse(file_path: str, chunk_size: int = 8192):
    """Yield text lines from end of file to start without loading entire file."""
    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_pos = f.tell()
        buffer = b""

        while file_pos > 0:
            read_size = min(chunk_size, file_pos)
            file_pos -= read_size
            f.seek(file_pos)
            chunk = f.read(read_size)
            buffer = chunk + buffer
            lines = buffer.split(b"\n")
            buffer = lines[0]

            for raw_line in reversed(lines[1:]):
                yield raw_line.decode("utf-8", errors="ignore")

        if buffer:
            yield buffer.decode("utf-8", errors="ignore")

def resolve_foam_solver_path(path: str) -> str:
    """
    If filename ends with FOAMSOLVER, try replacing the postfix with
    slurryFoam, buoyantFoam, pimpleFoam, isoFoam.
    Returns the first existing alternative path, or the original path.
    """
    base = os.path.basename(path)

    if not base.endswith("FOAMSOLVER"):
        return path

    case_dir = os.path.dirname(path)
    prefix = base[:-len("FOAMSOLVER")]  # everything before postfix

    alternatives = [
        "slurryFoam",
        "buoyantFoam",
        "pimpleFoam",
        "isoFoam",
    ]

    for solver in alternatives:
        alt_file = prefix + solver
        alt_path = os.path.join(case_dir, alt_file)
        if os.path.isfile(alt_path):
            return alt_path

    return path

def find_number_by_keyword(config: dict, casePath):
    """
    Reads a file and extracts a number based on the given mode.
    Falls back to default value if no number is found.
    
    Args:
        config (dict): Dictionary with keys:
            - 'token': identifier for this config
            - 'path': path to the file
            - 'keyword': keyword to search for
            - 'mode': one of [ 'FirstInColumn','LastInColumn', 'FromLastString', 'AfterKeyword', 'WordAfterKeyword']
            - 'default': default value if number not found
    
    Returns:
        tuple: (token, number)
    """
    token = config.get("token")
    path = casePath + config.get("path")
    keyword = config.get("keyword")
    mode = config.get("mode")
    default = config.get("default")

    number_re = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"

    if not token or not path or not keyword or not mode:
        raise ValueError("Config must contain 'token', 'path', 'keyword', and 'mode'.")

    result = default

    path = resolve_foam_solver_path(path)
    if not os.path.isfile(path):
        return (token, result)

    if mode == "FirstInColumn":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            raise IOError(f"Error reading file: {e}")

        header = lines[0].strip().split()
        header_line_idx = 0

        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                header = line.strip()[1:].strip().split()
                header_line_idx = i

        if keyword in header:
            col_index = header.index(keyword)

            for line in lines[header_line_idx + 1:]:
                if not line.strip():
                    continue

                if line.strip().startswith("#"):
                    continue

                parts = line.strip().split()

                if len(parts) <= col_index:
                    continue

                try:
                    result = float(parts[col_index])
                    break
                except ValueError:
                    continue

    elif mode == "LastInColumn":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            raise IOError(f"Error reading file: {e}")

        header = lines[0].strip().split()

        for line in reversed(lines): 
            if line.strip().startswith("#"): 
                header = line.strip()[1:].strip().split()
#                print(header) 
                break

        if keyword in header:
            col_index = header.index(keyword)
            #print(col_index)
            for line in reversed(lines[1:]):
                if line.strip().startswith("#"): 
                    break

                parts = line.strip().split()
                #print(parts)
                if len(parts) > col_index:
                    try:
                        result = float(parts[col_index])
                        break
                    except ValueError:
                        continue

    elif mode == "FromLastString":
        is_large_foam_log = re.match(r"log\..*Foam$", os.path.basename(path)) is not None
        line_iter = iter_lines_reverse(path) if is_large_foam_log else None

        if line_iter is None:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    line_iter = reversed(f.readlines())
            except Exception as e:
                raise IOError(f"Error reading file: {e}")

        lines_scanned = 0
        max_lines = 100 if is_large_foam_log else None

        for line in line_iter:
            lines_scanned += 1
            if max_lines is not None and lines_scanned > max_lines:
                break
            variants = [keyword, f'"{keyword}"', f"'{keyword}'"]
            for key_variant in variants:
                if key_variant in line:
                    idx = line.find(key_variant) + len(key_variant)
                    substring = line[idx:]
                    match = re.search(number_re, substring)
                    if match:
                        result = float(match.group())
                        break
            if result != default:
                break

    elif mode == "AfterKeyword":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            raise IOError(f"Error reading file: {e}")

        for line in lines:
            variants = [keyword, f'"{keyword}"', f"'{keyword}'"]
            for key_variant in variants:
                if key_variant in line:
                    idx = line.find(key_variant) + len(key_variant)
                    substring = line[idx:]
                    match = re.search(
                        r"\(?\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*\)?",
                        substring,
                    )
                    if match:
                        result = float(match.group(1))
                        break
            if result != default:
                break

    elif mode == "WordAfterKeyword":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            raise IOError(f"Error reading file: {e}")

        for line in lines:
            variants = [keyword, f'"{keyword}"', f"'{keyword}'"]
            for key_variant in variants:
                if key_variant in line:
                    # substring after keyword
                    idx = line.find(key_variant) + len(key_variant)
                    substring = line[idx:].strip()

                    # split into words
                    parts = substring.split()
                    if not parts:
                        continue

                    next_word = parts[0]
                    next_word = next_word.rstrip(";")

                    # try to convert to float, otherwise return string
                    try:
                        result = float(next_word)
                    except ValueError:
                        result = next_word

                    break
            if result != default:
                break

    else:
        raise ValueError(f"Unknown mode: {mode}")

    if result is None:
        result = default

    return (token, result)


def read_config_file(config_path: str):
    """
    Reads a config file with free columns:
    token, path, keyword, mode, default
    
    Columns are separated by tabs or spaces.
    Quoted values are supported, so keyword can contain spaces.
    
    Args:
        config_path (str): Path to the config file.
    
    Returns:
        list[dict]: List of config dictionaries.
    """
    configs = []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.strip().startswith("#"):
                    continue
                parts = shlex.split(line.strip())
                if len(parts) < 4:
                    continue
                token, path, keyword, mode = parts[0], parts[1], parts[2], parts[3]
                default_val = None
                if len(parts) >= 5:
                    try:
                        default_val = float(parts[4])
                    except ValueError:
                        default_val = parts[4]  # keep as string if not numeric
                configs.append({
                    "token": token,
                    "path": path,
                    "keyword": keyword,
                    "mode": mode,
                    "default": default_val
                })
    except Exception as e:
        raise IOError(f"Error reading config file: {e}")
    return configs


if __name__ == "__main__":
    import sys
    import csv
    import subprocess
    from datetime import datetime

    # argv[1]: mandatory cases folder
    # argv[2]: optional config file (default: cfd_clect.cfg)
    if len(sys.argv) < 2:
        raise SystemExit("Usage: collectMCR.py <cases_folder> [config_file]")

    cases_root = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else "cfd_collect.cfg"
    config_file = resolve_script_local_path(config_file)

    if not os.path.isdir(cases_root):
        raise SystemExit(f"Error: cases folder not found: {cases_root}")

    if not os.path.isfile(config_file):
        raise SystemExit(f"Error: config file not found: {config_file}")

    # Read all configurations once
    configs = read_config_file(config_file)
    print(configs)
    no_format_tokens = {
        cfg["token"]
        for cfg in configs
        if str(cfg.get("mode", "")) == "WordAfterKeyword"
    }


    # Collect and sort immediate subfolders
    subfolders = sorted(
        entry.path
        for entry in os.scandir(cases_root)
        if entry.is_dir()
    )

    if not subfolders:
        raise SystemExit(f"Error: no subfolders found in: {cases_root}")

    # Process each subfolder as a separate case
    all_results = {}
    for case_dir in subfolders:
        casePath = case_dir + "/"
        print(f"\n--- Case: {casePath} ---")
        results = {}
        for cfg in configs:
#            print("CONFIG")
#            print(cfg)
            token, value = find_number_by_keyword(cfg, casePath)
            results[token] = value
#            print(f"{token}: {value}")

        all_results[case_dir] = results

    # If you want to use results later:
    formatted_results = {
        case_dir: {
            k: format_collected_value(v, enabled=k not in no_format_tokens)
            for k, v in res.items()
        }
        for case_dir, res in all_results.items()
    }
    print(formatted_results)
    print(f"Collected {len(all_results)} cases, total scanned {len(subfolders)}")

    # Save to CSV: <cases_root_name>_YYYYMMDD.csv
    date_str = datetime.now().strftime("%Y%m%d")
    folder_name = os.path.basename(os.path.normpath(cases_root))
    csv_path = f"Summary_{folder_name}_{date_str}.csv"

    # Get solver version from first case's log file
    solver_version = "unknown"
    solver_name = "slurryFoam"
    log_file = f"log.{solver_name}"
    try:
        # Search for log file in the first subfolder
        if subfolders:
            log_path = os.path.join(subfolders[0], log_file)
            if os.path.isfile(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "version:" in line:
                            # Extract just the version string after "version:"
                            solver_version = line.split("version:")[-1].strip()
                            break
    except Exception as e:
        print(f"Warning: could not retrieve solver version: {e}")

    tokens = [cfg["token"] for cfg in configs]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"{csv_path} | Solver: {solver_name} {solver_version}"])
        writer.writerow(["case"] + tokens)
        for case_dir, results in all_results.items():
            row = [os.path.basename(case_dir)] + [
                format_collected_value(results.get(t), enabled=t not in no_format_tokens)
                for t in tokens
            ]
            writer.writerow(row)

    print(f"Results saved to: {csv_path}")

    try:
    #    plot_path = generate_crossplot_from_csv(Path(csv_path))
        print(f"Crossplot saved to: {plot_path}")
    except Exception as exc:
        print(f"Warning: crossplot generation skipped: {exc}")
