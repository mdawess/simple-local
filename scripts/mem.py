import platform
import re
import subprocess
import sys

PAGE_SIZE = 4096
GB = 1024**3
SWAP_WARN_RATIO = 0.75


def run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def total_ram() -> int:
    return int(run("sysctl", "-n", "hw.memsize") or 0)


def vm_pages() -> dict[str, int]:
    pages = {}
    for line in run("vm_stat").splitlines():
        match = re.match(r"(.+?):\s+(\d+)", line)
        if match:
            pages[match.group(1)] = int(match.group(2)) * PAGE_SIZE
    return pages


def free_percentage() -> str:
    for line in reversed(run("memory_pressure").splitlines()):
        if "percentage" in line:
            return line.split(":")[-1].strip()
    return "?"


def swap_usage() -> tuple[float, float]:
    raw = run("sysctl", "-n", "vm.swapusage")
    used = re.search(r"used = ([\d.]+)M", raw)
    total = re.search(r"total = ([\d.]+)M", raw)
    return (
        float(used.group(1)) / 1024 if used else 0.0,
        float(total.group(1)) / 1024 if total else 0.0,
    )


def processes() -> tuple[list[tuple[str, str, int]], int]:
    """Served models (one llama-server per model, plus the api process) and the
    total of every other process over a gigabyte."""
    served, other = [], 0
    for line in run("ps", "-Ao", "rss=,command=").splitlines():
        rss, _, command = line.strip().partition(" ")
        if not rss.isdigit():
            continue
        size = int(rss) * 1024
        if "llama-server" in command:
            alias = re.search(r"--alias (\S+)", command)
            artifact = re.search(r"--model \S*/([^/\s]+)", command)
            served.append(
                (
                    alias.group(1) if alias else "?",
                    artifact.group(1) if artifact else "",
                    size,
                )
            )
        elif "simple-local serve" in command and "/bin/sh" not in command:
            served.append(("(api process)", "", size))
        elif size > GB:
            other += size
    return served, other


def main() -> None:
    if platform.system() != "Darwin":
        print("mem: macOS only for now (uses vm_stat/memory_pressure)", file=sys.stderr)
        return

    pages = vm_pages()
    evictable = (
        pages.get("Pages free", 0)
        + pages.get("Pages inactive", 0)
        + pages.get("Pages speculative", 0)
    )
    swap_used, swap_total = swap_usage()

    print(
        f"RAM        {total_ram() / GB:5.1f} GB total | "
        f"{evictable / GB:5.1f} GB free+inactive | {free_percentage()} free"
    )
    swapping = swap_total and swap_used / swap_total > SWAP_WARN_RATIO
    warning = "   <- swapping, models are competing for RAM" if swapping else ""
    print(f"Swap       {swap_used:5.1f} GB used of {swap_total:.1f} GB{warning}")
    print(f"Compressed {pages.get('Pages occupied by compressor', 0) / GB:5.1f} GB")

    served, other = processes()
    print()
    if not served:
        print("No models running.")
        return

    print(f"{'Model':<28} {'RSS':>9}   artifact")
    for name, artifact, size in sorted(served, key=lambda row: -row[2]):
        print(f"  {name:<26} {size / GB:6.2f} GB  {artifact}")
    print(
        f"\n  served total: {sum(row[2] for row in served) / GB:.2f} GB"
        f"   other >1GB processes: {other / GB:.2f} GB"
    )
    print(
        "\nllama.cpp mmaps GGUF files, so RSS counts file-backed pages the kernel can"
        "\nevict — treat it as an upper bound. Set mlock: true to pin weights instead."
    )


if __name__ == "__main__":
    main()
