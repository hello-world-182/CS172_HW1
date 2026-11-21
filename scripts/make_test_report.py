"""Write a short PDF with the per-folder test0-test4 results from scripts/evaluate.py."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_json", type=Path, help="JSON written by scripts/evaluate.py")
    parser.add_argument("--output", type=Path, default=Path("output/pdf/test_results.pdf"))
    return parser.parse_args()


def score(digit_acc: float, image_acc: float) -> float:
    return 20 * max(digit_acc - 0.95, 0.0) / 0.05 + 20 * max(image_acc - 0.90, 0.0) / 0.10


def main() -> None:
    args = parse_args()
    report = json.loads(args.evaluation_json.read_text(encoding="utf-8"))
    results = report["results"]

    rows, total_images, total_digits_ok, total_images_ok = [], 0, 0.0, 0.0
    for name in sorted(results):
        row = results[name]
        n = row["num_images"]
        failures = n - round(row["image_acc"] * n)
        rows.append([name, str(n), f"{row['digit_acc']:.2%}", f"{row['image_acc']:.2%}", str(failures)])
        total_images += n
        total_digits_ok += row["digit_acc"] * n
        total_images_ok += row["image_acc"] * n
    digit_acc = total_digits_ok / total_images
    image_acc = total_images_ok / total_images
    rows.append(
        ["Aggregate", str(total_images), f"{digit_acc:.2%}", f"{image_acc:.2%}", str(total_images - round(total_images_ok))]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "CAPTCHA recognition: test0-test4 results", fontsize=16, weight="bold")
        fig.text(0.08, 0.91, f"Checkpoint: {report['checkpoint']}", fontsize=8)
        fig.text(0.08, 0.895, f"Model: {report['model']}   Input size: {report['image_size']}   Device: {report['device']}", fontsize=8)

        table_ax = fig.add_axes([0.08, 0.62, 0.84, 0.25])
        table_ax.axis("off")
        table = table_ax.table(
            cellText=rows,
            colLabels=["Test folder", "Images", "Per-digit acc.", "Full-image acc.", "Failures"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        for column in range(5):
            table[len(rows), column].set_text_props(weight="bold")

        fig.text(
            0.08,
            0.59,
            f"Evaluation score = 20*max(d-0.95,0)/0.05 + 20*max(f-0.90,0)/0.10 = {score(digit_acc, image_acc):.2f} / 40",
            fontsize=10,
        )

        chart_ax = fig.add_axes([0.1, 0.18, 0.82, 0.33])
        names = [row[0] for row in rows[:-1]]
        x = range(len(names))
        digit_values = [results[name]["digit_acc"] * 100 for name in names]
        image_values = [results[name]["image_acc"] * 100 for name in names]
        chart_ax.bar([i - 0.2 for i in x], digit_values, width=0.4, label="per-digit")
        chart_ax.bar([i + 0.2 for i in x], image_values, width=0.4, label="full-image")
        for i, (d, f) in enumerate(zip(digit_values, image_values)):
            chart_ax.text(i - 0.2, d + 0.2, f"{d:.1f}", ha="center", fontsize=7)
            chart_ax.text(i + 0.2, f + 0.2, f"{f:.1f}", ha="center", fontsize=7)
        chart_ax.set_xticks(list(x), names)
        chart_ax.set_ylim(min(80.0, min(image_values) - 2), 102)
        chart_ax.set_ylabel("accuracy (%)")
        chart_ax.set_title("Accuracy per test folder")
        chart_ax.grid(axis="y", alpha=0.3)
        chart_ax.legend(loc="lower right")
        pdf.savefig(fig)
        plt.close(fig)
    print(f"report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
