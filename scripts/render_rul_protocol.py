"""Render the simple vertical draw.io protocol when Desktop export is unavailable.

Reads labels, geometry and colors from native XML; supports this diagram only.
The PNG is a static preview, while the .drawio remains the editable source.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "assets/diagrams/rul-improvement-plan.drawio"
    tree = ET.parse(source)
    cells = tree.findall(".//mxCell")
    vertices = {cell.attrib["id"]: cell for cell in cells if cell.get("vertex") == "1"}
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set(xlim=(0, 1000), ylim=(1100, 0))
    ax.axis("off")
    geometry = {}
    for name, cell in vertices.items():
        geo = cell.find("mxGeometry")
        x, y, w, h = [float(geo.get(key)) for key in ("x", "y", "width", "height")]
        geometry[name] = (x, y, w, h)
        style = dict(item.split("=", 1) for item in cell.get("style", "").split(";") if "=" in item)
        if "fillColor" in style:
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=9", facecolor=style["fillColor"], edgecolor=style.get("strokeColor", "gray"), linestyle="--" if name == "group" else "-", zorder=0 if name == "group" else 2))
        top = name == "group"
        ax.text(x+w/2, y+20 if top else y+h/2, cell.get("value", "").replace("<br>", "\n"), ha="center", va="top" if top else "center", fontsize=float(style.get("fontSize", 17))*.72, fontfamily="Arial", weight="bold" if name == "title" else "normal", zorder=3)
    for cell in cells:
        if cell.get("edge") != "1":
            continue
        x,y,w,h = geometry[cell.get("source")]
        tx,ty,tw,th = geometry[cell.get("target")]
        ax.annotate("", xy=(tx+tw/2,ty), xytext=(x+w/2,y+h), arrowprops=dict(arrowstyle="-|>", color="#4b5563", lw=1.3), zorder=4)
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    fig.savefig(source.with_suffix(".png"), dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
