"""Data-driven cut-point detection for a ranked score curve.

Given per-feature scores sorted in *descending* order, the sorted curve looks
like a few highly-scored features followed by a long, flat tail of near-noise.
The "elbow" is the index where the marginal value of keeping the
next feature collapses -- the natural place to cut.

This module finds that index from the geometry of the curve itself, so the
number of features kept adapts to each dataset instead of being a fixed
fraction (the arbitrary 50th-percentile cut).

Nothing here depends on WL, the vocab, or labels: it consumes a score array and
returns where to cut.
"""

from collections import namedtuple

import numpy as np

# A resolved cut on the sorted curve: keep ``n_keep`` features (indices
# ``0..idx``), whose last kept score is ``threshold``. ``idx == n_keep - 1``.
Cut = namedtuple("Cut", "n_keep idx threshold")


def find_elbow_cut(scores, sorted_desc=True, min_keep=1):
    """Find the elbow of a decreasing score curve.

    Method: the classic "max distance to the chord". Normalise the sorted
    curve to the unit square, draw the straight line from the first point to
    the last, and pick the point farthest from that line -- for a convex
    decreasing curve that is exactly the elbow.

    Parameters
    ----------
    scores : array-like of float
        Feature scores, assumed sorted descending (highest first). Pass
        ``sorted_desc=False`` to have them sorted here.
    min_keep : int, default 1
        Floor on how many features to keep, for near-degenerate curves.

    Returns
    -------
    (n_keep, threshold) : (int, float)
        Keep ``n_keep`` features; ``threshold`` is the score of the last kept
        one (equivalently: keep every feature with ``score >= threshold``).
    """
    scores = np.asarray(scores, dtype=float)
    y = scores if sorted_desc else np.sort(scores)[::-1]
    n = y.size

    if n < 3 or (y[0] - y[-1]) == 0:          # too short or flat -> keep all
        return n, float(y[-1])

    # normalise both axes to [0, 1] so distance is scale-invariant
    x_n = np.arange(n, dtype=float) / (n - 1)
    y_n = (y - y[-1]) / (y[0] - y[-1])

    # perpendicular distance of every point to the chord (0,1)->(1,0)
    x1, y1, x2, y2 = x_n[0], y_n[0], x_n[-1], y_n[-1]
    denom = np.hypot(y2 - y1, x2 - x1)
    dist = np.abs((y2 - y1) * x_n - (x2 - x1) * y_n + (x2 * y1 - y2 * x1)) / denom

    elbow = int(np.argmax(dist))
    n_keep = min(max(elbow + 1, int(min_keep)), n)
    return n_keep, float(y[n_keep - 1])


def find_energy_cut(scores, energy=0.99, sorted_desc=True, min_keep=1):
    """Keep the top features that together hold a fraction of the total score.

    An alternative to the geometric elbow. Where ``find_elbow_cut`` asks *where
    does the marginal value of the next feature collapse*, this asks *how many
    of the top features do I need to retain X% of the signal*. On the brutally
    front-loaded curves this pipeline produces, the elbow is statistically
    conservative -- it cuts as soon as individual scores go flat, discarding a
    long tail of weak-but-not-worthless features that still lift downstream AUC.
    An energy threshold reaches deliberately into that tail: it walks the sorted
    curve accumulating score and stops once the cumulative sum first reaches
    ``energy`` of the grand total.

    The score used here is exactly the ranking score -- ``total_presence *
    discriminative_score`` in the WL encoder -- so "energy" means "fraction of
    the summed discriminative mass", not an information-theoretic quantity.

    Parameters
    ----------
    scores : array-like of float
        Feature scores, assumed sorted descending (highest first). Pass
        ``sorted_desc=False`` to have them sorted here. Assumed non-negative,
        as the WL scores are; a cumulative fraction is only meaningful then.
    energy : float, default 0.99
        Target fraction of the total score to retain, in (0, 1]. 0.99 keeps
        enough features to cover 99% of the summed score. Larger keeps more.
    min_keep : int, default 1
        Floor on how many features to keep, for near-degenerate curves.

    Returns
    -------
    (n_keep, threshold) : (int, float)
        Keep ``n_keep`` features; ``threshold`` is the score of the last kept
        one (equivalently: keep every feature with ``score >= threshold``).
    """
    scores = np.asarray(scores, dtype=float)
    y = scores if sorted_desc else np.sort(scores)[::-1]
    n = y.size

    total = float(y.sum())
    if n == 0:
        return 0, 0.0
    if total <= 0:                            # flat / all-zero -> keep all
        return n, float(y[-1])

    # smallest prefix whose cumulative score first reaches energy * total.
    # searchsorted finds the first index where the running sum crosses the
    # target; +1 turns that 0-based index into a count.
    target = float(energy) * total
    idx = int(np.searchsorted(np.cumsum(y), target))
    n_keep = min(max(idx + 1, int(min_keep)), n)
    return n_keep, float(y[n_keep - 1])


# --- plotting -------------------------------------------------------------
# The curve is brutally front-loaded: a couple of hundred features carry the
# signal and the remaining ~97% is a flat tail. One linear plot cannot show
# both ends, so every run renders five views of the same curve:
#
#   elbow.png         linear, full range. The reference figure -- the chord and
#                     the max-distance geometry are only truthful here.
#   elbow_zoom.png    the same linear axes cropped to the steep head.
#   elbow_logx.png    log rank: spreads the head across the page.
#   elbow_loglog.png  log rank + log score: additionally exposes the tail's
#                     decay, which every linear view pins to zero.
#   elbow_cdf.png     the *cumulative* view ("Curve B"): the running score mass
#                     normalised to end at 1. The four above plot the raw score
#                     (Curve A); this plots its integral, so the height at any
#                     rank reads directly as "fraction of total signal kept" --
#                     the quantity the energy cut thresholds.
#
# (log-x + log-y *is* log-log -- one figure, not two.)
#
# The chord is drawn only on the linear figures. It is a straight line in
# linear space by construction, so on a log axis it would render as a curve
# that invites reading the elbow off geometry the method never used.

PCT_COLORS = {25: "#9467bd", 50: "#d62728", 75: "#8c564b"}
ELBOW_COLOR = "#ff7f0e"          # orange: the geometric elbow
ENERGY_COLOR = "#17becf"         # teal:   the cumulative-energy cut


def _plt():
    import matplotlib
    matplotlib.use("Agg")            # headless: no display needed
    import matplotlib.pyplot as plt
    return plt


def _prepare(scores, sorted_desc, min_keep, selection="elbow", energy=0.99):
    """Sorted curve + both candidate cuts: the geometry every figure shares.

    Every figure draws *both* the geometric elbow and the cumulative-energy cut
    so they can be compared, and shades the one named by ``selection`` as the
    cut actually applied. ``selection`` therefore only decides emphasis, not
    which cuts are computed.

    ``selection="none"`` means no cut was applied at all (the encoder kept its
    full scored vocabulary -- see the comparison arms in ``pca/``). Both cuts are
    still drawn, for reference, but neither is flagged as selected and the shaded
    "kept" region covers the whole curve, which is the truth for such a run.

    Returns
    -------
    (y, n, elbow, energy_cut, active) : (ndarray, int, Cut, Cut, str)
        ``active`` is ``selection`` ("elbow", "energy" or "none") -- the cut that
        was really used to trim the vocab, shaded green in the figures.
    """
    scores = np.asarray(scores, dtype=float)
    y = scores if sorted_desc else np.sort(scores)[::-1]
    n = y.size

    ek, et = find_elbow_cut(y, sorted_desc=True, min_keep=min_keep)
    elbow = Cut(ek, ek - 1, et)

    gk, gt = find_energy_cut(y, energy=energy, sorted_desc=True, min_keep=min_keep)
    energy_cut = Cut(gk, gk - 1, gt)

    active = selection if selection in ("elbow", "energy", "none") else "elbow"
    return y, n, elbow, energy_cut, active


def _active_cut(active, elbow, energy_cut, n, y):
    """The cut the encoder actually applied, as a ``Cut``.

    ``"none"`` resolves to the whole curve, so every figure's shaded region and
    zoom window follow from one rule instead of each view special-casing it.
    """
    if active == "energy":
        return energy_cut
    if active == "none":
        return Cut(n, n - 1, float(y[-1]))
    return elbow


def _summary_box(ax, n, elbow, energy_cut, active, energy, x=0.985, y=0.60):
    """Both data-driven cuts vs the fixed-fraction cuts, as a monospace panel.

    The active cut (the one actually applied to the vocab) is flagged with a
    ``<`` marker so the figure and the trained model cannot silently disagree.
    """
    def _tag(name):
        return " <-- selected" if name == active else ""

    lines = [
        f"total features : {n}",
        f"elbow keep     : {elbow.n_keep} ({100.0 * elbow.n_keep / n:.1f}%)"
        + _tag("elbow"),
        f"energy {energy:<5.4g} keep: {energy_cut.n_keep} "
        f"({100.0 * energy_cut.n_keep / n:.1f}%)" + _tag("energy"),
        f"energy threshold: {energy_cut.threshold:.6g}",
        f"25th pct keep  : {min(int(round(n * 0.25)), n)}",
        f"50th pct keep  : {min(int(round(n * 0.50)), n)}",
        f"75th pct keep  : {min(int(round(n * 0.75)), n)}",
    ]
    if active == "none":
        # Neither cut carries the "selected" tag in this case, so say plainly
        # what was applied rather than leaving the reader to infer it.
        lines.insert(1, f"cut applied    : none (all {n} kept)")
    ax.text(
        x, y, "\n".join(lines), transform=ax.transAxes,
        fontsize=8, family="monospace", ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )


def _save(fig, save_path):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    _plt().close(fig)
    return save_path


def _draw_linear(ax, y, n, elbow, energy_cut, active, energy, title, hi=None):
    """The shared linear figure: curve, chord, percentiles, both cuts.

    Draws the geometric elbow (orange) and the cumulative-energy cut (teal), and
    shades [0, active] green for whichever one was actually applied. ``hi`` crops
    the rank axis to [0, hi] for the zoomed view; marks outside that window are
    skipped rather than drawn off-screen.
    """
    ranks = np.arange(n)
    span = y[0] - y[-1] + 1e-12
    x_span = hi if hi else n         # scales the annotation offsets to the view

    active_cut = _active_cut(active, elbow, energy_cut, n, y)

    ax.plot(ranks, y, color="#1f77b4", lw=1.6, label="feature score (sorted)")
    ax.axvspan(0, active_cut.idx, color="#2ca02c", alpha=0.08)

    # chord used by the max-distance elbow method (provenance detail)
    ax.plot([0, n - 1], [y[0], y[-1]], color="#7f7f7f", ls=":", lw=1.2,
            label="chord (max-distance method)")

    # 25th / 50th / 75th percentile margins (rank-based fixed-fraction cuts)
    for p, c in PCT_COLORS.items():
        idx = min(int(round(n * p / 100.0)), n - 1)
        if hi and idx > hi:          # off the cropped view entirely
            continue
        ax.axvline(idx, color=c, ls="--", lw=1.0, alpha=0.8)
        ax.annotate(
            f"{p}th pct\n(keep {idx})",
            xy=(idx, y[idx]), xytext=(idx, y[idx] + 0.06 * span),
            color=c, fontsize=8, ha="center", va="bottom",
        )

    # elbow marker (orange)
    sel = " (selected)" if active == "elbow" else ""
    ax.scatter([elbow.idx], [y[elbow.idx]], color=ELBOW_COLOR, zorder=5, s=70,
               edgecolor="black", linewidth=0.6,
               label=f"elbow: keep {elbow.n_keep} "
                     f"({100.0 * elbow.n_keep / n:.1f}%){sel}")
    ax.annotate(
        f"elbow @ {elbow.idx}\nthreshold={elbow.threshold:.4g}",
        xy=(elbow.idx, y[elbow.idx]),
        xytext=(elbow.idx + 0.04 * x_span, y[elbow.idx] + 0.15 * span),
        arrowprops=dict(arrowstyle="->", color=ELBOW_COLOR),
        fontsize=9, color=ELBOW_COLOR,
    )

    # energy cut (teal): vertical line + marker, unless cropped out of view
    if not (hi and energy_cut.idx > hi):
        sel = " (selected)" if active == "energy" else ""
        ax.axvline(energy_cut.idx, color=ENERGY_COLOR, ls="--", lw=1.1, alpha=0.9)
        ax.scatter([energy_cut.idx], [y[energy_cut.idx]], color=ENERGY_COLOR,
                   zorder=5, s=70, edgecolor="black", linewidth=0.6,
                   label=f"energy {energy:g}: keep {energy_cut.n_keep} "
                         f"({100.0 * energy_cut.n_keep / n:.1f}%){sel}")
        ax.annotate(
            f"energy {energy:g} @ {energy_cut.idx}\nthreshold={energy_cut.threshold:.4g}",
            xy=(energy_cut.idx, y[energy_cut.idx]),
            xytext=(energy_cut.idx + 0.02 * x_span, y[energy_cut.idx] + 0.28 * span),
            arrowprops=dict(arrowstyle="->", color=ENERGY_COLOR),
            fontsize=9, color=ENERGY_COLOR,
        )

    if hi:
        ax.axhline(active_cut.threshold, color="#2ca02c", ls=":", lw=0.9, alpha=0.8)
        ax.set_xlim(0, hi)

    ax.set_xlabel("feature rank (sorted by score, descending)")
    ax.set_ylabel("discriminative score")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    _summary_box(ax, n, elbow, energy_cut, active, energy)


def _title(kind, title, n):
    return (f"WL feature-selection elbow{kind}"
            + (f" — {title}" if title else "")
            + f"\ntotal features: {n}")


def plot_elbow_curve(scores, save_path, sorted_desc=True, min_keep=1, title=None,
                     selection="elbow", energy=0.99):
    """Render the sorted-score curve with both cuts and save it to disk.

    The figure shows, on the decreasing score curve:
      * the elbow point and the number of features it keeps,
      * the cumulative-energy cut and the number *it* keeps,
      * whichever of the two ``selection`` names is shaded green as the cut
        actually applied to the vocab,
      * the 25th / 50th / 75th percentile margins (rank-based cuts) so both
        data-driven cuts can be compared against fixed-fraction cuts such as
        the old 50th-percentile rule,
      * the chord used by the max-distance method, for provenance,
      * a summary box of counts / thresholds.

    Parameters
    ----------
    scores : array-like of float
        Feature scores (assumed sorted descending; pass ``sorted_desc=False``
        otherwise). The full curve, *not* just the kept portion.
    save_path : str
        Where to write the PNG. Parent directories are created if missing.
    title : str, optional
        Extra context (e.g. "wl_fddl_gpu / nci_full") shown in the title.
    selection : {"elbow", "energy"}, default "elbow"
        Which cut was actually applied; shaded green and flagged "selected".
    energy : float, default 0.99
        The energy fraction whose cut is drawn (matches the encoder's setting).

    Returns
    -------
    str
        The path the figure was written to.
    """
    plt = _plt()
    y, n, elbow, energy_cut, active = _prepare(scores, sorted_desc, min_keep,
                                               selection, energy)

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_linear(ax, y, n, elbow, energy_cut, active, energy, _title("", title, n))
    return _save(fig, save_path)


def plot_elbow_curve_zoom(scores, save_path, sorted_desc=True, min_keep=1, title=None,
                          zoom_to=None, selection="elbow", energy=0.99):
    """The linear figure cropped to the steep head.

    Identical geometry to ``plot_elbow_curve``, just a narrower rank axis. The
    point of zooming rather than reaching for a log axis is that the chord
    stays straight and the cuts stay where their methods actually put them, so
    nothing in the figure has to be mentally un-distorted.

    Parameters
    ----------
    zoom_to : int, optional
        Highest rank shown. Defaults to 4x the *active* cut, which keeps that
        cut about a quarter of the way into the window. Basing the window on
        the active cut matters when energy is selected: its keep count can be
        an order of magnitude past the elbow, and a window sized to the elbow
        would crop the energy marker straight off the figure.
    """
    plt = _plt()
    y, n, elbow, energy_cut, active = _prepare(scores, sorted_desc, min_keep,
                                               selection, energy)
    # With no cut applied the "active" region is the whole curve, which would
    # make the zoom window the full range and defeat the point of this view.
    # Fall back to framing the elbow, which is what the head of the curve is.
    active_keep = (elbow.n_keep if active == "none"
                   else _active_cut(active, elbow, energy_cut, n, y).n_keep)
    hi = min(int(zoom_to) if zoom_to else max(4 * active_keep, 50), n - 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_linear(ax, y, n, elbow, energy_cut, active, energy,
                 _title(f" — zoom on ranks 0–{hi}", title, n), hi=hi)
    return _save(fig, save_path)


def plot_elbow_curve_logx(scores, save_path, sorted_desc=True, min_keep=1, title=None,
                          selection="elbow", energy=0.99):
    """Same curve on a logarithmic rank axis.

    Adjustments the log axis forces:
      * ranks are shifted to 1-based, because log(0) is undefined;
      * the chord is not drawn -- it is straight only in linear space, so here
        it would render as a curve and invite reading the elbow off geometry
        the method never used. It stays on the linear figures.
      * the percentile labels are staggered vertically -- log rank crushes the
        25/50/75% marks into the last third of the page, where they would
        otherwise overprint each other.

    Both cuts are still the linear-space ones: ``find_elbow_cut`` /
    ``find_energy_cut`` are computed on the linear curve and this figure only
    re-draws their result. Log rank is a lens on the curve, not a different cut
    rule -- hence the footnote.
    """
    plt = _plt()
    y, n, elbow, energy_cut, active = _prepare(scores, sorted_desc, min_keep,
                                               selection, energy)
    x = np.arange(1, n + 1)          # 1-based: log(0) is undefined
    span = y[0] - y[-1] + 1e-12
    active_idx = _active_cut(active, elbow, energy_cut, n, y).idx

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")

    ax.plot(x, y, color="#1f77b4", lw=1.6, label="feature score (sorted)")
    ax.axvspan(1, active_idx + 1, color="#2ca02c", alpha=0.08)

    for (p, c), lift in zip(PCT_COLORS.items(), (0.22, 0.14, 0.06)):
        idx = min(int(round(n * p / 100.0)), n - 1)
        ax.axvline(idx + 1, color=c, ls="--", lw=1.0, alpha=0.8)
        ax.annotate(
            f"{p}th pct (keep {idx})",
            xy=(idx + 1, y[idx]), xytext=(idx + 1, y[idx] + lift * span),
            color=c, fontsize=8, ha="right", va="bottom",
        )

    sel = " (selected)" if active == "elbow" else ""
    ax.scatter([elbow.idx + 1], [y[elbow.idx]], color=ELBOW_COLOR, zorder=5, s=70,
               edgecolor="black", linewidth=0.6,
               label=f"elbow: keep {elbow.n_keep} "
                     f"({100.0 * elbow.n_keep / n:.1f}%){sel}")
    ax.annotate(
        f"elbow @ {elbow.idx}\nthreshold={elbow.threshold:.4g}",
        xy=(elbow.idx + 1, y[elbow.idx]),
        # sits above the staggered percentile labels, not across them
        xytext=(max((elbow.idx + 1) * 1.3, 2.0), y[elbow.idx] + 0.45 * span),
        arrowprops=dict(arrowstyle="->", color=ELBOW_COLOR),
        fontsize=9, color=ELBOW_COLOR, ha="left",
    )

    sel = " (selected)" if active == "energy" else ""
    ax.axvline(energy_cut.idx + 1, color=ENERGY_COLOR, ls="--", lw=1.1, alpha=0.9)
    ax.scatter([energy_cut.idx + 1], [y[energy_cut.idx]], color=ENERGY_COLOR,
               zorder=5, s=70, edgecolor="black", linewidth=0.6,
               label=f"energy {energy:g}: keep {energy_cut.n_keep} "
                     f"({100.0 * energy_cut.n_keep / n:.1f}%){sel}")
    ax.annotate(
        f"energy {energy:g} @ {energy_cut.idx}\nthreshold={energy_cut.threshold:.4g}",
        xy=(energy_cut.idx + 1, y[energy_cut.idx]),
        xytext=(max((energy_cut.idx + 1) * 1.3, 2.0), y[energy_cut.idx] + 0.28 * span),
        arrowprops=dict(arrowstyle="->", color=ENERGY_COLOR),
        fontsize=9, color=ENERGY_COLOR, ha="left",
    )

    ax.set_xlabel("feature rank (1-based, log scale)")
    ax.set_ylabel("discriminative score")
    ax.set_title(_title(" — log rank", title, n))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)
    _summary_box(ax, n, elbow, energy_cut, active, energy)
    ax.text(
        0.5, -0.13,
        "log rank only rescales the view: both cuts are computed in linear rank space "
        "(see the linear figures for the chord the elbow is measured against).",
        transform=ax.transAxes, fontsize=7.5, color="#666666", ha="center", va="top",
    )

    return _save(fig, save_path)


def plot_elbow_curve_loglog(scores, save_path, sorted_desc=True, min_keep=1, title=None,
                            selection="elbow", energy=0.99):
    """Same curve on log rank *and* log score (i.e. log-log).

    Adjustments the log-log axes force, on top of the log-rank ones:
      * scores of exactly 0 -- a feature equally present in both classes scores
        0 -- cannot be drawn on a log axis at all. They are dropped and counted
        in the footnote rather than silently clipped to some invented floor.
        A cut whose own threshold is 0 (its last kept feature carries no signal)
        therefore has no marker either -- only its vertical rank line is drawn.
      * the chord is not drawn, for the same reason as on the log-rank figure.

    This is the only view in which the tail is legible; every linear view pins
    it to zero. It is therefore the view that shows whether the tail decays
    like a power law (a straight line here) or dies off faster -- and where the
    energy cut, which lives out in that tail, is most clearly placed.
    """
    plt = _plt()
    y, n, elbow, energy_cut, active = _prepare(scores, sorted_desc, min_keep,
                                               selection, energy)
    x = np.arange(1, n + 1)

    pos = y > 0                      # log(<=0) is undefined -> cannot be shown
    n_dropped = int((~pos).sum())
    if not pos.any():                # degenerate: nothing to draw on log-y
        pos = np.ones_like(y, dtype=bool)
        n_dropped = 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    active_idx = _active_cut(active, elbow, energy_cut, n, y).idx

    ax.plot(x[pos], y[pos], color="#1f77b4", lw=1.6, label="feature score (sorted)")
    ax.axvspan(1, active_idx + 1, color="#2ca02c", alpha=0.08)

    # labels omitted here: log-log leaves no clear band for them
    for p, c in PCT_COLORS.items():
        idx = min(int(round(n * p / 100.0)), n - 1)
        ax.axvline(idx + 1, color=c, ls="--", lw=1.0, alpha=0.8)

    if y[elbow.idx] > 0:
        sel = " (selected)" if active == "elbow" else ""
        ax.scatter([elbow.idx + 1], [y[elbow.idx]], color=ELBOW_COLOR, zorder=5,
                   s=70, edgecolor="black", linewidth=0.6,
                   label=f"elbow: keep {elbow.n_keep} "
                         f"({100.0 * elbow.n_keep / n:.1f}%){sel}")
        ax.axhline(elbow.threshold, color=ELBOW_COLOR, ls=":", lw=0.9, alpha=0.7)
        ax.annotate(
            f"elbow @ {elbow.idx}\nthreshold={elbow.threshold:.4g}",
            xy=(elbow.idx + 1, y[elbow.idx]),
            xytext=(max((elbow.idx + 1) * 3.0, 4.0), y[elbow.idx] * 4.0),
            arrowprops=dict(arrowstyle="->", color=ELBOW_COLOR),
            fontsize=9, color=ELBOW_COLOR,
        )

    # energy cut: the rank line always renders; the marker/threshold need a
    # positive score to sit on the log-y axis.
    sel = " (selected)" if active == "energy" else ""
    ax.axvline(energy_cut.idx + 1, color=ENERGY_COLOR, ls="--", lw=1.1, alpha=0.9,
               label=f"energy {energy:g}: keep {energy_cut.n_keep} "
                     f"({100.0 * energy_cut.n_keep / n:.1f}%){sel}")
    if y[energy_cut.idx] > 0:
        ax.scatter([energy_cut.idx + 1], [y[energy_cut.idx]], color=ENERGY_COLOR,
                   zorder=5, s=70, edgecolor="black", linewidth=0.6)
        ax.annotate(
            f"energy {energy:g} @ {energy_cut.idx}\nthreshold={energy_cut.threshold:.4g}",
            xy=(energy_cut.idx + 1, y[energy_cut.idx]),
            xytext=(max((energy_cut.idx + 1) * 1.5, 4.0), y[energy_cut.idx] * 4.0),
            arrowprops=dict(arrowstyle="->", color=ENERGY_COLOR),
            fontsize=9, color=ENERGY_COLOR,
        )

    ax.set_xlabel("feature rank (1-based, log scale)")
    ax.set_ylabel("discriminative score (log scale)")
    ax.set_title(_title(" — log rank / log score", title, n))
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)
    _summary_box(ax, n, elbow, energy_cut, active, energy, y=0.98)

    note = ("log axes only rescale the view: both cuts are computed in linear rank "
            "space (see the linear figures for the chord the elbow is measured against).")
    if n_dropped:
        note += (f"\n{n_dropped} of {n} features score exactly 0 and cannot be drawn "
                 f"on a log axis; they are omitted from this view.")
    ax.text(
        0.5, -0.13, note, transform=ax.transAxes,
        fontsize=7.5, color="#666666", ha="center", va="top",
    )

    return _save(fig, save_path)


def plot_cdf_curve(scores, save_path, sorted_desc=True, min_keep=1, title=None,
                   selection="elbow", energy=0.99):
    """Render "Curve B": the cumulative score-mass CDF, and save it to disk.

    Every other figure in this module plots the raw score against rank ("Curve
    A"). This one plots its *running integral, normalised by the total*::

        F(k) = (sum of the top-k scores) / (sum of all scores)

    which is a genuine CDF over the score-mass measure: non-decreasing, and 1.0
    at the last feature by construction (dividing by the total is exactly what
    pins the right end to 1). The height at rank ``k`` therefore reads directly
    as "fraction of the total discriminative mass held by the top ``k``
    features" -- the quantity the energy cut thresholds. The energy cut is just
    the horizontal line ``F = energy`` and where it first meets the curve.

    Drawn on a log rank axis: these curves are so front-loaded that on linear
    ranks the CDF shoots to ~0.9 within the first couple hundred of ~10k
    features and then looks like a flat line at 1.0, hiding everything. Log rank
    spreads the head where all the action is. Both cuts are marked, each
    annotated with the mass fraction it actually covers, so the gap between "how
    many features" and "how much signal" is visible at a glance.
    """
    plt = _plt()
    y, n, elbow, energy_cut, active = _prepare(scores, sorted_desc, min_keep,
                                               selection, energy)
    total = float(y.sum())
    # F is the CDF; the degenerate all-zero curve has no mass to accumulate, so
    # fall back to a uniform ramp purely to keep the axes drawable.
    F = np.cumsum(y) / total if total > 0 else np.linspace(1.0 / n, 1.0, n)
    x = np.arange(1, n + 1)                   # 1-based ranks: log(0) undefined
    active_idx = _active_cut(active, elbow, energy_cut, n, y).idx

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")

    ax.plot(x, F, color="#1f77b4", lw=1.8, label="Curve B: cumulative score fraction (CDF)")
    ax.axvspan(1, active_idx + 1, color="#2ca02c", alpha=0.08)

    # the right end is pinned to 1.0 -- the whole reason we divide by the total
    ax.axhline(1.0, color="#999999", ls=":", lw=1.0)
    ax.text(n, 1.005, "ends at 1.0  (=100% of total score mass)",
            ha="right", va="bottom", fontsize=8, color="#666666")

    # the energy target: a horizontal line at F = energy, met at the energy cut
    ax.axhline(energy, color=ENERGY_COLOR, ls="--", lw=1.2)

    # faint percentile references, for parity with the other figures
    for p, c in PCT_COLORS.items():
        idx = min(int(round(n * p / 100.0)), n - 1)
        ax.axvline(idx + 1, color=c, ls="--", lw=0.8, alpha=0.5)

    # elbow cut: keeps few features, but see how much mass they already hold
    sel = " (selected)" if active == "elbow" else ""
    ax.axvline(elbow.idx + 1, color=ELBOW_COLOR, ls="--", lw=1.0, alpha=0.85)
    ax.scatter([elbow.idx + 1], [F[elbow.idx]], color=ELBOW_COLOR, zorder=5, s=80,
               edgecolor="black", linewidth=0.6,
               label=f"elbow: keep {elbow.n_keep} ({100.0 * elbow.n_keep / n:.1f}%) "
                     f"-> {100.0 * F[elbow.idx]:.1f}% of mass{sel}")

    # energy cut: the smallest prefix whose mass first reaches `energy`
    sel = " (selected)" if active == "energy" else ""
    ax.axvline(energy_cut.idx + 1, color=ENERGY_COLOR, ls="--", lw=1.2)
    ax.scatter([energy_cut.idx + 1], [F[energy_cut.idx]], color=ENERGY_COLOR, zorder=5,
               s=80, edgecolor="black", linewidth=0.6,
               label=f"energy {energy:g}: keep {energy_cut.n_keep} "
                     f"({100.0 * energy_cut.n_keep / n:.1f}%) "
                     f"-> {100.0 * F[energy_cut.idx]:.1f}% of mass{sel}")
    ax.annotate(
        f"{energy:g} of mass\nreached at rank {energy_cut.n_keep}",
        xy=(energy_cut.idx + 1, energy),
        xytext=(max((energy_cut.idx + 1) * 1.6, 3.0), energy - 0.18),
        arrowprops=dict(arrowstyle="->", color=ENERGY_COLOR),
        fontsize=9, color=ENERGY_COLOR, ha="left",
    )

    ax.set_xlabel("feature rank (1-based, log scale)")
    ax.set_ylabel("cumulative fraction of total score mass (CDF)")
    ax.set_title(_title(" — cumulative mass (Curve B / CDF)", title, n))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)
    _summary_box(ax, n, elbow, energy_cut, active, energy, y=0.55)
    ax.text(
        0.5, -0.13,
        "height = fraction of the summed discriminative score held by the top-k features; "
        "the energy cut keeps the fewest features whose height reaches the dashed line.",
        transform=ax.transAxes, fontsize=7.5, color="#666666", ha="center", va="top",
    )

    return _save(fig, save_path)


def plot_elbow_suite(scores, analytics_dir, stem="wl_feature_selection_elbow",
                     sorted_desc=True, min_keep=1, title=None,
                     selection="elbow", energy=0.99):
    """Write all four views of the curve into ``analytics_dir``.

    Every view draws both the elbow and the energy cut; ``selection`` (the
    method the encoder actually applied) and ``energy`` (its fraction) decide
    which cut is shaded as selected and where the energy marker lands. Pass the
    encoder's own ``selection`` / ``energy`` so the figure and the trained vocab
    agree.

    Returns the written paths, in the order described at the top of this
    section (linear, zoomed linear, log rank, log-log, cumulative CDF).
    """
    import os
    variants = (
        (plot_elbow_curve, f"{stem}.png"),
        (plot_elbow_curve_zoom, f"{stem}_zoom.png"),
        (plot_elbow_curve_logx, f"{stem}_logx.png"),
        (plot_elbow_curve_loglog, f"{stem}_loglog.png"),
        (plot_cdf_curve, f"{stem}_cdf.png"),
    )
    return [
        fn(scores, os.path.join(analytics_dir, filename),
           sorted_desc=sorted_desc, min_keep=min_keep, title=title,
           selection=selection, energy=energy)
        for fn, filename in variants
    ]
