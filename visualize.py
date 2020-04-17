import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from prices import RandomPattern, Decreasing


def func1(val, tick_number):
    if int(val) not in range(0, 12):
        return ""
    return ["%s %s" % (d, t) for d in ["MO", "DI", "MI", "DO", "FR", "SA"] for t in ["AM", "PM"]][int(val)]


def half_get(l, i):
    if i - int(i) < 0.5:
        return l[int(i)]
    else:
        return (l[int(i)] + l[int(i+1)])/2


def ab(x):
    if x % 2 == 0:
        return x - 1
    else:
        return x - 2


def visual(ax, prediction, quantile=[0.8], cols=None):

    """
    # Rescale day-wise
    max_vals = [max(p.keys(), default=0) for p in prediction]
    scaler = [min(max(max_vals), 2 * x) or 1 for x in max_vals]

    show_prediction = [p.copy() for p in prediction]
    # Rescale predictions day-wise
    for (d, pre), sc in zip(enumerate(show_prediction), scaler):
        show_prediction[d] = {k: v / sc for k, v in pre.items()}

    """

    # Plot colors and labels
    if cols is None:
        cols = ["darkorange"] * len(quantile)
    cols = [c for c in cols for _ in range(2)]
    labs = [x for q in quantile for x in [{"label": q}, dict()]]

    start = 0
    end = 12
    for d in range(0, 12):
        if len(prediction[d]) == 0:
            end = d
            break
        elif len(prediction[d]) == 1:
            start = d
        else:
            break

    max_val = max([max([x for x in p.values() if x < 0.20], default=0) for p in prediction])
    min_key = min([x for x in [min(p.keys(), default=None) for p in prediction] if x is not None], default=0)
    max_key = max([x for x in [max(p.keys(), default=None) for p in prediction] if x is not None], default=220)

    # Extrema, Expectation, (std deviation)
    max_vals = [max(p.keys(), default=None) for p in prediction]
    min_vals = [min(p.keys(), default=None) for p in prediction]
    exep = np.array([sum([val * pro for val, pro in p.items()]) for p in prediction])
    vari = np.sqrt(np.array([sum([(val - e)**2 * pro for val, pro in p.items()]) for p, e in zip(prediction, exep)]))
    # print(exep)
    # print(vari)

    # Quantile
    tracks = []
    for q in quantile:
        a = []
        b = []
        for i in range(0, end):
            abv = np.ceil(exep[i])
            bel = np.floor(exep[i])
            abv_sum = prediction[i].get(abv, 0)
            bel_sum = prediction[i].get(bel, 0) if abv != bel else 0
            while abv_sum + bel_sum < q:
                if abv_sum < bel_sum and abv < max_vals[i] or bel <= min_vals[i]:
                    abv += 1
                    abv_sum += prediction[i].get(abv, 0)
                else:
                    bel -= 1
                    bel_sum += prediction[i].get(bel, 0)
            a += [abv]
            b += [bel]
        tracks += [a]
        tracks += [b]

    x_pol = np.arange(start, end)
    x_pol_half = np.arange(start, end - 0.9, 0.5)
    all_x = np.arange(0, end)
    all_x_half = np.arange(0, end - 0.9, 0.5)
    x_new = np.linspace(start, end - 1, 200)
    all_x_new = np.linspace(0, end - 1, 200)

    max_vals_pol = [half_get(max_vals, i) for i in x_pol_half]
    min_vals_pol = [half_get(min_vals, i) for i in x_pol_half]
    exep_vals_pol = [half_get(exep, i) for i in all_x_half]
    tracks_vals_pol = [[half_get(tr, i) for i in x_pol_half] for tr in tracks]

    # Interpolate
    if len(x_pol_half) > 1:
        k = min(3, ab(len(x_pol_half)))
        maxpol = interp1d(x_pol_half, max_vals_pol, kind=k)
        minpol = interp1d(x_pol_half, min_vals_pol, kind=k)
        tracks_pol = [interp1d(x_pol_half, tr, kind=k) for tr in tracks_vals_pol]
    if len(all_x_half) > 1:
        exep_pol = interp1d(all_x_half, exep_vals_pol, kind=min(3, ab(len(all_x_half))))

    x = np.arange(1, 13)
    y = np.arange(0, max_key + 22)
    z = np.array([prediction[i - 1].get(j, 0) for j in y for i in x])
    Z = z.reshape(len(y), len(x))

    ax.imshow(Z, aspect=6/len(y), interpolation='nearest', vmin=0, vmax=max_val, cmap='Spectral_r', alpha=0.5)
    ax.plot(x_pol, max_vals[start: end], "o", c='gold')
    ax.plot(x_pol, min_vals[start: end], "o", c='gold')
    [ax.plot(x_pol, tr[start:], "o", c=col) for (tr, col) in zip(tracks, cols)]
    if len(x_pol_half) > 1:
        [ax.plot(x_new, f(x_new), c=col, **lab) for (f, col, lab) in zip(tracks_pol, cols, labs)]
        ax.plot(x_new, maxpol(x_new), c='gold', label='max')
        ax.plot(x_new, minpol(x_new), c='gold', label='min')
    if len(all_x_half) > 1:
        ax.plot(all_x_new, exep_pol(all_x_new), c='darkred', label='expectation')
    ax.plot(all_x, exep[:end], "o", c='darkred')

    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(func1))
    ax.set_ylim(max(0, min_key - 20), max_key + 21)
    ax.legend()


if __name__ == "__main__":
    base_price = 100
    knowns = {1: 87, 2: 83, 3: 78, 4: 74, 5: 80}
    speed = 0.0005
    prediction = [Decreasing(base_price=base_price, knowns=knowns, tol=speed).probs(i) for i in range(1, 13)]

    visual(prediction, quantile=[0.5, 0.8], cols=["crimson", "orangered"])

    # Only Expectation
    visual(prediction)
