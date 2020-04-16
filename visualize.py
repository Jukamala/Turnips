import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from prices import RandomPattern


def func1(val, tick_number):
    if int(val) not in range(0, 12):
        return ""
    return ["%s %s" % (d, t) for d in ["MO", "DI", "MI", "DO", "FR", "SA"] for t in ["AM", "PM"]][int(val)]


def half_get(l, i):
    if i - int(i) < 0.5:
        return l[int(i)]
    else:
        return (l[int(i)] + l[int(i+1)])/2


def visual(base_price, knowns, speed=0.001, quantile=(0.2, 0.5, 0.8), cols=None):
    # Plot colors and labels
    if cols is None:
        cols = ["darkorange"] * len(quantile)
    cols = [c for c in cols for _ in range(2)]
    labs = [x for q in quantile for x in [{"label": q}, dict()]]

    prediction = [RandomPattern(base_price=base_price, knowns=knowns, tol=speed).probs(i) for i in range(1, 13)]

    start = 0
    for d in range(0, 12):
        if len(prediction[d]) == 0:
            raise ValueError("Invalid inputs")
        elif len(prediction[d]) == 1:
            start = d
        else:
            break

    max_val = max([max([x for x in p.values() if x < 0.99], default=0) for p in prediction])
    min_key = min([min(p.keys()) for p in prediction])
    max_key = max([max(p.keys()) for p in prediction])

    # Extrema, Expectation, (std deviation)
    max_vals = [max(p.keys()) for p in prediction]
    min_vals = [min(p.keys()) for p in prediction]
    exep = np.array([sum([val * pro for val, pro in p.items()]) for p in prediction])
    vari = np.sqrt(np.array([sum([(val - e)**2 * pro for val, pro in p.items()]) for p, e in zip(prediction, exep)]))
    # print(exep)
    # print(vari)

    # Quantile
    tracks = []
    for q in quantile:
        a = []
        b = []
        for i in range(0, 12):
            abv = np.ceil(exep[i])
            bel = np.floor(exep[i])
            abv_sum = prediction[i].get(abv, 0)
            bel_sum = prediction[i].get(bel, 0) if abv != bel else 0
            while abv_sum + bel_sum < q:
                print(abv, bel, abv_sum + bel_sum)
                if abv_sum < bel_sum and abv < max_vals[i]:
                    abv += 1
                    abv_sum += prediction[i].get(abv, 0)
                else:
                    bel -= 1
                    bel_sum += prediction[i].get(bel, 0)
            a += [abv]
            b += [bel]
        tracks += [a]
        tracks += [b]

    fig, ax = plt.subplots()

    x_pol = np.arange(start, 12)
    x_pol_half = np.arange(start, 11.1, 0.5)
    all_x = np.arange(0, 12)
    all_x_half = np.arange(0, 11.1, 0.5)
    x_new = np.linspace(start, 11, 200)
    all_x_new = np.linspace(0, 11, 200)

    max_vals_pol = [half_get(max_vals, i) for i in x_pol_half]
    min_vals_pol = [half_get(min_vals, i) for i in x_pol_half]
    exep_vals_pol = [half_get(exep, i) for i in all_x_half]
    tracks_vals_pol = [[half_get(tr, i) for i in x_pol_half] for tr in tracks]
    maxpol = interp1d(x_pol_half, max_vals_pol, kind=5)
    minpol = interp1d(x_pol_half, min_vals_pol, kind=5)
    exep_pol = interp1d(all_x_half, exep_vals_pol, kind=5)
    tracks_pol = [interp1d(x_pol_half, tr, kind=5) for tr in tracks_vals_pol]

    x = np.arange(1, 13)
    y = np.arange(0, max_key + 22)
    z = np.array([prediction[i - 1].get(j, 0) for j in y for i in x])
    Z = z.reshape(len(y), len(x))

    ax.imshow(Z, aspect=6/len(y), vmin=0, vmax=max_val, cmap='Spectral_r', alpha=0.5)
    ax.plot(x_new, maxpol(x_new), c='gold', label='max')
    ax.plot(x_pol, max_vals[start:], "o", c='gold')
    ax.plot(x_new, minpol(x_new), c='gold', label='min')
    ax.plot(x_pol, min_vals[start:], "o", c='gold')
    ax.plot(all_x_new, exep_pol(all_x_new), c='darkred', label='expectation')
    ax.plot(all_x, exep, "o", c='darkred')
    [(ax.plot(x_new, f(x_new), c=col, **lab),
      ax.plot(x_pol, tr[start:], "o", c=col)) for (f, tr_pol, tr, col, lab) in
     zip(tracks_pol, tracks_vals_pol, tracks, cols, labs)]

    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(func1))
    ax.set_ylim(max(0, min_key - 20), max_key + 21)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    visual(base_price=98, knowns={1: 94, 2: 95, 3: 75, 4: 67, 5: 59, 6: 102, 7: 77},
           quantile=[0.5, 0.8], cols=["crimson", "orangered"], speed=0.001)

    # Only Expectation
    visual(base_price=98, knowns={1: 94, 2: 95, 3: 75, 4: 67, 5: 59, 6: 102, 7: 77}, quantile=[], speed=0.001)
