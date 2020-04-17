from prices import RandomPattern, LargeSpike, Decreasing, SmallSpike, generate_pattern
from visualize import visual
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    tol = 0.001

    # Last pattern
    pat = -1
    while pat not in range(5):
        try:
            pat = int(input("What Pattern did you have last week?\n"
                            "Rollercoaster[0], Large Spike[1], Decreasing[2], Small Spike[3], Don't know [4]"))
        except ValueError:
            continue
    if pat < 4:
        pattern = generate_pattern(last_pattern=pat)
    else:
        len_ = -2
        while len_ < -1:
            try:
                len_ = int(input("When did you buy your first turnips?\n"
                                 "This week[0], one week ago[1], ..., Don't know[-1]"))
            except ValueError:
                continue
        if len_ != -1:
            pattern = generate_pattern(weeks=len_)
        else:
            pattern = generate_pattern()

    base_price = int(input("Buy Price"))
    knowns = dict()
    nmb = -1
    while nmb < 0:
        try:
            nmb = int(input("How many prices are known?"))
        except ValueError:
            continue
    print("Enter day and price separated by a space \"<day> <price>\"")
    for _ in range(nmb):
        try:
            d, p = map(int, input().split(" "))
            if d in range(1, 13):
                knowns[d] = p
        except ValueError:
            continue

    predictors = [RandomPattern(base_price=base_price, knowns=knowns, tol=tol),
                  LargeSpike(base_price=base_price, knowns=knowns, tol=tol),
                  Decreasing(base_price=base_price, knowns=knowns, tol=tol),
                  SmallSpike(base_price=base_price, knowns=knowns, tol=tol)]
    rand_pred = [predictors[0].probs(i) for i in range(1, 13)]
    lasp_pred = [predictors[1].probs(i) for i in range(1, 13)]
    decr_pred = [predictors[2].probs(i) for i in range(1, 13)]
    smsp_pred = [predictors[3].probs(i) for i in range(1, 13)]

    ges_pred = [dict() for _ in range(12)]
    for pre, pro in zip([rand_pred, lasp_pred, decr_pred, smsp_pred], pattern):
        tmp = np.array([sum(list(p.values())) for p in pre])
        # print(tmp)
        # Pattern is possible
        if np.all(tmp > 0):
            for d in range(12):
                for price in pre[d]:
                    ges_pred[d][price] = ges_pred[d].get(price, 0) + pre[d][price] * pro

    # Rescale predictions day-wise
    for pred in [rand_pred, lasp_pred, decr_pred, smsp_pred, ges_pred]:
        part = np.array([sum(list(p.values())) for p in pred])
        for (d, pre), prt in zip(enumerate(pred), part):
            if prt == 0:
                pred[d] = dict()
            else:
                pred[d] = {k: v / prt for k, v in pre.items()}

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2)
    visual(ax1, rand_pred, quantile=[0.5, 0.8], cols=["crimson", "orangered"])
    visual(ax2, lasp_pred, quantile=[0.5, 0.8], cols=["crimson", "orangered"])
    visual(ax3, decr_pred, quantile=[0.5, 0.8], cols=["crimson", "orangered"])
    visual(ax4, smsp_pred, quantile=[0.5, 0.8], cols=["crimson", "orangered"])
    plt.show()

    fig, ax = plt.subplots()
    visual(ax, ges_pred, quantile=[0.5, 0.8], cols=["crimson", "orangered"])
    plt.show()
