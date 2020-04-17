import random
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import chain

from tools import shifted_irwin_hall


def decreasing_phase(rate, len_, dec_range, base, end=None, end_rate=None, tol=0.00001):
    """
    Calculates the probabilities on a given day for a decreasing phase
    in the period [start, end]
    At start the price is known or in a possible range (uniformly distributed)
    the price at end can (optionally) be given (to influence probability)

    begin     - the day of the last known price of the phase / starting day
    rate      - price at <begin> or rate range to sample from
    len_      - the # of days from begin to the day to be calculated
    dec_range - how much the rate decreases each halfday
    base      - the base_price of this week

    end       - the day of the next know price of the phase / end day
    end_rate  - price at <end>
    tol       - resolution of the sampling

    Returns a dict where each price is assigned a probability
    """
    percents = dict()

    # Convert price to rate(s) that result in this price
    if not isinstance(rate, list):
        rate = [rate/base, (rate+1)/base]

    # Sample rates (including right border)
    rates = np.arange(rate[0], rate[1] + tol/100, tol)

    # Sample decreases and their probability
    if len_ == 0:
        decs = np.array([0])
        probs = np.array([1])
    else:
        decs = np.arange(dec_range[0] * len_, dec_range[1] * len_ + tol/100, tol)
        probs = shifted_irwin_hall(*dec_range, len_, decs) / len(decs)

    if end is None:
        # Only previous prices influence the wanted day
        for r in rates:
            for price, prob in zip([int(p) for p in np.ceil((r - decs) * base)], probs):
                percents[price] = percents.get(price, 0) + 1/len(rates) * prob
        return percents
    else:
        len2 = end - len_
        # Also sample decreases till end

        if len2 == 0:
            decs2 = np.array([0])
            probs2 = np.array([1])
        else:
            decs2 = np.arange(dec_range[0] * len2, dec_range[1] * len2 + tol/100, tol)
            probs2 = shifted_irwin_hall(*dec_range, len2, decs2) / len(decs2)

        for r in rates:
            # Only use parts of decs that are possible
            pos_prices = [(int(np.ceil((r - d1) * base)), p1*p2)
                          for d1, p1 in zip(decs, probs) for d2, p2 in zip(decs2, probs2)
                          if math.ceil((r - d1 - d2) * base) == end_rate]
            for price, prob in pos_prices:
                percents[price] = percents.get(price, 0) + 1/len(rates) * prob

        # Because some combinations from dec, dec2 could be ruled out rescaling is necessary
        scale = sum(list(percents.values()))
        return {k: v/scale for k, v in percents.items()}


def generate_pattern(last_pattern=None, weeks=None, last_known_pattern=3):
    nxt_pattern_chances = np.array([[0.2, 0.5, 0.25, 0.45],
                                    [0.3, 0.05, 0.45, 0.25],
                                    [0.15, 0.2, 0.05, 0.15],
                                    [0.35, 0.25, 0.25, 0.15]])
    if last_pattern is not None:
        return list(nxt_pattern_chances[:, last_pattern])
    elif weeks is not None:
        chances = nxt_pattern_chances.copy()
        for i in range(weeks):
            chances = np.dot(chances, nxt_pattern_chances)
        return list(chances[:, last_known_pattern])
    else:
        # nxt_pattern_chances**1000
        return [0.346277, 0.247362, 0.147607, 0.258752]


"""
X: "prices in knowns recorded"
B: "price == Z at day"
A_Y: "pattern Y this week"

P(B|X)
= P(B and X) / P(X)
= sum(all Y, P(A_Y) * P(B and X| A_Y)) / P(X)
:= C_B / P(X)

Because:
sum(all B, P(B|X)) = sum(all B, P(B and X)) / P(X) = P(X) / P(X) = 1
We conclude
sum(all B, C_B) / P(X) = 1
And so:
P(B|X) = C_B / sum(all B, C_B)

This justifies the rescaling after computing all C_B
This argument is used for
Pattern - All Patterns
SubPattern - Pattern
Sampling - SubPattern
"""


class RandomPattern:
    """
    Random pattern

    HIGH - DEC - HIGH - DEC - HIGH
    """
    def __init__(self, base_price, knowns=None, tol=0.001):
        self.base_price = base_price
        self.knowns = knowns or dict()
        self.tol = tol

    def gen(self):
        # Phase lengths
        dec1_len = random.randint(2, 3)
        dec2_len = 5 - dec1_len
        inc1_len = random.randint(0, 6)
        inc2_len = random.randint(1, 7 - inc1_len)
        inc3_len = 7 - inc1_len - inc2_len

        # starting rates for decreasing
        rate1 = random.uniform(0.6, 0.8)
        rate2 = random.uniform(0.6, 0.8)

        # decreases from starting rate
        decs1 = np.cumsum([0] + [random.uniform(0.04, 0.1) for _ in range(dec1_len - 1)])
        decs2 = np.cumsum([0] + [random.uniform(0.04, 0.1) for _ in range(dec2_len - 1)])

        # Get prices
        prices = [math.ceil(random.uniform(0.9, 1.4) * self.base_price) for _ in range(inc1_len)]
        prices += [math.ceil((rate1 - dec) * self.base_price) for dec in decs1]
        prices += [math.ceil(random.uniform(0.9, 1.4) * self.base_price) for _ in range(inc2_len)]
        prices += [math.ceil((rate2 - dec) * self.base_price) for dec in decs2]
        prices += [math.ceil(random.uniform(0.9, 1.4) * self.base_price) for _ in range(inc3_len)]

        return prices

    def probs(self, day, verbose=True, plot=False):
        if verbose:
            print(day)
        if day in self.knowns:
            return {self.knowns[day]: 1}
        else:
            percents = dict()
            # All possible phase layouts
            for dec1_len in range(2, 4):
                dec2_len = 5 - dec1_len
                for inc1_len in range(0, 7):
                    for inc2_len in range(1, 8 - inc1_len):
                        inc3_len = 7 - inc1_len - inc2_len
                        starts = np.cumsum([inc1_len, dec1_len, inc2_len, dec2_len, inc3_len])
                        inc_days = list(chain(range(1, starts[0] + 1),
                                        range(starts[1] + 1, starts[2] + 1),
                                        range(starts[3] + 1, starts[4] + 1)))

                        # [P(B|Y)]
                        inc_chances = dict()
                        for rate in np.arange(0.9 + self.tol / 2, 1.4 + self.tol / 2, self.tol):
                            price = math.ceil(rate * self.base_price)
                            inc_chances[price] = inc_chances.get(price, 0) + self.tol / 0.5

                        chance = 1 / 2 * 1 / 7 * 1 / (7 - inc1_len)

                        # Adjust chance to reflect knowns
                        for d in self.knowns:
                            val = self.knowns[d]
                            # INC
                            if d in inc_days:
                                if val not in inc_chances:
                                    chance = 0
                                else:
                                    chance *= inc_chances[self.knowns[d]]
                            # DEC1
                            elif d in range(starts[0] + 1, starts[1] + 1):
                                # Only depends on what was before
                                begin = max([d_ for d_ in range(starts[0] + 1, starts[1] + 1)
                                             if d_ < d and d_ in self.knowns], default=None)
                                phase = decreasing_phase(rate=self.knowns.get(begin, None) or [0.6, 0.8],
                                                         len_=d - (begin or starts[0] + 1), dec_range=[0.04, 0.1],
                                                         base=self.base_price, tol=self.tol)
                                if val not in phase:
                                    chance = 0
                                else:
                                    chance *= phase[val]
                            # DEC2
                            elif d in range(starts[2] + 1, starts[3] + 1):
                                # Only depends on what was before
                                begin = max([d_ for d_ in range(starts[2] + 1, starts[3] + 1)
                                             if d_ < d and d_ in self.knowns], default=None)
                                phase = decreasing_phase(rate=self.knowns.get(begin, None) or [0.6, 0.8],
                                                         len_=d - (begin or starts[2] + 1), dec_range=[0.04, 0.1],
                                                         base=self.base_price, tol=self.tol)
                                if val not in phase:
                                    chance = 0
                                else:
                                    chance *= phase[val]
                            else:
                                raise ValueError

                        if chance > 0:
                            # INC
                            if day in inc_days:
                                # Sample at tol (e.g 0.01) and use mid <x+tol/2> of interval [x, x+tol]
                                for price in inc_chances:
                                    percents[price] = percents.get(price, 0) + chance * inc_chances[price]
                            # DEC1
                            elif day in range(starts[0] + 1, starts[1] + 1):
                                begin = max([d for d in range(starts[0] + 1, starts[1] + 1)
                                             if d < day and d in self.knowns], default=None)
                                end = min([d for d in range(starts[0] + 1, starts[1] + 1)
                                           if d > day and d in self.knowns], default=None)
                                phase = decreasing_phase(rate=self.knowns.get(begin, None) or [0.6, 0.8],
                                                         end=end, end_rate=self.knowns.get(end, None),
                                                         len_=day - (begin or starts[0] + 1), dec_range=[0.04, 0.1],
                                                         base=self.base_price, tol=self.tol)
                                for price in phase:
                                    percents[price] = percents.get(price, 0) + chance * phase[price]
                            # DEC2
                            elif day in range(starts[2] + 1, starts[3] + 1):
                                begin = max([d for d in range(starts[2] + 1, starts[3] + 1)
                                             if d < day and d in self.knowns], default=None)
                                end = min([d for d in range(starts[2] + 1, starts[3] + 1)
                                           if d > day and d in self.knowns], default=None)
                                phase = decreasing_phase(rate=self.knowns.get(begin, None) or [0.6, 0.8],
                                                         end=end, end_rate=self.knowns.get(end, None),
                                                         len_=day - (begin or starts[2] + 1), dec_range=[0.04, 0.1],
                                                         base=self.base_price, tol=self.tol)
                                for price in phase:
                                    percents[price] = percents.get(price, 0) + chance * phase[price]
                            else:
                                raise ValueError

            if plot:
                plt.bar(list(percents.keys()), list(percents.values()))
                plt.show()

            return percents


class LargeSpike:
    """
    Pattern with large spike

    DEC MID - HIGH - LOW
    """
    def __init__(self, base_price, knowns=None, tol=0.001):
        self.base_price = base_price
        self.knowns = knowns or dict()
        self.tol = tol

    def gen(self):
        mid_len = random.randint(1, 7)
        rate = random.uniform(0.85, 0.9)
        decs = np.cumsum([0] + [random.uniform(0.03, 0.05) for _ in range(mid_len - 1)])

        # Peak borders
        peak_limits = [[0.9, 1.4], [1.4, 2], [2, 6], [1.4, 2], [0.9, 1.4]]

        prices = [math.ceil((rate - dec) * self.base_price) for dec in decs]
        prices += [math.ceil(random.uniform(low, up) * self.base_price) for low, up in peak_limits]
        prices += [math.ceil(random.uniform(0.4, 0.9) * self.base_price) for _ in range(7 - mid_len)]

        return prices

    def probs(self, day, verbose=True, plot=False):
        if verbose:
            print(day)
        if day in self.knowns:
            return {self.knowns[day]: 1}
        else:
            percents = dict()
            # All possible phase layouts
            for mid_len in range(1, 8):
                inc_days = list(range(mid_len + 1, 13))

                # Borders for univariate case
                limits = [[0.9, 1.4], [1.4, 2], [2, 6], [1.4, 2], [0.9, 1.4]] + [[0.4, 0.9]] * (7 - mid_len)
                borders = {d: b for d, b in zip(range(mid_len + 1, 13), limits)}

                # Chances for univariate case
                uni_chances = dict()
                for d, b in borders.items():
                    uni_chances[d] = dict()
                    rates = np.arange(b[0] + self.tol / 2, b[1] + self.tol / 2, self.tol)
                    for rate in rates:
                        price = math.ceil(rate * self.base_price)
                        uni_chances[d][price] = uni_chances[d].get(price, 0) + 1 / len(rates)

                chance = 1 / 7

                # Adjust chance to reflect knowns
                for d in self.knowns:
                    val = self.knowns[d]
                    # UNI
                    if d in inc_days:
                        if val not in uni_chances[d]:
                            chance = 0
                        else:
                            chance *= uni_chances[d][val]
                    # DEC
                    elif d in range(1, mid_len + 1):
                        # Only depends on what was before
                        begin = max([d_ for d_ in range(1, mid_len + 1)
                                     if d_ < d and d_ in self.knowns], default=None)
                        phase = decreasing_phase(rate=self.knowns.get(begin, None) or [0.85, 0.9],
                                                 len_=d - (begin or 1), dec_range=[0.03, 0.05],
                                                 base=self.base_price, tol=self.tol)
                        if val not in phase:
                            chance = 0
                        else:
                            chance *= phase[val]
                    else:
                        raise ValueError

                if chance > 0:
                    # UNI
                    if day in inc_days:
                        # Sample at tol (e.g 0.01) and use mid <x+tol/2> of interval [x, x+tol]
                        for price in uni_chances[day]:
                            percents[price] = percents.get(price, 0) + chance * uni_chances[day][price]
                    # DEC
                    elif day in range(1, mid_len + 1):
                        begin = max([d for d in range(1, mid_len + 1)
                                     if d < day and d in self.knowns], default=None)
                        end = min([d for d in range(1, mid_len + 1)
                                   if d > day and d in self.knowns], default=None)
                        phase = decreasing_phase(rate=self.knowns.get(begin, None) or [0.85, 0.9],
                                                 end=end, end_rate=self.knowns.get(end, None),
                                                 len_=day - (begin or 1), dec_range=[0.03, 0.05],
                                                 base=self.base_price, tol=self.tol)
                        for price in phase:
                            percents[price] = percents.get(price, 0) + chance * phase[price]
                    else:
                        raise ValueError

            if plot:
                plt.bar(list(percents.keys()), list(percents.values()))
                plt.show()

            return percents


class Decreasing:
    """
    Decreasing pattern

    DEC
    """
    def __init__(self, base_price, knowns=None, tol=0.001):
        self.base_price = base_price
        self.knowns = knowns or dict()
        self.tol = tol

    def gen(self):
        rate = random.uniform(0.85, 0.9)
        decs = np.cumsum([0] + [random.uniform(0.03, 0.05) for _ in range(12)])

        return [math.ceil((rate - dec) * self.base_price) for dec in decs]

    def probs(self, day, verbose=True, plot=False):
        if verbose:
            print(day)
        if day in self.knowns:
            return {self.knowns[day]: 1}
        else:
            percents = dict()

            chance = 1

            # Adjust chance to reflect knowns
            for d in self.knowns:
                val = self.knowns[d]
                # DEC
                if d in range(1, 13):
                    # Only depends on what was before
                    begin = max([d_ for d_ in range(1, 13)
                                 if d_ < d and d_ in self.knowns], default=None)
                    phase = decreasing_phase(rate=self.knowns.get(begin, None) or [0.85, 0.9],
                                             len_=d - (begin or 1), dec_range=[0.03, 0.05],
                                             base=self.base_price, tol=self.tol)
                    if val not in phase:
                        chance = 0
                    else:
                        chance *= phase[val]
                else:
                    raise ValueError

            if chance > 0:
                # DEC
                if day in range(1, 13):
                    begin = max([d for d in range(1, 13)
                                 if d < day and d in self.knowns], default=None)
                    end = min([d for d in range(1, 13)
                               if d > day and d in self.knowns], default=None)
                    phase = decreasing_phase(rate=self.knowns.get(begin, None) or [0.85, 0.9],
                                             end=end, end_rate=self.knowns.get(end, None),
                                             len_=day - (begin or 1), dec_range=[0.03, 0.05],
                                             base=self.base_price, tol=self.tol)
                    for price in phase:
                        percents[price] = percents.get(price, 0) + chance * phase[price]
                else:
                    raise ValueError

        if plot:
            plt.bar(list(percents.keys()), list(percents.values()))
            plt.show()

        return percents


class SmallSpike:
    """
    Pattern with small spike

    DEC - HIGH - DEC
    """
    def __init__(self, base_price, knowns=None, tol=0.001):
        self.base_price = base_price
        self.knowns = knowns or dict()
        self.tol = tol

        self.peak = dict()
        for i in range(2):
            self.peak[i] = dict()
            peak_rate = np.arange(1.4, 2, self.tol)
            for prate in peak_rate:
                rates = np.arange(1.4, prate, self.tol)
                for rate in rates:
                    price = math.ceil(rate * self.base_price)
                    self.peak[i][price] = self.peak[i].get(price, 0) + \
                                            1 / (len(list(rates)) * len(list(peak_rate)))

    def gen(self):
        mid_len = random.randint(0, 7)
        rate1 = random.uniform(0.4, 0.9)
        rate2 = random.uniform(0.4, 0.9)
        decs1 = np.cumsum([0] + [random.uniform(0.03, 0.05) for _ in range(mid_len - 1)])
        decs2 = np.cumsum([0] + [random.uniform(0.03, 0.05) for _ in range(6 - mid_len)])

        # Peak borders
        peak_rate = random.uniform(1.4, 2)
        peak_limits = [[0.9, 1.4], [0.9, 1.4], [1.4, 2], [1.4, peak_rate], [1.4, peak_rate]]

        prices = [math.ceil((rate1 - dec) * self.base_price) for dec in decs1]
        prices += [math.ceil(random.uniform(low, up) * self.base_price) for low, up in peak_limits]
        prices += [math.ceil((rate2 - dec) * self.base_price) for dec in decs2]

        return prices

    def probs(self, day, verbose=True, plot=False):
        if verbose:
            print(day)
        if day in self.knowns:
            return {self.knowns[day]: 1}
        else:
            percents = dict()
            # All possible phase layouts
            for mid_len in range(0, 8):
                inc_days = list(range(mid_len + 1, mid_len + 6))

                # Borders for univariate case
                limits = [[0.9, 1.4], [0.9, 1.4], [1.4, 2]]
                borders = {d: b for d, b in zip(range(mid_len + 1, mid_len + 4), limits)}

                # Chances for univariate case
                uni_chances = dict()
                for d, b in borders.items():
                    uni_chances[d] = dict()
                    rates = np.arange(b[0] + self.tol / 2, b[1] + self.tol / 2, self.tol)
                    for rate in rates:
                        price = math.ceil(rate * self.base_price)
                        uni_chances[d][price] = uni_chances[d].get(price, 0) + 1 / len(list(rates))

                for i, j in enumerate(range(mid_len + 4, mid_len + 6)):
                    uni_chances[j] = self.peak[i]

                chance = 1 / 8

                # Adjust chance to reflect knowns
                for d in self.knowns:
                    val = self.knowns[d]
                    # UNI
                    if d in inc_days:
                        if val not in uni_chances[d]:
                            chance = 0
                        else:
                            chance *= uni_chances[d][val]
                    # DEC1
                    elif d in range(1, mid_len + 1):
                        # Only depends on what was before
                        begin = max([d_ for d_ in range(1, mid_len + 1)
                                     if d_ < d and d_ in self.knowns], default=None)
                        phase = decreasing_phase(rate=self.knowns.get(begin, None) or [0.4, 0.9],
                                                 len_=d - (begin or 1), dec_range=[0.03, 0.05],
                                                 base=self.base_price, tol=self.tol)
                        if val not in phase:
                            chance = 0
                        else:
                            chance *= phase[val]
                    # DEC2
                    elif d in range(mid_len + 6, 13):
                        # Only depends on what was before
                        begin = max([d_ for d_ in range(mid_len + 6, 13)
                                     if d_ < d and d_ in self.knowns], default=None)
                        phase = decreasing_phase(rate=self.knowns.get(begin, None) or [0.4, 0.9],
                                                 len_=d - (begin or mid_len + 6), dec_range=[0.03, 0.05],
                                                 base=self.base_price, tol=self.tol)
                        if val not in phase:
                            chance = 0
                        else:
                            chance *= phase[val]
                    else:
                        raise ValueError

                if chance > 0:
                    # UNI
                    if day in inc_days:
                        # Sample at tol (e.g 0.01) and use mid <x+tol/2> of interval [x, x+tol]
                        for price in uni_chances[day]:
                            percents[price] = percents.get(price, 0) + chance * uni_chances[day][price]
                    # DEC
                    elif day in range(1, mid_len + 1):
                        begin = max([d for d in range(1, mid_len + 1)
                                     if d < day and d in self.knowns], default=None)
                        end = min([d for d in range(1, mid_len + 1)
                                   if d > day and d in self.knowns], default=None)
                        phase = decreasing_phase(rate=self.knowns.get(begin, None) or [0.4, 0.9],
                                                 end=end, end_rate=self.knowns.get(end, None),
                                                 len_=day - (begin or 1), dec_range=[0.03, 0.05],
                                                 base=self.base_price, tol=self.tol)
                        for price in phase:
                            percents[price] = percents.get(price, 0) + chance * phase[price]
                    # DEC
                    elif day in range(mid_len + 6, 13):
                        begin = max([d for d in range(mid_len + 6, 13)
                                     if d < day and d in self.knowns], default=None)
                        end = min([d for d in range(mid_len + 6, 13)
                                   if d > day and d in self.knowns], default=None)
                        phase = decreasing_phase(rate=self.knowns.get(begin, None) or [0.4, 0.9],
                                                 end=end, end_rate=self.knowns.get(end, None),
                                                 len_=day - (begin or mid_len + 6), dec_range=[0.03, 0.05],
                                                 base=self.base_price, tol=self.tol)
                        for price in phase:
                            percents[price] = percents.get(price, 0) + chance * phase[price]
                    else:
                        raise ValueError

            if plot:
                plt.bar(list(percents.keys()), list(percents.values()))
                plt.show()

            return percents


if __name__ == "__main__":
    base_price = random.randint(90, 110)

    print(SmallSpike(base_price=100, knowns={1: 88}, tol=0.0005).gen())

    [SmallSpike(base_price=100, knowns={1: 87}, tol=0.001).probs(i, plot=True) for i in range(1, 13)]
