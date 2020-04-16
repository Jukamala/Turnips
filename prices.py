import random
import numpy as np
import math

from itertools import chain


def comb(n, k):
    return math.factorial(n) / math.factorial(k) / math.factorial(n - k)


def shifted_irwin_hall(start, end, n, x):
    """
    Let X_1, .., X_n iid with X_1 ~ U[start, end]
    Let Z = sum(i=1, n, X_i)
    Sample the pdf of Z
    """
    if end <= start:
        raise ValueError

    # Reshift so X_i ~ U[0,1]
    x_shift = (x - n * start) / (end - start)

    # Irwin-Hall-pdf (https://en.wikipedia.org/wiki/Irwin–Hall_distribution)
    y = sum([1 / (2 * math.factorial(n - 1)) * (-1)**k * comb(n, k) *
             (x_shift - k)**(n-1) * np.sign(x_shift - k) for k in range(n + 1)])
    y[(x < start * n) | (x > end * n)] = 0

    return y


def decreasing_phase(begin, rate, len_, dec_range, base, end=None, end_rate=None, tol=0.01):
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
    # Convert price to rate(s) that result in this price
    if not isinstance(rate, list):
        # Must be in [rate/base, (rate-1)/base], make an interval with len % tol == 0
        rate = [rate/base, rate/base - tol*int(1/tol*base)]
    percents = dict()
    if end is None:
        # Only previous prices influence the wanted day
        if isinstance(rate, list):
            # Sample the rate
            for r in np.arange(rate[0] + tol/2, rate[1] + tol/2, tol):
                # Sample Irwin-Hall density
                steps = np.arange(dec_range[0] * len_ + tol/2,
                                  dec_range[1] * len_ + tol/2, tol)
                vals = shifted_irwin_hall(*dec_range, len_, steps)

                for price in [int(p) for p in np.ceil((r - vals) * base)]:
                    # Probabilty of r and price assuming this phase is happening
                    mult = tol**2 / ((dec_range[1] - dec_range[0]) * (rate[1] - rate[0]) * len_)
                    percents[price] = percents.get(price, 0) + mult
        else:
            # Sample Irwin-Hall density
            steps = np.arange(dec_range[0] * len_ + tol / 2,
                              dec_range[1] * len_ + tol / 2, tol)
            vals = shifted_irwin_hall(*dec_range, len_, steps)

            for price in [int(p) for p in np.ceil((rate - vals) * base)]:
                # Probabilty of price assuming this phase is happening
                mult = tol / ((dec_range[1] - dec_range[0]) * len_)
                percents[price] = percents.get(price, 0) + mult
    else:
        len2 = end - len_

        # Only previous prices influence the wanted day
        if isinstance(rate, list):
            # Sample the rate
            for r in np.arange(rate[0] + tol / 2, rate[1] + tol / 2, tol):
                # Sample Irwin-Hall density
                steps = np.arange(dec_range[0] * len_ + tol / 2,
                                  dec_range[1] * len_ + tol / 2, tol)
                # Also for the time till end
                steps2 = np.arange(dec_range[0] * len2 + tol / 2,
                                   dec_range[1] * len2 + tol / 2, tol)
                vals1 = shifted_irwin_hall(*dec_range, len_, steps)
                vals2 = shifted_irwin_hall(*dec_range, len2, steps2)

                # Only use parts of vals1 that are possible
                vals = [v1 for v1 in vals1 for v2 in vals2 if math.ceil((r - v1 - v2) * base) == end_rate]

                for price in [int(p) for p in np.ceil((r - vals) * base)]:
                    # Probabilty of r and p assuming this phase is happening
                    mult = tol**3 / ((dec_range[1] - dec_range[0])**2 * (rate[1] - rate[0]) * len_ * len2)
                    percents[price] = percents.get(price, 0) + mult
        else:
            # Sample Irwin-Hall density
            steps = np.arange(dec_range[0] * len_ + tol / 2,
                              dec_range[1] * len_ + tol / 2, tol)
            # Also for the time till end
            steps2 = np.arange(dec_range[0] * len2 + tol / 2,
                               dec_range[1] * len2 + tol / 2, tol)
            vals1 = shifted_irwin_hall(*dec_range, len_, steps)
            vals2 = shifted_irwin_hall(*dec_range, len2, steps2)

            # Only use parts of vals1 that are possible
            vals = [v1 for v1 in vals1 for v2 in vals2 if math.ceil((rate - v1 - v2) * base) == end_rate]

            for price in [int(p) for p in np.ceil((rate - vals) * base)]:
                # Probabilty of price assuming this phase is happening
                mult = tol**2 / ((dec_range[1] - dec_range[0])**2 * len_ * len2)
                percents[price] = percents.get(price, 0) + mult
    return percents


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


class RandomPattern:
    """
    Random pattern

    HIGH - DEC - HIGH - DEC - HIGH
    """
    def __init__(self, base_price, knowns=None, tol=0.01):
        self.base_price = base_price
        self.knowns = knowns or dict()
        self.tol = tol

    def probs(self, day):
        """
        X: "prices in knowns recorded"
        B: "price == X at day"
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
        """
        if day in self.knowns:
            return {day: self.knowns[day]}
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
                            inc_chances[price] = inc_chances.get(price, 0) * self.tol / 0.5

                        chance = 1 / 2 * 1 / 7 * 1 / (7 - inc1_len)
                        # Adjust chance to reflect knowns
                        for d in self.knowns:
                            # INC
                            if d in inc_days:
                                chance *= inc_chances[self.knowns[d]]
                            # DEC1
                            elif d in range(starts[0] + 1, starts[1] + 1):
                                # Only depends on what was before
                                begin = max([d for d in range(starts[0] + 1, starts[1] + 1)
                                             if d < day and d in self.knowns], default=None)
                                phase = decreasing_phase(begin=begin or starts[0], base=base_price,
                                                         rate=self.knowns.get(begin, None) or [0.6, 0.8],
                                                         len_=day - (begin or starts[0]), dec_range=[0.04, 1])
                                chance *= phase[self.knowns[d]]
                            # DEC2
                            elif d in range(starts[2] + 1, starts[3] + 1):
                                # Only depends on what was before
                                begin = max([d for d in range(starts[2] + 1, starts[3] + 1)
                                             if d < day and d in self.knowns], default=None)
                                phase = decreasing_phase(begin=begin or starts[2], base=base_price,
                                                         rate=self.knowns.get(begin, None) or [0.6, 0.8],
                                                         len_=day - (begin or starts[2]), dec_range=[0.04, 1])
                                chance *= phase[self.knowns[d]]
                            else:
                                raise ValueError

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
                            phase = decreasing_phase(begin=begin or starts[0],
                                                     rate=self.knowns.get(begin, None) or [0.6, 0.8],
                                                     len_=day - (begin or starts[0]), dec_range=[0.04, 1],
                                                     base=base_price, end=end, end_rate=self.knowns.get(end, None))
                            for price in phase:
                                percents[price] = percents.get(price, 0) + chance * phase[price]
                        # DEC2
                        elif day in range(starts[2] + 1, starts[3] + 1):
                            begin = max([d for d in range(starts[2] + 1, starts[3] + 1)
                                         if d < day and d in self.knowns], default=None)
                            end = min([d for d in range(starts[2] + 1, starts[3] + 1)
                                       if d > day and d in self.knowns], default=None)
                            phase = decreasing_phase(begin=begin or starts[2],
                                                     rate=self.knowns.get(begin, None) or [0.6, 0.8],
                                                     len_=day - (begin or starts[2]), dec_range=[0.04, 1],
                                                     base=base_price, end=end, end_rate=self.knowns.get(end, None))
                            for price in phase:
                                percents[price] = percents.get(price, 0) + phase[price] * chance
                        else:
                            raise ValueError

            # Rescale
            scale = sum(list(percents.values()))
            return {k: v/scale for k, v in percents.items()}

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


class LargeSpike:
    """
    Pattern with large spike

    DEC MID - HIGH - LOW
    """
    def __init__(self, base_price, knowns=None):
        self.base_price = base_price
        self.knowns = knowns or dict()

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


class Decreasing:
    """
    Decreasing pattern

    DEC
    """
    def __init__(self, base_price, knowns=None):
        self.base_price = base_price
        self.knowns = knowns or dict()

    def gen(self):
        rate = random.uniform(0.85, 0.9)
        decs = np.cumsum([0] + [random.uniform(0.03, 0.05) for _ in range(12)])

        return [math.ceil((rate - dec) * self.base_price) for dec in decs]


class SmallSpike:
    """
    Pattern with small spike

    DEC - HIGH - DEC
    """
    def __init__(self, base_price, knowns=None):
        self.base_price = base_price
        self.knowns = knowns or dict()

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


if __name__ == "__main__":
    base_price = random.randint(90, 110)
    print(RandomPattern(base_price).probs(3))
    print(RandomPattern(base_price).probs(5))
    print(RandomPattern(base_price).probs(8))
    print(generate_pattern(last_pattern=2))
    print(generate_pattern(weeks=1000))
    print(RandomPattern(base_price).gen())
    print(LargeSpike(base_price).gen())
    print(Decreasing(base_price).gen())
    print(SmallSpike(base_price).gen())
