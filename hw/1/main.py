import math
import random

class Col:
    n = 0

class Sym(Col):
    pass

class Some(Col):
    pass

class Num(Col):
    mu = 0
    m2 = 0
    sd = 0
    lo = float('inf')
    hi = -float('inf')
    cachemu = []
    cachesd = []

    def _NumSd(self):
        if self.m2 < 0:
            return 0
        if self.n < 2:
            return 0
        return math.sqrt(self.m2/(self.n -1))

    def Num1(self, v):
        self.n += 1
        if v < self.lo:
            self.lo = v
        if v > self.hi:
            self.hi = v
        d = v - self.mu
        self.mu += d / self.n
        self.m2 = d * (v - self.mu)
        self.sd = self._NumSd()
        return v

    def NumNorm(self, x):
        return (x - self.lo)/(self.hi - self.lo + 10e-32)

    def NumLess(self,v):
        if self.n < 2:
            self.sd = 0
            return v
        self.n  -= 1
        d        = v - self.mu
        self.mu -= d / self.n
        self.m2 -= d*(v - self.mu)
        self.sd = self._NumSd()
        return v

    def NumCache(self):
        self.cachemu.append(self.mu)
        self.cachesd.append(self.sd)

    def NumCompare(self):
        if self.cachesd.pop() == self.sd and self.cachemu.pop() == self.mu:
            print("Iguais!")


def main():
    N = Num();
    numbers = []
    for i in range(100):
        numbers.append(random.randint(1, 10000))
    for i in range(100):
        if (i+1) % 10 == 0:
            N.NumCache()
        N.Num1(numbers[i])
    for i in range(100):
        if (i+1) % 10 == 0:
            N.NumCompare()
        N.NumLess(numbers[(100 - i - 1)])




if __name__ == '__main__':
    main()