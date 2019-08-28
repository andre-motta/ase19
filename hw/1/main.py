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
        self.m2 += d * (v - self.mu)
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

    def NumCompare(self, f, i):
        cursd = self.cachesd.pop()
        curmu = self.cachemu.pop()
        if math.isclose(cursd,self.sd,rel_tol=1e-6) and math.isclose(curmu, self.mu,rel_tol=1e-6):
            f.write("At iteration "+str(i)+" found : SD = %.4f"%self.sd + " == CachedSd = %.4f"%cursd + " MU = %.4f"%self.mu + " == CachedMu = %.4f"%curmu + '\n' )


def main():
    N = Num()
    numbers = []
    for i in range(100):
        numbers.append(random.randint(10, 1000))
    for i in range(100):
        N.Num1(numbers[i])
        if (i+1) % 10 == 0:
            N.NumCache()
    with open("out.txt", 'w') as f:
        count = 100
        for i in range(100):
            if (i+10) % 10 == 0:
                N.NumCompare(f, count)
                count -= 10
            N.NumLess(numbers[(100 - i - 1)])




if __name__ == '__main__':
    main()