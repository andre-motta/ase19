import math
import re


class Num:
    n = 0
    mu = 0
    m2 = 0
    sd = 0
    lo = float('inf')
    hi = -float('inf')

    def _numSd(self):
        if self.m2 < 0:
            return 0
        if self.n < 2:
            return 0
        return math.sqrt(self.m2/(self.n - 1))

    def numNorm(self, x):
        return (x - self.lo)/(self.hi - self.lo + 10e-32)

    def __sub__(self, v):
        if self.n < 2:
            self.sd = 0
            return v
        self.n -= 1
        d = v - self.mu
        self.mu -= d / self.n
        self.m2 -= d*(v - self.mu)
        self.sd = self._numSd()
        return v

    def __add__(self, v):
        self.n += 1
        if v < self.lo:
            self.lo = v
        if v > self.hi:
            self.hi = v
        d = v - self.mu
        self.mu += d / self.n
        self.m2 += d * (v - self.mu)
        self.sd = self._numSd()
        return v


class Row:
    def __init__(self, oid, cells, cooked, dom):
        self.oid, self.cells, self.cooked, self.dom = oid, cells, cooked, dom


class Col:
    def __init__(self, oid, txt, num):
        self.oid, self.txt, self.num = oid, txt, num

    def __add__(self, v):
        self.num + v

    def __sub__(self, v):
        self.num - v


class Tbl:
    def __init__(self, oid, file, string):
        self.oid = oid
        self.count = 0
        self.cols = []
        self.rows = []
        if string:
            lines = self.string(file)
        else:
            lines = self.read(file)
        lines = self.linemaker(lines)
        first = True
        for line in lines:
            if first:
                first = False
                self.create_cols(line)
            else:
                self.insert_row(line)
        print("Success")

    @staticmethod
    def read(file):
        lines = []
        with open(file) as f:
            for line in f:
                lines.append(line)
        return lines

    @staticmethod
    def linemaker(src, sep=",", doomed=r'([\n\t\r ]|#.*)'):
        "convert lines into lists, killing whitespace and comments"
        for line in src:
            line = line.strip()
            line = re.sub(doomed, '', line)
            if line:
                yield line.split(sep)

    @staticmethod
    def string(s):
        lines = []
        for line in s.splitlines():
            lines.append(line)
        return lines

    @staticmethod
    def compiler(x):
        try:
            int(x)
            return int
        except:
            try:
                float(x)
                return float
            except ValueError:
                return str

    def convert(self, x):
        f = self.compiler(x)
        return f(x)

    def create_cols(self, line):
        index = 0
        for val in line:
            self.cols.append(Col(index, str(val), Num()))
            index += 1

    def insert_row(self, line):
        self.rows.append(Row(self.count, [self.convert(x) for x in line], [], 0))
        self.count += 1
        index = 0
        for val in line:
            self.cols[index].num + self.convert(val)
            index += 1

    def dump(self):
        print("Dump table:")
        print("t.cols")
        for col in self.cols:
            print("|  "+str(col.oid))
            print("|  |  add: Num1")
            print("|  |  col: "+str(col.oid))
            print("|  |  hi: "+str(col.num.hi))
            print("|  |  lo: "+str(col.num.lo))
            print("|  |  m2: "+str(col.num.m2))
            print("|  |  mu: "+str(col.num.mu))
            print("|  |  n: "+str(col.num.n))
            print("|  |  oid: "+str(col.oid))
            print("|  |  sd: "+str(col.num.sd))
            print("|  |  txt: "+str(col.txt))
        print("t.oid: " + str(self.oid))
        print("t.rows")
        for row in self.rows:
            print("|  "+str(row.oid))
            print("|  |  cells")
            index = 0
            for val in row.cells:
                print("|  |  |  "+str(index)+": "+str(val))
                index += 1
            print("|  |  cooked")
            print("|  |  dom:"+str(row.dom))
            print("|  |  oid:"+str(row.oid))





def NumCache(cachemu, mu, cachesd, sd):
    cachemu.append(mu)
    cachesd.append(sd)


def NumCompare(cachemu, mu, cachesd, sd, f, i):
    cursd = cachesd.pop()
    curmu = cachemu.pop()
    if math.isclose(cursd,sd,rel_tol=1e-6) and math.isclose(curmu, mu, rel_tol=1e-6):
        if i == 100:
            f.write(
                    "At iteration "+str(i)+" found : SD = %.4f" % sd + " equal to Saved SD = %.4f" % cursd +
                    "\n" +
                    "At iteration "+str(i)+" found : MU = %.4f" % mu + " equal to Saved MU = %.4f" % curmu + '\n'
                    )
        else:
            f.write(
                "At iteration " + str(i) + "  found : SD = %.4f" % sd + " equal to Saved SD = %.4f" % cursd +
                "\n" +
                "At iteration " + str(i) + "  found : MU = %.4f" % mu + " equal to Saved MU = %.4f" % curmu + '\n'
            )
def main():
    s = """$cloudCover, $temp, $humid, $wind,  $playHours
      100,        68,    80,    0,    3   # comments
      0,          85,    85,    0,    0
      0,          80,    90,    10,   0
      60,         83,    86,    0,    4
      100,        70,    96,    0,    3
      100,        65,    70,    20,   0
      70,         64,    65,    15,   5
      0,          72,    95,    0,    0
      0,          69,    70,    0,    4
      80,          75,    80,    0,    3  
      0,          75,    70,    18,   4
      60,         72,    90,    10,   4
      40,         81,    75,    0,    2    
      100,        71,    91,    15,   0
      """
    tbl = Tbl(0, s, True)
    tbl.dump()


if __name__ == '__main__':
    main()
