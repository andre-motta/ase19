import math
import re
from collections import defaultdict
import random


class Abcd:

    def __init__(self, data="data", rx="rx"):
        self.db = data
        self.rx = rx
        self.num = 0
        self.yes = 0
        self.no = 0
        self.a = defaultdict(int)
        self.b = defaultdict(int)
        self.c = defaultdict(int)
        self.d = defaultdict(int)
        self.known = defaultdict(bool)

    def Abcd1(self, want, got):
        if not self.known[want]:
            self.known[want] = True
            self.a[want] = self.yes + self.no
        if not self.known[got]:
            self.known[got] = True
            self.a[got] = self.yes + self.no
        if want == got:
            self.yes += 1
        else:
            self.no += 1
        for x, v in self.known.items():
            if v:
                if want == x:
                    if want == got:
                        self.d[x] += 1
                    else:
                        self.b[x] += 1
                else:
                    if got == x:
                        self.c[x] += 1
                    else:
                        self.a[x] += 1
        self.num += 1

    def AbcdReport(self, filename="reportoutput.txt"):
        file = open(filename, "w")
        file.write("    db |    rx |   num |     a |     b |     c |     d |  acc |  pre |   pd |   pf |    f |    g | class\n")
        file.write("  ---- |  ---- |  ---- |  ---- |  ---- |  ---- |  ---- | ---- | ---- | ---- | ---- | ---- | ---- | -----\n")
        for x, v in self.known.items():
            if v:
                pd = pf = pn = prec = g = f = acc = 0
                a = self.a[x]
                b = self.b[x]
                c = self.c[x]
                d = self.d[x]
                if b + d > 0:
                    pd = d/(b + d)
                if a + c > 0:
                    pf = c/(a + c)
                    pn = (b + d)/(a + c)
                if c + d > 0:
                    prec = d/(c + d)
                if 1 - pf + pd > 0:
                    g = (2*(1 - pf)*pd)/(1 - pf + pd)
                if prec + pd > 0:
                    f = (2*prec*pd)/(prec + pd)
                if self.yes + self.no > 0:
                    acc = self.yes / (self.yes + self.no)
                file.write("{:7s}|".format(self.db) + "{:7s}|".format(self.rx) + "{:7d}|".format(self.num) + "{:7d}|".format(a) + "{:7d}|".format(b) + "{:7d}|".format(c) + "{:7d}|".format(d) + "{:6.2f}|".format(acc) + "{:6.2f}|".format(prec) + "{:6.2f}|".format(pd) + "{:6.2f}|".format(pf) + "{:6.2f}|".format(f) + "{:6.2f}|".format(g) + "{:7s}".format(x) + '\n')
        file.close()


class Sym:
    n=0
    most = 0
    mode = ""

    def __init__(self):
        self.cnt = defaultdict(int)

    def __add__(self, v):
        self.n += 1
        self.cnt[v] += 1
        tmp = self.cnt[v]
        if tmp > self.most:
            self.most = tmp
            self.mode = v
        return v

    def syment(self):
        e = 0
        for k, v in self.cnt.items():
            p = v/self.n
            e -= p*math.log(p)/math.log(2)
        return e

    def symany(self, without):
        r = random.randint()
        for k, v in self.cnt.items():
            m = self.n - v if without else v
            r -= m/self.n
            if r <= 0:
                return k
        return k

    def symlike(self,x,prior,m):
        f = self.cnt[x]
        return (f + m*prior)/(self.n + m)


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
    def __init__(self, oid, txt, obj, isnum):
        self.oid, self.txt, self.obj, self.isnum = oid, txt, obj, isnum

    def __add__(self, v):
        self.obj + v

    def __sub__(self, v):
        self.obj - v


class Tbl:
    def __init__(self, oid, file, string):
        self.oid = oid
        self.count = 0
        self.cols = []
        self.rows = []
        self.fileline = 0
        self.linesize = 0
        self.bannedcols = []
        self.goals = []
        self.nums = []
        self.syms = []
        self.w = defaultdict(int)
        self.xnums = []
        self.xs = []
        self.xsyms = []
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
        print("Table Created Successfully")

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
        cols = []
        for val in line:
            val = self.convert(val)
            if val[0] == "?":
                self.bannedcols.append(index+1)
            if "$" in val or "<" in val or ">" in val:
                self.nums.append(index + 1)

                cols.append(''.join(c for c in val if not c in ['$', '?']))
                self.cols.append(Col(index+1, str(''.join(c for c in val if not c in ['$', '?'])), Num(), True))
            else:
                self.syms.append(index + 1)
                cols.append(''.join(c for c in val if not c in ['$', '?']))
                self.cols.append(Col(index+1, str(''.join(c for c in val if not c in ['$', '?'])), Sym(), False))

            if "!" in val or "<" in val or ">" in val:
                self.goals.append(index+1)
                if "<" in val:
                    self.w[index+1] = -1
            if "<" not in val and ">" not in val and "!" not in val:
                self.xs.append(index + 1)
                if "$" in val:
                    self.xnums.append(index + 1)
                else:
                    self.xsyms.append(index + 1)
            index += 1
        self.linesize = index
        self.fileline += 1

    def insert_row(self, line):
        self.fileline += 1
        if len(line) < self.linesize:
            return
        realline = []
        realindex = 0
        index = 0
        for val in line:
            if index + 1 not in self.bannedcols:
                if val == "?":
                    realline.append(val)
                    continue
                self.cols[realindex].obj + self.convert(val)
                realline.append(val)
                realindex +=1
            else:
                realindex+=1
            index += 1
        self.rows.append(Row(self.count, [self.convert(x) for x in realline], [], 0))
        self.count += 1

    def dump(self, filename = "tabledump.txt"):
        f = open(filename, 'w')
        f.write("Dump table:"+"\n")
        f.write("t.cols"+"\n")
        for i, col in enumerate(self.cols):
            if i+1 in self.bannedcols:
                continue
            if col.isnum:
                f.write("|  "+str(col.oid)+"\n")
                f.write("|  |  add: Num1"+"\n")
                f.write("|  |  col: "+str(col.oid)+"\n")
                f.write("|  |  hi: "+str(col.obj.hi)+"\n")
                f.write("|  |  lo: "+str(col.obj.lo)+"\n")
                f.write("|  |  m2: "+str(col.obj.m2)+"\n")
                f.write("|  |  mu: "+str(col.obj.mu)+"\n")
                f.write("|  |  n: "+str(col.obj.n)+"\n")
                f.write("|  |  oid: "+str(col.oid)+"\n")
                f.write("|  |  sd: "+str(col.obj.sd)+"\n")
                f.write("|  |  txt: "+str(col.txt)+"\n")
            else:
                f.write("|  " + str(col.oid) + "\n")
                f.write("|  |  add: Sym1" + "\n")
                for k, v in col.obj.cnt.items():
                    f.write("|  |  |  " + str(k) + ": " + str(v) + "\n")
                f.write("|  |  col: "+str(col.oid)+"\n")
                f.write("|  |  mode: "+str(col.obj.mode)+"\n")
                f.write("|  |  most: "+str(col.obj.most)+"\n")
                f.write("|  |  n: " + str(col.obj.n) + "\n")
                f.write("|  |  oid: " + str(col.oid) + "\n")
                f.write("|  |  txt: " + str(col.txt) + "\n")

        f.write("t.my: "+"\n")
        f.write("|  class: " + str(len(self.cols))+"\n")
        f.write("|  goals" + "\n")
        for v in self.goals:
            if v not in self.bannedcols:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  nums" + "\n")
        for v in self.nums:
            if v not in self.bannedcols:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  syms" + "\n")
        for v in self.syms:
            if v not in self.bannedcols:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  w" + "\n")
        for k, v in self.w.items():
            if v not in self.bannedcols:
                f.write("|  |  " + str(k) + ": "+str(v)+"\n")
        f.write("|  xnums" + "\n")
        for v in self.xnums:
            if v not in self.bannedcols:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  xs" + "\n")
        for v in self.xs:
            if v not in self.bannedcols:
                f.write("|  |  " + str(v) + "\n")
        f.write("|  xsyms" + "\n")
        for v in self.xsyms:
            if v not in self.bannedcols:
                f.write("|  |  " + str(v) + "\n")
        f.close()

def _abcd():
    abcd = Abcd()
    for i in range(6):
        abcd.Abcd1("yes", "yes")
    for i in range(2):
        abcd.Abcd1("no",  "no")
    for i in range(5):
            abcd.Abcd1("maybe",  "maybe")
    abcd.Abcd1("maybe","no")
    abcd.AbcdReport("abcdtest.txt")

def _symTest(s, filename = "symTestOutput.txt"):
    file = open(filename, 'w')
    sym = Sym()
    for letter in s:
        sym + letter
    file.write("Entropy is {:3.2f}".format(sym.syment()))
    file.close



def main():
    s = """   outlook, ?$temp,  <humid, wind, !play
              rainy, 68, 80, FALSE, yes # comments
              sunny, 85, 85,  FALSE, no
              sunny, 80, 90, TRUE, no
              overcast, 83, 86, FALSE, yes
              rainy, 70, 96, FALSE, yes
              rainy, 65, 70, TRUE, no
              overcast, 64, 65, TRUE, yes
              sunny, 72, 95, FALSE, no
              sunny, 69, 70, FALSE, yes
              rainy, 75, 80, FALSE, yes
              sunny, 75, 70, TRUE, yes
              overcast, 72, 90, TRUE, yes
              overcast, 81, 75, FALSE, yes
              rainy, 71, 91, TRUE, no
  """
    tbl = Tbl(0, s, True)
    tbl.dump()
    _abcd()
    _symTest("aaaabbc")


if __name__ == '__main__':
    main()
