import math
import re
from collections import defaultdict
import random


class NB:

    def __init__(self, oid, file, string):
        self.tbl = Tbl(oid, file, string)
        self.tbls = {}
        self.n = -1
        self.count = 0
        self.m = 1
        self.k = 0

    def train(self, line):
        self.n += 1
        self.tbl.insert_row(line)
        val = line[self.tbl.goal]
        if val not in self.tbls:
            self.tbls[val] = Tbl(self.count, "", True)
            self.tbls[val].create_cols(self.tbl.header)
        self.count += 1
        self.tbls[val].insert_row(line)

    def classify(self, line):
        most = -10**64
        guess = ""
        for k, v in self.tbls.items():
            like = self.bayesTheorem(line, v)
            if like > most:
                most = like
                guess = k
        return guess

    def bayesTheorem(self, line, tbl):
        like = prior = ((len(tbl.rows) + self.k)/(self.n + self.k * len(self.tbls)))
        like = math.log(like)
        for c in tbl.xs:
            if c not in self.tbl.bannedcols:
                x = tbl.convert(line[c-1])
                if c in tbl.nums:
                    like += math.log(Num.numlike(tbl.cols[c-1].obj, x))
                else:
                    like += math.log(Sym.symlike(tbl.cols[c-1].obj, x, prior, self.m))
        return like

class ZeroR:

    def __init__(self, oid, file, string):
        self.tbl = Tbl(oid, file, string)

    def train(self, line):
        self.tbl.insert_row(line)

    def classify(self, line):
        return self.tbl.cols[self.tbl.goal].obj.mode


class Abcds:

    def __init__(self, classifier, abcd, wait):
        self.classifier = classifier
        self.abcd = abcd
        self.wait = wait

    def add(self, line, r):
        if r > self.wait:
            print("classifiying", r)
            got = self.classifier.classify(line)
            want = line[self.classifier.tbl.goal]
            self.abcd.Abcd1(want, got)
        print("training", r)
        self.classifier.train(line)


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
        self.num += 1
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

    def AbcdReport(self, filename="reportoutput.txt", mode="w", classname = "default"):
        file = open(filename, mode)
        if mode == "w":
            file.write("#--- "+classname+"ok -----------------------\n\nweathernon\n")
        else:
            file.write("\ndiabetes\n")
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

    @staticmethod
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

    @staticmethod
    def numlike(self, x):
        var = self.sd**2
        denom = math.sqrt(math.pi * 2 * var)
        num = (2.71828**(-(x-self.mu)**2)/(2*var+0.0001))
        return num/(denom+10**(-64)) + 10**(-64)


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
        self.goal = -1
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
        self.header = ""

    @staticmethod
    def read(file):
        lines = []
        with open(file) as f:
            for line in f:
                lines.append(line)
        return lines

    @staticmethod
    def linemaker(src, sep=",", doomed=r'([\n\t\r ]|#.*)'):
        lines = []
        "convert lines into lists, killing whitespace and comments"
        for line in src:
            line = line.strip()
            line = re.sub(doomed, '', line)
            if line:
                lines.append(line.split(sep))
        return lines

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
        self.header = line
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
                if "!" in val:
                    self.goal = index
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

def testClassifiers():
    with open("weathernon.csv", 'r') as f:
        s = f.read()
        reporter = Abcds(NB(0, s, True), Abcd(), 4)
        lines = reporter.classifier.tbl.linemaker(reporter.classifier.tbl.string(s))
        first = True
        count = 1
        for line in lines:
            if first:
                reporter.classifier.tbl.create_cols(line)
                first = False
            else:
                reporter.add(line, count)
                count += 1
        reporter.abcd.AbcdReport("NB.txt", "w", "nb")
        reporter = Abcds(ZeroR(0, s, True), Abcd(), 3)
        lines = reporter.classifier.tbl.linemaker(reporter.classifier.tbl.string(s))
        first = True
        count = 1
        for line in lines:
            if first:
                reporter.classifier.tbl.create_cols(line)
                first = False
            else:
                reporter.add(line, count)
                count += 1
        reporter.abcd.AbcdReport("ZeroR.txt", "w", "zeror")

    with open("diabetes.csv", 'r') as f:
        s = f.read()
        reporter = Abcds(NB(0, s, True), Abcd(), 20)
        lines = reporter.classifier.tbl.linemaker(reporter.classifier.tbl.string(s))
        first = True
        count = 1
        for line in lines:
            if first:
                reporter.classifier.tbl.create_cols(line)
                first = False
            else:
                reporter.add(line, count)
                count += 1
        reporter.abcd.AbcdReport("NB.txt", "a", "nb")
        reporter = Abcds(ZeroR(0, s, True), Abcd(), 3)
        lines = reporter.classifier.tbl.linemaker(reporter.classifier.tbl.string(s))
        first = True
        count = 1
        for line in lines:
            if first:
                reporter.classifier.tbl.create_cols(line)
                first = False
            else:
                reporter.add(line, count)
                count += 1
        reporter.abcd.AbcdReport("ZeroR.txt", "a", "zeror")

def main():
    testClassifiers()



if __name__ == '__main__':
    main()
