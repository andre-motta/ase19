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
        self.fileline = 0
        self.linesize = 0
        self.bannedcols = []
        if string:
            lines = self.string(file)
        else:
            lines = self.read(file)
        lines = self.linemaker(lines)
        first = True
        with open("outpart1.txt", "w") as f:
            for line in lines:
                f.write(str([self.convert(x) for x in line]))
                f.write("\n")
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
            if self.convert(val)[0] == "?":
                self.bannedcols.append(index)
            else:
                cols.append(val)
                self.cols.append(Col(index, str(val), Num()))
            index += 1
        with open("outpart2.txt", "w") as f:
            f.write(str(cols)+ "\n")
        self.linesize = index
        self.fileline += 1

    def insert_row(self, line):
        f = open("outpart2.txt", "a")
        self.fileline += 1
        if len(line) < self.linesize:
            f.write("E > skipping line " + str(self.fileline) + "\n")
            f.close()
            return
        realline = []
        realindex = 0
        index = 0
        for val in line:
            if index not in self.bannedcols:
                if val == "?":
                    realline.append(val)
                    continue
                self.cols[realindex].num + self.convert(val)
                realline.append(val)
                realindex +=1
            index += 1
        f.write(str([self.convert(x) for x in realline])+ "\n")
        self.rows.append(Row(self.count, [self.convert(x) for x in realline], [], 0))
        self.count += 1
        f.close()

    def dump(self, f):
        f.write("Dump table:"+"\n")
        f.write("t.cols"+"\n")
        for col in self.cols:
            f.write("|  "+str(col.oid)+"\n")
            f.write("|  |  add: Num1"+"\n")
            f.write("|  |  col: "+str(col.oid)+"\n")
            f.write("|  |  hi: "+str(col.num.hi)+"\n")
            f.write("|  |  lo: "+str(col.num.lo)+"\n")
            f.write("|  |  m2: "+str(col.num.m2)+"\n")
            f.write("|  |  mu: "+str(col.num.mu)+"\n")
            f.write("|  |  n: "+str(col.num.n)+"\n")
            f.write("|  |  oid: "+str(col.oid)+"\n")
            f.write("|  |  sd: "+str(col.num.sd)+"\n")
            f.write("|  |  txt: "+str(col.txt)+"\n")
        f.write("t.oid: " + str(self.oid)+"\n")
        f.write("t.rows"+"\n")
        for row in self.rows:
            f.write("|  "+str(row.oid)+"\n")
            f.write("|  |  cells"+"\n")
            index = 0
            for val in row.cells:
                f.write("|  |  |  "+str(index)+": "+str(val)+"\n")
                index += 1
            f.write("|  |  cooked"+"\n")
            f.write("|  |  dom:"+str(row.dom)+"\n")
            f.write("|  |  oid:"+str(row.oid)+"\n")






def main():
    s="""$cloudCover, $temp, ?$humid, <wind,  $playHours
  100,        68,    80,    0,    3   # comments
  0,          85,    85,    0,    0

  0,          80,    90,    10,   0
  60,         83,    86,    0,    4
  100,        70,    96,    0,    3
  100,        65,    70,    20,   0
  70,         64,    65,    15,   5
  0,          72,    95,    0,    0
  0,          69,    70,    0,    4
  ?,          75,    80,    0,    ?
  0,          75,    70,    18,   4
  60,         72,
  40,         81,    75,    0,    2
  100,        71,    91,    15,   0
  """
    tbl = Tbl(0, s, True)
    with open("outpart3.txt", "w") as f:
        tbl.dump(f)


if __name__ == '__main__':
    main()
