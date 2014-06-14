# coding=utf-8
from array import array
import codecs
import csv
import numpy
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")

class UnicodeReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, "utf-8") for s in row]

    def __iter__(self):
        return self

with open('data/Mitglieder.csv') as csvfile:
    csvreader = UnicodeReader(csvfile, delimiter=';')
    i = 0
    for row in csvreader:
        if i == 0:
            i += 1
            continue

        if i%size != rank:
            i += 1
            continue

        comm.send(row[1], dest=0)
        i += 1

if rank == 0:
    names = {}

    for j in range(1, i-1):
        buf = array('c', '\0') * 256
        r = comm.Irecv(buf, ANY_SOURCE)
        status = MPI.Status()
        r.Wait(status)
        n = status.Get_count(MPI.CHAR)
        s = buf[:n].tostring()

        if s in names:
            names[s] += 1
        else:
            names[s] = 1

    fig = plt.figure()
    ax = fig.add_subplot(211)

    plt.title(r'Identical Firstnames')
    rects1 = ax.hist(names.values(), color='r')

    fig.savefig('histograms/hist_identical_names.png')
