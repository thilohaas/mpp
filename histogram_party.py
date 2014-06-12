# coding=utf-8
# mpiexec -n 3 python histogram_party.py
import csv
import numpy
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dataset = []
parties = [
    'SP',
    'CVP',
    'SVP',
    'GLP',
    'FDP',
    'AL',
    'CSP',
    'BDP',
    'Grüne',
    'EDU',
    'EVP'
]

with open('Mitglieder.csv') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';')
    i = 0
    for row in csvreader:
        if i == 0:
            i += 1
            continue

        dataset.append(
            {
                'firstname': row[1],
                'lastname': row[0],
                'birthdate': row[2],
                'party': row[11],
            }
        )

comm.Barrier()

local_result = numpy.array([[[0]*26]*11]*3)
for i in xrange(rank, len(dataset), size):
    for letter in dataset[i]['firstname']:
        index = ord(letter.upper()) - 65
        if 0 <= index <= 26:
            local_result[0][parties.index(dataset[i]['party'])][index] += 1

    for letter in dataset[i]['lastname']:
        index = ord(letter.upper()) - 65
        if 0 <= index <= 26:
            local_result[1][parties.index(dataset[i]['party'])][index] += 1

    for letter in dataset[i]['birthdate']:
        index = ord(letter) - 48
        if 0 <= index <= 9:
            local_result[2][parties.index(dataset[i]['party'])][index] += 1

result = numpy.array([[[0]*26]*11]*3)
comm.Reduce(
    [local_result, MPI.DOUBLE],
    [result, MPI.DOUBLE],
    op=MPI.SUM,
    root=0
)

comm.Barrier()

if rank == 0:

    alphabet = map(chr, xrange(65,91))
    resultChars = []

    i = 0
    for party in parties:
        fig = plt.figure()
        ax = fig.add_subplot(211)

        ind = numpy.arange(26) # the x locations for the groups
        width = 0.40 # the width of the bars
        rects1 = ax.bar(ind, result[0][i], width, color='r')
        rects2 = ax.bar(ind+width, result[1][i], width, color='b')

        plt.title(r'First-/Lastname Letters')
        ax.set_xticks(ind + width)
        ax.set_xticklabels( alphabet )
        ax.legend((rects1[0], rects2[0]), ('Firstname', 'Lastname') )

        ax = fig.add_subplot(212)

        birthdayBars = ax.bar(ind+width, result[2][i], width, color='r')
        ind = numpy.arange(10)

        plt.title(r'Birthday Numbers')
        ax.set_xticks(ind + width)
        ax.set_xticklabels( range(10) )

        i += 1

        plotfilename = 'histograms/hist_' + party + '.png'
        fig.savefig(plotfilename)


