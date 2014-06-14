import csv
import numpy
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = []
with open('data/Mitglieder.csv') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';')
    for row in csvreader:
        data.append(
            {
                'firstname': row[1],
                'lastname': row[0],
                'birthdate': row[2]
            }
        )

comm.Barrier()

local_result = numpy.array([[0]*26]*3)
for i in xrange(rank, len(data), size):
    for letter in data[i]['firstname']:
        index = ord(letter.upper()) - 65
        if 0 <= index <= 26:
            local_result[0][index] += 1

    for letter in data[i]['lastname']:
        index = ord(letter.upper()) - 65
        if 0 <= index <= 26:
            local_result[1][index] += 1

    for letter in data[i]['birthdate']:
        index = ord(letter) - 48
        if 0 <= index <= 9:
            local_result[2][index] += 1

result = numpy.array([[0]*26]*3)
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

    fig = plt.figure()
    ax = fig.add_subplot(211)

    ind = numpy.arange(26) # the x locations for the groups
    width = 0.40 # the width of the bars
    rects1 = ax.bar(ind, result[0], width, color='r')
    rects2 = ax.bar(ind+width, result[1], width, color='b')

    plt.title(r'First-/Lastname Letters')
    ax.set_xticks(ind + width)
    ax.set_xticklabels( alphabet )
    ax.legend((rects1[0], rects2[0]), ('Firstname', 'Lastname') )

    ax = fig.add_subplot(212)

    birthdayBars = ax.bar(ind+width, result[2], width, color='r')
    ind = numpy.arange(10)

    plt.title(r'Birthday Numbers')
    ax.set_xticks(ind + width)
    ax.set_xticklabels( range(10) )

    fig.savefig('histograms/hist_members.png')

