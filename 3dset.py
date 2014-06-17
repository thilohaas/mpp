from functools import partial
import multiprocessing
import numpy
import random
import operator

processes           = 4
numberOfParticles   = 2**16
maxSpace            = 2**16
chunkSizes          = 2**4
randomSample        = 2**10
randomLoops         = 2**10

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

#########################
# Assignment Part 2 (a) #
#########################
def calculate_mass_center(items):
    x = [i[0] for i in items]
    y = [i[1] for i in items]
    z = [i[2] for i in items]
    return (sum(x) / len(items), sum(y) / len(items), sum(z) / len(items))

def mp_calculate_mass_center(pool, particles):
    results = pool.map(calculate_mass_center, chunks(particles, chunkSizes))
    while len(results) > 1:
        results = pool.map(calculate_mass_center, chunks(results, chunkSizes))
    return results[0]

###########################
# Assignment Part 2 (b/c) #
###########################
def shift_to_center(randParticles, centerOfMass):
    shiftedParticles = numpy.empty(len(randParticles), dtype=tuple)
    i = 0
    for particle in randParticles:
        direction = tuple(map(operator.sub, centerOfMass, particle))
        randomDistance = (random.random(),) * len(direction)
        randomDistance = tuple(map(operator.mul, randomDistance, direction))
        shiftedParticles[i] = tuple(map(operator.add, particle, randomDistance))
        i += 1
    return shiftedParticles


def mp_shift_to_center(pool, randParticles, centerOfMass):
    result = numpy.empty(0, dtype=tuple)

    partial_shift_to_center = partial(shift_to_center, centerOfMass=centerOfMass)
    mapResults = pool.map(partial_shift_to_center, chunks(randParticles, chunkSizes))
    for mapResult in mapResults:
        result = numpy.append(result, mapResult)
    return result

#########################
# Assignment Part 2 (d) #
#########################
def find_particles_within_range(particles, location, distance):
    particlesInDistance = []
    i = 0
    for particle in particles:
        direction = tuple(map(operator.sub, particle, location))
        x = numpy.array(direction)
        particleDistance = numpy.linalg.norm(x)
        if distance > particleDistance:
            particlesInDistance.append(particle)
            i += 1
    return numpy.array(particlesInDistance, dtype=tuple)

def mp_find_particles_within_range(pool, particles, location, distance):
    result = []

    partial_find_particles_within_range = partial(find_particles_within_range, location=location, distance=distance)
    mapResults = pool.map(partial_find_particles_within_range, chunks(particles, chunkSizes))
    for mapResult in mapResults:
        for res in mapResult:
            result.append(res)
    return numpy.array(result, dtype=tuple)



# Initializing Pool
pool = multiprocessing.Pool(processes=processes)

# Setup particle sample
particles = numpy.empty(numberOfParticles, dtype=tuple)
for i in xrange(numberOfParticles):
    particles[i] = (random.randrange(maxSpace),random.randrange(maxSpace),random.randrange(maxSpace))


########################################################################
# Assignment Part 2 (a) compute the position of the centre of the mass #
########################################################################
print "Assignment Part 2 (a)"

centerOfMass = mp_calculate_mass_center(pool, particles)
print "Center of the mass:", centerOfMass


#################################################################################################################################
# Assignment Part 2 (b) 2^10 times randomly select 2^10 particles and compute the position of the center of the selected masses #
#################################################################################################################################
print "Assignment Part 2 (b)"

for i in xrange(randomLoops):
    print "Assignment Part 2 (b) - round ", i
    randParticles = random.sample(particles, randomSample)
    result = mp_calculate_mass_center(pool, randParticles)
    print result


################################################################################################################################
# Assignment Part 2 (c) shift a randomly selected sub set of particles a random distance towards the center of the mass system #
################################################################################################################################
print "Assignment Part 2 (c)"

randParticles = random.sample(particles, randomSample)
resultC = mp_shift_to_center(pool, randParticles, centerOfMass)
print resultC


############################################################################################
# Assignment Part 2 (d) find all particles at a given distance range from a given location #
############################################################################################
print "Assignment Part 2 (d)"

print mp_find_particles_within_range(pool, particles, (32669, 32824, 32726), 1500)