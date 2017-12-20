# Simulation

Since we know the priority of the tasks (i.e. FTP), we
can use the periodicity of the schedule using the feasability
interval [0, Sn + P).

We can only have on job per 
task at a time (it is constrained deadline)


We provide a discrete simulation. Meaning that the CPU is simulated for each
t. There is no formula to avoid doing the simulation. It is indead possible
to simulate only [0, Sn + P).

Provide a two-implementation of the simulator, one discrete, and one periodic.

# Arrivals and Deadlines

Since those action are periodic, we can calculate them using modulo


Formulas:

(t - offset) // period

((t - offset) // deadline) - 1

## Output

if in t:
-- if something finshed
---- print it
-- if something happens
---- if job wa running
------ print it
---- print it