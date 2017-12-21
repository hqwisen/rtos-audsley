import argparse
import logging
import math
import sys
import copy
from math import gcd
from random import randint

log = logging.getLogger()
logging.basicConfig(level=logging.FATAL)

# Tasks keys
TASK_KEYS = ['offset', 'period', 'deadline', 'WCET']
O, T, D, C = 0, 1, 2, 3  # Indices of TASK_KEYS
N_TASK_KEYS = len(TASK_KEYS)


def add_task_arg(parser):
    parser.add_argument(dest='tasks_file', type=str,
                        help='File containing tasks: '
                             ' Offset, Period, Deadline and WCET.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='RTOS 2017-18: Project 1 Audsley.')
    subparsers = parser.add_subparsers(title='Valid commands',
                                       description='',
                                       help='Valid options to be ran.')
    interval_parser = subparsers.add_parser('interval',
                                            help='Return the feasibility '
                                                 'interval of given tasks.')
    add_task_arg(interval_parser)
    interval_parser.set_defaults(action='interval')

    sim_parser = subparsers.add_parser('sim',
                                       help='FTP simulator that simulates '
                                            'the system for a given period.')
    sim_parser.add_argument(dest='start', type=int,
                            help='Start point of the simulation.')
    sim_parser.add_argument(dest='stop', type=int,
                            help='Stop point of the simulation.')
    sim_parser.add_argument("--preemptive", action="store_true", default=False,
                            help="Activate preemptive output mode.")
    sim_parser.add_argument("--hard", action="store_true", default=False,
                            help="Considers deadlines as hard.")
    add_task_arg(sim_parser)
    sim_parser.set_defaults(action="sim")

    plot_parser = subparsers.add_parser('plot',
                                        help='Plot the FTP simulation '
                                             'for a given period.')
    plot_parser.add_argument(dest='start', type=int,
                             help='Start point of the simulation.')
    plot_parser.add_argument(dest='stop', type=int,
                             help='Stop point of the simulation.')
    add_task_arg(plot_parser)
    plot_parser.set_defaults(action="plot")

    audsley_parser = subparsers.add_parser('audsley',
                                           help='Depicts the research made by '
                                                'the Audsley’s algorithm to '
                                                'find a schedulable priorities'
                                                ' assignment.')
    audsley_parser.add_argument(dest='first', type=int,
                                help='First point of the interval.')
    audsley_parser.add_argument(dest='last', type=int,
                                help='Last point of the interval.')
    add_task_arg(audsley_parser)
    audsley_parser.set_defaults(action="audsley")

    gen_parser = subparsers.add_parser('gen',
                                       help='Generate a random  '
                                            'random periodic, asynchronous '
                                            'systems with '
                                            'constrained deadlines.')

    gen_parser.add_argument(dest='number_of_tasks', type=int,
                            help='Number of tasks')
    gen_parser.add_argument(dest='utilisation_factor', type=int,
                            help='Utilisation factor of the system')

    gen_parser.add_argument(dest='tasks_file', type=str,
                            help='File containing genearted tasks where '
                                 'every task has this format: '
                                 ' Offset, Period, Deadline and WCET.')
    gen_parser.set_defaults(action="gen")

    return vars(parser.parse_args())


def exec_func(func_name, **kwargs):
    """
    Execute a function from a string.
    :param func_name: Name of the function (str)
    :raise NotImplementedError if function doesn't exist
    """
    possibles = globals().copy()
    possibles.update(locals())
    method = possibles.get(func_name)
    if not method:
        raise NotImplementedError("Function '%s' not implemented" % func_name)
    method(**kwargs)


def parse_task(ti, elems):
    """
    Parse a task from a list of elements
    :param ti: task index (starts at 0)
    :param elems: List of elements containing O, T, D, C of a task
    :raise ValueError if too much elements,
    not enough elements or invalid elements.
    :return:
    """
    log.debug("Parse task #%s with %s" % (ti + 1, elems))
    task = []
    if len(elems) > N_TASK_KEYS:
        raise ValueError("Task #%s must only contain %s" % (ti + 1, TASK_KEYS))
    try:
        for i in range(len(elems)):
            task.append(int(elems[i]))
    except ValueError:
        raise ValueError(
            "Expecting integer for %s of task #%s, got '%s' instead." % (
                TASK_KEYS[i], ti + 1, elems[i]))
    if len(elems) < N_TASK_KEYS:
        raise ValueError(
            "Error: No %s given for task #%s" % (TASK_KEYS[len(elems)], ti + 1))
    return task


def parse_tasks(filename):
    tasks = []
    try:
        with open(filename, 'r') as f:
            ti = 0
            for line in f.readlines():
                elems = line.strip().split()
                if len(elems) == 0:  # Ignoring empty lines
                    continue
                tasks.append(parse_task(ti, elems))
                ti += 1
    except (FileNotFoundError, ValueError) as e:
        log.error(e)
        sys.exit(e)
    log.info("Tasks successfully parsed from '%s'" % filename)
    log.debug("Results tasks %s " % tasks)
    return tasks


def __lcm(a, b):
    result = a * b // gcd(a, b)
    log.debug("LCM(%s, %s) = %s" % (a, b, result))
    return result


def lcm(*elems):
    if len(elems) < 2:
        raise ValueError("Cannot return lcm of %s" % str(elems))
    if len(elems) == 2:
        return __lcm(*elems)
    else:
        log.debug("LCM(%s, %s)" % (elems[0], str(elems[1:])))
        return __lcm(elems[0], lcm(*elems[1:]))


def hyper_period(tasks):
    if len(tasks) == 0:
        return None
    elif len(tasks) == 1:
        return tasks[0][T]
    return lcm(*[task[T] for task in tasks])


def calculate_s(tasks, i=None):
    """
    Process S value of the Feasibility Interval of an asynchronous
    constrained deadline system, where tasks are FTP.
    :param tasks: list of tasks (ordered by priority i.e. FTP)
    :param i: S_{i} value to calculate. If set to None, will return S_{n}.
    :return: S_{i} value
    """

    if len(tasks) == 0:
        return None
    if i is None:
        return calculate_s(tasks, len(tasks) - 1)
    elif i == 0:
        return tasks[i][O]
    else:
        offset, period = tasks[i][O], tasks[i][T]
        previous_s = calculate_s(tasks, i - 1)
        return offset + \
               (math.ceil(max(previous_s - offset, 0) / period) * period)


class FTPSimulationException(Exception):
    pass


class Job:
    def __init__(self, task_id, job_id, computation):
        self.task_id = task_id
        self.job_id = job_id
        if computation < 0:
            raise FTPSimulationException("Cannot create a job with a negative "
                                         "computation.")
        self.computation = computation
        self.remaining_computation = computation

    def compute(self, computation=1):
        if self.done():
            raise FTPSimulationException("Cannot compute a job that is done.")
        self.remaining_computation -= computation

    def done(self):
        return self.remaining_computation == 0

    def __eq__(self, other):
        if other is None or not isinstance(other, Job):
            return False
        else:
            return self.task_id == other.task_id \
                   and self.job_id == other.job_id

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        """
        The task_id and job_id are printed with the first id
        as 1 (not 0).
        :return: String representation of a job
        """
        return "T%sJ%s" % (self.task_id + 1, self.job_id + 1)

    def __repr__(self):
        return str(self)


class Event:
    def __init__(self, t):
        self.t = t
        self.computed_job = None
        self.deadlines, self.arrivals = [], []
        self.missed_deadlines = []

    def has_requests(self):
        return self.has_deadlines() \
               or self.has_arrivals() or self.has_missed_deadlines()

    def has_arrivals(self):
        return len(self.arrivals) != 0

    def has_deadlines(self):
        return len(self.deadlines) != 0

    def has_missed_deadlines(self):
        return len(self.missed_deadlines) != 0

    def has_all_deadlines(self):
        return self.has_deadlines() or self.has_missed_deadlines()

    def print_arrivals(self):
        for job in self.arrivals:
            print("%s: Arrival of job %s" % (self.t, job))

    def print_deadlines(self):
        for job in self.deadlines:
            print("%s: Deadline of job %s" % (self.t, job))

    def print_missed_deadlines(self):
        for job in self.missed_deadlines:
            print("%s: Job %s misses a deadline" % (self.t, job))

    def print_all_deadlines(self):
        self.print_deadlines()
        self.print_missed_deadlines()

    def print(self):
        self.print_arrivals()
        self.print_all_deadlines()

    def __str__(self):
        string = ""
        string += "Job=" + str(self.computed_job)
        string += " Arrivals=" + str(self.arrivals)
        string += " Deadlines=" + str(self.deadlines)
        string += " Missed D.=" + str(self.missed_deadlines)

        return string

    def __repr__(self):
        return str(self)


class FTPSimulation:
    def __init__(self, start, stop, tasks, hard=False):
        self.start = start
        self.stop = stop
        self.tasks = tasks
        self.hard = hard
        self.S = calculate_s(self.tasks)
        self.P = hyper_period(self.tasks)
        self.pending_jobs = [[] for _ in range(self.tasks_count)]
        self.missed_jobs = [0 for _ in range(self.tasks_count)]
        self.scheduling = [[] for _ in range(self.tasks_count)]
        self.events = {}

    @property
    def tasks_count(self):
        return len(self.tasks)

    def is_arrival_for(self, t, task_id):
        """
        Return job_id if a job request occurs at time t for task_id.
        :param t: time step
        :param task_id
        :return: (job_id, True) if task_id request a new job at time t
        occurs, (None, False) otherwise. (first job_id is 0)
        """
        offset, period = self.tasks[task_id][O], self.tasks[task_id][T]
        cond = (t - offset >= 0) and ((t - offset) % period) == 0
        job_id = (t - offset) // period  # module == 0 i.e. no decimals
        return (job_id, True) if cond else (None, False)

    def is_deadline_for(self, t, task_id):
        """
        Return job_id if a deadline of a job occurs at time t for task_id.
        :param t: time step
        :param task_id
        :return: (job_id, True) if task_id have a deadline at time t,
        (None, False) otherwise. (first job_id is 0)
        """
        deadline = self.tasks[task_id][D]
        return self.is_arrival_for(t - deadline, task_id)

    def _jobs_for(self, t, test_func):
        """
        Return a list of list of jobs base on test_func
        :param t:
        :param test_func:
        :return: [[jobs of T1], ... [jobs of T2]] that correspond to test_func
        (usually either arrivals jobs, or deadlines)
        """
        jobs = [[] for _ in range(self.tasks_count)]
        for task_id in range(self.tasks_count):
            job_id, is_action = test_func(t, task_id)
            if is_action:
                jobs[task_id].append(Job(task_id, job_id,
                                         self.tasks[task_id][C]))
        return jobs

    def get_job_arrivals(self, t):
        return self._jobs_for(t, self.is_arrival_for)

    def get_job_deadlines(self, t):
        return self._jobs_for(t, self.is_deadline_for)

    def miss_deadline(self, job):
        return job in self.pending_jobs[job.task_id]

    def add_arrivals(self, t):
        requested_jobs = self.get_job_arrivals(t)
        for task_id in range(self.tasks_count):
            self.pending_jobs[task_id].extend(requested_jobs[task_id])
        return requested_jobs

    def get_active_job(self):
        """
        Return the active jobs i.e the first jobs of the
        highest tasks, if it exists.
        :return: active job (first job of higher prio. task), None if
        no jobs to compute.
        """
        for sub_jobs in self.pending_jobs:
            # As soon as a pending job is detected return it
            # Since it is sorted by tasks, and arrivals
            for job in sub_jobs:
                return job
        return None

    def add_scheduling_data(self, active_job):
        for task_id in range(self.tasks_count):
            # 0 will be used as the idle points in the plots
            value = task_id + 1 if active_job is not None and \
                                   task_id == active_job.task_id else 0
            self.scheduling[task_id].append(value)

    def compute(self, t):
        """
        Compute the higher priority job. If jobs is finished
        it's removed from pending_jobs.
        """
        active_job = self.get_active_job()
        self.add_scheduling_data(active_job)
        self.events[t].computed_job = active_job
        if active_job is not None:
            log.debug("Computing %s" % active_job)
            active_job.compute()
            if active_job.done():
                log.debug("%s: Removing job from pending: %s" % (t, active_job))
                self.pending_jobs[active_job.task_id].remove(active_job)

    def run(self):
        finterval = (0, self.S + self.P)
        log.debug("Sn = %s ; P = %s" % finterval)
        log.info("Feasibility/Periodicity"
                 " interval of simulation (0, Sn + P) = %s " % str(finterval))
        miss_deadlines = False
        for t in range(self.start, self.stop + 1):
            log.debug("%s: pending jobs: %s" % (t, self.pending_jobs))
            self.events[t] = Event(t)
            job_deadlines = self.get_job_deadlines(t)
            log.debug("%s: job deadlines: %s" % (t, job_deadlines))

            for task_id in range(self.tasks_count):
                for job in job_deadlines[task_id]:
                    if self.miss_deadline(job):
                        self.events[t].missed_deadlines.append(job)
                        self.missed_jobs[job.task_id] += 1
                        miss_deadlines = True
                    else:
                        self.events[t].deadlines.append(job)

            requested_jobs = self.add_arrivals(t)
            log.debug("%s: requested jobs: %s" % (t, requested_jobs))

            self.compute(t)

            for task_id in range(self.tasks_count):
                for job in requested_jobs[task_id]:
                    self.events[t].arrivals.append(job)

            if self.hard and miss_deadlines:
                self.stop = t
                break;
            # log.debug("Scheduling data: %s" % self.scheduling)

    def output(self, preemptive_mode):
        if preemptive_mode:
            self._output_preemptive_mode()
        else:
            self._output_request_mode()

    def _output_request_mode(self):
        print("Schedule from: %d to: %d ; %d tasks"
              % (self.start, self.stop, self.tasks_count)
              + (" (hard deadlines)" if self.hard else ''))
        compute_start, compute_shift = self.start, 0

        if len(self.events) == 0:
            print("(Nothing to process)")
            return;

        computed_job = self.events[self.start].computed_job
        self.events[self.start].print()
        for t in range(self.start + 1, self.stop + 1):

            if computed_job is not None:
                compute_shift += 1
                if computed_job != self.events[t].computed_job \
                        or self.events[t].has_requests():
                    print("%s-%s: %s" % (compute_start,
                                         compute_start + compute_shift,
                                         computed_job))
                    compute_start, compute_shift = t, 0
            else:
                compute_start, compute_shift = t, 0

            computed_job = self.events[t].computed_job
            if t != self.stop:
                self.events[t].print_arrivals()
            self.events[t].print_all_deadlines()

        if computed_job is not None and compute_shift > 0:
            print("%s-%s: %s" % (compute_start,
                                 compute_start + compute_shift,
                                 computed_job))

    def _output_preemptive_mode(self):
        print("Schedule from: %d to: %d ; %d tasks  (preemptive mode)"
              % (self.start, self.stop, self.tasks_count)
              + (" (hard deadlines)" if self.hard else ''))
        compute_start, compute_shift = self.start, 0

        if len(self.events) == 0:
            print("(Nothing to process)")
            return;

        computed_job = self.events[self.start].computed_job
        self.events[self.start].print()
        for t in range(self.start, self.stop + 1):
            if computed_job is not None:
                compute_shift += 1

                if computed_job != self.events[t].computed_job:
                    print("%s-%s: %s" % (compute_start,
                                         compute_start + compute_shift,
                                         computed_job))

                    for i in range(compute_start + 1, compute_start + compute_shift):
                         if i != self.stop:
                             self.events[i].print_arrivals()
                         self.events[i].print_all_deadlines()
                    compute_start, compute_shift = t, 0

            else:
                compute_start, compute_shift = t, 0
                if t != self.stop:
                    self.events[t].print_arrivals()
                self.events[t].print_all_deadlines()

            computed_job = self.events[t].computed_job

        if computed_job is not None and compute_shift > 0:
            print("%s-%s: %s" % (compute_start,
                                 compute_start + compute_shift,
                                 computed_job))

    def plot(self, filename):
        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for y, row in enumerate(self.scheduling):
            for x, col in enumerate(row):
                x1 = [x + self.start, x + self.stop + 1]
                y1 = np.array([y, y])
                y2 = y1 + 1
                if col != 0:
                    plt.fill_between(x1, y1, y2=y2, color='black')
                else:
                    plt.fill_between(x1, y1, y2=y2, color='white')

        plt.xlim(self.start, self.stop)
        ylabels = ["T%d" % (t + 1) for t in range(self.tasks_count)]
        plt.yticks([t + 0.5 for t in range(self.tasks_count)], ylabels)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


class Generator:
    def __init__(self, utilisation_factor, number_of_tasks):
        self.utilisation_factor = utilisation_factor
        self.number_of_tasks = number_of_tasks
        # This value are chosen arbitrary
        self.maximum_arbitrary_offset = 300
        self.maximum_arbitrary_period = 100
        self.minimum_offset = 0
        # Period cannot be equal to 0
        self.minimum_period = 1
        self.minimum_wcet = 1

    def generate(self):
        """
        The System must just respect this inequality
        Ci <= Di <= Ti for every task i
        """
        task = [0 for _ in range(N_TASK_KEYS)]
        all_tasks = [task[:] for i in range(self.number_of_tasks)]
        # adequate_system = False
        use_of_system = 0
        t = 0
        while t < self.number_of_tasks:
            all_tasks[t][O] = randint(self.minimum_offset,
                                      self.maximum_arbitrary_offset)

            all_tasks[t][T] = randint(self.minimum_period,
                                      self.maximum_arbitrary_period)
            all_tasks[t][C] = randint(self.minimum_wcet,
                                      all_tasks[t][T])  # Ci <= Di <= Ti
            all_tasks[t][D] = randint(all_tasks[t][C],
                                      all_tasks[t][T])  # Ci <= Di <= Ti
            use_of_system += (self.task_utilisation(all_tasks[t]))
            t += 1
            if (use_of_system * 100) > self.utilisation_factor or \
                    (t == self.number_of_tasks and not self.close_to(
                        self.utilisation_factor,
                        use_of_system * 100)):
                # if the system utilisation asked is not reached
                # in the generated system we reinitialize parameters
                t = 0
                # we must do task[:] because of references lists in Python
                # FIXME make sure that there is no trouble with [:]
                task[:] = [0 for _ in range(N_TASK_KEYS)]
                all_tasks[:] = [task[:] for i in range(self.number_of_tasks)]
                use_of_system = 0
        return all_tasks

    def gen(self):
        """
        The System must just respect this inequality
         Ci <= Di <= Ti for every task i
        """
        task = [0 for _ in range(N_TASK_KEYS)]
        all_tasks = [task[:] for i in range(self.number_of_tasks)]
        use_of_system = 0
        t = 0
        while not self.close_to_util(self.utilisation_factor,
                                     use_of_system * 100):
            task = [0 for _ in range(N_TASK_KEYS)]
            all_tasks = [task[:] for i in range(self.number_of_tasks)]
            use_of_system = 0
            t = 0
            for i in range(self.number_of_tasks):
                all_tasks[i][O] = randint(self.minimum_offset,
                                          self.maximum_arbitrary_offset)
                all_tasks[i][T] = randint(self.minimum_period,
                                          self.maximum_arbitrary_period)
                all_tasks[i][C] = randint(self.minimum_wcet, all_tasks[i][T])
                all_tasks[i][D] = randint(all_tasks[i][C], all_tasks[i][T])
                use_of_system += (self.task_utilisation(all_tasks[i]))
                log.info("use_of_system  '%s'" % use_of_system)
        return all_tasks

    def task_utilisation(self, task):
        """
        Calculate the utilisation of task WCET/PERIOD
        """
        return task[C] / task[T]

    def close_to(self, utilisation, founded_util):
        """
        Return True if the utilisation founded is close to utilisation asked
        """
        # the maximum of difference between the utilisation given
        # on parameter and the founded one
        max_difference = 2
        # here ( we followed the logic of round)
        valid_approximation = founded_util >= (
                utilisation - max_difference)
        return valid_approximation

    def close_to_util(self, utilisation, founded_util):
        """
        Return True if the utilisation founded is on a good interval
        """
        # 4 is an arbitrary choice ( we follow the logic of round)
        # the maximum of difference between the
        # utilisation given on parameter and the founded one
        max_difference = 2
        # here ( we followed the logic of round)
        valid_approximation = founded_util >= (
                utilisation - max_difference) and founded_util < utilisation
        return valid_approximation

    def system_utilisation(self, tasks):
        """
        Calculate the utilisation of the system
        which it's the sum of all tasks utilisation
        """
        result = 0
        for task in tasks:
            result += self.task_utilisation(task)
        result *= 100
        return result

    def generate_tasks_on_file(self, filename):
        """
        generate the tasks and write them on a file
        """
        f = open(filename, "w")
        all_tasks = self.generate()
        for task in all_tasks:
            for element in task:
                f.write(str(element) + "  ")
            f.write("\n")
        f.close()
        log.info("System utilisation of the generated tasks  '%s'"
                 % self.system_utilisation(all_tasks))


def lowest_priority_viable(tasks, task_id, start, stop):
    """
    Uses missed_jobs wihch contains
    tasks and number of missed jobs for every task return false if the
    task given on index miss a job
    """
    tasks_copy = tasks[:task_id] + tasks[task_id + 1:] + [tasks[task_id]]
    tasks_copy = copy.deepcopy(tasks_copy)
    simulation = FTPSimulation(start, stop, tasks_copy)
    simulation.run()
    # task_id is at the end when we test the lowest priority viability
    missed_jobs = simulation.missed_jobs[-1]
    log.info("Tasks missed jobs '%s'" % simulation.missed_jobs)
    return missed_jobs == 0


def filter_tasks_list(tasks, task_id, includes):
    """
    Return a list of tasks filters, by including only the indices from includes.
    :param tasks: list of tasks
    :param task_id: task_id to use for the new index
    :param includes: indices of elements to filter
    :return: the sub_tasks list with the corresponding sub_task_id of the
    element with index task_id in tasks.
    Example: tasks = [A, B, C] includes = [0, 2] task_id = 2 (index of C)
    will return [A, C], 1 (only the include element and the new index)
    """
    sub_tasks, sub_task_id = [], None
    for i in range(len(tasks)):
        if i in includes:
            sub_tasks.append(tasks[i])
        if task_id == i:
            sub_task_id = len(sub_tasks) - 1
    return sub_tasks, sub_task_id


def audsley_search(first, last, tasks, includes, level=0):
    """
    Implementation of the audsley algorithm
    :param first: Start point of the simulation
    :param last: last point of the simulation
    :param tasks: list of all tasks, where index of element is the task_id
    :param includes: task_ids of element to run audsley algo on
    :param level: level of recursion of algorithm
    :return: None
    """
    for task_id in includes:
        sub_tasks, sub_task_id = filter_tasks_list(tasks, task_id, includes)
        if lowest_priority_viable(sub_tasks, sub_task_id, first, last):
            print((" " * level) + "Task %d is lowest priority viable"
                  % (task_id + 1))
            new_indices = includes[:]
            new_indices.remove(task_id)
            audsley_search(first, last, tasks, new_indices,
                           level + 1)
        else:
            print((" " * level) + "Task %d is not lowest priority viable"
                  % (task_id + 1))


def audsley(first, last, tasks_file):
    log.info("Running audsley command with '%s'" % tasks_file)
    tasks = parse_tasks(tasks_file)
    indices = [i for i in range(len(tasks))]
    audsley_search(first, last, tasks, indices)


def gen(utilisation_factor, number_of_tasks, tasks_file):
    log.info("Generation of  '%s'" % number_of_tasks)
    Generator(utilisation_factor,
              number_of_tasks).generate_tasks_on_file(tasks_file)


def plot(start, stop, tasks_file):
    log.info("Running simulation (for plot) on '%s'" % tasks_file)
    simulation = FTPSimulation(start, stop, parse_tasks(tasks_file))
    simulation.run()
    filename = "scheduler.png"
    print("Plotting to '%s'" % filename)
    simulation.plot(filename)


def sim(start, stop, tasks_file, preemptive, hard):
    log.info("Simulation for '%s'" % tasks_file)
    simulation = FTPSimulation(start, stop, parse_tasks(tasks_file), hard=hard)
    simulation.run()
    simulation.output(preemptive)


def interval(tasks_file):
    log.info("Feasibility interval of '%s'" % tasks_file)
    tasks = parse_tasks(tasks_file)
    omax = max([task[O] for task in tasks])
    P = hyper_period(tasks)
    print(omax, omax + (2 * P))


def main():
    kwargs = parse_args()
    action = kwargs['action']
    del kwargs['action']
    exec_func(action, **kwargs)


if __name__ == "__main__":
    main()
