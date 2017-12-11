import argparse
import logging
import sys
from math import gcd

import math

log = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

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
    add_task_arg(sim_parser)
    sim_parser.set_defaults(action="sim")
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
    return lcm(*[task[T] for task in tasks])


def calculate_s(tasks, i=None):
    """
    Process S value of the Feasibility Interval of an asynchronous
    constrained deadline system, where tasks are FTP.
    :param tasks: list of tasks (ordered by priority i.e. FTP)
    :param i: S_{i} value to calculate. If set to None, will return S_{n}.
    :return: S_{i} value
    """
    if i is None:
        return calculate_s(tasks, len(tasks) - 1)
    elif i == 0:
        return tasks[i][O]
    else:
        offset, period = tasks[i][O], tasks[i][T]
        previous_s = calculate_s(tasks, i - 1)
        # FIXME check that the formula is correct
        return offset + \
               (math.ceil(max(previous_s - offset, 0) / period) * period)


def interval(tasks_file):
    log.info("Feasibility interval of '%s'" % tasks_file)
    tasks = parse_tasks(tasks_file)
    omax = max([task[O] for task in tasks])
    P = hyper_period(tasks)
    print(omax, omax + (2 * P))


def sim(start, stop, tasks_file):
    log.info("Simulation for '%s'" % tasks_file)
    FTPSimulation(start, stop, parse_tasks(tasks_file)).run()


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

    def __str__(self):
        """
        The task_id and job_id are printed with the first id
        as 1 (not 0).
        :return: String representation of a job
        """
        return "T%sJ%s" % (self.task_id + 1, self.job_id + 1)

    def __repr__(self):
        return str(self)


class FTPSimulation:
    def __init__(self, start, stop, tasks):
        self.start = start
        self.stop = stop
        self.tasks = tasks
        self.active_jobs = [0 for t in tasks]
        self.S = calculate_s(self.tasks)
        self.P = hyper_period(self.tasks)
        # '[[]] * x' will reference  the same list i.e. the for loop instead
        self.pending_jobs = [[] for _ in range(self.tasks_count)]

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
        offset, deadline = self.tasks[task_id][O], self.tasks[task_id][D]
        cond = (t - offset > 0) and ((t - offset) % deadline) == 0
        # There is no deadline  at (t - offset) = 0, since there is no job
        #  before, so the deadline is for the job that was
        # running previously, i.e -1
        #  module == 0 i.e. no decimals
        job_id = ((t - offset) // deadline) - 1  #
        return (job_id, True) if cond else (None, False)

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

    def handle_arrivals(self, t):
        requested_jobs = self.get_job_arrivals(t)
        for task_id in range(self.tasks_count):
            # Adding jobs to be computed
            self.pending_jobs[task_id].extend(requested_jobs[task_id])
            for job in requested_jobs[task_id]:
                print("%s: Arrival of job %s" % (t, job))

    def handle_deadlines(self, t):
        jobs = self.get_job_deadlines(t)
        print(jobs)
        for job in jobs:
            print("%s: Deadline of job %s" % (t, job))

    def get_active_jobs(self):
        """
        Return the active jobs i.e the first jobs of the
        highest tasks, if it exists.
        :return: active job (first job of higher prio. task)
        """
        for sub_jobs in self.pending_jobs:
            # As soon as a pending job is detected return it
            # Since it is sorted by tasks, and arrivals
            for job in sub_jobs:
                return job

    def compute(self):
        """
        Compute the higher priority job. If jobs is finished
        it's removed from pending_jobs.
        """
        active_job = self.get_active_jobs()
        active_job.compute()
        if active_job.done():
            self.pending_jobs[active_job.task_id].remove(active_job)

    def run(self):
        print("Schedule from: %d to: %d ; %d tasks"
              % (self.start, self.stop, self.tasks_count))
        finterval = (0, self.S + self.P)
        log.debug("Sn = %s ; P = %s" % finterval)
        log.info("Feasibility/Periodicity"
                 " interval of simulation (0, Sn + P) = %s " % str(finterval))

        for t in range(self.start, self.stop + 1):
            self.handle_arrivals(t)
            self.compute()
            # self.handle_deadlines(t)


def lowest_priority_viable(tasks, start, stop, index):
    raise NotImplementedError(
        "Function 'lowest_priority_viable' not implemented")


def audsley(first, last, tasks_file):
    log.info("Audsley algorithm for '%s'" % tasks_file)
    tasks = parse_tasks(tasks_file)
    raise NotImplementedError("Function 'audsley' not implemented")


def main():
    kwargs = parse_args()
    action = kwargs['action']
    del kwargs['action']
    exec_func(action, **kwargs)


if __name__ == "__main__":
    main()
