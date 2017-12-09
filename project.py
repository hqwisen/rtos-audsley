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
                                            help='Return the feasibility interval of given tasks.')
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
                                           help='Depicts the research made by the Audsley’s'
                                                ' algorithm to find a schedulable priorities'
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
    :raise ValueError if too much elements, not enough elements or invalid elements.
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
        # FIXME I'm not sure about the  * period at the end of the calc. see slide 56
        return offset + (math.ceil(max(previous_s - offset, 0) / period) * period)


def interval(tasks_file):
    log.info("Feasibility interval of '%s'" % tasks_file)
    tasks = parse_tasks(tasks_file)
    omax = max([task[O] for task in tasks])
    P = hyper_period(tasks)
    print(omax, omax + (2 * P))


def sim(start, stop, tasks_file):
    log.info("Simulation for '%s'" % tasks_file)
    FTPSimulation(start, stop, parse_tasks(tasks_file)).run()


class FTPSimulation:
    def __init__(self, start, stop, tasks):
        self.start = start
        self.stop = stop
        self.tasks = tasks
        self.active_jobs = [0 for t in tasks]
        self.S = calculate_s(self.tasks)
        self.P = hyper_period(self.tasks)

    # def transient_task(self, higher_tasks=None):
    #     compute = []
    #     if higher_tasks is None:  # First task
    #         if self.tasks[0][O] < self.S:
    #             return []  # No jobs to return
    #         else:
    #
    #             job = [self.tasks[0][O], self.tasks[0][D]]
    #             job.append((self.tasks[0][O], self.tasks[0][O] + self.tasks[0][T]))

    # def transient(self):
    #     """
    #     (0, Sn) transient interval
    #     :return:
    #     """
    #     times = {}
    #     for t in range(0, self.S):
    #        print(i, end = '')

    def is_arrival_for(self, t, task_index):
        """
        Return job_index if a job request occurs at time t for task_index.
        :param t: time step
        :param task_index
        :return: (job_index, True) if task_index request a new job at time t occurs,
        (None, False) otherwise. (first job_index is 0)
        """
        offset, period = self.tasks[task_index][O], self.tasks[task_index][T]
        cond = (t - offset >= 0) and ((t - offset) % period) == 0
        job_index = (t - offset) // period  # module == 0 i.e. no decimals
        return (job_index, True) if cond else (None, False)

    def is_deadline_for(self, t, task_index):
        """
        Return job_index if a deadline of a job occurs at time t for task_index.
        :param t: time step
        :param task_index
        :return: (job_index, True) if task_index have a deadline at time t,
        (None, False) otherwise. (first job_index is 0)
        """
        offset, deadline = self.tasks[task_index][O], self.tasks[task_index][D]
        cond = (t - offset > 0) and ((t - offset) % deadline) == 0
        # There is no deadline  at (t - offset) = 0, since there is no job before
        # So the deadline is for the job that was running previously, i.e -1
        job_index = ((t - offset) // deadline) - 1  # module == 0 i.e. no decimals
        return (job_index, True) if cond else (None, False)

    def print_actions(self, t, name, test_func):
        for task_index in range(len(self.tasks)):
            job_index, is_action = test_func(t, task_index)
            if is_action:
                print("%s: %s of job T%sJ%s" % (t, name, task_index + 1,
                                                job_index + 1))

    def print_arrivals(self, t):
        self.print_actions(t, 'Arrival', self.is_arrival_for)

    def print_deadlines(self, t):
        self.print_actions(t, 'Deadline', self.is_deadline_for)

    def run(self):
        print("Schedule from: %d to: %d ; %d tasks" % (self.start, self.stop, len(self.tasks)))
        finterval = (0, self.S + self.P)
        log.debug("Sn = %s ; P = %s" % finterval)
        log.info("Feasibility/Periodicity"
                 " interval of simulation (0, Sn + P) = %s " % str(finterval))

        for t in range(self.start, self.stop + 1):
            self.print_arrivals(t)
            self.print_deadlines(t)


def lowest_priority_viable(tasks, start, stop, index):
    raise NotImplementedError("Function 'lowest_priority_viable' not implemented")


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
