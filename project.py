import argparse
import logging
import sys

log = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

# Tasks keys
TASK_KEYS = ['offset', 'period', 'deadline', 'WCET']
O, T, D, C = 0, 1, 2, 3
N_TASK_KEYS = len(TASK_KEYS)


def add_task_arg(parser):
    parser.add_argument(dest='tasks_file', type=str,
                        help='File containing tasks'
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


def parse_task(tn, elems):
    """
    Parse a task from a list of elements
    :param tn: Number of the task
    :param elems: List of elements containing O, T, D, C of a task
    :raise ValueError if too much elements, not enough elements or invalid elements.
    :return:
    """
    log.debug("Parse task #%s with %s" % (tn, elems))
    task = []
    if len(elems) > N_TASK_KEYS:
        raise ValueError("Task #%s must only contain %s" % (tn, TASK_KEYS))
    try:
        for i in range(len(elems)):
            task.append(int(elems[i]))
    except ValueError:
        raise ValueError(
            "Expecting integer for %s of task #%s, got '%s' instead." % (
                TASK_KEYS[i], tn, elems[i]))
    if len(elems) < N_TASK_KEYS:
        raise ValueError(
            "Error: No %s given for task #%s" % (TASK_KEYS[len(elems)], tn))
    return task


def parse_tasks(filename):
    tasks = []
    try:
        with open(filename, 'r') as f:
            tn = 0
            for line in f.readlines():
                elems = line.strip().split()
                if len(elems) == 0:  # Ignoring empty lines
                    continue
                tasks.append(parse_task(tn, elems))
                tn += 1
    except (FileNotFoundError, ValueError) as e:
        log.error(e)
        sys.exit(e)
    log.info("Tasks successfully parsed from '%s'" % filename)
    log.debug("Results tasks %s " % tasks)
    return tasks


def interval(tasks_file):
    log.info("Feasibility interval of '%s'" % tasks_file)
    tasks = parse_tasks(tasks_file)
    print(tasks)


def main():
    kwargs = parse_args()
    action = kwargs['action']
    del kwargs['action']
    exec_func(action, **kwargs)


if __name__ == "__main__":
    main()
