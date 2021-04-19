import logging
import ray
from time import sleep

logger = logging.getLogger(__name__)


@ray.remote
class L(object):

    def __init__(self, worker_id):
        self.logger = logging.getLogger(__name__)
        self.wid = worker_id

    def info(self, message):
        self.logger.info(f'W: {self.wid} :: {message}')

    def warning(self, message):
        self.logger.warning(f'W: {self.wid} :: {message}')

    def debug(self, message):
        self.logger.debug(f'W: {self.wid} :: {message}')

    def critical(self, message):
        self.logger.critical(f'W: {self.wid} :: {message}')


@ray.remote
def run_worker(j):
    log = L.remote(j)
    for _ in range(10):
        sleep(2)
        log.warning.remote("informative log")


def run():
    ray.init(address='auto', _redis_password="")
    print('''This cluster consists of
        {} nodes in total
        {} CPU resources in total
    '''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))
    lo = L.remote(1000)
    lo.warning.remote("Starting!")
    ray.get([run_worker.remote(i) for i in range(20)])
    # ray.get()
    lo.warning.remote("print from outside worker")
