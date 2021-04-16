import multiprocessing as mp


# inspired by https://stackoverflow.com/questions/43078980/python-multiprocessing-with-generator
def process(q, iolock):
    from time import sleep
    while True:
        stuff = q.get()
        if stuff is None:
            break
        with iolock:
            print("processing", stuff)
        sleep(stuff)


class TestExperiment:
    def __init__(self, ncores):
        self.ncores = ncores

    def run(self):
        q = mp.Queue(maxsize=self.ncores)
        iolock = mp.Lock()

        pool = mp.Pool(self.ncores, initializer=process, initargs=(q, iolock))
        for stuff in range(20):
            q.put(stuff)  # blocks until q below its max size
            with iolock:
                print("queued", stuff)
        for _ in range(self.ncores):  # tell workers we're done
            q.put(None)
        pool.close()
        pool.join()
