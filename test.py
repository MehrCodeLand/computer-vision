def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)



import threading
import time

def run_threads():
    start_time = time.time()
    
    threads = []
    for _ in range(2):  # Two threads, but still runs sequentially due to GIL
        t = threading.Thread(target=fib, args=(35,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Threads Execution Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    run_threads()


import multiprocessing
import time

def run_processes():
    start_time = time.time()
    
    processes = []
    for _ in range(2):  # Two processes, true parallel execution
        p = multiprocessing.Process(target=fib, args=(35,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"Processes Execution Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    run_processes()

