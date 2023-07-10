# -utils.py
import math, time, sys

class Stdout_progressbar:
    def __init__(self, max_num, max_length=50) -> None:
        interval = math.ceil(max_num / max_length)
        length = math.ceil(max_num / interval)
        
        self.interval = interval
        self.length = length
        self.max_num = max_num
        self.time = 0

    def reset(self, skip_n=0):
        self.time = time.perf_counter()
        self.skip_n = skip_n

    def update(self, now_num): # num从1开始
        if now_num == 1:
            elapse = time.perf_counter() - self.time
            if elapse < 1:
                print(f"{1/elapse:.2f}step(frame) per second, eta {elapse*(self.max_num - 1)/self.skip_n:.2f}s")
            else:
                print(f"{elapse:.2f}second per step(frame), eta {elapse*(self.max_num - 1)/self.skip_n:.2f}s")

        if now_num % self.interval == 0:
            elapse = time.perf_counter() - self.time

            i = now_num // self.interval
            percentage = round(now_num / self.max_num * 100)
            print("\r", end="")
            print("Progress: {}%: |".format(percentage), "-" * (i), end="")
            print(" "*(self.length - i),"|",f"use {elapse:.2f}s", end="")
            sys.stdout.flush()

        if now_num == self.max_num:
            elapse = time.perf_counter() - self.time
            if now_num % self.interval != 0:
                print("\r", end="")
                print(f"Progress: {percentage}%: |", "-" * self.length, "|", end="")

            print("\nprecess finished!")
            print(f"totaly use {elapse:.2f}s")
            sys.stdout.flush()
