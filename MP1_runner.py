# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 20:47:05 2021
Runner for Mini-Project 1

@author: Kevin
"""

import time;

#import config here to share its content.
import configMP1
import Kevin_Task1
import Kevin_Task2

def main():
    print("Task 1")
    Kevin_Task1.main()
    print("\nTask 2")
    Kevin_Task2.main()
    
if __name__ == "__main__":
    begin_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print("This script has taken", end_time - begin_time, "seconds to execute.")