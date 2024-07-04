#!/bin/bash
mpirun -np 1 gdb $@ : -np 15 ./main