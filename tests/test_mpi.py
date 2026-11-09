#!/usr/bin/env python3
"""Run only INSIDE a PBS allocation with at least 4 MPI slots."""
import csv, io, math, subprocess
for dim,grid in [(2,3),(2,5),(2,9),(3,3),(3,5)]:
    for solution in ('ones','smooth'):
        args=['--dim',str(dim),'--grid',str(grid),'--solution',solution]
        serial=subprocess.run(['./cg_serial',*args],capture_output=True,text=True,check=True)
        base=list(csv.DictReader(io.StringIO(serial.stdout)))[0]
        for ranks in (1,2,4):
            cmd=['mpirun','-np',str(ranks),'./cg_mpi',*args]
            proc=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=120)
            data=list(csv.DictReader(io.StringIO(proc.stdout)))[0]
            assert data['converged']=='1', (cmd,data)
            assert int(data['n'])==int(base['n']) and int(data['nnz'])==int(base['nnz'])
            assert float(data['relative_residual'])<1e-8
            assert float(data['relative_error'])<1e-6
            print(f"PASS dim={dim} grid={grid} solution={solution} ranks={ranks}",flush=True)
print('All 30 MPI tests passed.')
