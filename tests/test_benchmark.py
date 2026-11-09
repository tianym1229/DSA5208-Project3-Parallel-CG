import csv, io, subprocess
for dim,grid in [(2,9),(3,5),(2,24),(3,10)]:
    for exe,cmd in [('serial',['./cg_serial']),('mpi',['mpirun','-np','2','./cg_mpi'])]:
        out=subprocess.check_output(cmd+['--dim',str(dim),'--grid',str(grid),'--solution','benchmark'],text=True,stderr=subprocess.DEVNULL)
        row=next(csv.DictReader(io.StringIO(out)))
        assert row['converged']=='1', (dim,grid,exe,row)
        assert float(row['relative_residual']) < 1e-8, (dim,grid,exe,row)
        assert float(row['relative_error']) < 1e-5, (dim,grid,exe,row)
        assert int(row['iterations']) > 1, (dim,grid,exe,row)
        print('PASS benchmark',dim,grid,exe,'iterations',row['iterations'])
print('All 8 benchmark smoke tests passed.')
