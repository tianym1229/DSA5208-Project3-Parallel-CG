import csv
import io
import subprocess

for dim in (2, 3):
    for grid in (3, 5, 9):
        for solution in ('ones', 'smooth'):
            p = subprocess.run(['./cg_serial', '--dim', str(dim), '--grid', str(grid), '--solution', solution], capture_output=True, text=True, check=True)
            row = next(csv.DictReader(io.StringIO(p.stdout)))
            assert int(row['converged']) == 1, row
            assert float(row['relative_residual']) < 1e-8, row
            assert float(row['relative_error']) < 1e-6, row
            n = grid ** dim
            expected_nnz = (5*grid*grid-4*grid) if dim==2 else (7*grid**3-6*grid*grid)
            assert int(row['n']) == n and int(row['nnz']) == expected_nnz, row
            print(f"PASS {dim}D grid={grid} solution={solution}")
print('All 12 serial tests passed.')
