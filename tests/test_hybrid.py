import csv
import io
import os
import subprocess


def run(exe, ranks, threads, dim, grid, comm):
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads)
    env['OMP_DYNAMIC'] = 'FALSE'
    cmd = ['mpirun', '-np', str(ranks), './'+exe, '--dim', str(dim),
           '--grid', str(grid), '--solution', 'benchmark', '--comm', comm]
    result = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=180)
    if result.returncode:
        raise AssertionError(f'{cmd} failed: {result.stderr[-2000:]}')
    row = next(csv.DictReader(io.StringIO(result.stdout)))
    assert row['converged'] == '1' and float(row['relative_residual']) < 1e-8, row
    return row


cases = [(2, 9), (2, 24), (3, 5), (3, 10)]
for dim, grid in cases:
    for comm in ('baseline', 'overlap'):
        for ranks, threads in ((1, 1), (1, 2), (2, 2), (4, 1)):
            reference = run('cg_mpi', ranks, 1, dim, grid, comm)
            hybrid = run('cg_hybrid', ranks, threads, dim, grid, comm)
            assert int(hybrid['omp_threads']) == threads, hybrid
            assert hybrid['iterations'] == reference['iterations'], (reference, hybrid)
            assert abs(float(hybrid['relative_error']) - float(reference['relative_error'])) < 1e-7, (reference, hybrid)
            print(f'PASS hybrid dim={dim} grid={grid} comm={comm} ranks={ranks} threads={threads}', flush=True)
print('All 32 hybrid comparisons passed.')
