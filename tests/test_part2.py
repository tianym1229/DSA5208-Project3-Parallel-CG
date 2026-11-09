import csv,io,subprocess
for dim,grid in [(2,9),(3,5),(2,24),(3,10)]:
 for ranks in (1,2,4):
  rows=[]
  for comm in ('baseline','overlap'):
   cmd=['mpirun','-np',str(ranks),'./cg_mpi','--dim',str(dim),'--grid',str(grid),'--solution','benchmark','--comm',comm]
   out=subprocess.check_output(cmd,text=True,stderr=subprocess.DEVNULL)
   row=next(csv.DictReader(io.StringIO(out)))
   assert row['converged']=='1' and float(row['relative_residual'])<1e-8,(cmd,row)
   rows.append(row)
  assert abs(float(rows[0]['relative_error'])-float(rows[1]['relative_error']))<1e-10,rows
  assert rows[0]['iterations']==rows[1]['iterations'],rows
  print('PASS communication',dim,grid,ranks)
for comm in ('baseline','overlap'):
 cmd=['mpirun','-np','2','./cg_mpi','--dim','2','--grid','24','--solution','benchmark','--comm',comm,'--fixed-iters','10']
 result=subprocess.run(cmd,text=True,capture_output=True)
 assert result.returncode in (0,2),result.stderr
 row=next(csv.DictReader(io.StringIO(result.stdout)))
 assert row['iterations']== '10' and row['fixed_iters']=='10',row
 print('PASS fixed iterations',comm)
print('All 26 Part 2 tests passed.')
