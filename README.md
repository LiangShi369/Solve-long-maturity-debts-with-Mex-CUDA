# Solve-long-maturity-debts-with-Mex-CUDA

This folder includes functions to implement the solving of sovereign default model wit long-maturity debts (Chartterjee and and Eyigungor, 2012, AER). 
To efficiently use Mex CUDA, we should avoid the "stack memory" problem, which can be identified by the reported issued with Matlab GPU coder "cfg.GenerateReport = true; "
The use of W(i,ib,iy) is to address the stack memory issue. 
Tip: 
Wrtie loops that explicitly visit all elements in the matrices. 

File long_seq_loasMex.m is the main script. 
Function solver_loasMex.m executes Mex parfor. 
