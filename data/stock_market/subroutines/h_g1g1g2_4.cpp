#include <iostream>
#include <armadillo>
#include <math.h>
#include "armaMex.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int i,j;
	arma::mat Idx_mat = armaGetPr(prhs[0]);
	int N = Idx_mat.n_rows;
	int r = Idx_mat.n_cols;
	
	double *h_vec = mxGetPr(prhs[1]);
	int n = mxGetScalar(prhs[2]);
	
	mat a_vec = mat(n,n,fill::zeros);
	for(i=0; i<N; i++){
			int x = Idx_mat(i,0)-1;
			int y = Idx_mat(i,1)-1;
			int z = Idx_mat(i,2)-1;	
			int w = Idx_mat(i,3)-1;				
			a_vec(x,y) += h_vec[i];
			a_vec(y,x) += h_vec[i];
			a_vec(x,z) += h_vec[i];
			a_vec(z,x) += h_vec[i];
			a_vec(y,z) += h_vec[i];
			a_vec(z,y) += h_vec[i];	
			a_vec(x,w) += h_vec[i];
			a_vec(y,w) += h_vec[i];
			a_vec(z,w) += h_vec[i];	
			a_vec(w,x) += h_vec[i];
			a_vec(w,y) += h_vec[i];
			a_vec(w,z) += h_vec[i];			
	}
	
	plhs[0] = mxCreateDoubleMatrix(a_vec.n_rows, a_vec.n_cols, mxREAL);
  
  // Return the cube C as plhs[0] in Matlab/Octave
    armaSetPr(plhs[0], a_vec);
	
	mat b_vec = mat(n,n,fill::zeros);
	for(i=0; i<N; i++){
			int x = Idx_mat(i,0)-1;
			int y = Idx_mat(i,1)-1;
			int z = Idx_mat(i,2)-1;	
			int w = Idx_mat(i,3)-1;	
			b_vec(x,y) += 1;
			b_vec(y,x) += 1;
			b_vec(x,z) += 1;
			b_vec(z,x) += 1;
			b_vec(y,z) += 1;
			b_vec(z,y) += 1;
			b_vec(x,w) += 1;
			b_vec(y,w) += 1;
			b_vec(z,w) += 1;	
			b_vec(w,x) += 1;
			b_vec(w,y) += 1;
			b_vec(w,z) += 1;				
	}
	
	plhs[1] = mxCreateDoubleMatrix(b_vec.n_rows, b_vec.n_cols, mxREAL);
  
  // Return the cube C as plhs[0] in Matlab/Octave
    armaSetPr(plhs[1], b_vec);
	


}