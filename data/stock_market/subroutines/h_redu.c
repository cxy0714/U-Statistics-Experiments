#include <mex.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int i,j;

	double *Idx_mat = mxGetPr(prhs[0]);
	const mwSize *dim_a = mxGetDimensions(prhs[0]);
	int N = dim_a[0];
	int r = dim_a[1];
	
	double *h_vec = mxGetPr(prhs[1]);
	int n = mxGetScalar(prhs[2]);
	
	plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
	double *a_vec = mxGetPr(plhs[0]);
	
	
	for(i=0; i<n; i++){
		a_vec[i] = 0;
	}
	
    for(i=0; i<N; i++){
        for(j=0; j<r; j++){
			a_vec[(int)Idx_mat[i+j*N]-1] += h_vec[i];
		}
    }

	plhs[1] = mxCreateDoubleMatrix(n, 1, mxREAL);
	double *b_vec = mxGetPr(plhs[1]);
	
	
	for(i=0; i<n; i++){
		b_vec[i] = 0;
	}
	
    for(i=0; i<N; i++){
        for(j=0; j<r; j++){
			b_vec[(int)Idx_mat[i+j*N]-1] += 1;
		}
    }


}