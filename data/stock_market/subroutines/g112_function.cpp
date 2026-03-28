#include <iostream>
#include <armadillo>
#include <math.h>
#include "armaMex.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int k,l;
	arma::vec al_weight = armaGetPr(prhs[0]);
	arma::vec al_number = armaGetPr(prhs[1]);
	arma::mat g2_up = armaGetPr(prhs[2]);
	arma::mat weight = armaGetPr(prhs[3]);
	double mu_hat = mxGetScalar(prhs[4]);
	int n = g2_up.n_cols;
	
	mat g112 = mat(n,n,fill::zeros);
	mat al_all = mat(n,n,fill::zeros);
	for(k=0; k<n; k++){
		for(l = 0; l<n; l++){
			if(k!=l){
				double g11 = al_weight(k) - mu_hat;
				double g12 = al_weight(l) - mu_hat;
				al_all(k,l) = weight(k,l)*al_number(k)*al_number(l);
				g112(k,l) = (g2_up(k,l) - mu_hat - g11 - g12)*g11*g12*al_all(k,l);
			}

		}		
	}
	
	double intera = sum(sum(g112))/sum(sum(al_all));
	
	plhs[0] = mxCreateDoubleScalar(intera);
  
  // Return the cube C as plhs[0] in Matlab/Octave

	

}