#include "tod_solve.h"

namespace libtensor {

tod_solve::tod_solve(tensor_i<2,double> &tmatrix, tensor_i<1,double> &vect) : m_tmatrix(tmatrix),
	m_tctrl_matrix(tmatrix), m_tvect(vect), m_tctrl_vect(vect) {
	//check dimensions of matrix and vector
	const dimensions<2> &dim_matrix(m_tmatrix.get_dims());
	const dimensions<1> &dim_vect(m_tvect.get_dims());
	if(dim_matrix[0]!=dim_matrix[1]) {
		throw_exc("tod_solve", "tod_solve(tensor_i<2,double> &, tensor_i<1,double> &)",
					"Matrix is not square");
	} else if(dim_matrix[0]!=dim_vect[0]) {
		throw_exc("tod_solve", "tod_solve(tensor_i<2,double> &, tensor_i<1,double> &)",
							"Matrix and input vector have different size");
	}

}

tod_solve::~tod_solve() {
}


void tod_solve::perform(tensor_i<1,double> &result_vect) throw(exception) {

	// check dimensions of vectors
	const dimensions<2> &dim_matrix(m_tmatrix.get_dims());
	const dimensions<1> &dim_vect(m_tvect.get_dims());
	const dimensions<1> &dim_rvect(result_vect.get_dims());
	if(dim_vect[0]!=dim_rvect[0]) {
		throw_exc("tod_solve", "perform(tensor_i<1,double> &)",
							"Result vector has wrong size");
	}

	tensor_ctrl<1,double> tctrl_rvect(result_vect);
	double *rvect = tctrl_rvect.req_dataptr();
	const double *matrix = m_tctrl_matrix.req_const_dataptr();
	const double *vect = m_tctrl_vect.req_const_dataptr();


	//A*X=B
	int *ipiv = new int[dim_matrix[0]]; //The pivot indices that define the permutation matrix P
	int lwork=dim_matrix[0]*dim_matrix[0];
	memset(ipiv, 0, sizeof(int)*dim_matrix[0]);//???????

	double *work = new double[lwork];
	memset(work, 0, sizeof(double)*lwork);

	float *swork = new float[lwork];
	memset(work, 0, sizeof(float)*lwork);

	int info, iter;
	int nrhs=1; //The number of right-hand sides, that is, the number of columns of the matrix B

	int matrix_size = dim_vect[0]; //number of linear equations = vector size


//	dsgesv_(integer *, integer *, doublereal *,
//		     integer *, integer *, doublereal *, integer *, doublereal *,
//		    integer *, doublereal *, real *, integer *, integer *);
	dsgesv_(&matrix_size, &nrhs, (double*)matrix, &matrix_size, ipiv, (double*)vect, &matrix_size,
			rvect, &matrix_size, work, swork, &iter, &info);

	delete [] work;
	delete [] ipiv;

	m_tctrl_matrix.ret_dataptr(matrix); matrix = NULL;
	m_tctrl_vect.ret_dataptr(vect); vect = NULL;
	tctrl_rvect.ret_dataptr(rvect); rvect = NULL;
}

} // namespace libtensor
