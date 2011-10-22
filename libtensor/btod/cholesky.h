#ifndef LIBTENSOR_CHOLESKY_H
#define LIBTENSOR_CHOLESKY_H

#include "../core/block_tensor_i.h"

namespace libtensor {

/**	\brief Computes Cholesky decomposition

	\ingroup libtensor_btod
 **/

class cholesky
{
public:
	cholesky(block_tensor_i<2, double> &bta, double tol = 1e-4, int maxiter = 100000);
	//!< bta - input matrix.
	~cholesky();
	virtual void decompose();
	//!< perform decomposition
	virtual void perform(block_tensor_i<2, double> &btb);
	//!< btb - output matrix. put data to the output matrix
	double get_tol(){return m_tol;};
	//!< Show the tolerance
	int get_maxiter(){return m_maxiter;};
	//!< Show the number of max iterations
	int get_iter(){return m_iter;};
	//!< Show the number of iterations have been done
	bool get_doneiter(){return doneiter;};
	//!< Check if iterations are converged
	int get_rank() {return m_rank;};
private:
	block_tensor_i<2, double> &m_bta; //!< Input block %tensor
	double m_tol;//!< Tolerance
	int m_maxiter;//!< Maximum number of iterations
	int m_iter;//!< Number of iterations have been done
	bool doneiter;//!< iterations are converged?
	double m_rank;
	block_tensor <2, double, std_allocator<double> > * pta;//!< buffer
};

}

#endif // LIBTENSOR_CHOLESKY_H
