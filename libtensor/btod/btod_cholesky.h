#ifndef LIBTENSOR_BTOD_CHOLESKY_H
#define LIBTENSOR_BTOD_CHOLESKY_H

#include "../core/block_tensor_i.h"

namespace libtensor {
/**	\brief Computes a Cholesky decomposition of square matrix using LAPACK routine

	\ingroup libtensor_btod
 **/
class btod_cholesky {
public:
	btod_cholesky(block_tensor_i<2, double> &bta, double tol = 1e-4);
	//!< bta - input symmetric matrix
	virtual void perform(block_tensor_i<2, double> &btb);
	//!< btb - output lower triangular matrix
private:
	block_tensor_i<2, double> &m_bta; //!< Input block %tensor
	double m_tol; //!< tolerance for decomsposition
};

}

#endif // LIBTENSOR_BTOD_CHOLESKY_H
