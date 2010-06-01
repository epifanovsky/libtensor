#ifndef LIBTENSOR_BTOD_TRIDIAGONALIZE_H
#define LIBTENSOR_BTOD_TRIDIAGONALIZE_H

#include "../core/block_tensor_i.h"

namespace libtensor {
/**	\brief Converts a symmetric matrix to the tridiagonal matrix using
 * 	Householder's reflections

	\ingroup libtensor_btod
 **/
class btod_tridiagonalize {
public:
	btod_tridiagonalize(block_tensor_i<2, double> &bta);
	//!< bta - input symmetric matrix
	virtual void perform(block_tensor_i<2, double> &btb,block_tensor_i<2, double> &S);
	//!< btb - output tridiag matrix,S - matrix of transformation
	virtual void print(block_tensor_i<2, double> &btb);
	//!< (Optional) prints the matrix
private:
	block_tensor_i<2, double> &m_bta; //!< Input block %tensor
};

}

#endif // LIBTENSOR_BTOD_TRIDIAGONALIZE_H
