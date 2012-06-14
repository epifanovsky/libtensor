#ifndef LIBTENSOR_CHOLESKY_AO_H
#define LIBTENSOR_CHOLESKY_AO_H

#include "../core/block_tensor_i.h"

namespace libtensor {

/**	\brief Computes Cholesky decomposition for two electron integrals in AO basis

	\ingroup libtensor_btod
 **/

typedef std::vector< block_tensor<2, double, std_allocator<double> > * > list;

class cholesky_ao
{
public:
	cholesky_ao(block_tensor_i<4, double> &bta, double tol = 1e-4);
	//!< bta - input two-elec integrals
	~cholesky_ao();
	virtual void decompose();
	//!< perform decomposition
	virtual void perform(block_tensor_i<3, double> &btb);
	//!< btb - output matrix of dimensions bbx where x - rank of cholesky decomposition
	double get_tol(){return m_tol;};
	//!< Show the tolerance
	int get_rank() {return m_rank;};
	//!< Get the rank of the decomposed matrix
private:
	block_tensor_i<4, double> &m_bta; //!< Input block two elec integrals
	block_tensor <2, double, std_allocator<double> > * diagonal; //!< Diagonal of two elect integrals minus cholesky vectors
	block_tensor <2, double, std_allocator<double> > * column;//!< Shell pair of two electron integrals
	list * chol_vecs; //!< List of Cholesky vectors
	double m_tol;//!< Tolerance
	int m_rank;//!< Rank of decomposition
	void get_diag();//!< Compute diagonal of two electron integrals
	void extract_column(index<4> &idxbl, index<4> &idxibl);//!< Extract shell pair from two electron integrals
	double sort_diag(index<2> &idxbl, index<2> &idxibl);//!< Find largest diagonal element
	void delete_chol_vecs();//!< Delete cholesky vectors
};

}

#endif // LIBTENSOR_CHOLESKY_AO_H
