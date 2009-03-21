#ifndef LIBTENSOR_TOD_CONTRACT2_H
#define LIBTENSOR_TOD_CONTRACT2_H

#include "defs.h"
#include "exception.h"
#include "tod_additive.h"

namespace libtensor {

/**	\brief Contracts two tensors (double)

	This operation contracts %tensor T1 permuted as P1 with %tensor T2
	permuted as P2 over n last indexes. The result is permuted as Pres
	and written or added to the resulting %tensor.

	Although it is convenient to define a contraction through permutations,
	it is not the most efficient way of calculating it. This class seeks
	to use algorithms tailored for different tensors to get the best
	performance. For more information, read the wiki section on %tensor
	contractions.

	\ingroup libtensor_tod
**/
template<size_t NC, size_t NA, size_t NB>
class tod_contract2 : public tod_additive<NC> {
private:
	size_t m_ncontr; //!< Number of indexes to contract over
	tensor_i<NA,double> &m_t1; //!< First tensor
	tensor_i<NB,double> &m_t2; //!< Second tensor
	permutation<NA> m_p1; //!< Permutation of the first tensor
	permutation<NB> m_p2; //!< Permutation of the second tensor

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the operation
	**/
	tod_contract2(const size_t n, tensor_i<NA,double> &t1,
		const permutation<NA> &p1, tensor_i<NB,double> &t2,
		const permutation<NB> &p2, const permutation<NC> &pres)
		throw(exception);

	/**	\brief Virtual destructor
	**/
	virtual ~tod_contract2();

	//@}

	//!	\name Implementation of direct_tensor_operation<T>
	//@{
	virtual void prefetch() throw(exception);
	//@}

	//!	\name Implementation of tod_additive
	//@{
	virtual void perform(tensor_i<NC,double> &t) throw(exception);
	virtual void perform(tensor_i<NC,double> &t, double c) throw(exception);
	//@}
};

template<size_t NC, size_t NA, size_t NB>
tod_contract2<NC,NA,NB>::tod_contract2(const size_t n, tensor_i<NA,double> &t1,
	const permutation<NA> &p1, tensor_i<NB,double> &t2,
	const permutation<NB> &p2, const permutation<NC> &pres)
	throw(exception) : m_ncontr(n), m_t1(t1), m_t2(t2), m_p1(p1), m_p2(p2) {
}

template<size_t NC, size_t NA, size_t NB>
tod_contract2<NC,NA,NB>::~tod_contract2() {
}

template<size_t NC, size_t NA, size_t NB>
void tod_contract2<NC,NA,NB>::prefetch() throw(exception) {
	tensor_ctrl<NA,double> ctrl_t1(m_t1);
	tensor_ctrl<NB,double> ctrl_t2(m_t2);
	ctrl_t1.req_prefetch();
	ctrl_t2.req_prefetch();
}

template<size_t NC, size_t NA, size_t NB>
void tod_contract2<NC,NA,NB>::perform(tensor_i<NC,double> &t) throw(exception) {
	dimensions<NA> dims_t1(m_t1.get_dims());
	dimensions<NB> dims_t2(m_t2.get_dims());
	dims_t1.permute(m_p1);
	dims_t2.permute(m_p2);
}

template<size_t NC, size_t NA, size_t NB>
void tod_contract2<NC,NA,NB>::perform(tensor_i<NC,double> &t, const double c)
	throw(exception) {
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_H

