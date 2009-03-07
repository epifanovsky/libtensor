#ifndef LIBTENSOR_TOD_CONTRACT2_H
#define LIBTENSOR_TOD_CONTRACT2_H

#include "defs.h"
#include "exception.h"
#include "direct_tensor_additive_operation.h"

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
class tod_contract2 : public direct_tensor_additive_operation<double> {
private:
	size_t m_ncontr; //!< Number of indexes to contract over
	tensor_i<T> &m_t1; //!< First tensor
	tensor_i<T> &m_t2; //!< Second tensor

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the operation
	**/
	tod_contract2(const size_t n, tensor_i<T> &t1, const permutation &p1,
		tensor_i<T> &t2, const permutation &p2,
		const permutation &pres) throw(exception);

	/**	\brief Virtual destructor
	**/
	virtual ~tod_contract2();

	//@}

	//!	\name Implementation of direct_tensor_operation<T>
	//@{
	virtual void prefetch() throw(exception);
	virtual void perform(tensor_i<T> &t) throw(exception);
	//@}

	//!	\name Implementation of direct_tensor_additive_operation<T>
	//@{
	virtual void perform(tensor_i<T> &t, const double c) throw(exception);
	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_H

