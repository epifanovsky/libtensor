#ifndef LIBTENSOR_TOD_CONTRACT1_H
#define LIBTENSOR_TOD_CONTRACT1_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../linalg.h"
#include "tod_additive.h"
#include "kernels/loop_list_node.h"


namespace libtensor {

/**	\brief Contracts two tensors (double)

	\tparam N Order of the source %tensor
	\tparam M Order of the result %tensor

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
template<size_t N, size_t M>
class tod_contract1 :
	public loop_list_add,
	public tod_additive<M>,
	public timings< tod_contract1<N, M> > {

public:
	static const char *k_clazz;

private:
	tensor_i<N, double> &m_ta; //!< Source tensor
	mask<N> m_msk; //!< Contraction mask.
	permutation<M> m_perm; //!< Permutation of result tensor
	double m_c; //!< Scaling coefficient
	dimensions<M> m_dimsb; //!< Dimensions of result tensor.

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the contraction operation

		\param contr Contraction.
		\param ta Tensor a (first argument).
		\param tb Tensor b (second argument).
	 **/
	tod_contract1(tensor_i<N, double> &ta, const mask<N> &msk,
			double coeff = 1.0);

	/**	\brief Initializes the contraction operation

		\param contr Contraction.
		\param ta Tensor a (first argument).
		\param tb Tensor b (second argument).
	 **/
	tod_contract1(tensor_i<N, double> &ta, const mask<N> &msk,
			const permutation<N> &perm, double coeff = 1.0);


	/**	\brief Virtual destructor
	 **/
	virtual ~tod_contract1();

	//@}

	//!	\name Implementation of direct_tensor_operation<T>
	//@{
	virtual void prefetch();
	//@}

	//!	\name Implementation of tod_additive<N>
	//@{
	virtual void perform(tensor_i<M, double> &tc);
	virtual void perform(tensor_i<M, double> &tc, double d);
	//@}

private:
	void do_perform(tensor_i<k_orderc, double> &tc, bool zero, double d);

};


} // namespace libtensor


#ifndef LIBTENSOR_INSTANTIATE_TEMPLATES
#include "tod_contract1_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_TOD_CONTRACT1_H

