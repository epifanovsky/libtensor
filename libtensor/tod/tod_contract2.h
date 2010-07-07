#ifndef LIBTENSOR_TOD_CONTRACT2_H
#define LIBTENSOR_TOD_CONTRACT2_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../linalg.h"
#include "tod_additive.h"
#include "contraction2.h"
#include "contraction2_list_builder.h"
#include "loop_list_mul.h"


namespace libtensor {

/**	\brief Contracts two tensors (double)

	\tparam N Order of the first %tensor (a) less the contraction degree
	\tparam M Order of the second %tensor (b) less the contraction degree
	\tparam K Contraction degree (the number of indexes over which the
		tensors are contracted)

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
template<size_t N, size_t M, size_t K>
class tod_contract2 : public loop_list_mul, public tod_additive<N + M> {

public:
	static const char *k_clazz;

private:
	class loop_list_adapter {
	private:
		typename loop_list_mul::list_t &m_list;

	public:
		loop_list_adapter(typename loop_list_mul::list_t &list) :
			m_list(list) { }
		void append(size_t weight, size_t inca, size_t incb,
			size_t incc) {
			typedef typename loop_list_mul::iterator_t iterator_t;
			typedef typename loop_list_mul::node node_t;
			iterator_t inode = m_list.insert(m_list.end(), node_t(weight));
			inode->stepa(0) = inca;
			inode->stepa(1) = incb;
			inode->stepb(0) = incc;
		}
	};

public:
	static const size_t k_ordera = N + K;
	static const size_t k_orderb = M + K;
	static const size_t k_orderc = N + M;

private:
	contraction2<N, M, K> m_contr; //!< Contraction
	tensor_i<k_ordera, double> &m_ta; //!< First tensor (a)
	tensor_i<k_orderb, double> &m_tb; //!< Second tensor (b)

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the contraction operation

		\param contr Contraction.
		\param ta Tensor a (first argument).
		\param tb Tensor b (second argument).
	 **/
	tod_contract2(const contraction2<N, M, K> &contr,
		tensor_i<k_ordera, double> &ta, tensor_i<k_orderb, double> &tb);


	/**	\brief Virtual destructor
	 **/
	virtual ~tod_contract2();

	//@}

	//!	\name Implementation of direct_tensor_operation<T>
	//@{
	virtual void prefetch();
	//@}

	//!	\name Implementation of tod_additive<N+M>
	//@{
	virtual void perform(tensor_i<k_orderc, double> &tc);
	virtual void perform(tensor_i<k_orderc, double> &tc, double d);
	//@}

private:
	void do_perform(tensor_i<k_orderc, double> &tc, bool zero, double d);

};


} // namespace libtensor


#ifndef LIBTENSOR_INSTANTIATE_TEMPLATES
#include "tod_contract2_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_TOD_CONTRACT2_H

