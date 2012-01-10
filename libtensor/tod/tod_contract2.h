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
#include "kernels/loop_list_node.h"


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
class tod_contract2 :
	public tod_additive<N + M>,
	public timings< tod_contract2<N, M, K> > {

public:
	static const char *k_clazz;

private:
	class loop_list_adapter {
	private:
		typedef std::list< loop_list_node<2, 1> > list_t;
		list_t &m_list;

	public:
		loop_list_adapter(list_t &list) : m_list(list) { }
		void append(size_t weight, size_t inca, size_t incb,
			size_t incc) {
			typedef typename list_t::iterator iterator_t;
			typedef loop_list_node<2, 1> node_t;
			iterator_t inode = m_list.insert(m_list.end(), node_t(weight));
			inode->stepa(0) = inca;
			inode->stepa(1) = incb;
			inode->stepb(0) = incc;
		}
	};

public:
	enum {
	    k_ordera = N + K, //!< Order of first argument (A)
	    k_orderb = M + K, //!< Order of second argument (B)
	    k_orderc = N + M //!< Order of result (C)
	};

private:
	contraction2<N, M, K> m_contr; //!< Contraction
	dense_tensor_i<k_ordera, double> &m_ta; //!< First tensor (a)
	dense_tensor_i<k_orderb, double> &m_tb; //!< Second tensor (b)

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the contraction operation

		\param contr Contraction.
		\param ta Tensor a (first argument).
		\param tb Tensor b (second argument).
	 **/
	tod_contract2(const contraction2<N, M, K> &contr,
		dense_tensor_i<k_ordera, double> &ta, dense_tensor_i<k_orderb, double> &tb);


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
    virtual void perform(cpu_pool &cpus, bool zero, double c,
        dense_tensor_i<k_orderc, double> &t);
	//@}

private:
	void do_perform(dense_tensor_i<k_orderc, double> &tc, bool zero, double d);

};


} // namespace libtensor


#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

    extern template class tod_contract2<0, 0, 1>;
    extern template class tod_contract2<0, 0, 2>;
    extern template class tod_contract2<0, 0, 3>;
    extern template class tod_contract2<0, 0, 4>;
    extern template class tod_contract2<0, 0, 5>;
    extern template class tod_contract2<0, 0, 6>;

    extern template class tod_contract2<0, 1, 1>;
    extern template class tod_contract2<0, 1, 2>;
    extern template class tod_contract2<0, 1, 3>;
    extern template class tod_contract2<0, 1, 4>;
    extern template class tod_contract2<0, 1, 5>;
    extern template class tod_contract2<1, 0, 1>;
    extern template class tod_contract2<1, 0, 2>;
    extern template class tod_contract2<1, 0, 3>;
    extern template class tod_contract2<1, 0, 4>;
    extern template class tod_contract2<1, 0, 5>;

    extern template class tod_contract2<0, 2, 1>;
    extern template class tod_contract2<0, 2, 2>;
    extern template class tod_contract2<0, 2, 3>;
    extern template class tod_contract2<0, 2, 4>;
    extern template class tod_contract2<1, 1, 0>;
    extern template class tod_contract2<1, 1, 1>;
    extern template class tod_contract2<1, 1, 2>;
    extern template class tod_contract2<1, 1, 3>;
    extern template class tod_contract2<1, 1, 4>;
    extern template class tod_contract2<1, 1, 5>;
    extern template class tod_contract2<2, 0, 1>;
    extern template class tod_contract2<2, 0, 2>;
    extern template class tod_contract2<2, 0, 3>;
    extern template class tod_contract2<2, 0, 4>;

    extern template class tod_contract2<0, 3, 1>;
    extern template class tod_contract2<0, 3, 2>;
    extern template class tod_contract2<0, 3, 3>;
    extern template class tod_contract2<1, 2, 0>;
    extern template class tod_contract2<1, 2, 1>;
    extern template class tod_contract2<1, 2, 2>;
    extern template class tod_contract2<1, 2, 3>;
    extern template class tod_contract2<1, 2, 4>;
    extern template class tod_contract2<2, 1, 0>;
    extern template class tod_contract2<2, 1, 1>;
    extern template class tod_contract2<2, 1, 2>;
    extern template class tod_contract2<2, 1, 3>;
    extern template class tod_contract2<2, 1, 4>;
    extern template class tod_contract2<3, 0, 1>;
    extern template class tod_contract2<3, 0, 2>;
    extern template class tod_contract2<3, 0, 3>;

    extern template class tod_contract2<0, 4, 1>;
    extern template class tod_contract2<0, 4, 2>;
    extern template class tod_contract2<1, 3, 0>;
    extern template class tod_contract2<1, 3, 1>;
    extern template class tod_contract2<1, 3, 2>;
    extern template class tod_contract2<1, 3, 3>;
    extern template class tod_contract2<2, 2, 0>;
    extern template class tod_contract2<2, 2, 1>;
    extern template class tod_contract2<2, 2, 2>;
    extern template class tod_contract2<2, 2, 3>;
    extern template class tod_contract2<2, 2, 4>;
    extern template class tod_contract2<3, 1, 0>;
    extern template class tod_contract2<3, 1, 1>;
    extern template class tod_contract2<3, 1, 2>;
    extern template class tod_contract2<3, 1, 3>;
    extern template class tod_contract2<4, 0, 1>;
    extern template class tod_contract2<4, 0, 2>;

    extern template class tod_contract2<0, 5, 1>;
    extern template class tod_contract2<1, 4, 0>;
    extern template class tod_contract2<1, 4, 1>;
    extern template class tod_contract2<1, 4, 2>;
    extern template class tod_contract2<2, 3, 0>;
    extern template class tod_contract2<2, 3, 1>;
    extern template class tod_contract2<2, 3, 2>;
    extern template class tod_contract2<2, 3, 3>;
    extern template class tod_contract2<3, 2, 0>;
    extern template class tod_contract2<3, 2, 1>;
    extern template class tod_contract2<3, 2, 2>;
    extern template class tod_contract2<3, 2, 3>;
    extern template class tod_contract2<4, 1, 0>;
    extern template class tod_contract2<4, 1, 1>;
    extern template class tod_contract2<4, 1, 2>;
    extern template class tod_contract2<5, 0, 1>;

    extern template class tod_contract2<1, 5, 0>;
    extern template class tod_contract2<1, 5, 1>;
    extern template class tod_contract2<2, 4, 0>;
    extern template class tod_contract2<2, 4, 1>;
    extern template class tod_contract2<2, 4, 2>;
    extern template class tod_contract2<3, 3, 0>;
    extern template class tod_contract2<3, 3, 1>;
    extern template class tod_contract2<3, 3, 2>;
    extern template class tod_contract2<3, 3, 3>;
    extern template class tod_contract2<4, 2, 0>;
    extern template class tod_contract2<4, 2, 1>;
    extern template class tod_contract2<4, 2, 2>;
    extern template class tod_contract2<5, 1, 0>;
    extern template class tod_contract2<5, 1, 1>;

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "tod_contract2_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_TOD_CONTRACT2_H

