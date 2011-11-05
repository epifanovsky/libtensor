#ifndef LIBTENSOR_TOD_DIRSUM_H
#define LIBTENSOR_TOD_DIRSUM_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/tensor_i.h"
#include "kernels/loop_list_node.h"
#include "bad_dimensions.h"

namespace libtensor {


/**	\brief Computes the direct sum of two tensors
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.

	Given two tensors \f$ a_{ij\cdots} \f$ and \f$ b_{mn\cdots} \f$,
	the operation computes
	\f$ c_{ij\cdots mn\cdots} = k_a a_{ij\cdots} + k_b b_{mn\cdots} \f$.

	The order of %tensor indexes in the result can be specified using
	a permutation.

	\ingroup libtensor_tod
 **/
template<size_t N, size_t M>
class tod_dirsum : public timings< tod_dirsum<N, M> > {
public:
	static const char *k_clazz; //!< Class name

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
			iterator_t inode = m_list.insert(m_list.end(),
				node_t(weight));
			inode->stepa(0) = inca;
			inode->stepa(1) = incb;
			inode->stepb(0) = incc;
		}
	};

public:
	enum {
	    k_ordera = N, //!< Order of first argument (A)
	    k_orderb = M, //!< Order of second argument (B)
	    k_orderc = N + M //!< Order of result (C)
	};

private:
	tensor_i<k_ordera, double> &m_ta; //!< First %tensor (A)
	tensor_i<k_orderb, double> &m_tb; //!< Second %tensor (B)
	double m_ka; //!< Coefficient A
	double m_kb; //!< Coefficient B
	permutation<k_orderc> m_permc; //!< Permutation of the result
	dimensions<k_orderc> m_dimsc; //!< Dimensions of the result

public:
	/**	\brief Initializes the operation
	 **/
	tod_dirsum(tensor_i<k_ordera, double> &ta, double ka,
		tensor_i<k_orderb, double> &tb, double kb);

	/**	\brief Initializes the operation
	 **/
	tod_dirsum(tensor_i<k_ordera, double> &ta, double ka,
		tensor_i<k_orderb, double> &tb, double kb,
		const permutation<k_orderc> &permc);

	/**	\brief Performs the operation
	 **/
	void perform(tensor_i<k_orderc, double> &tc);

	/**	\brief Performs the operation (additive)
	 **/
	void perform(tensor_i<k_orderc, double> &tc, double kc);

private:
	static dimensions<N + M> mk_dimsc(tensor_i<k_ordera, double> &ta,
		tensor_i<k_orderb, double> &tb);
	void do_perform(tensor_i<k_orderc, double> &tc, bool zero, double d);

};


} // namespace libtensor


#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

    extern template class tod_dirsum<1, 1>;

    extern template class tod_dirsum<1, 2>;
    extern template class tod_dirsum<2, 1>;

    extern template class tod_dirsum<1, 3>;
    extern template class tod_dirsum<2, 2>;
    extern template class tod_dirsum<3, 1>;

    extern template class tod_dirsum<1, 4>;
    extern template class tod_dirsum<2, 3>;
    extern template class tod_dirsum<3, 2>;
    extern template class tod_dirsum<4, 1>;

    extern template class tod_dirsum<1, 5>;
    extern template class tod_dirsum<2, 4>;
    extern template class tod_dirsum<3, 3>;
    extern template class tod_dirsum<4, 2>;
    extern template class tod_dirsum<5, 1>;

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "tod_dirsum_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENOSR_TOD_DIRSUM_H

