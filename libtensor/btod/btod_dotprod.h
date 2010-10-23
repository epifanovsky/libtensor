#ifndef LIBTENSOR_BTOD_DOTPROD_H
#define LIBTENSOR_BTOD_DOTPROD_H

#include <list>
#include <vector>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/abs_index.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../core/permutation.h"

namespace libtensor {


/**	\brief Computes the dot product of two block tensors
	\tparam N Tensor order.

	The dot product of two tensors is defined as the sum of elements of
	the element-wise product:

	\f[ c = \sum_i a_i b_i \f]

	This operation computes the dot product for a series of arguments.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_dotprod : public timings< btod_dotprod<N> > {
public:
	static const char *k_clazz; //!< Class name

private:
	struct arg {
		block_tensor_i<N, double> &bt1;
		block_tensor_i<N, double> &bt2;
		permutation<N> perm1;
		permutation<N> perm2;

		arg(block_tensor_i<N, double> &bt1_,
			block_tensor_i<N, double> &bt2_) :
			bt1(bt1_), bt2(bt2_) { }

		arg(block_tensor_i<N, double> &bt1_,
			const permutation<N> &perm1_,
			block_tensor_i<N, double> &bt2_,
			const permutation<N> &perm2_) :
			bt1(bt1_), bt2(bt2_), perm1(perm1_), perm2(perm2_) { }
	};

private:
	block_index_space<N> m_bis; //!< Block %index space of arguments
	std::list<arg> m_args; //!< Arguments

public:
	/**	\brief Initializes the first argument pair
			(identity permutation)
	 **/
	btod_dotprod(block_tensor_i<N, double> &bt1,
		block_tensor_i<N, double> &bt2);

	/**	\brief Initializes the first argument pair
	 **/
	btod_dotprod(block_tensor_i<N, double> &bt1,
		const permutation<N> &perm1, block_tensor_i<N, double> &bt2,
		const permutation<N> &perm2);

	/**	\brief Adds a pair of arguments (identity permutation)
	 **/
	void add_arg(block_tensor_i<N, double> &bt1,
		block_tensor_i<N, double> &bt2);

	/**	\brief Adds a pair of arguments
	 **/
	void add_arg(block_tensor_i<N, double> &bt1,
		const permutation<N> &perm1, block_tensor_i<N, double> &bt2,
		const permutation<N> &perm2);

	/**	\brief Returns the dot product of the first argument pair
	 **/
	double calculate();

	/**	\brief Computes the dot product for all argument pairs
	 **/
	void calculate(std::vector<double> &v);

private:
	double calc_in_orbit(block_tensor_ctrl<N, double> &ctrl1,
		const orbit_list<N, double> &ol1, const permutation<N> &pinv1,
		block_tensor_ctrl<N, double> &ctrl2,
		const orbit_list<N, double> &ol2, const permutation<N> &pinv2,
		const symmetry<N, double> &sym, const dimensions<N> &bidims,
		const index<N> &idx);

private:
	btod_dotprod(const btod_dotprod<N>&);
	const btod_dotprod<N> &operator=(const btod_dotprod<N>&);

};


} // namespace libtensor


#ifndef LIBTENSOR_INSTANTIATE_TEMPLATES
#include "btod_dotprod_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_BTOD_DOTPROD_H
