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
#include "../core/permutation.h"
#include "../tod/tod_dotprod.h"
#include "bad_block_index_space.h"

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
	btod_dotprod(const btod_dotprod<N>&);
	const btod_dotprod<N> &operator=(const btod_dotprod<N>&);

};


template<size_t N>
const char *btod_dotprod<N>::k_clazz = "btod_dotprod<N>";


template<size_t N>
btod_dotprod<N>::btod_dotprod(block_tensor_i<N, double> &bt1,
	block_tensor_i<N, double> &bt2) : m_bis(bt1.get_bis()) {

	add_arg(bt1, bt2);
}


template<size_t N>
btod_dotprod<N>::btod_dotprod(block_tensor_i<N, double> &bt1,
	const permutation<N> &perm1, block_tensor_i<N, double> &bt2,
	const permutation<N> &perm2) : m_bis(bt1.get_bis()) {

	m_bis.permute(perm1);
	add_arg(bt1, perm1, bt2, perm2);
}


template<size_t N>
void btod_dotprod<N>::add_arg(block_tensor_i<N, double> &bt1,
	block_tensor_i<N, double> &bt2) {

	static const char *method = "add_arg(block_tensor_i<N, double>&, "
		"block_tensor_i<N, double>&)";

	if(!m_bis.equals(bt1.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "bt1");
	}
	if(!m_bis.equals(bt2.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "bt2");
	}

	m_args.push_back(arg(bt1, bt2));
}


template<size_t N>
void btod_dotprod<N>::add_arg(block_tensor_i<N, double> &bt1,
	const permutation<N> &perm1, block_tensor_i<N, double> &bt2,
	const permutation<N> &perm2) {

	static const char *method = "add_arg(block_tensor_i<N, double>&, "
		"const permutation<N>&, block_tensor_i<N, double>&, "
		"const permutation<N>&)";

	block_index_space<N> bis1(bt1.get_bis());
	block_index_space<N> bis2(bt2.get_bis());
	bis1.permute(perm1);
	bis2.permute(perm2);
	if(!m_bis.equals(bis1)) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "bt1");
	}
	if(!m_bis.equals(bis2)) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "bt2");
	}

	m_args.push_back(arg(bt1, perm1, bt2, perm2));
}


template<size_t N>
double btod_dotprod<N>::calculate() {

	std::vector<double> v(1);
	calculate(v);
	return v[0];
/*
	btod_dotprod<N>::start_timer();

	// No-symmetry implementation

	block_tensor_ctrl<N, double> ctrl1(m_bt1);
	block_tensor_ctrl<N, double> ctrl2(m_bt2);
	dimensions<N> bidims(m_bt1.get_bis().get_block_index_dims());

	abs_index<N> ai1(bidims);
	double d = 0.0;
	do {
		index<N> i1(ai1.get_index()), i2(ai1.get_index());
		i1.permute(m_perm1);
		i2.permute(m_perm2);

		if(!ctrl1.req_is_zero_block(i1) &&
			!ctrl2.req_is_zero_block(i2)) {

			tensor_i<N, double> &t1(ctrl1.req_block(i1));
			tensor_i<N, double> &t2(ctrl2.req_block(i2));
			d += tod_dotprod<N>(
				t1, m_perm1, t2, m_perm2).calculate();
			ctrl1.ret_block(i1);
			ctrl2.ret_block(i2);
		}
	} while(ai1.inc());

	btod_dotprod<N>::stop_timer();

	return d;
*/
}


template<size_t N>
void btod_dotprod<N>::calculate(std::vector<double> &v) {

	static const char *method = "calculate(std::vector<double>&)";

	size_t narg = m_args.size(), i;

	if(v.size() != narg) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"v");
	}

	btod_dotprod<N>::start_timer();

	dimensions<N> bidims(m_bis.get_block_index_dims());

	std::vector< block_tensor_ctrl<N, double>* > ctrl1(narg), ctrl2(narg);
	std::vector< tod_dotprod<N>* > tod(narg, (tod_dotprod<N>*)0);

	typename std::list<arg>::const_iterator j;

	for(i = 0, j = m_args.begin(); i < narg; i++, j++) {
		v[i] = 0.0;
		ctrl1[i] = new block_tensor_ctrl<N, double>(j->bt1);
		ctrl2[i] = new block_tensor_ctrl<N, double>(j->bt2);
	}

	abs_index<N> ai(bidims);
	do {

		for(i = 0, j = m_args.begin(); i < narg; i++, j++) {

			index<N> i1(ai.get_index()), i2(ai.get_index());
			i1.permute(j->perm1);
			i2.permute(j->perm2);
			
			if(!ctrl1[i]->req_is_zero_block(i1) &&
				!ctrl2[i]->req_is_zero_block(i2)) {

				tensor_i<N, double> &t1 =
					ctrl1[i]->req_block(i1);
				tensor_i<N, double> &t2 =
					ctrl2[i]->req_block(i2);
				tod[i] = new tod_dotprod<N>(t1, j->perm1,
					t2, j->perm2);
				tod[i]->prefetch();

			} else {
				tod[i] = 0;
			}
		}

		for(i = 0, j = m_args.begin(); i < narg; i++, j++) {

			if(tod[i] == 0) continue;

			index<N> i1(ai.get_index()), i2(ai.get_index());
			i1.permute(j->perm1);
			i2.permute(j->perm2);

			v[i] += tod[i]->calculate();
			delete tod[i];
			tod[i] = 0;

			ctrl1[i]->ret_block(i1);
			ctrl2[i]->ret_block(i2);
		}

	} while(ai.inc());

	for(i = 0; i < narg; i++) {
		delete ctrl1[i];
		delete ctrl2[i];
	}

	btod_dotprod<N>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DOTPROD_H
