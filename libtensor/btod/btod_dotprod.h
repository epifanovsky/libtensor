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
#include "../symmetry/so_add.h"
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

	try {

	dimensions<N> bidims(m_bis.get_block_index_dims());

	std::vector< block_tensor_ctrl<N, double>* > ctrl1(narg), ctrl2(narg);
	std::vector< symmetry<N, double>* > sym(narg);
	std::vector< tod_dotprod<N>* > tod(narg, (tod_dotprod<N>*)0);

	typename std::list<arg>::const_iterator j;

	for(i = 0, j = m_args.begin(); i < narg; i++, j++) {
		v[i] = 0.0;
		ctrl1[i] = new block_tensor_ctrl<N, double>(j->bt1);
		ctrl2[i] = new block_tensor_ctrl<N, double>(j->bt2);
		sym[i] = new symmetry<N, double>(block_index_space<N>(
			j->bt1.get_bis()).permute(j->perm1));
		so_add<N, double>(ctrl1[i]->req_const_symmetry(), j->perm1,
			ctrl2[i]->req_const_symmetry(), j->perm2).
			perform(*sym[i]);
	}

	for(i = 0, j = m_args.begin(); i < narg; i++, j++) {

		orbit_list<N, double> ol1(ctrl1[i]->req_const_symmetry());
		orbit_list<N, double> ol2(ctrl2[i]->req_const_symmetry());
		orbit_list<N, double> ol(*sym[i]);

		permutation<N> pinv1(j->perm1, true), pinv2(j->perm2, true);

		for(typename orbit_list<N, double>::iterator io = ol.begin();
			io != ol.end(); io++) {

			index<N> i1(ol.get_index(io)), i2(ol.get_index(io));
			i1.permute(pinv1);
			i2.permute(pinv2);

			v[i] += calc_in_orbit(*ctrl1[i], ol1, pinv1, *ctrl2[i],
				ol2, pinv2, *sym[i], bidims, ol.get_index(io));
		}
	}

	for(i = 0; i < narg; i++) {
		delete sym[i];
		delete ctrl1[i];
		delete ctrl2[i];
	}

	} catch(...) {
		btod_dotprod<N>::stop_timer();
		throw;
	}

	btod_dotprod<N>::stop_timer();
}


template<size_t N>
double btod_dotprod<N>::calc_in_orbit(block_tensor_ctrl<N, double> &ctrl1,
	const orbit_list<N, double> &ol1, const permutation<N> &pinv1,
	block_tensor_ctrl<N, double> &ctrl2, const orbit_list<N, double> &ol2,
	const permutation<N> &pinv2, const symmetry<N, double> &sym,
	const dimensions<N> &bidims, const index<N> &idx) {

	orbit<N, double> orb(sym, idx);
	double c = 0.0;
	for(typename orbit<N, double>::iterator io = orb.begin();
		io != orb.end(); io++) c += orb.get_transf(io).get_coeff();

	if(c == 0.0) return 0.0;

	dimensions<N> bidims1(bidims), bidims2(bidims);
	bidims1.permute(pinv1);
	bidims2.permute(pinv2);

	index<N> i1(idx), i2(idx);
	i1.permute(pinv1);
	i2.permute(pinv2);

	orbit<N, double> orb1(ctrl1.req_const_symmetry(), i1),
		orb2(ctrl2.req_const_symmetry(), i2);

	const transf<N, double> &tr1 = orb1.get_transf(i1);
	const transf<N, double> &tr2 = orb2.get_transf(i2);

	abs_index<N> aci1(orb1.get_abs_canonical_index(), bidims1),
		aci2(orb2.get_abs_canonical_index(), bidims2);
	if(ctrl1.req_is_zero_block(aci1.get_index()) ||
		ctrl2.req_is_zero_block(aci2.get_index())) return 0.0;

	tensor_i<N, double> &blk1 = ctrl1.req_block(aci1.get_index());
	tensor_i<N, double> &blk2 = ctrl2.req_block(aci2.get_index());

	permutation<N> perm1, perm2;
	perm1.permute(tr1.get_perm()).permute(permutation<N>(pinv1, true));
	perm2.permute(tr2.get_perm()).permute(permutation<N>(pinv2, true));

	double d = tod_dotprod<N>(blk1, perm1, blk2, perm2).calculate() *
		tr1.get_coeff() * tr2.get_coeff();

	ctrl1.ret_block(aci1.get_index());
	ctrl2.ret_block(aci2.get_index());

	return c * d;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DOTPROD_H
