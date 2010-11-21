#ifndef LIBTENSOR_BTOD_SYMMETRIZE3_H
#define LIBTENSOR_BTOD_SYMMETRIZE3_H

#include <algorithm>
#include "../exception.h"
#include "../timings.h"
#include "../core/block_index_subspace_builder.h"
#include "../core/permutation_builder.h"
#include "../symmetry/so_concat.h"
#include "../symmetry/so_copy.h"
#include "../symmetry/so_proj_down.h"
#include "../symmetry/so_symmetrize.h"
#include "additive_btod.h"

namespace libtensor {


/**	\brief (Anti-)symmetrizes the result of a block %tensor operation
		over three groups of indexes
	\tparam N Tensor order.

	The operation symmetrizes or anti-symmetrizes the result of another
	block %tensor operation over three indexes or groups of indexes.

	\f[
		b_{ijk} = P_{\pm} a_{ijk} = a_{ijk} \pm a_{jik} \pm a_{kji} \pm
			a_{ikj} + a_{jki} + a_{kij}
	\f]

	The constructor takes three different indexes to be symmetrized.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_symmetrize3 :
	public additive_btod<N>,
	public timings< btod_symmetrize3<N> > {

public:
	static const char *k_clazz; //!< Class name

private:
	struct schrec {
		size_t ai;
		transf<N, double> tr;
		schrec() : ai(0) { }
		schrec(size_t ai_, const transf<N, double> &tr_) :
			ai(ai_), tr(tr_) { }
	};
	typedef std::pair<size_t, schrec> sym_schedule_pair_t;
	typedef std::multimap<size_t, schrec> sym_schedule_t;

private:
	additive_btod<N> &m_op; //!< Symmetrized operation
	size_t m_i1; //!< First %index
	size_t m_i2; //!< Second %index
	size_t m_i3; //!< Third %index
	bool m_symm; //!< Symmetrization/anti-symmetrization
	symmetry<N, double> m_sym; //!< Symmetry of the result
	assignment_schedule<N, double> m_sch; //!< Schedule

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the operation
		\param op Operation to be symmetrized.
		\param i1 First %index.
		\param i2 Second %index.
		\param i3 Third %index.
		\param symm True for symmetrization, false for
			anti-symmetrization.
	 **/
	btod_symmetrize3(additive_btod<N> &op, size_t i1, size_t i2, size_t i3,
		bool symm);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_symmetrize3() { }

	//@}


	//!	\name Implementation of direct_block_tensor_operation<N, double>
	//@{

	virtual const block_index_space<N> &get_bis() const {
		return m_op.get_bis();
	}

	virtual const symmetry<N, double> &get_symmetry() const {
		return m_sym;
	}

	virtual const assignment_schedule<N, double> &get_schedule() const {
		return m_sch;
	}

	virtual void sync_on() {
		m_op.sync_on();
	}

	virtual void sync_off() {
		m_op.sync_off();
	}

	//@}

protected:
	//!	\brief Implementation of additive_btod<N>
	//@{

	virtual void compute_block(tensor_i<N, double> &blk, const index<N> &i);

	virtual void compute_block(tensor_i<N, double> &blk, const index<N> &i,
		const transf<N, double> &tr, double c);

	//@}

private:
	void make_symmetry();
	void make_schedule();
	void make_schedule_blk(const abs_index<N> &ai,
		sym_schedule_t &sch) const;

private:
	btod_symmetrize3(const btod_symmetrize3<N>&);
	const btod_symmetrize3<N> &operator=(const btod_symmetrize3<N>&);

};


template<size_t N>
const char *btod_symmetrize3<N>::k_clazz = "btod_symmetrize3<N>";


template<size_t N>
btod_symmetrize3<N>::btod_symmetrize3(additive_btod<N> &op, size_t i1,
	size_t i2, size_t i3, bool symm) :

	m_op(op), m_i1(i1), m_i2(i2), m_i3(i3), m_symm(symm),
	m_sym(op.get_bis()),
	m_sch(op.get_bis().get_block_index_dims()) {

	static const char *method = "btod_symmetrize3(additive_btod<N>&, "
		"size_t, size_t, size_t, bool)";

	if(i1 == i2 || i2 == i3 || i1 == i3) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"i1,i2,i3");
	}

	if(m_i1 > m_i2) std::swap(m_i1, m_i2);
	if(m_i2 > m_i3) std::swap(m_i2, m_i3);
	if(m_i1 > m_i2) std::swap(m_i1, m_i2);

	make_symmetry();
	make_schedule();
}


template<size_t N>
void btod_symmetrize3<N>::compute_block(tensor_i<N, double> &blk,
	const index<N> &i) {

	typedef typename sym_schedule_t::iterator iterator_t;

	dimensions<N> bidims(m_op.get_bis().get_block_index_dims());
	abs_index<N> ai(i, bidims);

	sym_schedule_t sch;
	make_schedule_blk(ai, sch);

	tod_set<N>().perform(blk);

	std::pair<iterator_t, iterator_t> jr =
		sch.equal_range(ai.get_abs_index());
	for(iterator_t j = jr.first; j != jr.second; j++) {

		abs_index<N> aj(j->second.ai, bidims);
		transf<N, double> trj(j->second.tr);
		additive_btod<N>::compute_block(m_op, blk, aj.get_index(),
			j->second.tr, 1.0);
	}
}


template<size_t N>
void btod_symmetrize3<N>::compute_block(tensor_i<N, double> &blk,
	const index<N> &i, const transf<N, double> &tr, double c) {

	typedef typename sym_schedule_t::iterator iterator_t;

	dimensions<N> bidims(m_op.get_bis().get_block_index_dims());
	abs_index<N> ai(i, bidims);

	sym_schedule_t sch;
	make_schedule_blk(ai, sch);

	std::pair<iterator_t, iterator_t> jr =
		sch.equal_range(ai.get_abs_index());
	for(iterator_t j = jr.first; j != jr.second; j++) {

		abs_index<N> aj(j->second.ai, bidims);
		transf<N, double> trj(j->second.tr);
		trj.transform(tr);
		additive_btod<N>::compute_block(m_op, blk, aj.get_index(),
			trj, c);
	}
}


template<size_t N>
void btod_symmetrize3<N>::make_symmetry() {

	//	1. Separate symmetrized dimensions from all other dimensions.
	//	2. Add perm symmetry to the symmetrized dimensions.
	//	3. Concatenate back into the full space.

	mask<N> m1, m2;
	for(size_t i = 0; i < N; i++) m1[i] = true;
	m1[m_i1] = false; m1[m_i2] = false; m1[m_i3] = false;
	m2[m_i1] = true; m2[m_i2] = true; m2[m_i3] = true;

	block_index_subspace_builder<N - 3, 3> bisbldr1(m_op.get_bis(), m1);
	block_index_subspace_builder<3, N - 3> bisbldr2(m_op.get_bis(), m2);

	symmetry<N - 3, double> sym1(bisbldr1.get_bis());
	symmetry<3, double> sym2(bisbldr2.get_bis()),
		sym2tmp(bisbldr2.get_bis());

	so_proj_down<N, 3, double>(m_op.get_symmetry(), m1).perform(sym1);
	so_proj_down<N, N - 3, double>(m_op.get_symmetry(), m2).perform(sym2);

	so_symmetrize<3, double>(sym2, permutation<3>().permute(0, 1), m_symm).
		perform(sym2tmp);
	so_symmetrize<3, double>(sym2tmp, permutation<3>().permute(0, 2),
		m_symm).perform(sym2);

	sequence<N, size_t> seq1(0), seq2(0);
	for(size_t i = 0; i < N; i++) seq1[i] = i;
	for(size_t i = 0, j = 0; i < N; i++) {
		if(i == m_i1) seq2[i] = N - 3;
		else if(i == m_i2) seq2[i] = N - 2;
		else if(i == m_i3) seq2[i] = N - 1;
		else seq2[i] = j++;
	}
	permutation_builder<N> pb(seq2, seq1);

	so_concat<N - 3, 3, double>(sym1, sym2, pb.get_perm()).perform(m_sym);
}


template<size_t N>
void btod_symmetrize3<N>::make_schedule() {

	btod_symmetrize3<N>::start_timer("make_schedule");

	dimensions<N> bidims(m_op.get_bis().get_block_index_dims());
	orbit_list<N, double> ol(m_sym);

	for(typename orbit_list<N, double>::iterator io = ol.begin();
		io != ol.end(); io++) {

		abs_index<N> ai(ol.get_index(io), bidims);
		sym_schedule_t sch;
		make_schedule_blk(ai, sch);
		if(!sch.empty()) m_sch.insert(ai.get_abs_index());
	}

	btod_symmetrize3<N>::stop_timer("make_schedule");
}


template<size_t N>
void btod_symmetrize3<N>::make_schedule_blk(const abs_index<N> &ai,
	sym_schedule_t &sch) const {

	permutation<N> perm1, perm2, perm3;
	perm1.permute(m_i1, m_i2);
	perm2.permute(m_i1, m_i3);
	perm3.permute(m_i2, m_i3);
	double scal = m_symm ? 1.0 : -1.0;

	index<N> idx0(ai.get_index()), idx1(idx0), idx2(idx0), idx3(idx0),
		idx4(idx0), idx5(idx0);
	idx1.permute(perm1);
	idx2.permute(perm2);
	idx3.permute(perm3);
	idx4.permute(perm1).permute(perm2);
	idx5.permute(perm1).permute(perm3);

	const symmetry<N, double> &sym0 = m_op.get_symmetry();
	const assignment_schedule<N, double> &sch0 = m_op.get_schedule();

	orbit<N, double> o0(sym0, idx0), o1(sym0, idx1), o2(sym0, idx2),
		o3(sym0, idx3), o4(sym0, idx4), o5(sym0, idx5);

	if(sch0.contains(o0.get_abs_canonical_index())) {
		transf<N, double> tr(o0.get_transf(idx0));
		sch.insert(sym_schedule_pair_t(ai.get_abs_index(),
			schrec(o0.get_abs_canonical_index(), tr)));
	}
	if(sch0.contains(o1.get_abs_canonical_index())) {
		transf<N, double> tr(o1.get_transf(idx1));
		tr.permute(perm1);
		tr.scale(scal);
		sch.insert(sym_schedule_pair_t(ai.get_abs_index(),
			schrec(o1.get_abs_canonical_index(), tr)));
	}
	if(sch0.contains(o2.get_abs_canonical_index())) {
		transf<N, double> tr(o2.get_transf(idx2));
		tr.permute(perm2);
		tr.scale(scal);
		sch.insert(sym_schedule_pair_t(ai.get_abs_index(),
			schrec(o2.get_abs_canonical_index(), tr)));
	}
	if(sch0.contains(o3.get_abs_canonical_index())) {
		transf<N, double> tr(o3.get_transf(idx3));
		tr.permute(perm3);
		tr.scale(scal);
		sch.insert(sym_schedule_pair_t(ai.get_abs_index(),
			schrec(o3.get_abs_canonical_index(), tr)));
	}
	if(sch0.contains(o4.get_abs_canonical_index())) {
		transf<N, double> tr(o4.get_transf(idx4));
		tr.permute(perm1);
		tr.permute(perm3);
		sch.insert(sym_schedule_pair_t(ai.get_abs_index(),
			schrec(o4.get_abs_canonical_index(), tr)));
	}
	if(sch0.contains(o5.get_abs_canonical_index())) {
		transf<N, double> tr(o5.get_transf(idx5));
		tr.permute(perm1);
		tr.permute(perm2);
		sch.insert(sym_schedule_pair_t(ai.get_abs_index(),
			schrec(o5.get_abs_canonical_index(), tr)));
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMMETRIZE3_H
