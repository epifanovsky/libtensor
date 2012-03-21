#ifndef LIBTENSOR_BTOD_SYMMETRIZE_H
#define LIBTENSOR_BTOD_SYMMETRIZE_H

#include <list>
#include <map>
#include "../timings.h"
#include "../core/allocator.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../symmetry/so_permute.h"
#include "../symmetry/so_symmetrize.h"
#include "../tod/tod_set.h"
#include "additive_btod.h"
#include "transf_double.h"

namespace libtensor {


/**	\brief Symmetrizes the result of another block %tensor operation
	\tparam N Tensor order.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_symmetrize :
	public additive_btod<N>,
	public timings< btod_symmetrize<N> > {

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
	permutation<N> m_perm1; //!< First symmetrization permutation
	bool m_symm; //!< Symmetrization sign
	block_index_space<N> m_bis; //!< Block %index space of the result
	symmetry<N, double> m_sym; //!< Symmetry of the result
	assignment_schedule<N, double> m_sch; //!< Schedule
	sym_schedule_t m_sym_sch; //!< Symmetrization schedule

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the operation to symmetrize two indexes
		\param op Symmetrized operation.
		\param i1 First %tensor %index.
		\param i2 Second %tensor %index.
		\param symm True for symmetric, false for anti-symmetric.
	 **/
	btod_symmetrize(additive_btod<N> &op, size_t i1, size_t i2, bool symm);

	/**	\brief Initializes the operation using a unitary %permutation
			(P = P^-1)
		\param op Symmetrized operation.
		\param perm Unitary %permutation.
		\param symm True for symmetric, false for anti-symmetric.
	 **/
	btod_symmetrize(additive_btod<N> &op, const permutation<N> &perm,
		bool symm);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_symmetrize() { }

	//@}


	//!	\name Implementation of direct_block_tensor_operation<N, double>
	//@{

	virtual const block_index_space<N> &get_bis() const {
		return m_bis;
	}

	virtual const symmetry<N, double> &get_symmetry() const {
		return m_sym;
	}

	virtual const assignment_schedule<N, double> &get_schedule() const {
		return m_sch;
	}

	virtual void sync_on();
	virtual void sync_off();

	//@}

protected:
	//!	\brief Implementation of additive_btod<N>
	//@{

	virtual void compute_block(bool zero, dense_tensor_i<N, double> &blk,
	    const index<N> &i, const transf<N, double> &tr, double c,
	    cpu_pool &cpus);

	//@}

private:
	/**	\brief Constructs the %symmetry of the result
	 **/
	void make_symmetry();

	/**	\brief Constructs the assignment schedule of the operation
	 **/
	void make_schedule();

private:
	btod_symmetrize(const btod_symmetrize<N>&);
	const btod_symmetrize<N> &operator=(const btod_symmetrize<N>&);

};


template<size_t N>
const char *btod_symmetrize<N>::k_clazz = "btod_symmetrize<N>";


template<size_t N>
btod_symmetrize<N>::btod_symmetrize(additive_btod<N> &op, size_t i1, size_t i2,
	bool symm) :

	m_op(op), m_symm(symm), m_bis(op.get_bis()), m_sym(m_bis),
	m_sch(m_bis.get_block_index_dims()) {

	static const char *method =
		"btod_symmetrize(additive_btod<N>&, size_t, size_t, bool)";

	if(i1 == i2) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"i");
	}
	m_perm1.permute(i1, i2);
	make_symmetry();
	make_schedule();
}


template<size_t N>
btod_symmetrize<N>::btod_symmetrize(additive_btod<N> &op,
	const permutation<N> &perm, bool symm) :

	m_op(op), m_symm(symm), m_perm1(perm), m_bis(op.get_bis()),
	m_sym(m_bis), m_sch(m_bis.get_block_index_dims()) {

	static const char *method = "btod_symmetrize(additive_btod<N>&, "
		"const permutation<N>&, bool)";

	permutation<N> p1(perm); p1.permute(perm);
	if(perm.is_identity() || !p1.is_identity()) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "perm");
	}
	make_symmetry();
	make_schedule();
}


template<size_t N>
void btod_symmetrize<N>::sync_on() {

	m_op.sync_on();
}


template<size_t N>
void btod_symmetrize<N>::sync_off() {

	m_op.sync_off();
}

/*
template<size_t N>
void btod_symmetrize<N>::compute_block(dense_tensor_i<N, double> &blk,
	const index<N> &i) {

	typedef typename sym_schedule_t::iterator iterator_t;

	dimensions<N> bidims(m_bis.get_block_index_dims());
	abs_index<N> ai(i, bidims);

	tod_set<N>().perform(blk);
	compute_block(blk, i, transf<N, double>(), 1.0);
}*/


template<size_t N>
void btod_symmetrize<N>::compute_block(bool zero, dense_tensor_i<N, double> &blk,
	const index<N> &idx, const transf<N, double> &tr, double c,
	cpu_pool &cpus) {

	typedef typename sym_schedule_t::iterator iterator_t;

    if(zero) tod_set<N>().perform(cpus, blk);

	dimensions<N> bidims(m_bis.get_block_index_dims());
	abs_index<N> aidx(idx, bidims);

	std::list<schrec> sch1;
	std::pair<iterator_t, iterator_t> jr =
		m_sym_sch.equal_range(aidx.get_abs_index());
	for(iterator_t j = jr.first; j != jr.second; ++j) {
		sch1.push_back(j->second);
	}

	while(!sch1.empty()) {
		abs_index<N> ai(sch1.front().ai, bidims);
		size_t n = 0;
		for(typename std::list<schrec>::iterator j = sch1.begin();
			j != sch1.end(); ++j) {
			if(j->ai == ai.get_abs_index()) n++;
		}

		transf<N, double> tri(sch1.front().tr);
		tri.transform(tr);

		if(n == 1) {
			additive_btod<N>::compute_block(m_op, false, blk,
				ai.get_index(), tri, c, cpus);
			sch1.pop_front();
		} else {
			dimensions<N> dims(blk.get_dims());
			// TODO: replace with "temporary block" feature
			dense_tensor< N, double, allocator<double> > tmp(dims);
			additive_btod<N>::compute_block(m_op, true, tmp,
				ai.get_index(), tri, c, cpus);
			transf<N, double> tri_inv(tri);
			tri_inv.invert();
			for(typename std::list<schrec>::iterator j =
				sch1.begin(); j != sch1.end();) {
				if(j->ai != ai.get_abs_index()) {
					++j; continue;
				}
				transf<N, double> trj(tri_inv);
				trj.transform(j->tr);
				trj.transform(tr);
				tod_copy<N>(tmp, trj.get_perm(),
					trj.get_coeff()).perform(cpus, false, 1.0, blk);
				j = sch1.erase(j);
			}
		}
	}
}


template<size_t N>
void btod_symmetrize<N>::make_symmetry() {

    sequence<N, size_t> seq2(0), idxgrp(0), symidx(0);
    for (register size_t i = 0; i < N; i++) seq2[i] = i;
    m_perm1.apply(seq2);

    size_t idx = 1;
    for (register size_t i = 0; i < N; i++) {
        if (seq2[i] <= i) continue;

        idxgrp[i] = 1;
        idxgrp[seq2[i]] = 2;
        symidx[i] = symidx[seq2[i]] = idx++;
    }
	so_symmetrize<N, double>(m_op.get_symmetry(),
	        idxgrp, symidx, m_symm).perform(m_sym);
}


template<size_t N>
void btod_symmetrize<N>::make_schedule() {

	btod_symmetrize<N>::start_timer("make_schedule");

	dimensions<N> bidims(m_bis.get_block_index_dims());
	orbit_list<N, double> ol(m_sym);

	const assignment_schedule<N, double> &sch0 = m_op.get_schedule();
	for(typename assignment_schedule<N, double>::iterator i = sch0.begin();
		i != sch0.end(); i++) {

		abs_index<N> ai0(sch0.get_abs_index(i), bidims);
		orbit<N, double> o(m_op.get_symmetry(), ai0.get_index());

		for(typename orbit<N, double>::iterator j = o.begin();
			j != o.end(); j++) {

			abs_index<N> aj1(o.get_abs_index(j), bidims);
			index<N> j2(aj1.get_index()); j2.permute(m_perm1);
			abs_index<N> aj2(j2, bidims);

			if(ol.contains(aj1.get_abs_index())) {
				if(!m_sch.contains(aj1.get_abs_index())) {
					m_sch.insert(aj1.get_abs_index());
				}
				transf<N, double> tr1(o.get_transf(j));
				m_sym_sch.insert(sym_schedule_pair_t(
					aj1.get_abs_index(),
					schrec(ai0.get_abs_index(), tr1)));
			}
			if(ol.contains(aj2.get_abs_index())) {
				if(!m_sch.contains(aj2.get_abs_index())) {
					m_sch.insert(aj2.get_abs_index());
				}
				transf<N, double> tr2(o.get_transf(j));
				tr2.permute(m_perm1);
				tr2.scale(m_symm ? 1.0 : -1.0);
				m_sym_sch.insert(sym_schedule_pair_t(
					aj2.get_abs_index(),
					schrec(ai0.get_abs_index(), tr2)));
			}
		}
	}

	btod_symmetrize<N>::stop_timer("make_schedule");
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMMETRIZE_H
