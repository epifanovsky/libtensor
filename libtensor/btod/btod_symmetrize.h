#ifndef LIBTENSOR_BTOD_SYMMETRIZE_H
#define LIBTENSOR_BTOD_SYMMETRIZE_H

#include <map>
#include "../timings.h"
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
	permutation<N> m_perm; //!< Symmetrization permutation
	bool m_symm; //!< Symmetrization sign
	block_index_space<N> m_bis; //!< Block %index space of the result
	symmetry<N, double> m_sym; //!< Symmetry of the result
	assignment_schedule<N, double> m_sch; //!< Schedule
	sym_schedule_t m_sym_sch; //!< Symmetrization schedule

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the operation
		\param op Symmetrized operation.
		\param perm Permutation to be added to the %symmetry group.
		\param symm True for symmetric, false for anti-symmetric.
	 **/
	btod_symmetrize(additive_btod<N> &op, const permutation <N> &perm,
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

	//@}

protected:
	//!	\brief Implementation of additive_btod<N>
	//@{

	virtual void compute_block(tensor_i<N, double> &blk, const index<N> &i);

	virtual void compute_block(tensor_i<N, double> &blk, const index<N> &i,
		const transf<N, double> &tr, double c);

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
btod_symmetrize<N>::btod_symmetrize(additive_btod<N> &op,
	const permutation<N> &perm, bool symm) :

	m_op(op), m_perm(perm), m_symm(symm), m_bis(op.get_bis()), m_sym(m_bis),
	m_sch(m_bis.get_block_index_dims()) {

	make_symmetry();
	make_schedule();
}


template<size_t N>
void btod_symmetrize<N>::compute_block(tensor_i<N, double> &blk,
	const index<N> &i) {

	typedef typename sym_schedule_t::iterator iterator_t;

	dimensions<N> bidims(m_bis.get_block_index_dims());
	abs_index<N> ai(i, bidims);

	tod_set<N>().perform(blk);

	std::pair<iterator_t, iterator_t> jr =
		m_sym_sch.equal_range(ai.get_abs_index());
	for(iterator_t j = jr.first; j != jr.second; j++) {

		abs_index<N> aj(j->second.ai, bidims);
		additive_btod<N>::compute_block(m_op, blk, aj.get_index(),
			j->second.tr, 1.0);
	}
}


template<size_t N>
void btod_symmetrize<N>::compute_block(tensor_i<N, double> &blk,
	const index<N> &i, const transf<N, double> &tr, double c) {

	typedef typename sym_schedule_t::iterator iterator_t;

	dimensions<N> bidims(m_bis.get_block_index_dims());
	abs_index<N> ai(i, bidims);

	std::pair<iterator_t, iterator_t> jr =
		m_sym_sch.equal_range(ai.get_abs_index());
	for(iterator_t j = jr.first; j != jr.second; j++) {

		abs_index<N> aj(j->second.ai, bidims);
		transf<N, double> trj(j->second.tr);
		trj.transform(tr);
		additive_btod<N>::compute_block(m_op, blk, aj.get_index(),
			trj, c);
	}
}


template<size_t N>
void btod_symmetrize<N>::make_symmetry() {

	so_symmetrize<N, double>(m_op.get_symmetry(), m_perm, m_symm).
		perform(m_sym);
}


template<size_t N>
void btod_symmetrize<N>::make_schedule() {

	btod_symmetrize<N>::start_timer("make_schedule");

	dimensions<N> bidims(m_bis.get_block_index_dims());

	const assignment_schedule<N, double> &sch0 = m_op.get_schedule();
	for(typename assignment_schedule<N, double>::iterator i = sch0.begin();
		i != sch0.end(); i++) {

		abs_index<N> ai0(sch0.get_abs_index(i), bidims);
		orbit<N, double> o(m_sym, ai0.get_index());
		size_t aci = o.get_abs_canonical_index();
		if(!m_sch.contains(aci)) {
			m_sch.insert(aci);
		}
		transf<N, double> tr0(o.get_transf(ai0.get_abs_index()));
		tr0.invert();
		m_sym_sch.insert(sym_schedule_pair_t(aci,
			schrec(ai0.get_abs_index(), tr0)));

		index<N> i1(ai0.get_index());
		i1.permute(m_perm);
		if(ai0.get_index().equals(i1)) {
			transf<N, double> tr1;
			tr1.permute(m_perm);
			tr1.scale(m_symm ? 1.0 : -1.0);
			tr1.transform(tr0);
			m_sym_sch.insert(sym_schedule_pair_t(aci,
				schrec(ai0.get_abs_index(), tr1)));
		}
	}

	btod_symmetrize<N>::stop_timer("make_schedule");
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMMETRIZE_H
