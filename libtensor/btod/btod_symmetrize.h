#ifndef LIBTENSOR_BTOD_SYMMETRIZE_H
#define LIBTENSOR_BTOD_SYMMETRIZE_H

#include <map>
#include "../timings.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../symmetry/so_add.h"
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
	size_t m_nsym; //!< Number of symmetrized indexes
	permutation<N> m_perm1; //!< First symmetrization permutation
	permutation<N> m_perm2; //!< Second symmetrization permutation
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

	/**	\brief Initializes the operation to symmetrize three indexes
		\param op Symmetrized operation.
		\param i1 First %tensor %index.
		\param i2 Second %tensor %index.
		\param i3 Third %tensor %index.
		\param symm True for symmetric, false for anti-symmetric.
	 **/
	btod_symmetrize(additive_btod<N> &op, size_t i1, size_t i2, size_t i3,
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
btod_symmetrize<N>::btod_symmetrize(additive_btod<N> &op, size_t i1, size_t i2,
	bool symm) :

	m_op(op), m_nsym(2), m_symm(symm), m_bis(op.get_bis()), m_sym(m_bis),
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
btod_symmetrize<N>::btod_symmetrize(additive_btod<N> &op, size_t i1, size_t i2,
	size_t i3, bool symm) :

	m_op(op), m_nsym(3), m_symm(symm), m_bis(op.get_bis()), m_sym(m_bis),
	m_sch(m_bis.get_block_index_dims()) {

	static const char *method = "btod_symmetrize(additive_btod<N>&, "
		"size_t, size_t, size_t, bool)";

	if(i1 == i2 || i1 == i3 || i2 == i3) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"i");
	}
	m_perm1.permute(i1, i2);
	m_perm2.permute(i1, i3);
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
		//~ std::cout << i << " <- " << aj.get_index() << " " << j->second.tr.get_perm() << " " << j->second.tr.get_coeff() << std::endl;
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

	permutation<N> perm0;

	// Using so_permute to work around a bug in so_add

	if(m_nsym == 2) {
		symmetry<N, double> sym1(m_bis), sym2(m_bis);
		so_permute<N, double>(m_op.get_symmetry(), m_perm1).perform(sym1);
		so_add<N, double>(m_op.get_symmetry(), perm0, sym1, perm0).perform(sym2);
		so_symmetrize<N, double>(sym2, m_perm1, m_symm).perform(m_sym);
	} else if(m_nsym == 3) {
		symmetry<N, double> sym1(m_bis), sym2(m_bis), sym3(m_bis), sym4(m_bis);
		so_permute<N, double>(m_op.get_symmetry(), m_perm1).perform(sym1);
		so_permute<N, double>(m_op.get_symmetry(), m_perm2).perform(sym2);
		so_add<N, double>(sym1, perm0, sym2, perm0).perform(sym3);
		so_add<N, double>(m_op.get_symmetry(), perm0, sym3, perm0).perform(sym4);
		so_symmetrize<N, double>(sym4, m_perm1, m_symm).perform(sym1);
		so_symmetrize<N, double>(sym1, m_perm2, m_symm).perform(m_sym);
	}
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

			if(m_nsym == 2) continue;
			index<N> j3(aj1.get_index()); j3.permute(m_perm2);
			abs_index<N> aj3(j3, bidims);
			if(ol.contains(aj3.get_abs_index())) {
				if(!m_sch.contains(aj3.get_abs_index())) {
					m_sch.insert(aj3.get_abs_index());
				}
				transf<N, double> tr3(o.get_transf(j));
				tr3.permute(m_perm2);
				tr3.scale(m_symm ? 1.0 : -1.0);
				m_sym_sch.insert(sym_schedule_pair_t(
					aj3.get_abs_index(),
					schrec(ai0.get_abs_index(), tr3)));
			}
		}
	}

	btod_symmetrize<N>::stop_timer("make_schedule");
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMMETRIZE_H
