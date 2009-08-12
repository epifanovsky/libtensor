#ifndef LIBTENSOR_BTOD_COPY_H
#define LIBTENSOR_BTOD_COPY_H

#include <map>
#include "defs.h"
#include "exception.h"
#include "btod/btod_additive.h"
#include "tod/tod_copy.h"

namespace libtensor {

/**	\brief Makes a copy of a block %tensor, applying a permutation or
		a scaling coefficient
	\tparam N Tensor order.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_copy : public btod_additive<N> {
public:
	static const char *k_clazz; //!< Class name

private:
	block_tensor_i<N, double> &m_bt; //!< Source block %tensor
	permutation<N> m_perm; //!< Permutation
	double m_c; //!< Scaling coefficient
	block_index_space<N> m_bis; //!< Block %index space of the output

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the copy operation
		\param bt Source block %tensor.
		\param c Scaling coefficient.
	 **/
	btod_copy(block_tensor_i<N, double> &bt, double c = 1.0);

	/**	\brief Initializes the permuted copy operation
		\param bt Source block %tensor.
		\param p Permutation.
		\param c Scaling coefficient.
	 **/
	btod_copy(block_tensor_i<N, double> &bt, const permutation<N> &p,
		double c = 1.0);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_copy();
	//@}

	//!	\name Implementation of
	//!		libtensor::direct_block_tensor_operation<N, double>
	//@{
	virtual const block_index_space<N> &get_bis() const;
	virtual const symmetry<N, double> &get_symmetry() const;
	virtual void perform(block_tensor_i<N, double> &bt) throw(exception);
	//@}

	//!	\name Implementation of libtensor::btod_additive<N>
	//@{
	virtual void perform(block_tensor_i<N, double> &bt, double c)
		throw(exception);
	//@}

};


template<size_t N>
const char *btod_copy<N>::k_clazz = "btod_copy<N>";


template<size_t N>
btod_copy<N>::btod_copy(block_tensor_i<N, double> &bt, double c)
: m_bt(bt), m_c(c), m_bis(bt.get_bis()) {

}


template<size_t N>
btod_copy<N>::btod_copy(
	block_tensor_i<N, double> &bt, const permutation<N> &p, double c)
: m_bt(bt), m_perm(p), m_c(c), m_bis(bt.get_bis()) {

	m_bis.permute(m_perm);
}


template<size_t N>
btod_copy<N>::~btod_copy() {

}


template<size_t N>
const block_index_space<N> &btod_copy<N>::get_bis() const {
	throw_exc("btod_copy<N>", "get_bis()", "Not implemented");
}


template<size_t N>
const symmetry<N, double> &btod_copy<N>::get_symmetry() const {
	throw_exc("btod_copy<N>", "get_symmetry()", "Not implemented");
}


template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &bt) throw(exception) {

	static const char *method = "perform(block_tensor_i<N, double>&)";

	if(!m_bis.equals(bt.get_bis())) {
		throw bad_parameter("libtensor", k_clazz, method,
			__FILE__, __LINE__,
			"Incorrect block index space of the output tensor.");
	}

	block_tensor_ctrl<N, double> ctrl_src(m_bt), ctrl_dst(bt);
	dimensions<N> bidims = m_bis.get_block_index_dims();

	// Prefetch blocks here

	// Equalize the symmetries of input and output (to do)

	// Create a list of all non-zero canonical blocks in input
/*
	std::map< index<N>, tod_copy<N>* > map_todo, map_extra;
	orbit_iterator<N, double> oi_src = ctrl_src.req_orbits();
	while(!oi_src.end()) {
		index<N> idx(oi_src.get_index());
		idx.permute(m_perm);
		map_todo.insert(std::pair< index<N>, tod_copy<N>* >(
			idx, NULL));
		oi_src.next();
	}

	// Request the symmetry of input and output

	const symmetry_i<N, double> &sym_src = ctrl_src.req_symmetry();
	const symmetry_i<N, double> &sym_dst = ctrl_dst.req_symmetry();

	// Reconcile canonical blocks in input and output

	permutation<N> invperm(m_perm);
	invperm.invert();
	orbit_iterator<N, double> oi_dst(sym_dst.get_oi_handler(),
		sym_dst.get_bi_handler());
	while(!oi_dst.end()) {
		size_t absidx = bidims.abs_index(oi_dst.get_index());
		if(map_todo.find(oi_dst.get_index()) == map_todo.end()) {
			index<N> idx(oi_dst.get_index());
			idx.permute(invperm);
			if(sym_src.is_canonical(idx)) {
				ctrl_dst.req_zero_block(oi_dst.get_index());
			} else {
				map_extra.insert(
					std::pair< index<N>, tod_copy<N>* >(
						oi_dst.get_index(), NULL));
			}
		}
		oi_dst.next();
	}

	// Go over the todo list

	typename std::map< index<N>, tod_copy<N>* >::iterator iter =
		map_todo.begin();
	while(iter != map_todo.end()) {
		index<N> idx_src(iter->first), idx_dst(iter->first);
		idx_dst.permute(m_perm);
		tensor_i<N, double> &blk_src = ctrl_src.req_block(idx_src);
		tensor_i<N, double> &blk_dst = ctrl_dst.req_block(idx_dst);
		tod_copy<N> cp(blk_src, m_perm, m_c);
		cp.perform(blk_dst);
		ctrl_src.ret_block(idx_src);
		ctrl_dst.ret_block(idx_dst);
		iter++;
	}*/
}


template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &bt, double c)
	throw(exception) {
	block_tensor_ctrl<N, double> ctrl_src(m_bt), ctrl_dst(bt);
	index<N> i0;
	tensor_i<N, double> &tsrc(ctrl_src.req_block(i0));
	tensor_i<N, double> &tdst(ctrl_dst.req_block(i0));
	tod_copy<N> op(tsrc, m_perm, m_c);
	op.perform(tdst, c);
}


} // namespace libtensor

#endif // LITENSOR_BTOD_COPY_H
