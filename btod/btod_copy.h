#ifndef LIBTENSOR_BTOD_COPY_H
#define LIBTENSOR_BTOD_COPY_H

#include <map>
#include "defs.h"
#include "exception.h"
#include "core/orbit_list.h"
#include "tod/tod_copy.h"
#include "btod_additive.h"
#include "btod_so_copy.h"
#include "btod_so_equalize.h"

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
	block_index_space<N> m_bis; //!< Block %index space of output
	dimensions<N> m_bidims; //!< Block %index dimensions
	symmetry<N, double> m_symmetry; //!< Symmetry of output

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
: m_bt(bt), m_c(c), m_bis(bt.get_bis()), m_bidims(m_bis.get_block_index_dims()),
	m_symmetry(m_bis) {

}


template<size_t N>
btod_copy<N>::btod_copy(
	block_tensor_i<N, double> &bt, const permutation<N> &p, double c)
: m_bt(bt), m_perm(p), m_c(c), m_bis(bt.get_bis()),
	m_bidims(m_bis.get_block_index_dims()), m_symmetry(m_bis) {

	m_bis.permute(m_perm);
	m_bidims.permute(m_perm);
	m_symmetry.permute(m_perm);
}


template<size_t N>
btod_copy<N>::~btod_copy() {

}


template<size_t N>
inline const block_index_space<N> &btod_copy<N>::get_bis() const {

	return m_bis;
}


template<size_t N>
inline const symmetry<N, double> &btod_copy<N>::get_symmetry() const {

	return m_symmetry;
}


template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &bt) throw(exception) {

	static const char *method = "perform(block_tensor_i<N, double>&)";

	if(!m_bis.equals(bt.get_bis())) {
		throw bad_parameter("libtensor", k_clazz, method,
			__FILE__, __LINE__,
			"Incorrect block index space of the output tensor.");
	}

	btod_so_copy<N> symcopy(m_symmetry);
	symcopy.perform(bt);

	block_tensor_ctrl<N, double> src_ctrl(m_bt), dst_ctrl(bt);
	dimensions<N> bidims = m_bis.get_block_index_dims();

	orbit_list<N, double> orblst(src_ctrl.req_symmetry());
	typename orbit_list<N, double>::iterator iorbit = orblst.begin();
	for(; iorbit != orblst.end(); iorbit++) {

		orbit<N, double> orb(src_ctrl.req_symmetry(), *iorbit);
		index<N> src_blk_idx;
		bidims.abs_index(orb.get_abs_canonical_index(), src_blk_idx);
		if(src_ctrl.req_is_zero_block(src_blk_idx)) continue;
		index<N> dst_blk_idx(src_blk_idx);
		dst_blk_idx.permute(m_perm);

		tensor_i<N, double> &src_blk = src_ctrl.req_block(src_blk_idx);
		tensor_i<N, double> &dst_blk = dst_ctrl.req_block(dst_blk_idx);
		tod_copy<N> cp(src_blk, m_perm, m_c);
		cp.perform(dst_blk);
		src_ctrl.ret_block(src_blk_idx);
		dst_ctrl.ret_block(dst_blk_idx);

	}

}


template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &bt, double c)
	throw(exception) {

	static const char *method =
		"perform(block_tensor_i<N, double>&, double)";

	if(!m_bis.equals(bt.get_bis())) {
		throw bad_parameter("libtensor", k_clazz, method,
			__FILE__, __LINE__,
			"Incorrect block index space of the output tensor.");
	}

	btod_so_equalize<N> symeq(m_symmetry);
	symeq.perform(bt);

	block_tensor_ctrl<N, double> src_ctrl(m_bt), dst_ctrl(bt);
	dimensions<N> bidims = m_bis.get_block_index_dims();

	orbit_list<N, double> orblst(src_ctrl.req_symmetry());
	typename orbit_list<N, double>::iterator iorbit = orblst.begin();
	for(; iorbit != orblst.end(); iorbit++) {

		orbit<N, double> orb(src_ctrl.req_symmetry(), *iorbit);
		index<N> src_blk_idx;
		bidims.abs_index(orb.get_abs_canonical_index(), src_blk_idx);
		if(src_ctrl.req_is_zero_block(src_blk_idx)) continue;
		index<N> dst_blk_idx(src_blk_idx);
		dst_blk_idx.permute(m_perm);

		tensor_i<N, double> &src_blk = src_ctrl.req_block(src_blk_idx);
		tensor_i<N, double> &dst_blk = dst_ctrl.req_block(dst_blk_idx);
		tod_copy<N> cp(src_blk, m_perm, m_c);
		cp.perform(dst_blk);
		src_ctrl.ret_block(src_blk_idx);
		dst_ctrl.ret_block(dst_blk_idx);

	}

}


} // namespace libtensor

#endif // LITENSOR_BTOD_COPY_H
