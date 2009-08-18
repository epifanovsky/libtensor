#ifndef LIBTENSOR_BTOD_SO_EQUALIZE_H
#define LIBTENSOR_BTOD_SO_EQUALIZE_H

#include "defs.h"
#include "exception.h"
#include "core/block_tensor_i.h"
#include "core/block_tensor_ctrl.h"
#include "core/symmetry.h"

namespace libtensor {


/**	\brief Auxiliary block %tensor operation: equalize %symmetry

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_so_equalize {
private:
	const symmetry<N, double> &m_symmetry; //!< Symmetry

public:
	btod_so_equalize(const symmetry<N, double> &sym);
	void perform(block_tensor_i<N, double> &bt) throw(exception);
};


template<size_t N>
btod_so_equalize<N>::btod_so_equalize(const symmetry<N, double> &sym)
: m_symmetry(sym) {

}


template<size_t N>
void btod_so_equalize<N>::perform(block_tensor_i<N, double> &bt)
	throw(exception) {

	dimensions<N> bidims(bt.get_bis().get_block_index_dims());

	block_tensor_ctrl<N, double> ctrl(bt);
	symmetry<N, double> src_sym(m_symmetry);
	symmetry<N, double> dst_sym(ctrl.req_symmetry());
	src_sym.set_overlap(dst_sym);

	orbit_list<N, double> orblst(src_sym);
	typename orbit_list<N, double>::iterator iorbit = orblst.begin();
	for(; iorbit != orblst.end(); iorbit++) {
		orbit<N, double> orb(src_sym, *iorbit);
		index<N> blkidx;
		bidims.abs_index(orb.get_abs_canonical_index(), blkidx);
//		if(!dst_sym.is_canonical(blkidx)) {
//			throw_exc("btod_so_equalize<N>", "perform()",
//				"Symmetry lowering is not supported yet.");
//		}
	}

	ctrl.req_sym_clear_elements();
	size_t nelem = src_sym.get_num_elements();
	for(size_t ielem = 0; ielem < nelem; ielem++) {
		ctrl.req_sym_add_element(src_sym.get_element(ielem));
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SO_EQUALIZE_H
