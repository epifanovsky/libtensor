#ifndef LIBTENSOR_BTOD_SO_COPY_H
#define LIBTENSOR_BTOD_SO_COPY_H

#include "../defs.h"
#include "../exception.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/symmetry.h"

namespace libtensor {


/**	\brief Auxiliary block %tensor operation: copy %symmetry

	This operation is used by other block %tensor operations to install
	a certain %symmetry.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_so_copy {
private:
	const symmetry<N, double> &m_symmetry; //!< Symmetry

public:
	btod_so_copy(const symmetry<N, double> &sym);
	void perform(block_tensor_i<N, double> &bt) throw(exception);
};


template<size_t N>
inline btod_so_copy<N>::btod_so_copy(const symmetry<N, double> &sym)
: m_symmetry(sym) {

}


template<size_t N>
void btod_so_copy<N>::perform(block_tensor_i<N, double> &bt) throw(exception) {

	block_tensor_ctrl<N, double> ctrl(bt);

	ctrl.req_zero_all_blocks();
	ctrl.req_sym_clear_elements();
	typename symmetry<N, double>::iterator ielem = m_symmetry.begin();
	for(; ielem != m_symmetry.end(); ielem++) {
		ctrl.req_sym_add_element(m_symmetry.get_element(ielem));
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SO_COPY_H
