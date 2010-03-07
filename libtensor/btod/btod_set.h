#ifndef LIBTENSOR_BTOD_SET_H
#define LIBTENSOR_BTOD_SET_H

#include "../defs.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../tod/tod_set.h"
#include "bad_block_index_space.h"

namespace libtensor {


/**	\brief Sets all elements of a block %tensor to a value preserving
		%symmetry
	\tparam N Tensor order.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_set {
public:
	static const char *k_clazz; //!< Class name

private:
	double m_a; //!< Value

public:
	btod_set(double a = 0.0);

	void perform(block_tensor_i<N, double> &btb);

private:
	btod_set(const btod_set<N> &);
	const btod_set<N> &operator=(const btod_set<N> &);

};


template<size_t N>
const char *btod_set<N>::k_clazz = "btod_set<N>";


template<size_t N>
btod_set<N>::btod_set(double a) : m_a(a) {

}


template<size_t N>
void btod_set<N>::perform(block_tensor_i<N, double> &btb) {

	block_tensor_ctrl<N, double> ctrlb(btb);

	orbit_list<N, double> olstb(ctrlb.req_symmetry());

	for(typename orbit_list<N, double>::iterator iob = olstb.begin();
		iob != olstb.end(); iob++) {

		index<N> idxb(olstb.get_index(iob));
		if(m_a == 0.0) {
			ctrlb.req_zero_block(idxb);
		} else {
			tensor_i<N, double> &blkb = ctrlb.req_block(idxb);
			tod_set<N>(m_a).perform(blkb);
			ctrlb.ret_block(idxb);
		}
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_H
