#include "dimensions.h"

namespace libtensor {

bool dimensions::inc_index(index &idx) const throw(exception) {
	if(m_dims.get_order() != idx.get_order())
		throw_exc("inc_index(index&)", "Incompatible index");
	if(m_dims.less(idx) || m_dims.equals(idx)) return false;
	size_t n = m_dims.get_order() - 1;
	bool done = false;
	while(!done && n!=0) {
		if(idx[n] < m_dims[n]-1) {
			idx[n]++;
			for(size_t i=n+1; i<m_dims.get_order(); i++) idx[i]=0;
			done = true;
		} else {
			n--;
		}
	}
	return done;
}

size_t dimensions::abs_index(const index &idx) const throw(exception) {
	if(m_dims.get_order() != idx.get_order())
		throw_exc("abs_index(const index&)", "Incompatible index");
	size_t abs = 0;
	for(size_t i=0; i<m_dims.get_order(); i++) {
		if(idx[i] < m_dims[i]) {
			abs += m_incs[i]*idx[i];
		} else {
			throw_exc("abs_index(const index&)",
				"Index out of bounds");
		}
	}
	return abs;
}

} // namespace libtensor

