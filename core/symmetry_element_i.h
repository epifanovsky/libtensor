#ifndef LIBTENSOR_SYMMETRY_ELEMENT_I_H
#define LIBTENSOR_SYMMETRY_ELEMENT_I_H

#include "defs.h"
#include "exception.h"
#include "index.h"
#include "mask.h"

namespace libtensor {

/**	\brief Symmetry element interface
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
class symmetry_element_i {
public:
	virtual const mask<N> &get_mask() const = 0;
	virtual bool is_allowed(const index<N> &idx) const = 0;
	virtual void apply(index<N> &idx) const = 0;
	virtual bool equals(const symmetry_element_i<N, T> &se) const = 0;
	virtual symmetry_element_i<N, T> *clone() const = 0;
	virtual symmetry_element_i<N + 1, T> *project_up(
		const mask<N + 1> &msk) const = 0;
	virtual symmetry_element_i<N - 1, T> *project_down(
		const mask<N> &msk) const = 0;
};

} // namespace libtensor

#endif // SYMMETRY_ELEMENT_I_H
