#ifndef LIBTENSOR_SO_UNION_IMPL_PERM_H
#define LIBTENSOR_SO_UNION_IMPL_PERM_H

#include "symmetry_element_set_adapter.h"
#include "so_union.h"
#include "se_perm.h"

namespace libtensor {


/**	\brief Implementation of so_union<N, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_union_impl< se_perm<N, T> > {
public:
	void perform(const symmetry_operation_params< so_union<N, T> > &params,
		symmetry_element_set<N, T> &set);
};


template<size_t N, typename T>
void so_union_impl< se_perm<N, T> >::perform(
	const symmetry_operation_params< so_union<N, T> > &params,
	symmetry_element_set<N, T> &set) {

}


} // namespace libtensor

#endif // LIBTENSOR_SO_UNION_IMPL_PERM_H

