#ifndef LIBTENSOR_SO_PERMUTE_IMPL_PART_H
#define LIBTENSOR_SO_PERMUTE_IMPL_PART_H

#include "../defs.h"
#include "../exception.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_permute.h"
#include "se_part.h"

namespace libtensor {


/**	\brief Implementation of so_permute<N, T> for se_part<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_permute<N, T>, se_part<N, T> > :
	public symmetry_operation_impl_base< so_permute<N, T>, se_part<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_permute<N, T> operation_t;
	typedef se_part<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, typename T>
const char *symmetry_operation_impl< so_permute<N, T>, se_part<N, T> >::k_clazz =
	"symmetry_operation_impl< so_permute<N, T>, se_part<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_permute<N, T>, se_part<N, T> >::do_perform(
	symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(const symmetry_operation_params_t&)";

	typedef symmetry_element_set_adapter< N, T, se_part<N, T> > adapter_t;
	adapter_t adapter1(params.grp1);

	params.grp2.clear();

	for (typename adapter_t::iterator it1 = adapter1.begin();
			it1 != adapter1.end(); it1++) {

		se_part<N, T> se2(adapter1.get_elem(it1));
		se2.permute(params.perm);

		params.grp2.insert(se2);

	}
}


} // namespace libtensor

#endif // LIBTENSOR_SO_PERMUTE_IMPL_PERM_H
