#ifndef LIBTENSOR_SO_APPLY_IMPL_PART_H
#define LIBTENSOR_SO_APPLY_IMPL_PART_H

#include "../defs.h"
#include "../exception.h"
#include "partition_set.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_apply.h"
#include "se_part.h"

namespace libtensor {


/**	\brief Implementation of so_apply<N, T> for se_part<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_apply<N, T>, se_part<N, T> > :
	public symmetry_operation_impl_base< so_apply<N, T>, se_part<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_apply<N, T> operation_t;
	typedef se_part<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, typename T>
const char *symmetry_operation_impl< so_apply<N, T>, se_part<N, T> >::k_clazz =
	"symmetry_operation_impl< so_apply<N, T>, se_part<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_apply<N, T>, se_part<N, T> >::do_perform(
	symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(const symmetry_operation_params_t&)";

	typedef symmetry_element_set_adapter<N, T, element_t> adapter_t;

	params.grp2.clear();

	adapter_t adapter1(params.grp1);

	// If functor is asymmetric, only positive mappings survive.
	if (params.is_asym) {

		for (typename adapter_t::iterator it1 = adapter1.begin();
			it1 != adapter1.end(); it1++) {

			const element_t &se1 = adapter1.get_elem(it1);
			element_t se2(se1.get_bis(), se1.get_mask(), se1.get_npart());

			const dimensions<N> &pdims = se1.get_pdims();
			abs_index<N> ai(pdims);
			do {
				abs_index<N> bi(se1.get_direct_map(ai.get_index()), pdims);
				if (ai.get_abs_index() >= bi.get_abs_index())
					continue;

				if (se1.get_sign(ai.get_index(), bi.get_index())) {
					se2.add_map(ai.get_index(), bi.get_index());
				}
			} while (ai.inc());

			se2.permute(params.perm1);
			params.grp2.insert(se2);
		}
	}
	// If functor is symmetric with respect to the y-axis all negative
	// mappings become positive
	else if (params.sign) {
		for (typename adapter_t::iterator it1 = adapter1.begin();
			it1 != adapter1.end(); it1++) {

			const element_t &se1 = adapter1.get_elem(it1);
			element_t se2(se1.get_bis(), se1.get_mask(), se1.get_npart());

			const dimensions<N> &pdims = se1.get_pdims();
			abs_index<N> ai(pdims);
			do {
				abs_index<N> bi(se1.get_direct_map(ai.get_index()), pdims);
				if (ai.get_abs_index() >= bi.get_abs_index())
					continue;

				se2.add_map(ai.get_index(), bi.get_index());
			} while (ai.inc());

			se2.permute(params.perm1);
			params.grp2.insert(se2);
		}
	}
	else {
		for (typename adapter_t::iterator it1 = adapter1.begin();
			it1 != adapter1.end(); it1++) {

			const element_t &se1 = adapter1.get_elem(it1);
			element_t se2(se1);

			se2.permute(params.perm1);
			params.grp2.insert(se2);
		}
	}
}


} // namespace libtensor

#endif // LIBTENSOR_SO_APPLY_IMPL_PART_H
