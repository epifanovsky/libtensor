#ifndef LIBTENSOR_SO_PROJ_UP_IMPL_LABEL_H
#define LIBTENSOR_SO_PROJ_UP_IMPL_LABEL_H

#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_proj_up.h"
#include "se_label.h"

namespace libtensor {


/**	\brief Implementation of so_proj_up<N, M, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_proj_up<N, M, T>, se_label<N, T> > :
	public symmetry_operation_impl_base<
		so_proj_up<N, M, T>, se_label<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_proj_up<N, M, T> operation_t;
	typedef se_label<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, size_t M, typename T>
const char *symmetry_operation_impl< so_proj_up<N, M, T>,
	se_label<N, T> >::k_clazz =
	"symmetry_operation_impl< so_proj_up<N, M, T>, se_label<N, T> >";


template<size_t N, size_t M, typename T>
void symmetry_operation_impl< so_proj_up<N, M, T>,
	se_label<N, T> >::do_perform(symmetry_operation_params_t &params) const {

	static const char *method = "do_perform(symmetry_operation_params_t&)";

	// Adapter type for the input group
	typedef symmetry_element_set_adapter< N, T, se_label<N, T> > adapter_t;

	adapter_t g1(params.g1);
	params.g2.clear();

	//	Verify that the mask is valid
	const mask<N + M> &m = params.msk;
	size_t nm = 0;
	for(size_t i = 0; i < N + M; i++) if(m[i]) nm++;
	if(nm != N) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params.msk");
	}

	// map  result index -> input index
	sequence<N, size_t> map(0);
	for (size_t j = 0; j < N; j++) map[j] = j;
	params.perm.apply(map);

	//	Go over each element in the first source group
	for(typename adapter_t::iterator i = g1.begin(); i != g1.end(); i++) {

		const se_label<N, T> &e1 = g1.get_elem(i);

		// Create result se_label
		se_label<N + M, T> e2(params.bis.get_block_index_dims(),
				e1.get_table_id());

		// Assign labels to the dimensions stemming from sym1
		size_t j = 0;
		for (size_t k = 0; k < N + M; k++) {
			if (params.msk[k]) {
				mask<N + M> msk;
				msk[k] = true;
				size_t ktype = e1.get_dim_type(map[j]);
				for (size_t l = 0; l < e1.get_dim(ktype); l++) {
					if (! e1.is_valid(e1.get_label(ktype, l))) continue;

					e2.assign(msk, l, e1.get_label(ktype, l));
				}
				j++;
			}
		}

		e2.match_labels();

		// set target labels
		for (size_t k = 0; k < e1.get_n_targets(); k++)
			e2.add_target(e1.get_target(k));

		params.g2.insert(e2);
	}

}


} // namespace libtensor

#endif // LIBTENSOR_SO_PROJ_UP_IMPL_LABEL_H
