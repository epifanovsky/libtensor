#ifndef LIBTENSOR_SO_CONCAT_IMPL_LABEL_H
#define LIBTENSOR_SO_CONCAT_IMPL_LABEL_H

#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_concat.h"
#include "se_label.h"

namespace libtensor {


/**	\brief Implementation of so_concat<N, M, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_concat<N, M, T>, se_label<N, T> > :
	public symmetry_operation_impl_base<
		so_concat<N, M, T>, se_label<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_concat<N, M, T> operation_t;
	typedef se_label<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, size_t M, typename T>
const char *symmetry_operation_impl< so_concat<N, M, T>,
	se_label<N, T> >::k_clazz =
	"symmetry_operation_impl< so_concat<N, M, T>, se_label<N, T> >";


template<size_t N, size_t M, typename T>
void symmetry_operation_impl< so_concat<N, M, T>,
	se_label<N, T> >::do_perform(symmetry_operation_params_t &params) const {

	static const char *method = "do_perform(symmetry_operation_params_t&)";

	//	Adapter type for the input group
	typedef symmetry_element_set_adapter< N, T, se_label<N, T> > adapter1_t;
	typedef symmetry_element_set_adapter< M, T, se_label<M, T> > adapter2_t;

	size_t map[N + M];
	for (size_t j = 0; j < N + M; j++) map[j] = j;
	permutation<N + M> pinv(params.perm, true);
	pinv.apply(map);

	adapter1_t g1(params.g1);
	adapter2_t g2(params.g2);

	//	Go over each element in the first source group and project up
	for(typename adapter1_t::iterator i = g1.begin(); i != g1.end(); i++) {

		const se_label<N, T> &e1 = g1.get_elem(i);
		std::string id(e1.get_table_id());

		// create result se_label
		se_label<N + M, T> e3(params.bis.get_block_index_dims(), id);

		typename adapter2_t::iterator j = g2.begin();
		for(; j != g2.end(); j++)
			if (id.compare(g2.get_elem().get_table_id()) == 0) break;



		for (size_t k = 0; k < N; k++) {
			mask<N + M> msk;
			msk[map[k]] = true;

			size_t ktype = e1.get_dim_type(map[k]);
			for (size_t l = 0; l < e1.get_dim(ktype); l++)
				e3.assign(msk, l, e1.get_label(ktype, l));
		}


		//	Project the combined permutation onto the larger
		//	space and form a symmetry element
		size_t a1[N];
		size_t a2a[N + M], a2b[N + M];
		for (size_t j = 0; j < N; j++) a1[j] = j;
		e1.get_perm().apply(a1);

		size_t k = 0;
		for(; k < N; k++) {
			a2a[map[k]] = k; a2b[map[k]] = a1[k];
		}
		for(; k < N + M; k++) {
			a2a[map[k]] = a2b[map[k]] = k;
		}

		permutation_builder<N + M> pb(a2b, a2a);
		se_perm<N + M, T> e3(pb.get_perm(), e1.is_symm());
		params.g3.insert(e3);
	}



		const se_perm<M, T> &e2 = g2.get_elem(i);

		//	Project the combined permutation onto the larger
		//	space and form a symmetry element
		size_t a1[M];
		size_t a2a[N + M], a2b[N + M];
		for (size_t j = 0; j < M; j++) a1[j] = N + j;
		e2.get_perm().apply(a1);

		size_t k = 0;
		for(; k < N; k++) {
			a2a[map[k]] = a2b[map[k]] = k;
		}
		for(; k < N + M; k++) {
			a2a[map[k]] = k; a2b[map[k]] = a1[k - N];
		}

		permutation_builder<N + M> pb(a2b, a2a);
		se_perm<N + M, T> e3(pb.get_perm(), e2.is_symm());
		params.g3.insert(e3);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_SO_PROJ_UP_IMPL_PERM_H
