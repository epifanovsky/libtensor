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

	// Adapter type for the input groups
	typedef symmetry_element_set_adapter< N, T, se_label<N, T> > adapter1_t;
	typedef symmetry_element_set_adapter< M, T, se_label<M, T> > adapter2_t;

	adapter1_t g1(params.g1);
	adapter2_t g2(params.g2);
	params.g3.clear();

	// map result index to input index
	size_t map[N + M];
	for (size_t j = 0; j < N + M; j++) map[j] = j;
	permutation<N + M> pinv(params.perm, true);
	pinv.apply(map);


	//	Go over each element in the first source group
	for(typename adapter1_t::iterator i = g1.begin(); i != g1.end(); i++) {

		const se_label<N, T> &e1 = g1.get_elem(i);

		// Create result se_label
		se_label<N + M, T> e3(params.bis.get_block_index_dims(),
				e1.get_table_id());

		// Assign labels to the dimensions stemming from sym1
		for (size_t k = 0; k < N; k++) {
			mask<N + M> msk;
			msk[map[k]] = true;

			size_t ktype = e1.get_dim_type(k);
			for (size_t l = 0; l < e1.get_dim(ktype); l++) {
				typename se_label<N, T>::label_t label = e1.get_label(ktype, l);
				if (! e1.is_valid(label)) continue;

				e3.assign(msk, l, label);
			}
		}

		// check whether there is an se_label in set2 with the same
		// product table
		typename adapter2_t::iterator j = g2.begin();
		for(; j != g2.end(); j++) {
			if (e1.get_table_id() == g2.get_elem(j).get_table_id()) break;
		}
		if (j == g2.end()) {
			e3.match_labels();

			// set target labels
			for (size_t k = 0; k < e1.get_n_targets(); k++)
				e3.add_target(e1.get_target(k));
		}
		else {
			// assign labels to the remaining dimensions
			const se_label<M, T> &e2 = g2.get_elem(j);

			for (size_t k = 0; k < M; k++) {
				mask<N + M> msk;
				msk[map[N + k]] = true;

				size_t ktype = e2.get_dim_type(k);
				for (size_t l = 0; l < e2.get_dim(ktype); l++) {
					typename se_label<M, T>::label_t label =
							e2.get_label(ktype, l);
					if (! e2.is_valid(label)) continue;

					e3.assign(msk, l, label);
				}
			}

			e3.match_labels();

			// obtain product_table
			const product_table_i &pt = product_table_container::get_instance()
				.req_const_table(e1.get_table_id());

			// set target labels
			product_table_i::label_group lg(2);
			for (size_t k = 0; k < e1.get_n_targets(); k++) {
				for (size_t l = 0; l < e2.get_n_targets(); l++) {
					lg[0] = e1.get_target(k);
					lg[1] = e2.get_target(l);
					for (typename se_label<N + M,T>::label_t m = 0; m < pt.nlabels(); m++)
						if (pt.is_in_product(lg, m)) e3.add_target(m);
				}
			}
			product_table_container::get_instance().ret_table(e1.get_table_id());
		}

		params.g3.insert(e3);

	}

	//	Go over each element in the second source group
	for(typename adapter2_t::iterator i = g2.begin(); i != g2.end(); i++) {

		const se_label<M, T> &e2 = g2.get_elem(i);

		typename adapter1_t::iterator j = g1.begin();
		for(; j != g1.end(); j++) {
			if (e2.get_table_id() == g1.get_elem(j).get_table_id()) break;
		}
		if (j != g1.end()) continue;

		// Create result se_label
		se_label<N + M, T> e3(params.bis.get_block_index_dims(),
				e2.get_table_id());

		for (size_t k = 0; k < M; k++) {
			mask<N + M> msk;
			msk[map[N + k]] = true;

			size_t ktype = e2.get_dim_type(k);
			for (size_t l = 0; l < e2.get_dim(ktype); l++) {
				typename se_label<M, T>::label_t label = e2.get_label(ktype, l);
				if (! e2.is_valid(label)) continue;

				e3.assign(msk, l, label);
			}
		}

		e3.match_labels();
		for (size_t k = 0; k < e2.get_n_targets(); k++)
			e3.add_target(e2.get_target(k));

		params.g3.insert(e3);

	}


}


} // namespace libtensor

#endif // LIBTENSOR_SO_PROJ_UP_IMPL_PERM_H
