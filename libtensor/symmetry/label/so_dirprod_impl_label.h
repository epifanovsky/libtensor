#ifndef LIBTENSOR_SO_DIRPROD_IMPL_LABEL_H
#define LIBTENSOR_SO_DIRPROD_IMPL_LABEL_H

#include "../../defs.h"
#include "../../exception.h"
#include "../../core/permutation_builder.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_dirprod.h"
#include "../se_label.h"
#include "label_set.h"

namespace libtensor {


/**	\brief Implementation of so_dirprod<N, M, T> for se_label<N + M, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_dirprod<N, M, T>, se_label<N + M, T> > :
	public symmetry_operation_impl_base<
		so_dirprod<N, M, T>, se_label<N + M, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_dirprod<N, M, T> operation_t;
	typedef se_label<N + M, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_dirprod<N, M, T>, se_label<N + M, T> >::k_clazz =
	"symmetry_operation_impl< so_dirprod<N, M, T>, se_label<N + M, T> >";


template<size_t N, size_t M, typename T>
void
symmetry_operation_impl< so_dirprod<N, M, T>, se_label<N + M, T> >::do_perform(
        symmetry_operation_params_t &params) const {

	static const char *method = "do_perform(symmetry_operation_params_t&)";

	// Adapter type for the input groups
	typedef symmetry_element_set_adapter< N, T, se_label<N, T> > adapter1_t;
	typedef symmetry_element_set_adapter< M, T, se_label<M, T> > adapter2_t;

	adapter1_t g1(params.g1);
	adapter2_t g2(params.g2);
	params.g3.clear();

	// map result index to input index
	sequence<N + M, size_t> map(0);
	for (size_t j = 0; j < N + M; j++) map[j] = j;
	permutation<N + M> pinv(params.perm, true);
	pinv.apply(map);

	dimensions<N + M> bidims = params.bis.get_block_index_dims();
	//	Go over each element in the first source group
	for(typename adapter1_t::iterator i = g1.begin(); i != g1.end(); i++) {

		// Create result se_label
		se_label<N + M, T> e3(bidims);

		const se_label<N, T> &e1 = g1.get_elem(i);
		// Loop over all label sets in the source se_label and copy them
		for (typename se_label<N, T>::const_iterator iss = e1.begin();
		        iss != e1.end(); iss++) {

		    const label_set<N> &ss1 = e1.get_subset(iss);
		    const mask<N> &msk1 = ss1.get_mask();
		    const mask<N> &emsk1 = ss1.get_eval_msk();

		    mask<N + M> msk3, emsk3;
	        for (register size_t k = 0; k < N; k++) {
	            msk3[map[k]] = msk1[k];
	            emsk3[map[k]] = emsk1[k];
	        }

		    label_set<N + M> &ss3 = e3.create_subset(msk3, ss1.get_table_id());
		    ss3.set_eval_msk(emsk3);

            // Copy the intrinsic labels
		    for (typename label_set<N>::iterator ii = ss1.begin();
		            ii != ss1.end(); ii++) {
		        ss3.add_intrinsic(ss1.get_intrinsic(ii));
		    }

		    // Assign labels to the dimensions stemming from ss1
		    for (register size_t k = 0; k < N; k++) {
		        mask<N + M> msk;
		        msk[map[k]] = true;

		        size_t ktype = ss1.get_dim_type(k);
		        for (size_t kpos = 0; kpos < bidims[map[k]]; kpos++) {

		            typename label_set<N>::label_t label =
		                    ss1.get_label(ktype, kpos);

		            if (! ss1.is_valid(label)) continue;
		            ss3.assign(msk, kpos, label);
		        }
		    }

		    ss3.match_blk_labels();
		}

		params.g3.insert(e3);
	}
    //  Do the same for the second source group
    for(typename adapter2_t::iterator i = g2.begin(); i != g2.end(); i++) {

        // Create result se_label
        se_label<N + M, T> e3(bidims);

        const se_label<M, T> &e2 = g2.get_elem(i);
        // Loop over all label sets in the source se_label and copy them
        for (typename se_label<M, T>::const_iterator iss = e2.begin();
                iss != e2.end(); iss++) {

            const label_set<M> &ss2 = e2.get_subset(iss);
            const mask<M> &msk2 = ss2.get_mask();
            const mask<M> &emsk2 = ss2.get_eval_msk();

            mask<N + M> msk3, emsk3;
            for (register size_t k = 0; k < M; k++) {
                msk3[map[k + N]] = msk2[k];
                emsk3[map[k + N]] = emsk2[k];
            }

            label_set<N + M> &ss3 = e3.create_subset(msk3, ss2.get_table_id());
            ss3.set_eval_msk(emsk3);

            // Copy the intrinsic labels
            for (typename label_set<M>::iterator ii = ss2.begin();
                    ii != ss2.end(); ii++) {
                ss3.add_intrinsic(ss2.get_intrinsic(ii));
            }

            // Assign labels to the dimensions stemming from ss1
            for (register size_t k = 0; k < M; k++) {
                mask<N + M> msk;
                msk[map[k + N]] = true;

                size_t ktype = ss2.get_dim_type(k);
                for (size_t kpos = 0; kpos < bidims[map[k]]; kpos++) {

                    typename label_set<M>::label_t label =
                            ss2.get_label(ktype, kpos);

                    if (! ss2.is_valid(label)) continue;
                    ss3.assign(msk, kpos, label);
                }
            }

            ss3.match_blk_labels();
        }

        params.g3.insert(e3);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_SO_DIRPROD_IMPL_LABEL_H
