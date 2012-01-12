#ifndef LIBTENSOR_SO_MERGE_IMPL_LABEL_H
#define LIBTENSOR_SO_MERGE_IMPL_LABEL_H

#include "../../defs.h"
#include "../../exception.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_merge.h"
#include "../se_label.h"

namespace libtensor {


/**	\brief Implementation of so_merge<N, T> for se_label<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	This implementation sets the target label to all labels.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t K, typename T>
class symmetry_operation_impl< so_merge<N, M, K, T>, se_label<N, T> > :
public symmetry_operation_impl_base< so_merge<N, M, K, T>, se_label<N, T> > {

public:
    static const char *k_clazz; //!< Class name
    static const size_t k_order2 = N - M + K; //!< Dimension of result

public:
    typedef so_merge<N, M, K, T> operation_t;
    typedef se_label<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, size_t M, size_t K, typename T>
const char *
symmetry_operation_impl< so_merge<N, M, K, T>, se_label<N, T> >::k_clazz =
        "symmetry_operation_impl< so_merge<N, M, K, T>, se_label<N, T> >";

template<size_t N, size_t M, size_t K, typename T>
void symmetry_operation_impl< so_merge<N, M, K, T>, se_label<N, T> >
::do_perform(symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter<N, T, element_t> adapter_t;
    typedef se_label<k_order2, T> el2_t;

    //	Verify that the projection masks are correct
    size_t nm = 0;
    mask<N> tm;
    for (size_t k = 0; k < K; k++) {
        const mask<N> &m = params.msk[k];
        for (size_t i = 0; i < N; i++) {
            if (! m[i]) continue;
            if (tm[i]) {
                throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                        "params.msk[k]");
            }
            tm[i] = true;
            nm++;
        }
    }
    if(nm != M) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "params.msk");
    }

    params.grp2.clear();
    if (params.grp1.is_empty()) return;

    sequence<N, size_t> map;
    for (size_t i = 0, j = 0; i < N; i++) {
        // Check if this is the first masked index in one of the masks
        if (tm[i]) {
            size_t k = 0;
            for (; k < K; k++) { if (params.msk[k][i]) break; }
            const mask<N> &m = params.msk[k];

            size_t ii = 0;
            for (; ii < i; ii++) {  if (m[ii]) break; }
            if (ii != i) { map[i] = map[ii]; continue; }
        }

        map[i] = j++;
    }

    adapter_t g1(params.grp1);

    // Create block index dimensions of result se_label
    typename adapter_t::iterator it1 = g1.begin();
    const dimensions<N> &bidims1 =
            g1.get_elem(it1).get_labeling().get_block_index_dims();

    index<k_order2> idx1, idx2;
    for (size_t i = 0; i < N; i++) {
        idx2[map[i]] = bidims1[i] - 1;
    }
    dimensions<k_order2> bidims2(index_range<k_order2>(idx1, idx2));

    // Loop over all se_label elements and merge dimensions in each one
    for (; it1 != g1.end(); it1++) {

#ifdef LIBTENSOR_DEBUG
        // This should never happen!!!
        if (bidims1 !=
                g1.get_elem(it1).get_labeling().get_block_index_dims()) {
            throw bad_symmetry(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Incompatible se_labels in input.");
        }
#endif

        const element_t &se1 = g1.get_elem(it1);
        el2_t se2(bidims2, se1.get_table_id());
        transfer_labeling(se1.get_labeling(), map, se2.get_labeling());

        // Copy evaluation rule
        evaluation_rule r2(se1.get_rule());

        // Modify the evaluation orders according to merge
        for (evaluation_rule::rule_iterator ir = r2.begin();
                ir != r2.end(); ir++) {

            evaluation_rule::rule_id rid = r2.get_rule_id(ir);
            evaluation_rule::basic_rule &br = r2.get_rule(rid);
            for (size_t i = 0; i < br.order.size(); i++) {
                if (br.order[i] == evaluation_rule::k_intrinsic) continue;

                br.order[i] = map[br.order[i]];
            }
        }

        se2.set_rule(r2);
        params.grp2.insert(se2);

    } // Loop it1
}


} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_IMPL_PERM_H
