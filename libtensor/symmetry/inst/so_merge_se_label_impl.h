#ifndef LIBTENSOR_SO_MERGE_SE_LABEL_IMPL_H
#define LIBTENSOR_SO_MERGE_SE_LABEL_IMPL_H

#include <libtensor/defs.h>
#include "../bad_symmetry.h"
#include "../product_table_container.h"
#include "combine_label.h"
#include "er_merge.h"
#include "er_optimize.h"

namespace libtensor {

template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_merge<N, M, T>, se_label<N - M, T> >::k_clazz =
        "symmetry_operation_impl< so_merge<N, M, T>, se_label<N - M, T> >";

template<size_t N, size_t M, typename T>
void
symmetry_operation_impl< so_merge<N, M, T>, se_label<N - M, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method = "do_perform(symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter<N, T, el1_t> adapter1_t;

    params.grp2.clear();
    if (params.grp1.is_empty()) return;

    // Create some necessary index maps
    sequence<N, size_t> mmap, lmap((size_t) -1);
    {
        sequence<M, size_t> msteps((size_t) -1);
        for (register size_t i = 0, j = 0; i < N; i++) {
            if (params.msk[i]) {
                if (msteps[params.mseq[i]] == (size_t) -1) {
                    lmap[i] = msteps[params.mseq[i]] = j++;
                }
                mmap[i] = msteps[params.mseq[i]];
            }
            else mmap[i] = lmap[i] = j++;
        }
    }

    adapter1_t g1(params.grp1);

    // Create block index dimensions of result se_label
    typename adapter1_t::iterator it1 = g1.begin();
    const dimensions<N> &bidims1 =
            g1.get_elem(it1).get_labeling().get_block_index_dims();

    index<N - M> idx1, idx2;
    for (size_t i = 0; i < N; i++) {
        idx2[mmap[i]] = bidims1[i] - 1;
    }
    dimensions<N - M> bidims2(index_range<N - M>(idx1, idx2));

    // Loop over all se_label elements and merge dimensions in each one
    std::set<std::string> table_ids;
    for (; it1 != g1.end(); it1++) {

        const el1_t &se1 = g1.get_elem(it1);
        if (table_ids.count(se1.get_table_id()) != 0) continue;

#ifdef LIBTENSOR_DEBUG
        // This should never happen!!!
        if (bidims1 != se1.get_labeling().get_block_index_dims()) {
            throw bad_symmetry(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Incompatible se_labels in input.");
        }
#endif

        // Collect all label objects with the same product table
        combine_label<N, T> cl1(se1);
        table_ids.insert(cl1.get_table_id());

        { // Look for other se_label elements with the same table id
            typename adapter1_t::iterator it2 = it1; it2++;
            for (; it2 != g1.end(); it2++) {
                const el1_t &se1b = g1.get_elem(it2);
                if (se1b.get_table_id() != cl1.get_table_id()) continue;
                cl1.add(se1b);
            }
        }

        const block_labeling<N> &bl1 = cl1.get_labeling();

        // Check that merged dimensions have the same labeling...
        mask<N - M> smsk;
        try {
            const product_table_i &pt = product_table_container::get_instance().
                    req_const_table(se1.get_table_id());

            mask<N> done;
            for (size_t i = 0; i < N; i++) {

                if (! params.msk[i] || done[params.mseq[i]]) continue;

                size_t j = i + 1, typei = bl1.get_dim_type(i);
                for (size_t j = i + 1; j < N; j++) {
                    if (! params.msk[j] || params.mseq[j] != params.mseq[i])
                        continue;

                    size_t typej = bl1.get_dim_type(j);
                    if (typei == typej) continue;

#ifdef LIBTENSOR_DEBUG
                    for (size_t k = 0; k < bl1.get_dim(typei); k++) {
                        if (bl1.get_label(typei, k) != bl1.get_label(typej, k)
                                && bl1.get_label(typei, k) != product_table_i::k_invalid
                                && bl1.get_label(typej, k) != product_table_i::k_invalid)
                            throw bad_symmetry(g_ns, k_clazz, method,
                                    __FILE__, __LINE__, "Merge dimensions.");
                    }
#endif
                }

                for (size_t j = 0; j < bl1.get_dim(typei); j++) {
                    product_table_i::label_t l = bl1.get_label(typei, j);
                    if (l == product_table_i::k_invalid ||
                            l == product_table_i::k_identity) continue;

                    product_table_i::label_group_t lg(2, l);
                    product_table_i::label_set_t ls;
                    pt.product(lg, ls);
                    if (ls.size() != 1 ||
                            *(ls.begin()) != product_table_i::k_identity) break;
                }

                smsk[mmap[i]] = (j == bl1.get_dim(typei));
                done[params.mseq[i]] = true;
            }

            product_table_container::get_instance().
                    ret_table(se1.get_table_id());

        } catch (exception &e) {

            product_table_container::get_instance().
                    ret_table(se1.get_table_id());

            throw;
        }

        // Create the result
        el2_t se2(bidims2, cl1.get_table_id());

        // Transfer the labeling of the remaining dimensions
        block_labeling<N - M> &bl2 = se2.get_labeling();
        transfer_labeling(bl1, lmap, bl2);

        // Transfer the rule
        const evaluation_rule<N> &r1 = cl1.get_rule();
        evaluation_rule<N - M> r2a, r2b;
        er_merge<N, N - M>(r1, mmap, smsk).perform(r2a);
        er_optimize<N - M>(r2a, cl1.get_table_id()).perform(r2b);
        se2.set_rule(r2b);
        params.grp2.insert(se2);

    } // Loop it1
}

} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_SE_LABEL_IMPL_H
