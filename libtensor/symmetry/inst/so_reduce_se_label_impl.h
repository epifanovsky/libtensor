#ifndef LIBTENSOR_SO_REDUCE_SE_LABEL_IMPL_H
#define LIBTENSOR_SO_REDUCE_SE_LABEL_IMPL_H

#include <map>
#include <set>
#include <libtensor/defs.h>
#include <libtensor/core/abs_index.h>
#include "../bad_symmetry.h"
#include "../product_table_container.h"
#include "combine_label.h"
#include "er_reduce.h"
#include "er_optimize.h"

namespace libtensor {

template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_reduce<N, M, T>, se_label<N - M, T> >::k_clazz =
        "symmetry_operation_impl< so_reduce<N, M, T>, se_label<N - M, T> >";

template<size_t N, size_t M, typename T>
void
symmetry_operation_impl< so_reduce<N, M, T>, se_label<N - M, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter<k_order1, T, el1_t> adapter_t;
    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_group_t label_group_t;
    typedef product_table_i::label_set_t label_set_t;

    params.g2.clear();
    if (params.g1.is_empty()) return;

    // Create a map of remaining indexes
    sequence<k_order1, size_t> map(0), rmap(0);
    size_t nrsteps = 0;
    for (register size_t i = 0, j = 0; i < k_order1; i++) {
        if (params.msk[i]) {
            map[i] = size_t(-1);
            rmap[i] = params.rseq[i] + k_order2;
            nrsteps = std::max(nrsteps, params.rseq[i]);
        }
        else {
            rmap[i] = map[i] = j++;
        }
    }
    nrsteps++;

    adapter_t g1(params.g1);

    // Create block index dimensions of result se_label
    typename adapter_t::iterator it1 = g1.begin();
    const dimensions<N> &bidims1 =
            g1.get_elem(it1).get_labeling().get_block_index_dims();

    index<k_order2> idx1, idx2;
    for (size_t i = 0, j = 0; i < k_order1; i++) {
        if (params.msk[i]) {
#ifdef LIBTENSOR_DEBUG
            if (params.rblrange.get_end()[i] >= bidims1[i]) {
                throw bad_parameter(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "rblrange.");
            }
#endif
        }
        else {
            idx2[j++] = bidims1[i] - 1;
        }
    }
    dimensions<k_order2> bidims2(index_range<k_order2>(idx1, idx2));

    // Loop over all se_label elements and do the reduction in each one
    std::set<std::string> table_ids;
    for (; it1 != g1.end(); it1++) {

        const el1_t &se1 = g1.get_elem(it1);

#ifdef LIBTENSOR_DEBUG
        // Check that the block index dimensions are alright.
        if (bidims1 != se1.get_labeling().get_block_index_dims()) {
            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Incompatible se_labels in input.");
        }
#endif

        if (table_ids.count(se1.get_table_id()) != 0) continue;

        // Combine all sym elements with the same product table
        combine_label<k_order1, T> cl1(se1);
        table_ids.insert(cl1.get_table_id());

        { // Look for other se_label elements with the same table id
            typename adapter_t::iterator it2 = it1; it2++;
            for (; it2 != g1.end(); it2++) {
                const el1_t &se1b = g1.get_elem(it2);
                if (se1b.get_table_id() != cl1.get_table_id()) continue;

                cl1.add(se1b);
            }
        }

        // Create result se_label
        el2_t se2(bidims2, cl1.get_table_id());

        // Transfer the labeling
        const block_labeling<k_order1> &bl1 = cl1.get_labeling();
        transfer_labeling(bl1, map, se2.get_labeling());

        // Obtain the product table
        const product_table_i &pt = product_table_container::get_instance().
                req_const_table(cl1.get_table_id());
        label_t n_labels = pt.get_n_labels();

        // Collect the labels for each reduction step!
        sequence<M, label_group_t> blk_labels;
        for (size_t i = 0; i < nrsteps; i++) {

            // Find first dimensions of the current reduction step
            size_t j = 0;
            for (; j < k_order1; j++) {
                if (params.msk[j] && params.rseq[j] == i) break;
            }

            // Create a list of labels for the current reduction step
            size_t itype = bl1.get_dim_type(j);
            size_t imin = params.rblrange.get_begin()[j];
            size_t imax = params.rblrange.get_end()[j] + 1;

            bool is_complete = (imin == 0 && imax == bidims1[j]);
            bool has_invalid = false;

            // Compare all dimensions in the current reduction step w.r.t itype
            j++;
            for (; j < k_order1; j++) {
                if (! params.msk[j] || params.rseq[j] != i) continue;

                size_t jtype = bl1.get_dim_type(j);
                if (itype == jtype) continue;

                for (size_t k = imin; k < imax; k++) {
                    if (bl1.get_label(itype, k) == bl1.get_label(jtype, k)) {
                        continue;
                    }
                    if (bl1.get_label(itype, k) == product_table_i::k_invalid ||
                            bl1.get_label(jtype, k) == product_table_i::k_invalid) {
                        has_invalid = true;
                    }
                    else {
#ifdef LIBTENSOR_DEBUG
                        throw bad_symmetry(g_ns, k_clazz, method, __FILE__,
                                __LINE__, "Incompatible block labels.");
#endif
                    }
                }
            }

            // Transfer the vector of block labels into a sorted vector of
            // unique labels
            label_group_t &cbl = blk_labels[i];
            if (is_complete || has_invalid) {
                for (label_t l = 0; l < n_labels; l++) cbl.push_back(l);
            }
            else {
                label_set_t ls;
                for (size_t k = imin; k < imax; k++) {
                    ls.insert(bl1.get_label(itype, k));
                }
                for (label_set_t::const_iterator is = ls.begin();
                        is != ls.end(); is++) {
                    cbl.push_back(*is);
                }
            }
        }

        // Return the product table
        product_table_container::get_instance().ret_table(cl1.get_table_id());

        // Copy evaluation rule
        const evaluation_rule<N> &r1 = cl1.get_rule();
        evaluation_rule<k_order2> r2a, r2b;
        er_reduce<N, M>(r1, rmap, blk_labels, cl1.get_table_id()).perform(r2a);
        er_optimize<N - M>(r2a, cl1.get_table_id()).perform(r2b);
        se2.set_rule(r2b);
        params.g2.insert(se2);

    } // Loop it1
}

} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_SE_LABEL_IMPL_H
