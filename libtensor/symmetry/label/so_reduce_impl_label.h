#ifndef LIBTENSOR_SO_REDUCE_IMPL_LABEL_H
#define LIBTENSOR_SO_REDUCE_IMPL_LABEL_H

#include <map>
#include <set>
#include "../../defs.h"
#include "../../exception.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_reduce.h"
#include "../se_label.h"

namespace libtensor {


/**	\brief Implementation of so_reduce<N, T> for se_label<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	This implementation sets the target label to all labels.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t K, typename T>
class symmetry_operation_impl< so_reduce<N, M, K, T>, se_label<N, T> > :
public symmetry_operation_impl_base< so_reduce<N, M, K, T>, se_label<N, T> > {

public:
    static const char *k_clazz; //!< Class name
    static const size_t k_order2 = N - M + K; //!< Dimension of result

public:
    typedef so_reduce<N, M, K, T> operation_t;
    typedef se_label<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, size_t M, size_t K, typename T>
const char *symmetry_operation_impl< so_reduce<N, M, K, T>, se_label<N, T> >
::k_clazz = "symmetry_operation_impl< so_reduce<N, M, K, T>, se_label<N, T> >";

template<size_t N, size_t M, size_t K, typename T>
void symmetry_operation_impl< so_reduce<N, M, K, T>, se_label<N, T> >
::do_perform(symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter<N, T, element_t> adapter_t;
    typedef se_label<k_order2, T> el2_t;
    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_group label_group;
    typedef evaluation_rule::label_set label_set;
    typedef evaluation_rule::rule_id rule_id;
    typedef evaluation_rule::basic_rule basic_rule;

    //	Verify that the projection masks are correct
    mask<N> tm;
    size_t nm = 0;
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

    // Create a map of indexes
    sequence<N, size_t> map(0);
    for (size_t i = 0, j = 0; i < N; i++) {
        if (tm[i]) {
            for (size_t k = 0; k < K; k++) {
                if (! params.msk[k][i]) continue;

                map[i] = k_order2 + k;
            }
        }
        else {
            map[i] = j++;
        }
    }

    adapter_t g1(params.grp1);

    // Create block index dimensions of result se_label
    typename adapter_t::iterator it1 = g1.begin();
    const dimensions<N> &bidims1 =
            g1.get_elem(it1).get_labeling().get_block_index_dims();

    index<k_order2> idx1, idx2;
    for (size_t i = 0, j = 0; i < N; i++) {
        if (tm[i]) continue;
        idx2[j++] = bidims1[i] - 1;
    }

    dimensions<k_order2> bidims2(index_range<k_order2>(idx1, idx2));

#ifdef LIBTENSOR_DEBUG
    // Check the bidims for correctness
    for (size_t i = 0; i < N; i++) {
        if (bidims2[map[i]] != bidims1[i]) {
            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "bidims2");
        }
    }
#endif

    // Loop over all se_label elements and reduce dimensions in each one
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
        const block_labeling<N> &bl1 = se1.get_labeling();

        // Check if the reduced dimensions comprise all labels
        // (to allow later merges)
        label_set blk_labels[K];
        const product_table_i &pt = product_table_container::
                get_instance().req_const_table(se1.get_table_id());

        for (size_t k = 0; k < K; k++) {
            size_t i = 0;
            for (; i < N; i++) { if (params.msk[k][i]) break; }

            size_t type = bl1.get_dim_type(i);
            for (size_t j = 0; j < bl1.get_dim(type); j++) {
                blk_labels[k].insert(bl1.get_label(type, j));
            }
        }

        // Create result se_label
        el2_t se2(bidims2, se1.get_table_id());
        transfer_labeling(bl1, map, se2.get_labeling());

        // Copy evaluation rule
        const evaluation_rule &r1 = se1.get_rule();
        evaluation_rule r2;

        // Loop over all products
        std::map<rule_id, rule_id> rule_map;
        for (size_t i = 0; i < r1.get_n_products(); i++) {

            // Steps:
            // - Simplify "normal" contractions by merging products
            // - Create multiple products wherever the above simplification is
            //   not possible

            // First create sets of rules to merge for each mask
            std::multiset<rule_id> merge_set[K];
            for (evaluation_rule::product_iterator pit = r1.begin(i);
                    pit != r1.end(i); pit++) {

                rule_id rid1 = r1.get_rule_id(pit);
                const evaluation_rule::basic_rule &br1 = r1.get_rule(pit);
                bool added = false;
                for (size_t i = 0; i < br1.order.size(); i++) {

                    if (tm[br1.order[i]]) {
                        merge_set[map[br1.order[i]] - k_order2].insert(rid1);
                        added = true;
                    }
                }
                if (added) continue;

                // Those rules which do not contain any indexes that should be
                // reduced are just added to the result
                std::map<rule_id, rule_id>::iterator ir = rule_map.find(rid1);
                rule_id rid2;
                if (ir == rule_map.end()) {
                    rid2 = r2.add_rule(br1.intr, br1.order);
                    rule_map[rid1] = rid2;
                }
                else {
                    rid2 = ir->second;
                }
                if (i >= r2.get_n_products()) { r2.add_product(rid2); }
                else { r2.add_to_product(i, rid2); }
            }

            // Then collect the rules that may be merged into one
            for (size_t k = 0; k < K; k++) {
                if (merge_set[k].size() != 2 ||
                        blk_labels[k].size() != pt.nlabels()) {
                    continue;
                }

                // New rule
                evaluation_rule::basic_rule br;

                std::set<rule_id>::iterator ism = merge_set[k].begin();
                std::list<rule_id> to_merge; to_merge.push_back(*ism);
                std::set<rule_id> merged;
                do {
                    // Merge the next rule in the list to_merge
                    rule_id cur_rid = to_merge.pop_front();
                    const basic_rule &cur_br = r1.get_rule(cur_rid);

                    // Add the current rule to merged
                    merged.insert(cur_rid);

                    // Merge current order into br
                    bool merge_self = false;
                    std::set<size_t> self_idx;
                    for (size_t i = 0; i < cur_br.order.size(); i++) {
                        // Index not masked
                        if (! tm[cur_br.order[i]]) {
                            br.order.push_back(map[cur_br.order[i]]);
                            continue;
                        }

                        size_t kk = map[cur_br.order[i]] - k_order2;
                        // Cannot merge w.r.t. this index
                        if (merge_set[kk].size() != 2 ||
                                blk_labels[kk].size() != pt.nlabels()) {
                            br.order.push_back(kk + k_order2);
                            continue;
                        }

                        ism = merge_set[kk].begin();
                        if (cur_rid == *ism) ism++;
                        // Special case: A_kk
                        if (cur_rid == *ism) {
                            merge_self = true; self_idx.insert(kk);
                            continue;
                        }
                        if (merged.count(*ism) == 0) to_merge.push_back(*ism);
                    }

                    if (br.intr.empty()) {
                        br.intr = cur_br.intr;
                    }
                    else if (br.intr.size() == pt.nlabels()) {
                        continue;
                    }
                    else {
                        // Merge current intrinsic into br.intr
                        for (label_set::iterator is =
                                br.intr.begin(); is != br.intr.end(); is++) {

                            label_group lg(2);
                            lg[0] = *is;
                            for (label_set::iterator isc = cur_br.intr.begin();
                                    isc != cur_br.intr.end(); isc++) {

                                lg[1] = *isc;
                                for (label_t l = 0; l < pt.nlabels(); l++) {
                                    if (pt.is_in_product(lg, l))
                                        br.intr.insert(l);
                                }
                            }
                        }
                    }

                    // Special case
                    if (! merge_self) continue;
                    if (br.intr.size() == pt.nlabels()) continue;

                    for (std::set<size_t>::iterator ii = self_idx.begin();
                            ii != self_idx.end(); ii++) {

                        label_set self_intr;
                        for (label_set::iterator il = blk_labels[*ii].begin();
                                il != blk_labels[*ii].end(); il++) {

                            label_group lg(2, *ii);
                            for (label_t ll = 0; ll < pt.nlabels(); ll++) {
                                if (pt.is_in_product(lg, ll))
                                    self_intr.insert(ll);
                            }
                        }

                        if (self_intr.size() == pt.nlabels()) {
                            br.intr = self_intr; break;
                        }

                        for (label_set::iterator is = br.intr.begin();
                                is != br.intr.end(); is++) {

                            label_group lg(2);
                            lg[0] = *is;
                            for (label_set::iterator isc = self_intr.begin();
                                     isc != self_intr.end(); isc++) {

                                lg[1] = *isc;
                                for (label_t l = 0; l < pt.nlabels(); l++) {
                                    if (pt.is_in_product(lg, l))
                                        br.intr.insert(l);
                                }
                            }
                        }
                    }

                } while (! to_merge.empty());

                rule_id rid2 = r2.add_rule(br.intr, br.order);
                if (i >= r2.get_n_products()) { r2.add_product(rid2); }
                else { r2.add_to_product(i, rid2); }

            }

            // Now loop over product in r2 and multiply the product for those
            // indexes that have not been reduced yet.

            // TODO: finish implementing
            for (evaluation_rule::product_iterator pit = r2.begin(i);
                    pit != r2.end(i); pit++) {

                rule_id rid2 = r2.get_rule_id(pit);
                basic_rule &br2 = r2.get_rule(rid2);
                for (size_t i = 0; i < br2.order.size(); i++) {

                    if (tm[br2.order[i]]) {

                    }
                }
            }

        }

        se2.set_rule(r2);
        params.grp2.insert(se2);

    } // Loop it1
}


} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_IMPL_LABEL_H
