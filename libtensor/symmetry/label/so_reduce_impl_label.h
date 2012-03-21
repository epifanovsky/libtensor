#ifndef LIBTENSOR_SO_REDUCE_IMPL_LABEL_H
#define LIBTENSOR_SO_REDUCE_IMPL_LABEL_H

#include <map>
#include <set>
#include "../../defs.h"
#include "../../exception.h"
#include "../../core/abs_index.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_reduce.h"
#include "../se_label.h"
#include "product_table_container.h"

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
    static const size_t k_order2 = N - M; //!< Dimension of result

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

    //	Verify that the projection masks are correct
    size_t ntm = 0;
    sequence<K, size_t> nm;
    mask<N> tm;
    for (register size_t k = 0; k < K; k++) {
        const mask<N> &m = params.msk[k];
        for (register size_t i = 0; i < N; i++) {
            if (! m[i]) continue;
            if (tm[i]) {
                throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                        "params.msk[k]");
            }
            tm[i] = true;
            nm[k]++; ntm++;
        }
    }
    if(ntm != M) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "params.msk");
    }

    params.grp2.clear();
    if (params.grp1.is_empty()) return;

    // Create a map of indexes
    sequence<N, size_t> map, map2;
    for (size_t i = 0, j = 0; i < N; i++) {
        if (tm[i]) {
            for (size_t k = 0; k < K; k++) {
                if (! params.msk[k][i]) continue;

                map[i] = k_order2 + k;
                map2[i] = (size_t) -1;
            }
        }
        else {
            map[i] = map2[i] = j++;
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

    // Loop over all se_label elements and do the reduction in each one
    for (; it1 != g1.end(); it1++) {

#ifdef LIBTENSOR_DEBUG
        // Check that the block index dimensions are alright.
        if (bidims1 !=
                g1.get_elem(it1).get_labeling().get_block_index_dims()) {
            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Incompatible se_labels in input.");
        }
#endif

        const element_t &se1 = g1.get_elem(it1);
        const block_labeling<N> &bl1 = se1.get_labeling();

        // Create result se_label
        el2_t se2(bidims2, se1.get_table_id());
        transfer_labeling(bl1, map2, se2.get_labeling());

        // Collect the labels along each dimension that will be reduced!
        bool has_invalid[K];
        std::vector<product_table_i::label_t> blk_labels[K];
        for (size_t k = 0; k < K; k++) {

            product_table_i::label_set_t ls;
            has_invalid[k] = false;

            size_t i = 0;
            for (; i < N; i++) { if (params.msk[k][i]) break; }
            if (i == N) continue;

            size_t type = bl1.get_dim_type(i);
            for (size_t j = 0; j < bl1.get_dim(type); j++) {
                product_table_i::label_t ll = bl1.get_label(type, j);
                if (ll == product_table_i::k_invalid) {
                    has_invalid[k] = true; break;
                }
                ls.insert(ll);
            }

            for (product_table_i::label_set_t::const_iterator ii = ls.begin();
                    ii != ls.end(); ii++) {
                blk_labels[k].push_back(*ii);
            }
        }

        // Copy evaluation rule
        const evaluation_rule<N> &r1 = se1.get_rule();
        evaluation_rule<k_order2> r2;

        std::map<size_t, size_t> m1to2; // map of seq in 1 to seq in 2
        std::set<size_t> m1to0; // seq's with only internal indexes
        std::set<size_t> m1tox; // seq's with internal indexes that are
        // marked by has_invalid
        std::multimap<size_t, size_t> seq2k; // map which seq contains which k
        for (size_t sno = 0; sno < r1.get_n_sequences(); sno++) {

            const sequence<N, size_t> &seq1 = r1[sno];
            sequence<k_order2, size_t> seq2(0);
            bool nodims = true, all_allowed = false;
            for (register size_t i = 0; i < N; i++) {
                if (seq1[i] == 0) continue;
                if (map[i] < k_order2) {
                    nodims = false;
                    seq2[map[i]] = seq1[i];
                }
                else {
                    size_t k = map[i] - k_order2;
                    if (has_invalid[k]) {
                        all_allowed = true;
                        break;
                    }
                    seq2k.insert(
                            std::multimap<size_t,size_t>::value_type(sno, k));
                }
            }

            if (nodims) {
                m1to0.insert(sno); continue;
            }

            if (all_allowed) {
                m1tox.insert(sno);
                seq2k.erase(sno);
                continue;
            }

            m1to2[sno] = r2.add_sequence(seq2);
        }

        // Subsequently, the product table is required
        const product_table_i &pt = product_table_container::get_instance().
                req_const_table(se1.get_table_id());

        // Loop over the products
        product_table_i::label_t n_labels = pt.get_n_labels();
        for (size_t pno = 0; pno < r1.get_n_products(); pno++) {

            // Loop over all terms in current product and find the reduction
            // indexes present
            index<K> i1, i2;
            std::vector<size_t> seq2nos;
            std::vector<bool> has_internal;
            std::vector<typename evaluation_rule<N>::iterator> seq1to0, seq1to2;

            bool has_allowed = false;
            for (typename evaluation_rule<N>::iterator itp = r1.begin(pno);
                    itp != r1.end(pno); itp++) {

                size_t sno = r1.get_seq_no(itp);
                if (m1tox.count(sno) != 0) {
                    has_allowed = true; continue;
                }
                if (m1to0.count(sno) != 0) {
                    seq1to0.push_back(itp); continue;
                }

                seq1to2.push_back(itp);
                seq2nos.push_back(m1to2[sno]);
                has_internal.push_back(seq2k.count(sno) > 0);

                if (! has_internal.back()) continue;

                for (std::multimap<size_t, size_t>::const_iterator its =
                        seq2k.lower_bound(sno);
                        its != seq2k.upper_bound(sno); its++) {
                    i2[its->second] = blk_labels[its->second].size() - 1;
                }
            }

            // The current product is all allowed, i.e. the rule is all allowed.
            if (seq2nos.size() == 0 && has_allowed) {
                // set flag to handle this outside the loop
                break;
            }

            // No loop over all possible index combinations of the reduced
            // indexes and create possible combinations of result labels
            dimensions<K> rdims(index_range<K>(i1, i2));

            product_table_i::label_set_t tot_intr;
            abs_index<K> rx(rdims);
            do {
                const index<K> &idx = rx.get_index();

                // Loop over terms w/o external indexes and check,
                // if they are allowed. If not skip the index
                bool not_allowed = false;
                for (size_t sno = 0; sno < seq1to0.size(); sno++) {

                    typename evaluation_rule<N>::iterator itp = seq1to0[sno];
                    const sequence<N, size_t> &seq = r1.get_sequence(itp);

                    product_table_i::label_group_t lg;
                    for (size_t i = 0; i < N; i++) {
                        if (seq[i] == 0) continue;

                        size_t k = map[i] - k_order2;
                        lg.insert(lg.end(), seq[i], blk_labels[k][idx[k]]);
                    }
                    lg.push_back(r1.get_intrinsic(itp));

                    if (! pt.is_in_product(lg, r1.get_target(itp))) {
                        not_allowed = true; break;
                    }
                }
                if (not_allowed) continue;

                // Loop over all other rules
                product_table_i::label_set_t cur_intr1, cur_intr2;
                cur_intr1.insert(0);
                product_table_i::label_set_t *pi1 = &cur_intr1,
                        *pi2 = &cur_intr2;
                product_table_i::label_t nlx = 1;
                for (size_t sno = 0; sno < seq1to2.size(); sno++) {

                    // Skip those w/o internal indexes
                    if (! has_internal[sno]) continue;

                    typename evaluation_rule<N>::iterator itp = seq1to2[sno];
                    const sequence<N, size_t> &seq = r1.get_sequence(itp);

                    product_table_i::label_group_t lg;
                    for (size_t i = 0; i < N; i++) {
                        if (seq[i] == 0 || map[i] < k_order2) continue;

                        size_t k = map[i] - k_order2;
                        lg.insert(lg.end(), seq[i], blk_labels[k][idx[k]]);
                    }
                    lg.push_back(r1.get_intrinsic(itp));

                    product_table_i::label_set_t ls = pt.product(lg);

                    for (product_table_i::label_set_t::const_iterator ils =
                            ls.begin(); ils != ls.end(); ils++) {

                        product_table_i::label_t lx = *ils * nlx;
                        for (product_table_i::label_set_t::const_iterator it =
                                pi1->begin(); it != pi1->end(); it++) {

                            pi2->insert(*it + lx);
                        }
                    }
                    std::swap(pi1, pi2);
                    pi2->clear();

                    nlx *= n_labels;
                }

                tot_intr.insert(pi1->begin(), pi1->end());

            } while (rx.inc());

            for (product_table_i::label_set_t::const_iterator it =
                    tot_intr.begin(); it != tot_intr.end(); it++) {

                size_t pno = -1;
                size_t j = 0;
                product_table_i::label_t ll = *it;
                if (has_internal[j]) {
                    pno = r2.add_product(seq2nos[j], ll % n_labels,
                            r1.get_target(seq1to2[j]));
                    ll = ll / n_labels;
                }
                else {
                    pno = r2.add_product(seq2nos[j],
                            r1.get_intrinsic(seq1to2[j]),
                            r1.get_target(seq1to2[j]));
                }
                j++;

                for (; j < seq1to2.size(); j++) {
                    if (has_internal[j]) {
                        r2.add_to_product(pno, seq2nos[j],
                                ll % n_labels, r1.get_target(seq1to2[j]));
                        ll = ll / n_labels;
                    }
                    else {
                        r2.add_to_product(pno, seq2nos[j],
                                r1.get_intrinsic(seq1to2[j]),
                                r1.get_target(seq1to2[j]));
                    }
                }
            }
        }

        // Return the product table
        product_table_container::get_instance().ret_table(se1.get_table_id());

        se2.set_rule(r2);
        params.grp2.insert(se2);

    } // Loop it1
}


} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_IMPL_LABEL_H
