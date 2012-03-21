#ifndef LIBTENSOR_SO_MERGE_SE_LABEL_IMPL_H
#define LIBTENSOR_SO_MERGE_SE_LABEL_IMPL_H

#include <libtensor/defs.h>
#include "../bad_symmetry.h"
#include "../product_table_container.h"

namespace libtensor {

template<size_t N, size_t M, size_t K, typename T>
const char *
symmetry_operation_impl< so_merge<N, M, K, T>, se_label<N, T> >::k_clazz =
        "symmetry_operation_impl< so_merge<N, M, K, T>, se_label<N, T> >";

template<size_t N, size_t M, size_t K, typename T>
void
symmetry_operation_impl< so_merge<N, M, K, T>, se_label<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter<N, T, element_t> adapter_t;
    typedef se_label<k_order2, T> el2_t;

    //	Verify that the total projection mask is correct
    size_t ntm = 0;
    sequence<K, size_t> nm;
    mask<N> tm;
    for (register size_t k = 0; k < K; k++) {
        const mask<N> &m = params.msk[k];
        for (register size_t i = 0; i < N; i++) {
            if (! m[i]) continue;
            if (tm[i]) {
                throw bad_parameter(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "params.msk[k]");
            }
            tm[i] = true;
            nm[k]++; ntm++;
        }
    }
    if(ntm != M) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "params.tmsk");
    }

    params.grp2.clear();
    if (params.grp1.is_empty()) return;

    // Create some necessary index maps
    sequence<N, size_t> map;
    sequence<K, size_t> mapk;
    for (size_t i = 0, j = 0; i < N; i++) {
        // Check if this is the first masked index in one of the masks
        if (tm[i]) {
            size_t k = 0;
            for (; k < K; k++) { if (params.msk[k][i]) break; }
            const mask<N> &m = params.msk[k];

            size_t ii = 0;
            for (; ii < i; ii++) {  if (m[ii]) break; }
            if (ii != i) { map[i] = map[ii]; continue; }

            mapk[k] = j;
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
        const block_labeling<N> &bl1 = se1.get_labeling();

        // First check if any of the merges is reducible
        const product_table_i &pt = product_table_container::get_instance().
                req_const_table(se1.get_table_id());
        mask<K> rm;
        for (size_t k = 0; k < K; k++) {
            if (nm[k] < 2) continue;

            size_t type = bl1.get_dim_type(mapk[k]);
            size_t i = 0;
            for (; i < bl1.get_dim(type); i++) {
                product_table_i::label_t l = bl1.get_label(type, i);
                if (l == product_table_i::k_invalid ||
                        l == product_table_i::k_identity) continue;

                product_table_i::label_set_t ls = pt.product(l, l);
                if (ls.size() != 1 ||
                        *(ls.begin()) != product_table_i::k_identity) break;
            }

            rm[k] = (i == bl1.get_dim(type));
        }
        product_table_container::get_instance().ret_table(se1.get_table_id());

        // Create the result
        el2_t se2(bidims2, se1.get_table_id());
        // Copy the block labels
        transfer_labeling(bl1, map, se2.get_labeling());

        // Copy evaluation rule
        const evaluation_rule<N> r1 = se1.get_rule();

        evaluation_rule<k_order2> r2;
        std::vector<size_t> m1to2(r1.get_n_sequences(), 0);
        for (size_t i = 0; i < r1.get_n_sequences(); i++) {

            const sequence<N, size_t> &seq1 = r1[i];
            sequence<k_order2, size_t> seq2(0);
            for (register size_t j = 0; j < N; j++) seq2[map[j]] += seq1[j];

            for (register size_t k = 0; k < K; k++) {
                if (rm[k]) seq2[mapk[k]] %= 2;
            }
            size_t nidx = 0;
            for (register size_t j = 0; j < k_order2; j++) nidx += seq2[j];
            if (nidx == 0) m1to2[i] = (size_t) -1;
            else m1to2[i] = r2.add_sequence(seq2);
        }

        for (size_t i = 0; i < r1.get_n_products(); i++) {

            typename evaluation_rule<N>::iterator ip = r1.begin(i);
            size_t pno = r2.add_product(m1to2[r1.get_seq_no(ip)],
                    r1.get_intrinsic(ip), r1.get_target(ip));
            ip++;
            for (; ip != r1.end(i); ip++) {
                r2.add_to_product(pno, m1to2[r1.get_seq_no(ip)],
                        r1.get_intrinsic(ip), r1.get_target(ip));
            }
        }
        r2.optimize();

        se2.set_rule(r2);
        params.grp2.insert(se2);

    } // Loop it1
}

} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_SE_LABEL_IMPL_H
