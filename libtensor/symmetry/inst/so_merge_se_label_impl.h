#ifndef LIBTENSOR_SO_MERGE_SE_LABEL_IMPL_H
#define LIBTENSOR_SO_MERGE_SE_LABEL_IMPL_H

#include <libtensor/defs.h>
#include "../bad_symmetry.h"
#include "../combine_label.h"
#include "../product_table_container.h"

namespace libtensor {

template<size_t N, size_t M, size_t NM, typename T>
const char *
symmetry_operation_impl< so_merge<N, M, T>, se_label<NM, T> >::k_clazz =
        "symmetry_operation_impl< so_merge<N, M, T>, se_label<N - M, T> >";

template<size_t N, size_t M, size_t NM, typename T>
void
symmetry_operation_impl< so_merge<N, M, T>, se_label<NM, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method = "do_perform(symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter<N, T, el1_t> adapter1_t;

    params.grp2.clear();
    if (params.grp1.is_empty()) return;

    // Create some necessary index maps
    sequence<N, size_t> map, mapx((size_t) -1);
    sequence<M, size_t> mmap, msteps(0);
    for (register size_t i = 0, j = 0; i < N; i++) {
        if (params.msk[i]) {
            if (msteps[params.mseq[i]] == 0) mmap[params.mseq[i]] = j++;
            msteps[params.mseq[i]]++;
            map[i] = mmap[params.mseq[i]];
        }
        else {
            map[i] = mapx[i] = j++;
        }
    }

    adapter1_t g1(params.grp1);

    // Create block index dimensions of result se_label
    typename adapter1_t::iterator it1 = g1.begin();
    const dimensions<N> &bidims1 =
            g1.get_elem(it1).get_labeling().get_block_index_dims();

    index<N - M> idx1, idx2;
    for (size_t i = 0; i < N; i++) {
        idx2[map[i]] = bidims1[i] - 1;
    }
    dimensions<N - M> bidims2(index_range<N - M>(idx1, idx2));

    // Loop over all se_label elements and merge dimensions in each one
    std::set<std::string> id_done;
    for (; it1 != g1.end(); it1++) {

        const el1_t &se1 = g1.get_elem(it1);
        if (id_done.count(se1.get_table_id()) != 0) continue;

#ifdef LIBTENSOR_DEBUG
        // This should never happen!!!
        if (bidims1 != se1.get_labeling().get_block_index_dims()) {
            throw bad_symmetry(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Incompatible se_labels in input.");
        }
#endif


        // Collect all label objects with the same product table
        combine_label<N, T> cl1(se1);
        id_done.insert(cl1.get_table_id());
        typename adapter1_t::iterator it2 = it1; it2++;
        for (; it2 != g1.end(); it2++) {
            const el1_t &se1b = g1.get_elem(it2);
            if (se1b.get_table_id() != cl1.get_table_id()) continue;
            cl1.add(se1b);
        }

        const block_labeling<N> &bl1 = cl1.get_labeling();
        const evaluation_rule<N> &r1 = cl1.get_rule();

        // Create the result
        el2_t se2(bidims2, cl1.get_table_id());
        block_labeling<N - M> &bl2 = se2.get_labeling();
        evaluation_rule<N - M> r2;

        // Loop over all sequences and determine which of the merged indexes
        // are active
        mask<N> active;
        for (size_t sno = 0; sno < r1.get_n_sequences(); sno++) {
            for (register size_t i = 0; i < N; i++) {
                if (! params.msk[i]) continue;
                active[i] = active[i] || (r1[sno][i] != 0);
            }
        }

        const product_table_i &pt = product_table_container::get_instance().
                req_const_table(se1.get_table_id());

        // Now loop over all merge steps and determine how to transfer the
        // merged block labelings
        mask<N> rm;
        for (size_t i = 0; i < M && msteps[i] != 0; i++) {

            register size_t j = 0;
            for (; j < N; j++) {
                if (params.msk[j] && active[j] && params.mseq[j] == i) break;
            }
            if (j == N) {
                rm[i] = true; continue;
            }

            size_t j0 = j, typei = bl1.get_dim_type(j);
            j++;
            for (; j < N; j++) {
                if (! (params.msk[j] && active[j] && params.mseq[j] == i))
                    continue;

                size_t typej = bl1.get_dim_type(j);
                if (typej == typei) continue;

                register size_t k = 0;
                for (; k < bl1.get_dim(typej); k++) {
                    if (bl1.get_label(typej, k) != bl1.get_label(typei, k))
                        break;
                }
                if (k != bl1.get_dim(typej)) break;
            }
            if (j != N) {
                rm[i] = true; continue;
            }

            mapx[j0] = mmap[i];

            for (j = 0; j < bl1.get_dim(typei); j++) {
                product_table_i::label_t l = bl1.get_label(typei, j);
                if (l == product_table_i::k_invalid ||
                        l == product_table_i::k_identity) continue;

                product_table_i::label_set_t ls = pt.product(l, l);
                if (ls.size() != 1 ||
                        *(ls.begin()) != product_table_i::k_identity) break;
            }
            rm[i] = (j == bl1.get_dim(typei));
        }

        product_table_container::get_instance().ret_table(se1.get_table_id());

        // Transfer the labeling of the remaining dimensions
        transfer_labeling(bl1, mapx, bl2);

        // Copy evaluation rule
        std::vector<size_t> m1to2(r1.get_n_sequences(), 0);
        for (size_t sno = 0; sno < r1.get_n_sequences(); sno++) {

            const sequence<N, size_t> &seq1 = r1[sno];
            sequence<N - M, size_t> seq2(0);
            for (register size_t i = 0; i < N; i++) seq2[map[i]] += seq1[i];

            for (register size_t i = 0; i < M && msteps[i] != 0; i++) {
                if (rm[i]) seq2[mmap[i]] %= 2;
            }
            size_t nidx = 0;
            for (register size_t i = 0; i < N - M; i++) nidx += seq2[i];
            if (nidx == 0) m1to2[sno] = (size_t) -1;
            else m1to2[sno] = r2.add_sequence(seq2);
        }

        size_t pno = 0;
        for (; pno < r1.get_n_products(); pno++) {
            // First look for all-forbidden rules in product
            typename evaluation_rule<N>::iterator ip = r1.begin(pno);
            for (; ip != r1.end(pno); ip++) {
                if (r1.get_intrinsic(ip) == product_table_i::k_invalid)
                    continue;

                if ((r1.get_target(ip) == product_table_i::k_invalid) ||
                        ((m1to2[r1.get_seq_no(ip)] == (size_t) -1) &&
                                (r1.get_intrinsic(ip) != r1.get_target(ip))))
                    break;
            }
            // Skip product if there is an all-forbidden rule
            if (ip != r1.end(pno)) continue;

            // Then look for all-allowed rules in product
            ip = r1.begin(pno);
            for (; ip != r1.end(pno); ip++) {
                if ((m1to2[r1.get_seq_no(ip)] != (size_t) -1) &&
                        (r1.get_intrinsic(ip) != product_table_i::k_invalid))
                    break;
            }
            // Only all-allowed rules in product
            if (ip == r1.end(pno)) break;

            size_t cno = r2.add_product(m1to2[r1.get_seq_no(ip)],
                    r1.get_intrinsic(ip), r1.get_target(ip));
            ip++;
            for (; ip != r1.end(pno); ip++) {
                if ((m1to2[r1.get_seq_no(ip)] == (size_t) -1) ||
                        (r1.get_intrinsic(ip) == product_table_i::k_invalid))
                    continue;

                r2.add_to_product(cno, m1to2[r1.get_seq_no(ip)],
                        r1.get_intrinsic(ip), r1.get_target(ip));
            }
        }
        if (pno != r1.get_n_products())
            se2.set_rule(product_table_i::k_invalid);
        else
            se2.set_rule(r2);


        params.grp2.insert(se2);

    } // Loop it1
}

} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_SE_LABEL_IMPL_H
