#ifndef LIBTENSOR_SO_SYMMETRIZE_SE_LABEL_IMPL_H
#define LIBTENSOR_SO_SYMMETRIZE_SE_LABEL_IMPL_H

#include <libtensor/core/permutation_generator.h>
#include "../bad_symmetry.h"

namespace libtensor {

template<size_t N, typename T>
const char *
symmetry_operation_impl< so_symmetrize<N, T>, se_label<N, T> >::k_clazz =
    "symmetry_operation_impl< so_symmetrize<N, T>, se_perm<N, T> >";


template<size_t N, typename T>
void
symmetry_operation_impl< so_symmetrize<N, T>, se_label<N, T> >::do_perform(
    symmetry_operation_params_t &params) const {

    static const char *method =
        "do_perform(const symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter< N, T, se_label<N, T> > adapter_t;

    adapter_t g1(params.grp1);
    params.grp2.clear();

    size_t ngrp = 0, nidx = 0;
     for (register size_t i = 0; i < N; i++) {
         ngrp = std::max(ngrp, params.idxgrp[i]);
         nidx = std::max(nidx, params.symidx[i]);
     }
     if (ngrp < 2) return;

     // Mask for permutation generator: permuted are only index groups
     mask<N> msk;
     for (register size_t i = ngrp; i < N; i++) msk[i] = true;

     // Maps for symmetrization and transfer of block labels.
     sequence<N, size_t> map, idmap;
     for (register size_t i = 0; i < N; i++) {
         idmap[i] = i;
         if (params.idxgrp[i] == 0) continue;
         map[(params.idxgrp[i] - 1) * nidx + params.symidx[i] - 1] = i;
     }

     for(typename adapter_t::iterator it = g1.begin(); it != g1.end(); it++) {

         const se_label<N, T> &e1 = g1.get_elem(it);
         const block_labeling<N> &bl1 = e1.get_labeling();
         const evaluation_rule<N> &r1 = e1.get_rule();

         for (register size_t i = 0; i < nidx; i++) {

             size_t typei = bl1.get_dim_type(map[i]);
             for (register size_t j = 1, k = nidx; j < ngrp; j++, k+=nidx) {
                 if (bl1.get_dim_type(k) != typei)
                     throw bad_symmetry(g_ns, k_clazz, method,
                             __FILE__, __LINE__, "Incompatible dimensions.");
             }
         }

         se_label<N, T> e2(bl1.get_block_index_dims(), e1.get_table_id());
         block_labeling<N> &bl2 = e2.get_labeling();
         transfer_labeling(bl1, idmap, bl2);

         evaluation_rule<N> r2;
         // Symmetrize the sequences
         std::vector< std::vector<size_t> > perm_seq(r1.get_n_sequences());

         // Add the existing sequences to r2
         for (size_t sno = 0; sno < r1.get_n_sequences(); sno++) {
             perm_seq[sno].push_back(r2.add_sequence(r1[sno]));
         }

         // Now generate all permuted sequences
         size_t nperm = 1;
         permutation_generator<N> pg(msk);
         permutation<N> pprev, px;
         while (pg.next()) {
             // Determine pair which was permuted in this step
             const permutation<N> &p = pg.get_perm();
             register size_t i = 0, i0, i1;
             for (; i < N && p[i] == pprev[i]; i++) ;
             i0 = i++;
             for (; i < N && p[i] == pprev[i]; i++) ;
             i1 = i;
             pprev.permute(i0, i1);

             // Construct index permutation for this step
             i0 *= nidx; i1 *= nidx;
             for (i = 0; i < nidx; i++, i0++, i1++)
                 px.permute(map[i0], map[i1]);

             // Create permuted sequences
             for (size_t sno = 0; sno < r1.get_n_sequences(); sno++) {

                 const sequence<N, size_t> &seq = r1[sno];
                 sequence<N, size_t> pseq(seq);
                 px.apply(pseq);
                 for (i = 0; i < N; i++) { if (pseq[i] != seq[i]) break; }
                 if (i == N) {
                     perm_seq[sno].push_back(perm_seq[sno][0]);
                 }
                 else {
                     perm_seq[sno].push_back(r2.add_sequence(pseq));
                 }
             }
             nperm++;
         }

         // Symmetrize the products
         for (size_t pno = 0; pno < r1.get_n_products(); pno++) {

             // Product to be symmetrized
             typename evaluation_rule<N>::iterator it = r1.begin(pno);
             size_t iperm = 0;
             size_t ip = r2.add_product(perm_seq[r1.get_seq_no(it)][iperm],
                     r1.get_intrinsic(it), r1.get_target(it));
             it++;
             for (; it != r1.end(pno); it++) {
                 r2.add_to_product(ip, perm_seq[r1.get_seq_no(it)][iperm],
                         r1.get_intrinsic(it), r1.get_target(it));
             }
             iperm++;
             for ( ; iperm != nperm; iperm++) {
                 // Does the permuted product differ from the original?
                 for (it = r1.begin(pno); it != r1.end(pno); it++) {
                     if (perm_seq[r1.get_seq_no(it)][iperm] !=
                             perm_seq[r1.get_seq_no(it)][0]) break;
                 }
                 // If not continue
                 if (it == r1.end(pno)) continue;

                 it = r1.begin(pno);
                 ip = r2.add_product(perm_seq[r1.get_seq_no(it)][iperm],
                         r1.get_intrinsic(it), r1.get_target(it));
                 it++;
                 for (; it != r1.end(pno); it++) {
                     r2.add_to_product(ip, perm_seq[r1.get_seq_no(it)][iperm],
                             r1.get_intrinsic(it), r1.get_target(it));
                 }
             }
         }

         e2.set_rule(r2);

         params.grp2.insert(e2);
     }
}



} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_SE_LABEL_IMPL_H
