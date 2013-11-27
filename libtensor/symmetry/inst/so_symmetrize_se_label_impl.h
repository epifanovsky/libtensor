#ifndef LIBTENSOR_SO_SYMMETRIZE_SE_LABEL_IMPL_H
#define LIBTENSOR_SO_SYMMETRIZE_SE_LABEL_IMPL_H

#include <libtensor/core/permutation_builder.h>
#include <libtensor/core/permutation_generator.h>
#include "../bad_symmetry.h"
#include "er_optimize.h"

namespace libtensor {

template<size_t N, typename T>
const char *
symmetry_operation_impl< so_symmetrize<N, T>, se_label<N, T> >::k_clazz =
    "symmetry_operation_impl< so_symmetrize<N, T>, se_label<N, T> >";


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
             size_t k = i + nidx;
             for (register size_t j = 1; j < ngrp; j++) {
                 if (bl1.get_dim_type(map[k]) != typei)
                     throw bad_symmetry(g_ns, k_clazz, method,
                             __FILE__, __LINE__, "Incompatible dimensions.");

                 k += nidx;
             }
         }

         se_label<N, T> e2(bl1.get_block_index_dims(), e1.get_table_id());
         block_labeling<N> &bl2 = e2.get_labeling();
         transfer_labeling(bl1, idmap, bl2);

         evaluation_rule<N> r2a, r2b;
         for (typename evaluation_rule<N>::iterator ir = r1.begin();
                 ir != r1.end(); ir++) {

             const product_rule<N> &pr1 = r1.get_product(ir);

             permutation_generator<N> pg(msk);
             do {
                 const permutation<N> &p = pg.get_perm();
                 sequence<N, size_t> seq1(0), seq2(0);
                 for (register size_t i = 0; i < N; i++) seq1[i] = seq2[i] = i;

                 for (register size_t i = 0, k = 0; i < ngrp; i++) {
                     for (register size_t j = 0, kk = p[i] * nidx;
                             j < nidx; j++, k++, kk++) {
                         seq2[map[kk]] = seq1[map[k]];
                     }
                 }
                 permutation_builder<N> pb(seq2, seq1);

                 product_rule<N> &pr2 = r2a.new_product();
                 for (typename product_rule<N>::iterator ip = pr1.begin();
                         ip != pr1.end(); ip++) {

                     sequence<N, size_t> seq(pr1.get_sequence(ip));
                     pb.get_perm().apply(seq);
                     pr2.add(seq, pr1.get_intrinsic(ip));
                 }

             } while (pg.next());
         }
         er_optimize<N>(r2a, e1.get_table_id()).perform(r2b);
         e2.set_rule(r2b);

         params.grp2.insert(e2);
     }
}



} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_SE_LABEL_IMPL_H
