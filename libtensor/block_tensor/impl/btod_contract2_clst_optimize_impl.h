#ifndef LIBTENSOR_BTOD_CONTRACT2_CLST_OPTIMIZE_IMPL_H
#define LIBTENSOR_BTOD_CONTRACT2_CLST_OPTIMIZE_IMPL_H

#include <libtensor/core/scalar_transf_double.h>
#include "../btod_contract2_clst_optimize.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
void btod_contract2_clst_optimize<N, M, K>::perform(contr_list &clst) {

    iterator j1 = clst.begin();
    while(j1 != clst.end()) {

        iterator j2 = j1;
        ++j2;
        bool incj1 = true;
        while(j2 != clst.end()) {

            if(!check_same_blocks(j1, j2)) {
                ++j2; continue;
            }

            contraction2<N, M, K> contr1(m_contr), contr2(m_contr);
            contr1.permute_a(j1->get_transf_a().get_perm());
            contr1.permute_b(j1->get_transf_b().get_perm());
            contr2.permute_a(j2->get_transf_a().get_perm());
            contr2.permute_b(j2->get_transf_b().get_perm());
            if(!check_same_contr(contr1, contr2)) {
                ++j2; continue;
            }

            double d1 = j1->get_transf_a().get_scalar_tr().get_coeff() *
                    j1->get_transf_b().get_scalar_tr().get_coeff();
            double d2 = j2->get_transf_a().get_scalar_tr().get_coeff() *
                    j2->get_transf_b().get_scalar_tr().get_coeff();
            if (d1 + d2 == 0) {
                j1 = clst.erase(j1);
                if(j1 == j2) {
                    j1 = j2 = clst.erase(j2);
                } else {
                    j2 = clst.erase(j2);
                }
                incj1 = false;
                break;
            } else {
                j1->get_transf_a().get_scalar_tr().reset();
                j1->get_transf_b().get_scalar_tr().reset();
                j1->get_transf_a().get_scalar_tr().scale(d1 + d2);
                j2 = clst.erase(j2);
            }
        }
        if(incj1) ++j1;
    }
}


template<size_t N, size_t M, size_t K>
inline bool btod_contract2_clst_optimize<N, M, K>::check_same_blocks(
    const iterator &i1, const iterator &i2) {

    return
        i1->get_acindex_a() == i2->get_acindex_a() &&
        i1->get_acindex_b() == i2->get_acindex_b();
}


template<size_t N, size_t M, size_t K>
bool btod_contract2_clst_optimize<N, M, K>::check_same_contr(
    const contraction2<N, M, K> &contr1,
    const contraction2<N, M, K> &contr2) {

    const sequence<2 * (N + M + K), size_t> &conn1 = contr1.get_conn(),
        &conn2 = contr2.get_conn();
    for(size_t i = 0; i < 2 * (N + M + K); i++) {
        if(conn1[i] != conn2[i]) return false;
    }
    return true;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_CLST_OPTIMIZE_IMPL_H
