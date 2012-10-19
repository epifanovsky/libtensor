#ifndef LIBTENSOR_BTOD_CONTRACT2_CLST_OPTIMIZE_H
#define LIBTENSOR_BTOD_CONTRACT2_CLST_OPTIMIZE_H

#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_contract2_clst.h>

namespace libtensor {

/** \brief Optimizes the contraction block list

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, size_t M, size_t K>
class btod_contract2_clst_optimize {
public:
    typedef typename gen_bto_contract2_clst<N, M, K, double>::list_type
            contr_list;
public:
    void perform(contr_list &clst);
};


template<size_t N, size_t M, size_t K>
void btod_contract2_clst_optimize<N, M, K>::perform(contr_list &clst) {

    typename contr_list::iterator j1 = clst.begin();
    while(j1 != clst.end()) {

        typename contr_list::iterator j2 = j1;
        ++j2;
        bool incj1 = true;
        while(j2 != clst.end()) {
            if (j1->get_abs_index_a() != j2->get_abs_index_a() ||
                    j1->get_abs_index_b() != j2->get_abs_index_b()) {
                ++j2; continue;
            }
            if (! j1->get_transf_a().get_perm().equals(
                    j2->get_transf_a().get_perm()) ||
                    ! j1->get_transf_b().get_perm().equals(
                            j2->get_transf_b().get_perm())) {
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

} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_CLST_OPTIMIZE_H
