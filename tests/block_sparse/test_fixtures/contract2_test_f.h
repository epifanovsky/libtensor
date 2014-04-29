#ifndef CONTRACT2_TEST_F_H
#define CONTRACT2_TEST_F_H

#include <libtensor/block_sparse/sparse_bispace.h>
#include "util.h"
#include <string.h>

namespace libtensor {

class contract2_test_f 
{
private:
    static sparse_bispace<1> init_i(void);
    static sparse_bispace<1> init_j(void);
    static sparse_bispace<1> init_k(void);
    static sparse_bispace<1> init_l(void);
public:
    static const size_t ij_sparsity[4][2];
    static const size_t kl_sparsity[4][2];

    static const double s_A_arr[45];
    static const double s_B_arr[60];
    static const double s_C_arr[18];

    double A_arr[45];
    double B_arr[60];
    double C_arr[18];

    sparse_bispace<1> spb_i;
    sparse_bispace<1> spb_j;
    sparse_bispace<1> spb_k;
    sparse_bispace<1> spb_l;
    sparse_bispace<3> spb_A;
    sparse_bispace<3> spb_B;
    sparse_bispace<2> spb_C;

    contract2_test_f() : spb_i(init_i()),
                         spb_j(init_j()),
                         spb_k(init_k()),
                         spb_l(init_l()),
                         spb_A(spb_i % spb_j << get_sig_blocks(ij_sparsity,4) | spb_k),
                         spb_B(spb_j | spb_k % spb_l << get_sig_blocks(kl_sparsity,4)),
                         spb_C(spb_i|spb_l)

    {
        memcpy(A_arr,s_A_arr,sizeof(s_A_arr));
        memcpy(B_arr,s_B_arr,sizeof(s_B_arr));
        memcpy(C_arr,s_C_arr,sizeof(s_C_arr));
    }
};



} // namespace libtensor


#endif /* CONTRACT2_TEST_F_H */
