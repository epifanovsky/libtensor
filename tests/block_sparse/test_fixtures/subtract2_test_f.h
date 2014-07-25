#ifndef SUBTRACT2_TEST_F_H
#define SUBTRACT2_TEST_F_H

#include <libtensor/block_sparse/sparse_bispace.h>
#include "util.h"

namespace libtensor {

//The subtraction operation DESTROYS sparsity in this case, because the zero blocks of the sparse tensor
//may have nonzero blocks subtracted from them
class subtract2_test_f
{
private:
    static sparse_bispace<1> init_i(void);
    static sparse_bispace<1> init_j(void);
    static sparse_bispace<1> init_k(void);
public:
    static const size_t ij_sparsity[4][2];

    static const double s_A_arr[45];
    static const double s_B_arr[60];
    static const double s_C_arr[60];

    double A_arr[45];
    double B_arr[60];
    double C_arr[60];

    sparse_bispace<1> spb_i;
    sparse_bispace<1> spb_j;
    sparse_bispace<1> spb_k;

    sparse_bispace<3> spb_A;
    sparse_bispace<3> spb_B;
    sparse_bispace<3> spb_C;

    subtract2_test_f() : spb_i(init_i()),
                         spb_j(init_j()),
                         spb_k(init_k()),
                         spb_A(spb_i % spb_j << get_sig_blocks(ij_sparsity,4) | spb_k),
                         spb_B(spb_i | spb_j | spb_k),
                         spb_C(spb_B)
    {
        memcpy(A_arr,s_A_arr,sizeof(s_A_arr));
        memcpy(B_arr,s_B_arr,sizeof(s_B_arr));
        memcpy(C_arr,s_C_arr,sizeof(s_C_arr));
    }

};

} // namespace libtensor

#endif /* SUBTRACT2_TEST_F_H */
