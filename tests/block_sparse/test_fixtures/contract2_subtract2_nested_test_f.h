#ifndef CONTRACT2_SUBTRACT2_NESTED_TEST_F_H
#define CONTRACT2_SUBTRACT2_NESTED_TEST_F_H

#include <libtensor/block_sparse/sparse_bispace.h>
#include "util.h"
#include "contract2_test_f.h"

namespace libtensor {

// C = AB
// G = C - F
// E = D G^T
class contract2_subtract2_nested_test_f  : public contract2_test_f
{
private:
    static const double s_F_arr[18];
    static const double s_D_arr[21]; 
    static const double s_E_arr[18];

    static const size_t ml_sparsity[4][2];

    static sparse_bispace<1> init_m();


public:
    double F_arr[18];
    double D_arr[21]; 
    double E_arr[18];

    sparse_bispace<1> spb_m;
    sparse_bispace<2> spb_D;
    sparse_bispace<2> spb_E;

    contract2_subtract2_nested_test_f() : spb_m(init_m()),
                                          spb_D(spb_m % spb_l << get_sig_blocks(ml_sparsity,4)),
                                          spb_E(spb_m | spb_i)

                                          
    {
        memcpy(F_arr,s_F_arr,sizeof(s_F_arr));
        memcpy(D_arr,s_D_arr,sizeof(s_D_arr));
        memcpy(E_arr,s_E_arr,sizeof(s_E_arr));
    }


};

} // namespace libtensor



#endif /* CONTRACT2_SUBTRACT2_NESTED_TEST_F_H */
