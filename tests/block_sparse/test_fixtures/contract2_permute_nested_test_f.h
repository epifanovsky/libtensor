#ifndef CONTRACT2_PERMUTE_NESTED_TEST_F_H
#define CONTRACT2_PERMUTE_NESTED_TEST_F_H

#include <libtensor/block_sparse/sparse_bispace.h>
#include "util.h"
#include "contract2_test_f.h"
#include <string.h>

namespace libtensor {

class contract2_permute_nested_test_f  : public contract2_test_f
{
private:
    static sparse_bispace<1> init_i(void);
    static sparse_bispace<1> init_j(void);
    static sparse_bispace<1> init_k(void);
    static sparse_bispace<1> init_l(void);

public:
    static const double s_D_arr[18];
    double D_arr[18];
    sparse_bispace<2> spb_D;

    contract2_permute_nested_test_f() : spb_D(spb_l | spb_i)
    {
        memcpy(D_arr,s_D_arr,sizeof(s_D_arr));
    }
};



} // namespace libtensor


#endif /* CONTRACT2_PERMUTE_NESTED_TEST_F_H */
