#ifndef CONTRACT2_PERMUTE_NESTED_TEST_F_H
#define CONTRACT2_PERMUTE_NESTED_TEST_F_H

#include <libtensor/block_sparse/sparse_bispace.h>
#include "util.h"
#include "contract2_test_f.h"

namespace libtensor {

class contract2_permute_nested_test_f  : public contract2_test_f
{
private:
    static const double s_D_arr[18];
public:
    double D_arr[18];
    sparse_bispace<2> spb_D;

    contract2_permute_nested_test_f() : spb_D(spb_l | spb_i)
    {
        memcpy(D_arr,s_D_arr,sizeof(s_D_arr));
    }
};



} // namespace libtensor


#endif /* CONTRACT2_PERMUTE_NESTED_TEST_F_H */
