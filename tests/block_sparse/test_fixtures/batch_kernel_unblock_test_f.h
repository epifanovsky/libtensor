#ifndef BATCH_KERNEL_UNBLOCK_TEST_F_H
#define BATCH_KERNEL_UNBLOCK_TEST_F_H

#include "contract2_dense_dense_test_f.h"

namespace libtensor {

class batch_kernel_unblock_test_f
{
private:
    static const double s_correct_A_unblocked_arr_0[60];
    static const double s_correct_A_unblocked_arr_1[60];
    static const double s_correct_A_unblocked_arr_2[60];
    static const double s_correct_A_unblocked_arr_0_0_0[20];
    static const double s_correct_A_unblocked_arr_0_0_1[40];
    static const double s_correct_A_unblocked_arr_1_1_0[30];
    static const double s_correct_A_unblocked_arr_1_1_1[30];
    static const double s_correct_A_unblocked_arr_2_2_0[24];
    static const double s_correct_A_unblocked_arr_2_2_1[36];
    static const double s_correct_A_unblocked_arr_0_2_1[40];
    static const double s_correct_A_unblocked_arr_2_0_0[24];
    static const double s_correct_A_unblocked_arr_2_0_1[36];
    static const double s_A_arr_0_1[40];
    static const double s_A_arr_1_0[30];
    static const double s_A_arr_2_1[36];
public:
    sparse_bispace<3> spb_A;
    double A_arr[60];
    double correct_A_unblocked_arr_0[60];
    double correct_A_unblocked_arr_1[60];
    double correct_A_unblocked_arr_2[60];
    double correct_A_unblocked_arr_0_0_0[20];
    double correct_A_unblocked_arr_0_0_1[40];
    double correct_A_unblocked_arr_1_1_0[30];
    double correct_A_unblocked_arr_1_1_1[30];
    double correct_A_unblocked_arr_2_2_0[24];
    double correct_A_unblocked_arr_2_2_1[36];
    double correct_A_unblocked_arr_0_2_1[40];
    double correct_A_unblocked_arr_2_0_0[24];
    double correct_A_unblocked_arr_2_0_1[36];
    double A_arr_0_1[40];
    double A_arr_1_0[30];
    double A_arr_2_1[36];

    batch_kernel_unblock_test_f() : spb_A(contract2_dense_dense_test_f().spb_A)
    {
        memcpy(A_arr,contract2_dense_dense_test_f().A_arr,sizeof(A_arr));
        memcpy(correct_A_unblocked_arr_0,s_correct_A_unblocked_arr_0,sizeof(correct_A_unblocked_arr_0));
        memcpy(correct_A_unblocked_arr_1,s_correct_A_unblocked_arr_1,sizeof(correct_A_unblocked_arr_1));
        memcpy(correct_A_unblocked_arr_2,s_correct_A_unblocked_arr_2,sizeof(correct_A_unblocked_arr_2));
        memcpy(correct_A_unblocked_arr_0_0_0,s_correct_A_unblocked_arr_0_0_0,sizeof(correct_A_unblocked_arr_0_0_0));
        memcpy(correct_A_unblocked_arr_0_0_1,s_correct_A_unblocked_arr_0_0_1,sizeof(correct_A_unblocked_arr_0_0_1));
        memcpy(correct_A_unblocked_arr_1_1_0,s_correct_A_unblocked_arr_1_1_0,sizeof(correct_A_unblocked_arr_1_1_0));
        memcpy(correct_A_unblocked_arr_1_1_1,s_correct_A_unblocked_arr_1_1_1,sizeof(correct_A_unblocked_arr_1_1_1));
        memcpy(correct_A_unblocked_arr_2_2_0,s_correct_A_unblocked_arr_2_2_0,sizeof(correct_A_unblocked_arr_2_2_0));
        memcpy(correct_A_unblocked_arr_2_2_1,s_correct_A_unblocked_arr_2_2_1,sizeof(correct_A_unblocked_arr_2_2_1));
        memcpy(correct_A_unblocked_arr_0_2_1,s_correct_A_unblocked_arr_0_2_1,sizeof(correct_A_unblocked_arr_0_2_1));
        memcpy(correct_A_unblocked_arr_2_0_0,s_correct_A_unblocked_arr_2_0_0,sizeof(correct_A_unblocked_arr_2_0_0));
        memcpy(correct_A_unblocked_arr_2_0_1,s_correct_A_unblocked_arr_2_0_1,sizeof(correct_A_unblocked_arr_2_0_1));
        memcpy(A_arr_0_1,s_A_arr_0_1,sizeof(A_arr_0_1));
        memcpy(A_arr_1_0,s_A_arr_1_0,sizeof(A_arr_1_0));
        memcpy(A_arr_2_1,s_A_arr_2_1,sizeof(A_arr_2_1));
    }
};

} // namespace libtensor

#endif /* BATCH_KERNEL_UNBLOCK_TEST_F_H */
