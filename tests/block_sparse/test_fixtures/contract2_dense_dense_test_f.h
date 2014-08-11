#ifndef CONTRACT2_DENSE_DENSE_TEST_F_H
#define CONTRACT2_DENSE_DENSE_TEST_F_H

#include <libtensor/block_sparse/sparse_bispace.h>

namespace libtensor {

//Tensors are stored block-major for this test
//dimensions: i = 3,j = 4, k = 5,l = 6
//Contraction takes the form of A*B
//C(i|j|l) = A(i|j|k) B(k|l) 
class contract2_dense_dense_test_f 
{
protected:
    static sparse_bispace<1> init_i(void);
    static sparse_bispace<1> init_j(void);
    static sparse_bispace<1> init_k(void);
    static sparse_bispace<1> init_l(void);

    static const double s_A_arr[60];
    static const double s_B_arr[30];
    static const double s_C_correct_arr[72];
public:


    double A_arr[60];
    double B_arr[30];
    double C_correct_arr[72];

    sparse_bispace<1> spb_i;
    sparse_bispace<1> spb_j;
    sparse_bispace<1> spb_k;
    sparse_bispace<1> spb_l;
    sparse_bispace<3> spb_A;
    sparse_bispace<2> spb_B;
    sparse_bispace<3> spb_C;

    contract2_dense_dense_test_f() : spb_i(init_i()),
                                     spb_j(init_j()),
                                     spb_k(init_k()),
                                     spb_l(init_l()),
                                     spb_A(spb_i|spb_j|spb_k),
                                     spb_B(spb_k|spb_l),
                                     spb_C(spb_i|spb_j|spb_l)

    {
        memcpy(A_arr,s_A_arr,sizeof(s_A_arr));
        memcpy(B_arr,s_B_arr,sizeof(s_B_arr));
        memcpy(C_correct_arr,s_C_correct_arr,sizeof(s_C_correct_arr));
    }
};



} // namespace libtensor


#endif /* CONTRACT2_DENSE_DENSE_TEST_F_H */
