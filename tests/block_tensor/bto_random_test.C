#include <cmath>
#include <cstdlib>
#include <ctime>
#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/bto_random.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_btconv.h>
#include <libtensor/symmetry/se_perm.h>
#include "../compare_ref.h"
#include "bto_random_test.h"

namespace libtensor {

void bto_random_test::perform() throw(libtest::test_exception){
    std::cout << "Testing bto_random_test_x<double>   ";
    bto_random_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing bto_random_test_x<float>   ";
    bto_random_test_x<float> t_float;
    t_float.perform();
}

template<>
const double bto_random_test_x<double>::k_thresh = 1e-15;

template<>
const float bto_random_test_x<float>::k_thresh = 1e-7;

template<typename T>
void bto_random_test_x<T>::perform() throw(libtest::test_exception)
{
    allocator<T>::init(4, 16, 65536, 65536);

    typedef allocator<T> allocator_t;
    typedef dense_tensor<4, T, allocator_t> tensor_t;
    typedef block_tensor<4, T, allocator_t> block_tensor_t;
    typedef block_tensor_ctrl<4, T> block_tensor_ctrl_t;

    try {

    index<4> i1, i2;
    i2[0] = 3; i2[1] = 4;    i2[2] = 3; i2[3] = 4;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> msk;
    msk[0]=true; msk[1]=false; msk[2]=true; msk[3]=false;
    bis.split(msk,2);
    msk[0]=false; msk[1]=true; msk[2]=false; msk[3]=true;
    bis.split(msk,2);

    block_tensor_t bta(bis);
    block_tensor_ctrl_t btactrl(bta);

    permutation<4> perm1, perm2;
    perm1.permute(1, 3);
    perm2.permute(0, 2);
    scalar_transf<T> tr0, tr1(-1.);
    se_perm<4, T> cycle1(perm1, tr0);
    se_perm<4, T> cycle2(perm2, tr0);

    btactrl.req_symmetry().insert(cycle1);
    btactrl.req_symmetry().insert(cycle2);

    bto_random<4, T> randr;
    randr.perform(bta);

    tensor_t ta(bta.get_bis().get_dims());
    to_btconv<4, T> conv(bta);
    conv.perform(ta);

    tensor_t tb(ta), tc(ta), td(ta);
    permutation<4> permb, permc, permd;
    permb.permute(0,2);
    permc.permute(1,3);
    permd.permute(0,2);
    permd.permute(1,3);

    to_copy<4, T>(ta, permb, 1.0).perform(true, tb);
    compare_ref_x<4, T>::compare("bto_random_test::test_permb",ta,tb,k_thresh);

    to_copy<4, T>(ta, permc, 1.0).perform(true, tc);
    compare_ref_x<4, T>::compare("bto_random_test::test_permb",ta,tc,k_thresh);

    to_copy<4, T>(ta, permd, 1.0).perform(true, td);
    compare_ref_x<4, T>::compare("bto_random_test::test_permb",ta,td,k_thresh);

    } catch(exception &exc) {
        fail_test("bto_random_test", __FILE__, __LINE__, exc.what());
        allocator<T>::shutdown();
    }
    allocator<T>::shutdown();
}

template class bto_random_test_x<double>;
template class bto_random_test_x<float>;

} // namespace libtensor
