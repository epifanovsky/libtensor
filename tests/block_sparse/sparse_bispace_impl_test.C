#include "sparse_bispace_impl_test.h"
#include <libtensor/block_sparse/sparse_bispace_impl.h>

using namespace std;

namespace libtensor {

void sparse_bispace_impl_test::perform() throw(libtest::test_exception)
{
    test_nd_equality_true();
}


/* Tests equality operator for multidimensional block index spaces
 *
 */
void sparse_bispace_impl_test::test_nd_equality_true() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_test::test_nd_equality_true()";

    //First two
    vector<subspace> subspaces(1,subspace(5));
    vector<size_t> split_points_0; 
    split_points_0.push_back(1);
    split_points_0.push_back(3);
    subspaces[0].split(split_points_0);

    subspaces.push_back(subspace(6));
    vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    subspaces[1].split(split_points_1);

    sparse_bispace_impl spb_i_0(subspaces);
    sparse_bispace_impl spb_i_1(subspaces);

    if(!(spb_i_0 == spb_i_1))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::operator==(...) returned false");
    }
}

} // namespace libtensor
