#include "subspace_iterator_test.h"
/*#include "test_fixtures/index_group_test_f.h"*/
/*#include <libtensor/block_sparse/subspace_iterator.h>*/

using namespace std;

namespace libtensor {

void subspace_iterator_test::perform() throw(libtest::test_exception) 
{
#if 0
    test_get_block_index_dense();
    test_incr_dense();
    test_get_block_index_sparse();
    test_incr_sparse();
    test_done_dense();
    test_done_sparse();
    test_get_slice_size_dense();
    test_get_slice_size_sparse();
#endif
}

#if 0
void subspace_iterator_test::test_get_block_index_dense() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_iterator_test::test_get_block_index_dense()";

    index_groups_test_f tf = index_groups_test_f();
    subspace_iterator si(tf.bispace,0);

    if(si.get_block_index() != 0)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace_iterator::get_block_index(...) did not return correct value for subspace 0 (dense subspace)");
    }
}

void subspace_iterator_test::test_incr_dense() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_iterator_test::test_incr_dense()";

    index_groups_test_f tf = index_groups_test_f();
    subspace_iterator si(tf.bispace,0);

    ++si;
    ++si;
    if(si.get_block_index() != 2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace_iterator::get_block_index(...) did not return correct value for subspace 0 (dense subspace) after incr");
    }
}

void subspace_iterator_test::test_get_block_index_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_iterator_test::test_get_block_index_sparse()";

    index_groups_test_f tf = index_groups_test_f();
    subspace_iterator si(tf.bispace,3);

    if(si.get_block_index() != 1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace_iterator::get_block_index(...) did not return correct value for subspace 3 (sparse subspace)");
    }
}

void subspace_iterator_test::test_incr_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_iterator_test::test_incr_sparse()";

    index_groups_test_f tf = index_groups_test_f();
    subspace_iterator si(tf.bispace,6);

    ++si;
    if(si.get_block_index() != 3)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace_iterator::get_block_index(...) did not return correct value for subspace 6 (sparse subspace) after incr");
    }
}

void subspace_iterator_test::test_done_dense() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_iterator_test::test_done_dense()";

    index_groups_test_f tf = index_groups_test_f();
    subspace_iterator si(tf.bispace,0);

    ++si;
    if(si.done())
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace_iterator::get_block_index(...) claims done when not done");
    }
    for(size_t i = 0; i < tf.bispace[0].get_n_blocks() - 1; ++i)
    {
        ++si;
    }
    if(!si.done())
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace_iterator::get_block_index(...) claims not done when done");
    }
}

void subspace_iterator_test::test_done_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_iterator_test::test_done_sparse()";

    index_groups_test_f tf = index_groups_test_f();
    subspace_iterator si(tf.bispace,6);

    ++si;
    if(si.done())
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace_iterator::get_block_index(...) claims done when not done");
    }
    ++si;
    ++si;
    if(!si.done())
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace_iterator::get_block_index(...) claims not done when done");
    }
}

void subspace_iterator_test::test_get_slice_size_dense() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_iterator_test::test_get_slice_size_dense()";

    index_groups_test_f tf = index_groups_test_f();
    subspace_iterator si(tf.bispace,0);

    ++si;
    if(si.get_slice_size() != 129600)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace_iterator::get_slice_size(...) did not return correct value for subspace 0 (dense subspace)");
    }
}

void subspace_iterator_test::test_get_slice_size_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_iterator_test::test_get_slice_size_dense()";

    index_groups_test_f tf = index_groups_test_f();
    subspace_iterator si(tf.bispace,6);

    ++si;
    if(si.get_slice_size() != 216000)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace_iterator::get_slice_size(...) did not return correct value for subspace 6 (sparse subspace)");
    }
    
    //Test edge case of last block
    ++si;
    if(si.get_slice_size() != 216000)
    {
        cout << si.get_slice_size() << "\n";
        fail_test(test_name,__FILE__,__LINE__,
                "subspace_iterator::get_slice_size(...) did not return correct value for subspace 6 (sparse subspace)");
    }
}
#endif

} // namespace libtensor
