//TODO: remove block_loop.h,block_contract2_kernel.h
#include <libtensor/block_sparse/block_loop.h>
#include <libtensor/block_sparse/block_contract2_kernel.h>
#include <libtensor/block_sparse/block_loop_new.h>
#include <sstream>
#include "block_loop_test.h" 

namespace libtensor {

void block_loop_test::perform() throw(libtest::test_exception) {

    test_set_subspace_looped_invalid_bispace_idx();
    test_set_subspace_looped_invalid_subspace_idx();
    test_set_subspace_looped_not_matching_subspaces();

    test_get_subspace_looped_invalid_bispace_idx();
    test_get_subspace_looped();

    test_is_bispace_ignored_invalid_bispace_idx();
    test_is_bispace_ignored();
}

void block_loop_test::test_set_subspace_looped_invalid_bispace_idx()
		throw (libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_set_subspace_looped_invalid_bispace_idx()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    std::vector< sparse_bispace_any_order > bispaces(2,spb_1|spb_2);
    block_loop_new bl(bispaces);

    //Fails because there is no third bispace
    bool threw_exception = false;
    try
    {
		bl.set_subspace_looped(3,1);
    }
    catch(out_of_bounds&)
    {
    	threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::set_subspace_looped(...) did not throw exception when invalid bispace index specified");
    }
}

void block_loop_test::test_set_subspace_looped_invalid_subspace_idx()
		throw (libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_set_subspace_looped_invalid_bispace_idx()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    std::vector< sparse_bispace_any_order > bispaces(2,spb_1|spb_2);

    block_loop_new bl(bispaces);

    //Fails because 2nd subspace has no 3rd subspace
    bool threw_exception = false;
    try
    {
		bl.set_subspace_looped(1,2);
    }
    catch(out_of_bounds&)
    {
    	threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::set_subspace_looped(...) did not throw exception when invalid subspace index specified");
    }
}

void block_loop_test::test_set_subspace_looped_not_matching_subspaces()
		throw (libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_set_subspace_looped_not_matching_subspaces()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    std::vector< sparse_bispace_any_order > bispaces(2,spb_1|spb_2);

    block_loop_new bl(bispaces);

    //Dimension 4
    bl.set_subspace_looped(0,0);
    //Fails because loop is accessing incompatible subspaces of the two bispaces
    bool threw_exception = false;
    try
    {
    	//Dimension 5
		bl.set_subspace_looped(1,1);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::set_subspace_looped(...) did not throw exception when two incompatible subspace indices specified");
    }
}

void block_loop_test::test_get_subspace_looped_invalid_bispace_idx()
		throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_get_subspace_looped_invalid_bispace_idx()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(6);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    split_points_3.push_back(4);
    spb_3.split(split_points_3);

    std::vector< sparse_bispace_any_order > bispaces;
    bispaces.push_back(spb_1|spb_2|spb_3);
    bispaces.push_back(spb_2|spb_3);
    bispaces.push_back(spb_2);

    //This loop ignores the second bispace
    block_loop_new bl(bispaces);
    bl.set_subspace_looped(0,1);
    bl.set_subspace_looped(2,0);

    //Out of bounds, only 3 bispaces, should fail
    bool threw_exception = false;
    try
    {
    	bl.get_subspace_looped(3);
    }
    catch(out_of_bounds&)
    {
    	threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::get_subspace_looped(...) did not throw exception when bispace requested out of bounds");
    }

    //Second bispace is not looped, should throw exception
    threw_exception = false;
    try
    {
    	bl.get_subspace_looped(1);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::get_subspace_looped(...) did not throw exception when bispace requested not looped over");
    }
}

void block_loop_test::test_get_subspace_looped()
		throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_get_subspace_looped()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(6);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    split_points_3.push_back(4);
    spb_3.split(split_points_3);

    std::vector< sparse_bispace_any_order > bispaces;
    bispaces.push_back(spb_1|spb_2|spb_3);
    bispaces.push_back(spb_2|spb_3);
    bispaces.push_back(spb_2);

    //This loop ignores the second bispace
    block_loop_new bl(bispaces);
    bl.set_subspace_looped(0,1);
    bl.set_subspace_looped(2,0);

    if(bl.get_subspace_looped(0) != 1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::get_subspace_looped(...) returned incorrect value");
    }
}

void block_loop_test::test_is_bispace_ignored_invalid_bispace_idx()
		throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_is_bispace_ignored_invalid_bispace_idx()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(6);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    split_points_3.push_back(4);
    spb_3.split(split_points_3);

    std::vector< sparse_bispace_any_order > bispaces;
    bispaces.push_back(spb_1|spb_2|spb_3);
    bispaces.push_back(spb_2|spb_3);
    bispaces.push_back(spb_2);

    //This loop ignores the second bispace
    block_loop_new bl(bispaces);
    bl.set_subspace_looped(0,1);
    bl.set_subspace_looped(2,0);

    //Only three bispaces should fail
    bool threw_exception = false;
    try
    {
    	bl.is_bispace_ignored(3);
    }
    catch(out_of_bounds)
    {
    	threw_exception = true;
    }

    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::is_bispace_ignored(...) did not throw exception when bispace index out of bounds");
    }
}

void block_loop_test::test_is_bispace_ignored()
		throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_is_bispace_ignored()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(6);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    split_points_3.push_back(4);
    spb_3.split(split_points_3);

    std::vector< sparse_bispace_any_order > bispaces;
    bispaces.push_back(spb_1|spb_2|spb_3);
    bispaces.push_back(spb_2|spb_3);
    bispaces.push_back(spb_2);

    //This loop ignores the second bispace
    block_loop_new bl(bispaces);
    bl.set_subspace_looped(0,1);
    bl.set_subspace_looped(2,0);

    if(!bl.is_bispace_ignored(1))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::is_bispace_ignored(...) did not return true for ignored bispace");
    }

    if(bl.is_bispace_ignored(0))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::is_bispace_ignored(...) did returned true for non-ignored bispace");
    }
}

} // namespace libtensor
