#include <sstream>
#include <libtensor/dense_tensor/to_contract2_dims.h>
#include "to_contract2_dims_test.h"

namespace libtensor {


void to_contract2_dims_test::perform() throw(libtest::test_exception) {

    test_ij_i_j(1, 1);
    test_ij_i_j(5, 7);
    test_ij_j_i(1, 1);
    test_ij_j_i(5, 7);

    test_ij_ik_jk(1, 1, 1);
    test_ij_ik_jk(5, 6, 7);
    test_ij_ik_jk(3, 3, 4);
}


void to_contract2_dims_test::test_ij_i_j(size_t ni, size_t nj)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_contract2_dims_test::test_ij_i_j(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    try {

    index<1> ia1, ia2;
    ia2[0] = ni - 1;
    dimensions<1> dimsa(index_range<1>(ia1, ia2));

    index<1> ib1, ib2;
    ib2[0] = nj - 1;
    dimensions<1> dimsb(index_range<1>(ib1, ib2));

    index<2> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dimsc(index_range<2>(ic1, ic2));

    contraction2<1, 1, 0> contr;

    to_contract2_dims<1, 1, 0> tocd(contr, dimsa, dimsb);
    if(!tocd.get_dimsc().equals(dimsc)) {
        fail_test(tn.c_str(), __FILE__, __LINE__, "Bad dimsc.");
    }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void to_contract2_dims_test::test_ij_j_i(size_t ni, size_t nj)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_contract2_dims_test::test_ij_j_i(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    try {

    index<1> ia1, ia2;
    ia2[0] = nj - 1;
    dimensions<1> dimsa(index_range<1>(ia1, ia2));

    index<1> ib1, ib2;
    ib2[0] = ni - 1;
    dimensions<1> dimsb(index_range<1>(ib1, ib2));

    index<2> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dimsc(index_range<2>(ic1, ic2));

    permutation<2> permc;
    permc.permute(0, 1);
    contraction2<1, 1, 0> contr(permc);

    to_contract2_dims<1, 1, 0> tocd(contr, dimsa, dimsb);
    if(!tocd.get_dimsc().equals(dimsc)) {
        fail_test(tn.c_str(), __FILE__, __LINE__, "Bad dimsc.");
    }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void to_contract2_dims_test::test_ij_ik_jk(size_t ni, size_t nj, size_t nk)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_contract2_dims_test::test_ij_ik_jk(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    try {

    index<2> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nk - 1;
    dimensions<2> dimsa(index_range<2>(ia1, ia2));

    index<2> ib1, ib2;
    ib2[0] = nj - 1; ib2[1] = nk - 1;
    dimensions<2> dimsb(index_range<2>(ib1, ib2));

    index<2> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dimsc(index_range<2>(ic1, ic2));

    contraction2<1, 1, 1> contr;
    contr.contract(1, 1);

    to_contract2_dims<1, 1, 1> tocd(contr, dimsa, dimsb);
    if(!tocd.get_dimsc().equals(dimsc)) {
        fail_test(tn.c_str(), __FILE__, __LINE__, "Bad dimsc.");
    }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

