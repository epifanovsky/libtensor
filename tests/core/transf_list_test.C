#include <sstream>
#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/core/transf_list.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/se_perm.h>
#include "transf_list_test.h"

namespace libtensor {


void transf_list_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5a();
    test_5b();
    test_5c();
}


namespace transf_list_test_ns {


template<size_t N>
std::ostream &operator<<(std::ostream &os, const tensor_transf<N, double> &tr) {
    os << "[" << tr.get_perm() << "; " << tr.get_scalar_tr().get_coeff() << "]";
    return os;
}


template<size_t N, typename T>
std::string trlist_compare(const char *testname, const index<N> &idx,
    const transf_list<N, T> &trlist,
    const std::list< tensor_transf<N, T> > &trlist_ref) {

    typedef std::pair<tensor_transf<N, T>, bool> trpair_t;
    std::list<trpair_t> trplist_ref;
    for(typename std::list< tensor_transf<N, T> >::const_iterator i =
        trlist_ref.begin(); i != trlist_ref.end(); i++) {
        trplist_ref.push_back(trpair_t(*i, false));
    }

    size_t n = 0;
    for(typename transf_list<N, T>::iterator i = trlist.begin();
        i != trlist.end(); i++, n++) {

        typename std::list<trpair_t>::iterator iref =
            trplist_ref.begin();
        while(iref != trplist_ref.end()) {
            if(iref->first == trlist.get_transf(i)) break;
            iref++;
        }
        if(iref == trplist_ref.end()) {
            std::ostringstream ss;
            ss << "Transformation " << trlist.get_transf(i) <<
                " for " << idx
                << " not found in reference list.";
            return ss.str();
        }
        if(iref->second) {
            std::ostringstream ss;
            ss << "Duplicate transformation "
                << trlist.get_transf(i) << " for "
                << idx << ".";
            return ss.str();
        }
        iref->second = true;
    }
    if(n != trplist_ref.size()) {
        std::ostringstream ss;
        ss << "Wrong number " << n << " != " << trplist_ref.size()
            << " (ref) of transformations listed for "
            << idx << ".";
        return ss.str();
    }
    return std::string();
}


} // namespace transf_list_test_ns
using namespace transf_list_test_ns;


/** \brief Tests transformation lists for diagonal and non-diagonal blocks
        of a two-index tensor with empty symmetry.
 **/
void transf_list_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "transf_list_test::test_1()";

    try {

    index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    mask<2> msk;
    msk[0] = true; msk[1] = true;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(msk, 1);
    symmetry<2, double> sym(bis);

    //  Reference lists

    tensor_transf<2, double> trref;
    std::list< tensor_transf<2, double> > trlist00_ref, trlist01_ref;
    trlist00_ref.push_back(trref);
    trlist01_ref.push_back(trref);

    //  Make transformation lists

    index<2> i00, i01, i10, i11;
    i10[0] = 1; i01[1] = 1;
    i11[0] = 1; i11[1] = 1;
    transf_list<2, double> trl00(sym, i00), trl01(sym, i01),
        trl10(sym, i10), trl11(sym, i11);

    //  Check against the reference

    std::string s;
    s = trlist_compare(testname, i00, trl00, trlist00_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
    s = trlist_compare(testname, i01, trl01, trlist01_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
    s = trlist_compare(testname, i10, trl10, trlist01_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
    s = trlist_compare(testname, i11, trl11, trlist00_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \brief Tests transformation lists for diagonal and non-diagonal blocks
        of a two-index tensor with S2(+) symmetry.
 **/
void transf_list_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "transf_list_test::test_2()";

    try {

    index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    mask<2> msk;
    msk[0] = true; msk[1] = true;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(msk, 1);
    symmetry<2, double> sym(bis);

    scalar_transf<double> tr0;
    se_perm<2, double> se1(permutation<2>().permute(0, 1), tr0);
    sym.insert(se1);

    //  Reference lists

    tensor_transf<2, double> trref;
    std::list< tensor_transf<2, double> > trlist00_ref, trlist01_ref;

    trref.reset();
    trlist00_ref.push_back(trref);
    trref.reset();
    trref.permute(permutation<2>().permute(0, 1));
    trlist00_ref.push_back(trref);

    trref.reset();
    trlist01_ref.push_back(trref);

    //  Make transformation lists

    index<2> i00, i01, i10, i11;
    i10[0] = 1; i01[1] = 1;
    i11[0] = 1; i11[1] = 1;
    transf_list<2, double> trl00(sym, i00), trl01(sym, i01),
        trl10(sym, i10), trl11(sym, i11);

    //  Check against the reference

    std::string s;
    s = trlist_compare(testname, i00, trl00, trlist00_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
    s = trlist_compare(testname, i01, trl01, trlist01_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
    s = trlist_compare(testname, i10, trl10, trlist01_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
    s = trlist_compare(testname, i11, trl11, trlist00_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \brief Tests transformation lists for a diagonal block of a three-index
        tensor with S3(+) symmetry (indirect relation to self).
 **/
void transf_list_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "transf_list_test::test_3()";

    try {

    index<3> i1, i2;
    i2[0] = 2; i2[1] = 2; i2[2] = 2;
    mask<3> msk;
    msk[0] = true; msk[1] = true; msk[2] = true;
    dimensions<3> dims(index_range<3>(i1, i2));
    block_index_space<3> bis(dims);
    bis.split(msk, 1);
    symmetry<3, double> sym(bis);

    scalar_transf<double> tr0;
    se_perm<3, double> se1(permutation<3>().permute(0, 1).permute(1, 2), tr0);
    se_perm<3, double> se2(permutation<3>().permute(1, 2), tr0);
    sym.insert(se1);
    sym.insert(se2);

    //  Reference lists

    tensor_transf<3, double> trref;
    std::list< tensor_transf<3, double> > trlist000_ref, trlist010_ref;

    trref.reset();
    trlist000_ref.push_back(trref);
    trref.reset();
    trref.permute(permutation<3>().permute(0, 1));
    trlist000_ref.push_back(trref);
    trref.reset();
    trref.permute(permutation<3>().permute(0, 2));
    trlist000_ref.push_back(trref);
    trref.reset();
    trref.permute(permutation<3>().permute(1, 2));
    trlist000_ref.push_back(trref);
    trref.reset();
    trref.permute(permutation<3>().permute(0, 1).permute(1, 2));
    trlist000_ref.push_back(trref);
    trref.reset();
    trref.permute(permutation<3>().permute(1, 2).permute(0, 1));
    trlist000_ref.push_back(trref);

    trref.reset();
    trlist010_ref.push_back(trref);
    trref.reset();
    trref.permute(permutation<3>().permute(0, 2));
    trlist010_ref.push_back(trref);

    //  Make transformation lists

    index<3> i000, i010;
    i010[1] = 1;
    transf_list<3, double> trl000(sym, i000), trl010(sym, i010);

    //  Check against the reference

    std::string s;
    s = trlist_compare(testname, i000, trl000, trlist000_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
    s = trlist_compare(testname, i010, trl010, trlist010_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \brief Tests transformation lists for diagonal and non-diagonal blocks
        of a two-index tensor with partition symmetry.
 **/
void transf_list_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "transf_list_test::test_4()";

    try {

    index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    mask<2> msk;
    msk[0] = true; msk[1] = true;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(msk, 1);
    symmetry<2, double> sym(bis);

    index<2> i00, i01, i10, i11;
    i10[0] = 1; i01[1] = 1;
    i11[0] = 1; i11[1] = 1;
    scalar_transf<double> tr0;

    se_part<2, double> se1(bis, msk, 2);
    se1.add_map(i00, i11, tr0);
    se1.add_map(i01, i10, tr0);
    se1.add_map(i00, i01, tr0);
    sym.insert(se1);

    //  Reference lists

    tensor_transf<2, double> trref;
    std::list< tensor_transf<2, double> > trlist00_ref, trlist01_ref;

    trref.reset();
    trlist00_ref.push_back(trref);

    trref.reset();
    trlist01_ref.push_back(trref);

    //  Make transformation lists

    transf_list<2, double> trl00(sym, i00), trl01(sym, i01),
        trl10(sym, i10), trl11(sym, i11);

    //  Check against the reference

    std::string s;
    s = trlist_compare(testname, i00, trl00, trlist00_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
    s = trlist_compare(testname, i01, trl01, trlist01_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
    s = trlist_compare(testname, i10, trl10, trlist01_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
    s = trlist_compare(testname, i11, trl11, trlist00_ref);
    if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \brief Tests transformation lists for non-diagonal blocks
        of a two-index tensor with S2 (+) and partition symmetry.
 **/
void transf_list_test::test_5a() throw(libtest::test_exception) {

    static const char *testname = "transf_list_test::test_5a()";

    try {

    index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    mask<2> msk;
    msk[0] = true; msk[1] = true;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(msk, 1);

    scalar_transf<double> tr0;

    se_perm<2, double> se(permutation<2>().permute(0, 1), tr0);

    index<2> i00, i01, i10, i11;
    i10[0] = 1; i01[1] = 1;
    i11[0] = 1; i11[1] = 1;

    se_part<2, double> sp(bis, msk, 2);
    sp.add_map(i00, i01, tr0);
    sp.add_map(i01, i10, tr0);
    sp.add_map(i10, i11, tr0);

    symmetry<2, double> sym1(bis), sym2(bis);
    sym1.insert(se); sym1.insert(sp);
    sym2.insert(sp); sym2.insert(se);

    transf_list<2, double> trl1(sym1, i01), trl2(sym2, i01);
    for (transf_list<2, double>::iterator it = trl1.begin();
            it != trl1.end(); it++) {

        const tensor_transf<2, double> &tr = trl1.get_transf(it);
        if (! trl2.is_found(tr)) {
            std::ostringstream oss;
            oss << "Transformation " << tr;
            fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
        }
    }
    for (transf_list<2, double>::iterator it = trl2.begin();
            it != trl2.end(); it++) {

        const tensor_transf<2, double> &tr = trl2.get_transf(it);
        if (! trl1.is_found(tr)) {
            std::ostringstream oss;
            oss << "Transformation " << tr;
            fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \brief Tests transformation lists for non-diagonal blocks
        of a four-index tensor with S2 x S2 (+) and partition symmetry.
 **/
void transf_list_test::test_5b() throw(libtest::test_exception) {

    static const char *testname = "transf_list_test::test_5b()";

    try {

    index<4> i1, i2;
    i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
    mask<4> msk;
    msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    bis.split(msk, 1);
    scalar_transf<double> tr0;

    se_perm<4, double> se1(permutation<4>().permute(0, 1), tr0);
    se_perm<4, double> se2(permutation<4>().permute(2, 3), tr0);

    index<4> i0000, i0001, i0010, i0011, i0100, i0101, i0110, i0111,
        i1000, i1001, i1010, i1011, i1100, i1101, i1110, i1111;
    i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
    i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
    i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
    i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
    i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
    i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
    i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
    i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

    se_part<4, double> sp(bis, msk, 2);
    sp.add_map(i0000, i0001, tr0);
    sp.add_map(i0001, i0010, tr0);
    sp.add_map(i0010, i0011, tr0);
    sp.add_map(i0011, i0100, tr0);
    sp.add_map(i0100, i0101, tr0);
    sp.add_map(i0101, i0110, tr0);
    sp.add_map(i0110, i0111, tr0);
    sp.add_map(i0111, i1000, tr0);
    sp.add_map(i1000, i1001, tr0);
    sp.add_map(i1001, i1010, tr0);
    sp.add_map(i1010, i1011, tr0);
    sp.add_map(i1011, i1100, tr0);
    sp.add_map(i1100, i1101, tr0);
    sp.add_map(i1101, i1110, tr0);
    sp.add_map(i1110, i1111, tr0);

    symmetry<4, double> sym1(bis), sym2(bis);
    sym1.insert(se1); sym1.insert(se2); sym1.insert(sp);
    sym2.insert(sp); sym2.insert(se1); sym2.insert(se2);

    transf_list<4, double> trl1(sym1, i0101), trl2(sym2, i0101);
    for (transf_list<4, double>::iterator it = trl1.begin();
            it != trl1.end(); it++) {

        const tensor_transf<4, double> &tr = trl1.get_transf(it);
        if (! trl2.is_found(tr)) {
            std::ostringstream oss;
            oss << "Transformation " << tr;
            fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
        }
    }
    for (transf_list<4, double>::iterator it = trl2.begin();
            it != trl2.end(); it++) {

        const tensor_transf<4, double> &tr = trl2.get_transf(it);
        if (! trl1.is_found(tr)) {
            std::ostringstream oss;
            oss << "Transformation " << tr;
            fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \brief Tests transformation lists for non-diagonal blocks
        of a four-index tensor with S2 x S2 (+), label and partition symmetry.
 **/
void transf_list_test::test_5c() throw(libtest::test_exception) {

    static const char *testname = "transf_list_test::test_5c()";

    { // Setup point group table
    std::vector<std::string> im(4);
    im[0] = "A1"; im[1] = "A2"; im[2] = "B1"; im[3] = "B2";
    point_group_table pt("pg", im, im[0]);
    pt.add_product(1, 1, 0);
    pt.add_product(1, 2, 3);
    pt.add_product(1, 3, 2);
    pt.add_product(2, 2, 0);
    pt.add_product(2, 3, 1);
    pt.add_product(3, 3, 0);

    product_table_container::get_instance().add(pt);
    }

    try {

    index<4> i1, i2;
    i2[0] = 8; i2[1] = 8; i2[2] = 8; i2[3] = 8;
    mask<4> msk;
    msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    bis.split(msk, 1); bis.split(msk, 2); bis.split(msk, 3);
    bis.split(msk, 4);
    bis.split(msk, 5); bis.split(msk, 6); bis.split(msk, 7);

    scalar_transf<double> tr0;
    se_perm<4, double> se1(permutation<4>().permute(0, 1), tr0);
    se_perm<4, double> se2(permutation<4>().permute(2, 3), tr0);

    se_label<4, double> sl(bis.get_block_index_dims(), "pg");
    block_labeling<4> &bl = sl.get_labeling();
    bl.assign(msk, 0, 0); bl.assign(msk, 1, 1);
    bl.assign(msk, 2, 2); bl.assign(msk, 3, 3);
    bl.assign(msk, 4, 0); bl.assign(msk, 5, 1);
    bl.assign(msk, 6, 2); bl.assign(msk, 7, 3);
    evaluation_rule<4> r1;
    sequence<4, size_t> seq(1);
    r1.add_sequence(seq);
    r1.add_product(0, 0);
    r1.add_product(0, 1);
    r1.add_product(0, 2);
    r1.add_product(0, 3);
    sl.set_rule(r1);

    index<4> i0000, i0001, i0010, i0011, i0100, i0101, i0110, i0111,
        i1000, i1001, i1010, i1011, i1100, i1101, i1110, i1111;
    i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
    i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
    i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
    i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
    i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
    i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
    i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
    i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

    se_part<4, double> sp(bis, msk, 2);
    sp.add_map(i0000, i0001, tr0);
    sp.add_map(i0001, i0010, tr0);
    sp.add_map(i0010, i0011, tr0);
    sp.add_map(i0011, i0100, tr0);
    sp.add_map(i0100, i0101, tr0);
    sp.add_map(i0101, i0110, tr0);
    sp.add_map(i0110, i0111, tr0);
    sp.add_map(i0111, i1000, tr0);
    sp.add_map(i1000, i1001, tr0);
    sp.add_map(i1001, i1010, tr0);
    sp.add_map(i1010, i1011, tr0);
    sp.add_map(i1011, i1100, tr0);
    sp.add_map(i1100, i1101, tr0);
    sp.add_map(i1101, i1110, tr0);
    sp.add_map(i1110, i1111, tr0);

    symmetry<4, double> sym1(bis), sym2(bis);
    sym1.insert(se1); sym1.insert(se2); sym1.insert(sp); sym1.insert(sl);
    sym2.insert(sl); sym2.insert(sp); sym2.insert(se1); sym2.insert(se2);

    transf_list<4, double> trl1(sym1, i0101), trl2(sym2, i0101);
    for (transf_list<4, double>::iterator it = trl1.begin();
            it != trl1.end(); it++) {

        const tensor_transf<4, double> &tr = trl1.get_transf(it);
        if (! trl2.is_found(tr)) {
            std::ostringstream oss;
            oss << "Transformation " << tr;
            fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
        }
    }
    for (transf_list<4, double>::iterator it = trl2.begin();
            it != trl2.end(); it++) {

        const tensor_transf<4, double> &tr = trl2.get_transf(it);
        if (! trl1.is_found(tr)) {
            std::ostringstream oss;
            oss << "Transformation " << tr;
            fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
        }
    }

    } catch(exception &e) {
        product_table_container::get_instance().erase("pg");
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    product_table_container::get_instance().erase("pg");
}

} // namespace libtensor
