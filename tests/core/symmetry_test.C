#include <set>
#include <typeinfo>
#include <libtensor/core/symmetry.h>
#include "symmetry_test.h"

namespace libtensor {


void symmetry_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
}


namespace symmetry_test_ns {


class symelem1 : public symmetry_element_i<4, double> {
public:
    static const char *k_typ;
private:
    size_t n;
public:
    symelem1(size_t n_) : n(n_) { }
    symelem1(const symelem1 &e) : n(e.n) { }
    virtual ~symelem1() { }
    virtual const char *get_type() const { return k_typ; }
    virtual symmetry_element_i<4, double> *clone() const {
        return new symelem1(*this);
    }
    virtual const mask<4> &get_mask() const { throw 0; }
    virtual void permute(const permutation<4>&) { throw 0; }
    virtual bool is_valid_bis(const block_index_space<4>&) const {
        return true;
    }
    virtual bool is_allowed(const index<4>&) const { throw 0; }
    virtual void apply(index<4>&) const { throw 0; }
    virtual void apply(index<4>&, tensor_transf<4, double>&) const { throw 0; }
    size_t get_n() const { return n; }
};


class symelem2 : public symmetry_element_i<4, double> {
public:
    static const char *k_typ;
private:
    size_t m;
public:
    symelem2(size_t m_) : m(m_) { }
    symelem2(const symelem2 &e) : m(e.m) { }
    virtual ~symelem2() { }
    virtual const char *get_type() const { return k_typ; }
    virtual symmetry_element_i<4, double> *clone() const {
        return new symelem2(*this);
    }
    virtual const mask<4> &get_mask() const { throw 0; }
    virtual void permute(const permutation<4>&) { throw 0; }
    virtual bool is_valid_bis(const block_index_space<4>&) const {
        return true;
    }
    virtual bool is_allowed(const index<4>&) const { throw 0; }
    virtual void apply(index<4>&) const { throw 0; }
    virtual void apply(index<4>&, tensor_transf<4, double>&) const { throw 0; }
    size_t get_m() const { return m; }
};


const char *symelem1::k_typ = "symelem1";
const char *symelem2::k_typ = "symelem2";


} // namespace symmetry_test_ns
using namespace symmetry_test_ns;


/** \test Tests that a new symmetry doesn't contain any subsets
 **/
void symmetry_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "symmetry_test::test_1()";

    typedef symmetry<4, double> symmetry_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
    block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

    symmetry_t sym(bis);
    symmetry_t::iterator i = sym.begin();
    if(i != sym.end()) {
        fail_test(testname, __FILE__, __LINE__, "i != sym.end()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests the insertion of one symmetry element
 **/
void symmetry_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "symmetry_test::test_2()";

    typedef symmetry<4, double> symmetry_t;
    typedef symmetry_element_set<4, double> symmetry_element_set_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
    block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

    symmetry_t sym(bis);
    symelem1 elem1(22);
    sym.insert(elem1);
    symmetry_t::iterator i = sym.begin();
    if(i == sym.end()) {
        fail_test(testname, __FILE__, __LINE__, "i == sym.end()");
    }
    if(sym.get_subset(i).get_id().compare(symelem1::k_typ) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Bad symmetry id.");
    }

    const symmetry_element_set_t &subset1 = sym.get_subset(i);
    symmetry_element_set_t::const_iterator ii = subset1.begin();
    if(ii == subset1.end()) {
        fail_test(testname, __FILE__, __LINE__, "ii == subset1.end()");
    }
    try {
        const symelem1 &elem1i = dynamic_cast<const symelem1&>(
            subset1.get_elem(ii));
        if(elem1i.get_n() != 22) {
            fail_test(testname, __FILE__, __LINE__, "Bad elem1i.");
        }
    } catch(std::bad_cast &e) {
        fail_test(testname, __FILE__, __LINE__, "bad_cast for elem1i");
    }

    ii++;
    if(ii != subset1.end()) {
        fail_test(testname, __FILE__, __LINE__, "ii != subset1.end()");
    }

    i++;
    if(i != sym.end()) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected only one subset.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests the insertion of two symmetry element of the same type
 **/
void symmetry_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "symmetry_test::test_3()";

    typedef symmetry<4, double> symmetry_t;
    typedef symmetry_element_set<4, double> symmetry_element_set_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
    block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

    symmetry_t sym(bis);
    symelem1 elem1(22), elem2(33);
    sym.insert(elem1);
    sym.insert(elem2);
    symmetry_t::iterator i = sym.begin();
    if(i == sym.end()) {
        fail_test(testname, __FILE__, __LINE__, "i == sym.end()");
    }
    if(sym.get_subset(i).get_id().compare(symelem1::k_typ) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Bad symmetry id.");
    }

    std::set<size_t> refset;
    refset.insert(22); refset.insert(33);

    const symmetry_element_set_t &subset1 = sym.get_subset(i);
    symmetry_element_set_t::const_iterator ii = subset1.begin();
    if(ii == subset1.end()) {
        fail_test(testname, __FILE__, __LINE__,
            "Unexpected end of subset1 (1).");
    }
    try {
        const symelem1 &elem1i = dynamic_cast<const symelem1&>(
            subset1.get_elem(ii));
        std::set<size_t>::iterator j = refset.find(elem1i.get_n());
        if(j == refset.end()) {
            fail_test(testname, __FILE__, __LINE__, "Bad elem1i.");
        } else {
            refset.erase(j);
        }
    } catch(std::bad_cast &e) {
        fail_test(testname, __FILE__, __LINE__, "bad_cast for elem1i");
    }

    ii++;
    if(ii == subset1.end()) {
        fail_test(testname, __FILE__, __LINE__,
            "Unexpected end of subset1 (2).");
    }
    try {
        const symelem1 &elem2i = dynamic_cast<const symelem1&>(
            subset1.get_elem(ii));
        std::set<size_t>::iterator j = refset.find(elem2i.get_n());
        if(j == refset.end()) {
            fail_test(testname, __FILE__, __LINE__, "Bad elem2i.");
        } else {
            refset.erase(j);
        }
    } catch(std::bad_cast &e) {
        fail_test(testname, __FILE__, __LINE__, "bad_cast for elem2i");
    }

    ii++;
    if(ii != subset1.end()) {
        fail_test(testname, __FILE__, __LINE__, "ii != subset1.end()");
    }

    i++;
    if(i != sym.end()) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected only one subset.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests the insertion of two symmetry element of different types
 **/
void symmetry_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "symmetry_test::test_4()";

    typedef symmetry<4, double> symmetry_t;
    typedef symmetry_element_set<4, double> symmetry_element_set_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
    block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

    symmetry_t sym(bis);
    symelem1 elem1(22);
    symelem2 elem2(33);
    sym.insert(elem1);
    sym.insert(elem2);

    const symmetry_element_set_t *subset1_ptr = 0, *subset2_ptr = 0;
    symmetry_t::iterator i = sym.begin();
    for(size_t is = 0; is < 2; is++) {    
        if(i == sym.end()) {
            fail_test(testname, __FILE__, __LINE__,
                "Unexpected end of sym.");
        }
        if(sym.get_subset(i).get_id().compare(symelem1::k_typ) == 0) {
            if(subset1_ptr != 0) {
                fail_test(testname, __FILE__, __LINE__,
                    "Duplicate subset1.");
            }
            subset1_ptr = &sym.get_subset(i);
        }
        if(sym.get_subset(i).get_id().compare(symelem2::k_typ) == 0) {
            if(subset2_ptr != 0) {
                fail_test(testname, __FILE__, __LINE__,
                    "Duplicate subset2.");
            }
            subset2_ptr = &sym.get_subset(i);
        }
        i++;
    }
    if(i != sym.end()) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected only two subsets.");
    }
    if(subset1_ptr == 0 || subset2_ptr == 0) {
        fail_test(testname, __FILE__, __LINE__, "Bad subset found.");
    }

    const symmetry_element_set_t &subset1 = *subset1_ptr;
    const symmetry_element_set_t &subset2 = *subset2_ptr;

    symmetry_element_set_t::const_iterator ii1 = subset1.begin();
    if(ii1 == subset1.end()) {
        fail_test(testname, __FILE__, __LINE__,
            "Unexpected end of subset1.");
    }
    try {
        const symelem1 &elem1i = dynamic_cast<const symelem1&>(
            subset1.get_elem(ii1));
        if(elem1i.get_n() != 22) {
            fail_test(testname, __FILE__, __LINE__, "Bad elem1i.");
        }
    } catch(std::bad_cast &e) {
        fail_test(testname, __FILE__, __LINE__, "bad_cast for elem1i");
    }
    ii1++;
    if(ii1 != subset1.end()) {
        fail_test(testname, __FILE__, __LINE__, "ii1 != subset1.end()");
    }

    symmetry_element_set_t::const_iterator ii2 = subset2.begin();
    if(ii2 == subset2.end()) {
        fail_test(testname, __FILE__, __LINE__,
            "Unexpected end of subset2.");
    }
    try {
        const symelem2 &elem2i = dynamic_cast<const symelem2&>(
            subset2.get_elem(ii2));
        if(elem2i.get_m() != 33) {
            fail_test(testname, __FILE__, __LINE__, "Bad elem2i.");
        }
    } catch(std::bad_cast &e) {
        fail_test(testname, __FILE__, __LINE__, "bad_cast for elem2i");
    }
    ii2++;
    if(ii2 != subset2.end()) {
        fail_test(testname, __FILE__, __LINE__, "ii2 != subset2.end()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests the insertion of four symmetry element of two different
        types
 **/
void symmetry_test::test_5() throw(libtest::test_exception) {

    static const char *testname = "symmetry_test::test_5()";

    typedef symmetry<4, double> symmetry_t;
    typedef symmetry_element_set<4, double> symmetry_element_set_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
    block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

    symmetry_t sym(bis);
    symelem1 elem1(22);
    symelem2 elem2(44);
    symelem1 elem3(33);
    symelem2 elem4(55);
    sym.insert(elem1);
    sym.insert(elem2);
    sym.insert(elem3);
    sym.insert(elem4);

    const symmetry_element_set_t *subset1_ptr = 0, *subset2_ptr = 0;
    symmetry_t::iterator i = sym.begin();
    for(size_t is = 0; is < 2; is++) {    
        if(i == sym.end()) {
            fail_test(testname, __FILE__, __LINE__,
                "Unexpected end of sym.");
        }
        if(sym.get_subset(i).get_id().compare(symelem1::k_typ) == 0) {
            if(subset1_ptr != 0) {
                fail_test(testname, __FILE__, __LINE__,
                    "Duplicate subset1.");
            }
            subset1_ptr = &sym.get_subset(i);
        }
        if(sym.get_subset(i).get_id().compare(symelem2::k_typ) == 0) {
            if(subset2_ptr != 0) {
                fail_test(testname, __FILE__, __LINE__,
                    "Duplicate subset2.");
            }
            subset2_ptr = &sym.get_subset(i);
        }
        i++;
    }
    if(i != sym.end()) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected only two subsets.");
    }
    if(subset1_ptr == 0 || subset2_ptr == 0) {
        fail_test(testname, __FILE__, __LINE__, "Bad subset found.");
    }

    std::set<size_t> refset1, refset2;
    refset1.insert(22); refset1.insert(33);
    refset2.insert(44); refset2.insert(55);

    const symmetry_element_set_t &subset1 = *subset1_ptr;
    const symmetry_element_set_t &subset2 = *subset2_ptr;

    symmetry_element_set_t::const_iterator ii1 = subset1.begin();
    if(ii1 == subset1.end()) {
        fail_test(testname, __FILE__, __LINE__,
            "Unexpected end of subset1 (1).");
    }
    try {
        const symelem1 &elem1i = dynamic_cast<const symelem1&>(
            subset1.get_elem(ii1));
        std::set<size_t>::iterator j = refset1.find(elem1i.get_n());
        if(j == refset1.end()) {
            fail_test(testname, __FILE__, __LINE__, "Bad elem1i.");
        } else {
            refset1.erase(j);
        }
    } catch(std::bad_cast &e) {
        fail_test(testname, __FILE__, __LINE__, "bad_cast for elem1i");
    }
    ii1++;
    if(ii1 == subset1.end()) {
        fail_test(testname, __FILE__, __LINE__,
            "Unexpected end of subset1 (2).");
    }
    try {
        const symelem1 &elem2i = dynamic_cast<const symelem1&>(
            subset1.get_elem(ii1));
        std::set<size_t>::iterator j = refset1.find(elem2i.get_n());
        if(j == refset1.end()) {
            fail_test(testname, __FILE__, __LINE__, "Bad elem2i.");
        } else {
            refset1.erase(j);
        }
    } catch(std::bad_cast &e) {
        fail_test(testname, __FILE__, __LINE__, "bad_cast for elem2i");
    }
    ii1++;
    if(ii1 != subset1.end()) {
        fail_test(testname, __FILE__, __LINE__, "ii1 != subset1.end()");
    }

    symmetry_element_set_t::const_iterator ii2 = subset2.begin();
    if(ii2 == subset2.end()) {
        fail_test(testname, __FILE__, __LINE__,
            "Unexpected end of subset2 (1).");
    }
    try {
        const symelem2 &elem3i = dynamic_cast<const symelem2&>(
            subset2.get_elem(ii2));
        std::set<size_t>::iterator j = refset2.find(elem3i.get_m());
        if(j == refset2.end()) {
            fail_test(testname, __FILE__, __LINE__, "Bad elem3i.");
        } else {
            refset2.erase(j);
        }
    } catch(std::bad_cast &e) {
        fail_test(testname, __FILE__, __LINE__, "bad_cast for elem3i");
    }
    ii2++;
    if(ii2 == subset2.end()) {
        fail_test(testname, __FILE__, __LINE__,
            "Unexpected end of subset2 (2).");
    }
    try {
        const symelem2 &elem4i = dynamic_cast<const symelem2&>(
            subset2.get_elem(ii2));
        std::set<size_t>::iterator j = refset2.find(elem4i.get_m());
        if(j == refset2.end()) {
            fail_test(testname, __FILE__, __LINE__, "Bad elem4i.");
        } else {
            refset2.erase(j);
        }
    } catch(std::bad_cast &e) {
        fail_test(testname, __FILE__, __LINE__, "bad_cast for elem4i");
    }
    ii2++;
    if(ii2 != subset2.end()) {
        fail_test(testname, __FILE__, __LINE__, "ii2 != subset2.end()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
