#include <algorithm>
#include <memory>
#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/expr/node_product.h>
#include "node_product_test.h"

namespace libtensor {


void node_product_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_4();
}


namespace {

class nodep : public expr::node_product {
public:
    nodep(
        size_t n,
        const std::vector<size_t> &idx,
        const std::vector<size_t> &cidx = std::vector<size_t>()) :
        expr::node_product("nodep", n, idx, cidx)
    { }

    virtual expr::node *clone() const {
        return new nodep(*this);
    }

};

} // unnamed namespace


void node_product_test::test_1() {

    static const char testname[] = "node_product_test::test_1()";

    try {

    size_t idx[] = { 0, 1, 0, 1 };
    size_t oidx[] = { 0, 1 };
    std::vector<size_t> vidx(idx, idx + sizeof(idx) / sizeof(size_t));
    std::vector<size_t> vidx_ref(vidx);
    std::vector<size_t> vcidx_ref;
    std::vector<size_t> voidx_ref(oidx, oidx + sizeof(oidx) / sizeof(size_t));
    std::vector<size_t> voidx;

    nodep n1(2, vidx);

    if(vidx_ref.size() != n1.get_idx().size()) {
        fail_test(testname, __FILE__, __LINE__,
            "vidx_ref.size() != n1.get_idx().size()");
    }
    for(size_t i = 0; i < vidx_ref.size(); i++) {
        if(n1.get_idx()[i] != vidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__,
                "n1.get_idx()[i] != vidx_ref[i]");
        }
    }
    if(n1.get_cidx().size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "n1.get_cidx().size() != 0");
    }
    n1.build_output_indices(voidx);
    if(voidx_ref.size() != voidx.size()) {
        fail_test(testname, __FILE__, __LINE__,
            "voidx_ref.size() != voidx.size()");
    }
    for(size_t i = 0; i < voidx_ref.size(); i++) {
        if(voidx[i] != voidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__, "voidx[i] != voidx_ref[i]");
        }
    }

    std::auto_ptr<nodep> n2((nodep*)n1.clone());

    if(vidx_ref.size() != n2->get_idx().size()) {
        fail_test(testname, __FILE__, __LINE__,
            "vidx_ref.size() != n2->get_idx().size()");
    }
    for(size_t i = 0; i < vidx_ref.size(); i++) {
        if(n2->get_idx()[i] != vidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__,
                "n2->get_idx()[i] != vidx_ref[i]");
        }
    }
    if(n2->get_cidx().size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "n2->get_cidx().size() != 0");
    }
    n2->build_output_indices(voidx);
    if(voidx_ref.size() != voidx.size()) {
        fail_test(testname, __FILE__, __LINE__,
            "voidx_ref.size() != voidx.size()");
    }
    for(size_t i = 0; i < voidx_ref.size(); i++) {
        if(voidx[i] != voidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__, "voidx[i] != voidx_ref[i]");
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void node_product_test::test_2() {

    static const char testname[] = "node_product_test::test_2()";

    try {

    size_t idx[] = { 0, 2, 2, 1 };
    size_t cidx[] = { 2 };
    size_t oidx[] = { 0, 1 };
    std::vector<size_t> vidx(idx, idx + sizeof(idx) / sizeof(size_t));
    std::vector<size_t> vidx_ref(vidx);
    std::vector<size_t> vcidx(cidx, cidx + sizeof(cidx) / sizeof(size_t));
    std::vector<size_t> vcidx_ref(vcidx);
    std::vector<size_t> voidx_ref(oidx, oidx + sizeof(oidx) / sizeof(size_t));
    std::vector<size_t> voidx;

    nodep n1(2, vidx, vcidx);

    if(vidx_ref.size() != n1.get_idx().size()) {
        fail_test(testname, __FILE__, __LINE__,
            "vidx_ref.size() != n1.get_idx().size()");
    }
    for(size_t i = 0; i < vidx_ref.size(); i++) {
        if(n1.get_idx()[i] != vidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__,
                "n1.get_idx()[i] != vidx_ref[i]");
        }
    }
    if(vcidx_ref.size() != n1.get_cidx().size()) {
        fail_test(testname, __FILE__, __LINE__,
            "vcidx_ref.size() != n1.get_cidx().size()");
    }
    for(size_t i = 0; i < vcidx_ref.size(); i++) {
        if(n1.get_cidx()[i] != vcidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__,
                "n1.get_cidx()[i] != vcidx_ref[i]");
        }
    }
    n1.build_output_indices(voidx);
    if(voidx_ref.size() != voidx.size()) {
        fail_test(testname, __FILE__, __LINE__,
            "voidx_ref.size() != voidx.size()");
    }
    for(size_t i = 0; i < voidx_ref.size(); i++) {
        if(voidx[i] != voidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__, "voidx[i] != voidx_ref[i]");
        }
    }

    std::auto_ptr<nodep> n2((nodep*)n1.clone());

    if(vidx_ref.size() != n2->get_idx().size()) {
        fail_test(testname, __FILE__, __LINE__,
            "vidx_ref.size() != n2->get_idx().size()");
    }
    for(size_t i = 0; i < vidx_ref.size(); i++) {
        if(n2->get_idx()[i] != vidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__,
                "n2->get_idx()[i] != vidx_ref[i]");
        }
    }
    if(vcidx_ref.size() != n2->get_cidx().size()) {
        fail_test(testname, __FILE__, __LINE__,
            "vcidx_ref.size() != n2->get_cidx().size()");
    }
    for(size_t i = 0; i < vcidx_ref.size(); i++) {
        if(n2->get_cidx()[i] != vcidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__,
                "n2->get_cidx()[i] != vcidx_ref[i]");
        }
    }
    n2->build_output_indices(voidx);
    if(voidx_ref.size() != voidx.size()) {
        fail_test(testname, __FILE__, __LINE__,
            "voidx_ref.size() != voidx.size()");
    }
    for(size_t i = 0; i < voidx_ref.size(); i++) {
        if(voidx[i] != voidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__, "voidx[i] != voidx_ref[i]");
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void node_product_test::test_3() {

    static const char testname[] = "node_product_test::test_3()";

    try {

    size_t idx[] = { 0, 4, 1, 5, 2, 4, 3, 5 };
    size_t cidx[] = { 4, 5 };
    size_t oidx[] = { 0, 1, 2, 3 };
    std::vector<size_t> vidx(idx, idx + sizeof(idx) / sizeof(size_t));
    std::vector<size_t> vidx_ref(vidx);
    std::vector<size_t> vcidx(cidx, cidx + sizeof(cidx) / sizeof(size_t));
    std::vector<size_t> vcidx_ref(vcidx);
    std::vector<size_t> voidx_ref(oidx, oidx + sizeof(oidx) / sizeof(size_t));
    std::vector<size_t> voidx;

    nodep n1(4, vidx, vcidx);

    if(vidx_ref.size() != n1.get_idx().size()) {
        fail_test(testname, __FILE__, __LINE__,
            "vidx_ref.size() != n1.get_idx().size()");
    }
    for(size_t i = 0; i < vidx_ref.size(); i++) {
        if(n1.get_idx()[i] != vidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__,
                "n1.get_idx()[i] != vidx_ref[i]");
        }
    }
    if(vcidx_ref.size() != n1.get_cidx().size()) {
        fail_test(testname, __FILE__, __LINE__,
            "vcidx_ref.size() != n1.get_cidx().size()");
    }
    for(size_t i = 0; i < vcidx_ref.size(); i++) {
        if(n1.get_cidx()[i] != vcidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__,
                "n1.get_cidx()[i] != vcidx_ref[i]");
        }
    }
    n1.build_output_indices(voidx);
    if(voidx_ref.size() != voidx.size()) {
        fail_test(testname, __FILE__, __LINE__,
            "voidx_ref.size() != voidx.size()");
    }
    for(size_t i = 0; i < voidx_ref.size(); i++) {
        if(voidx[i] != voidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__, "voidx[i] != voidx_ref[i]");
        }
    }

    std::auto_ptr<nodep> n2((nodep*)n1.clone());

    if(vidx_ref.size() != n2->get_idx().size()) {
        fail_test(testname, __FILE__, __LINE__,
            "vidx_ref.size() != n2->get_idx().size()");
    }
    for(size_t i = 0; i < vidx_ref.size(); i++) {
        if(n2->get_idx()[i] != vidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__,
                "n2->get_idx()[i] != vidx_ref[i]");
        }
    }
    if(vcidx_ref.size() != n2->get_cidx().size()) {
        fail_test(testname, __FILE__, __LINE__,
            "vcidx_ref.size() != n2->get_cidx().size()");
    }
    for(size_t i = 0; i < vcidx_ref.size(); i++) {
        if(n2->get_cidx()[i] != vcidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__,
                "n2->get_cidx()[i] != vcidx_ref[i]");
        }
    }
    n2->build_output_indices(voidx);
    if(voidx_ref.size() != voidx.size()) {
        fail_test(testname, __FILE__, __LINE__,
            "voidx_ref.size() != voidx.size()");
    }
    for(size_t i = 0; i < voidx_ref.size(); i++) {
        if(voidx[i] != voidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__, "voidx[i] != voidx_ref[i]");
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void node_product_test::test_4() {

    static const char testname[] = "node_product_test::test_4()";

    try {

    size_t idx[] = { 0, 0 };
    size_t oidx[] = { 0 };
    std::vector<size_t> vidx(idx, idx + sizeof(idx) / sizeof(size_t));
    std::vector<size_t> vidx_ref(vidx);
    std::vector<size_t> vcidx_ref;
    std::vector<size_t> voidx_ref(oidx, oidx + sizeof(oidx) / sizeof(size_t));
    std::vector<size_t> voidx;

    nodep n1(1, vidx);

    if(vidx_ref.size() != n1.get_idx().size()) {
        fail_test(testname, __FILE__, __LINE__,
            "vidx_ref.size() != n1.get_idx().size()");
    }
    for(size_t i = 0; i < vidx_ref.size(); i++) {
        if(n1.get_idx()[i] != vidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__,
                "n1.get_idx()[i] != vidx_ref[i]");
        }
    }
    if(n1.get_cidx().size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "n1.get_cidx().size() != 0");
    }
    n1.build_output_indices(voidx);
    if(voidx_ref.size() != voidx.size()) {
        fail_test(testname, __FILE__, __LINE__,
            "voidx_ref.size() != voidx.size()");
    }
    for(size_t i = 0; i < voidx_ref.size(); i++) {
        if(voidx[i] != voidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__, "voidx[i] != voidx_ref[i]");
        }
    }

    std::auto_ptr<nodep> n2((nodep*)n1.clone());

    if(vidx_ref.size() != n2->get_idx().size()) {
        fail_test(testname, __FILE__, __LINE__,
            "vidx_ref.size() != n2->get_idx().size()");
    }
    for(size_t i = 0; i < vidx_ref.size(); i++) {
        if(n2->get_idx()[i] != vidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__,
                "n2->get_idx()[i] != vidx_ref[i]");
        }
    }
    if(n2->get_cidx().size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "n2->get_cidx().size() != 0");
    }
    n2->build_output_indices(voidx);
    if(voidx_ref.size() != voidx.size()) {
        fail_test(testname, __FILE__, __LINE__,
            "voidx_ref.size() != voidx.size()");
    }
    for(size_t i = 0; i < voidx_ref.size(); i++) {
        if(voidx[i] != voidx_ref[i]) {
            fail_test(testname, __FILE__, __LINE__, "voidx[i] != voidx_ref[i]");
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
