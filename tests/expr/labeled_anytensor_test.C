#include <complex>
#include <libtensor/core/allocator.h>
#include <libtensor/expr/anytensor.h>
#include <libtensor/expr/assignment_operator.h>
#include <libtensor/expr/operators.h>
#include <libtensor/expr/expression_dispatcher.h>
#include "labeled_anytensor_test.h"

namespace libtensor {


void labeled_anytensor_test::perform() throw(libtest::test_exception) {

    test_label();
    test_expr_int();
    test_expr_double();
    test_expr_complex_1();
    test_expr_complex_2();
}


namespace labeled_anytensor_test_ns {

template<size_t N, typename T>
class anytensor2_renderer : public expression_renderer_i<N, T> {
public:
    virtual ~anytensor2_renderer() { }
    virtual expression_renderer_i<N, T> *clone() const {
        return new anytensor2_renderer<N, T>();
    }
    virtual void render(const expression<N, T> &e, anytensor<N, T> &t) { }
};

template<size_t N, typename T>
class anytensor2 : public anytensor<N, T> {
public:
    static const char *k_tensor_type;

    anytensor2() {
        anytensor2_renderer<N, T> r;
        expression_dispatcher<N, T>::get_instance().register_renderer(
            k_tensor_type, r);
    }

    template<typename TensorType>
    explicit anytensor2(TensorType *pt) : anytensor<N, T>(pt) { }

    virtual const char *get_tensor_type() const {
        return k_tensor_type;
    }

};

template<size_t N, typename T>
const char *anytensor2<N, T>::k_tensor_type = "anytensor2";

} // namespace labeled_anytensor_test_ns
using namespace labeled_anytensor_test_ns;


void labeled_anytensor_test::test_label() throw(libtest::test_exception) {

    static const char *testname = "labeled_anytensor_test::test_label()";

    try {

    anytensor2<4, int> t;

    letter i, j, k, a, b, c;

    if(!t(i|j|a|b).get_label().contains(i)) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: t(i|j|a|b).contains(i)");
    }
    if(!t(i|j|a|b).get_label().contains(j)) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: t(i|j|a|b).contains(j)");
    }
    if(!t(i|j|a|b).get_label().contains(a)) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: t(i|j|a|b).contains(a)");
    }
    if(!t(i|j|a|b).get_label().contains(b)) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: t(i|j|a|b).contains(b)");
    }
    if(t(i|j|a|b).get_label().contains(k)) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: t(i|j|a|b).contains(k)");
    }

    if(t(i|j|a|b).get_label().index_of(i) != 0) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: t(i|j|a|b).index_of(i)");
    }
    if(t(i|j|a|b).get_label().index_of(j) != 1) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: t(i|j|a|b).index_of(j)");
    }
    if(t(i|j|a|b).get_label().index_of(a) != 2) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: t(i|j|a|b).index_of(a)");
    }
    if(t(i|j|a|b).get_label().index_of(b) != 3) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: t(i|j|a|b).index_of(b)");
    }

    if(t(i|j|a|b).get_label().letter_at(0) != i) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: t(i|j|a|b).letter_at(0)");
    }
    if(t(i|j|a|b).get_label().letter_at(1) != j) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: t(i|j|a|b).letter_at(1)");
    }
    if(t(i|j|a|b).get_label().letter_at(2) != a) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: t(i|j|a|b).letter_at(2)");
    }
    if(t(i|j|a|b).get_label().letter_at(3) != b) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: t(i|j|a|b).letter_at(3)");
    }

    anytensor2<1, int> s;

    if(s(i).get_label().index_of(i) != 0) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed label test: s(i).contains(i)");
    }

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void labeled_anytensor_test::test_expr_int() throw(libtest::test_exception) {

    static const char *testname = "labeled_anytensor_test::test_expr_int()";

    try {

    anytensor2<4, int> t1_ijab, t2_ijab;
    anytensor2<4, int> t3_jiab, t4_jiab;
    anytensor2<4, int> &t1i_ijab(t1_ijab);

    letter i, j, a, b;

    t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = -t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = -(t1_ijab(i|j|a|b) * 2);
    t3_jiab(j|i|a|b) = +t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = +(t1_ijab(i|j|a|b) * 2);
    t3_jiab(j|i|a|b) = t1i_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
    t2_ijab(i|j|a|b) = t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) - 2 * t3_jiab(j|i|a|b);
    2 * t1_ijab(i|j|a|b) - t3_jiab(j|i|a|b);
    2 * t1_ijab(i|j|a|b) - 3 * t3_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b));
    (t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)) +
        (t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b));

    t3_jiab(j|i|a|b) = 5 * t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b) * 5;
    t3_jiab(j|i|a|b) = 5 * t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
    t1_ijab(i|j|a|b) * 2 + t2_ijab(i|j|a|b);
    t1_ijab(i|j|a|b) + 5 * t2_ijab(i|j|a|b);
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) * 2;
    5 * t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) * 2;
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + 5 * t3_jiab(j|i|a|b);
    5 * t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b));
    5 * t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + 2 * t3_jiab(j|i|a|b));
    5 * (t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b));
    (t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)) * 5;
    5 * (t1_ijab(i|j|a|b) + 0 * t2_ijab(i|j|a|b));
    t4_jiab(j|i|a|b) = (t1_ijab(i|j|a|b) + 2 * t2_ijab(i|j|a|b)) * 5;
    3 * (t1_ijab(i|j|a|b) + 2 * t2_ijab(i|j|a|b)) * 5;

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void labeled_anytensor_test::test_expr_double() throw(libtest::test_exception) {

    static const char *testname = "labeled_anytensor_test::test_expr_double()";

    try {

    anytensor2<4, double> t1_ijab, t2_ijab;
    anytensor2<4, double> t3_jiab, t4_jiab;
    anytensor2<4, double> &t1i_ijab(t1_ijab);

    letter i, j, a, b;

    t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = -t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = -(t1_ijab(i|j|a|b) * 2.0);
    t3_jiab(j|i|a|b) = +t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = +(t1_ijab(i|j|a|b) * 2.0);
    t3_jiab(j|i|a|b) = t1i_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
    t2_ijab(i|j|a|b) = t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) - 2.0 * t3_jiab(j|i|a|b);
    2.0 * t1_ijab(i|j|a|b) - t3_jiab(j|i|a|b);
    2.0 * t1_ijab(i|j|a|b) - 3.0 * t3_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b));
    (t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)) +
        (t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b));

    t3_jiab(j|i|a|b) = 0.5 * t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b) * 0.5;
    t3_jiab(j|i|a|b) = 0.5 * t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
    t1_ijab(i|j|a|b) * 2.0 + t2_ijab(i|j|a|b);
    t1_ijab(i|j|a|b) + 0.5 * t2_ijab(i|j|a|b);
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) * 2.0;
    0.5 * t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) * 2.0;
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + 0.5 * t3_jiab(j|i|a|b);
    0.5 * t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b));
    0.5 * t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + 2.0 * t3_jiab(j|i|a|b));
    0.5 * (t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b));
    (t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)) * 0.5;
    0.5 * (t1_ijab(i|j|a|b) + 2.0 * t2_ijab(i|j|a|b));
    t4_jiab(j|i|a|b) = (t1_ijab(i|j|a|b) + 2.0 * t2_ijab(i|j|a|b)) * 0.5;
    2.0 * (t1_ijab(i|j|a|b) + 2.0 * t2_ijab(i|j|a|b)) * 0.5;

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void labeled_anytensor_test::test_expr_complex_1()
    throw(libtest::test_exception) {

    static const char *testname =
        "labeled_anytensor_test::test_expr_complex_1()";

    try {

    anytensor2< 4, std::complex<double> > t1_ijab, t2_ijab;
    anytensor2< 4, std::complex<double> > t3_jiab, t4_jiab;
    anytensor2< 4, std::complex<double> > &t1i_ijab(t1_ijab);

    letter i, j, a, b;

    t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = -t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = -(t1_ijab(i|j|a|b) * 2.0);
    t3_jiab(j|i|a|b) = +t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = +(t1_ijab(i|j|a|b) * 2.0);
    t3_jiab(j|i|a|b) = t1i_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
    t2_ijab(i|j|a|b) = t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) - 2.0 * t3_jiab(j|i|a|b);
    2.0 * t1_ijab(i|j|a|b) - t3_jiab(j|i|a|b);
    2.0 * t1_ijab(i|j|a|b) - 3.0 * t3_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b));
    (t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)) +
        (t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b));

    t3_jiab(j|i|a|b) = 0.5 * t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b) * 0.5;
    t3_jiab(j|i|a|b) = 0.5 * t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
    t1_ijab(i|j|a|b) * 2.0 + t2_ijab(i|j|a|b);
    t1_ijab(i|j|a|b) + 0.5 * t2_ijab(i|j|a|b);
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) * 2.0;
    0.5 * t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) * 2.0;
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + 0.5 * t3_jiab(j|i|a|b);
    0.5 * t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b));
    0.5 * t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + 2.0 * t3_jiab(j|i|a|b));
    0.5 * (t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b));
    (t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)) * 0.5;
    0.5 * (t1_ijab(i|j|a|b) + 2.0 * t2_ijab(i|j|a|b));
    t4_jiab(j|i|a|b) = (t1_ijab(i|j|a|b) + 2.0 * t2_ijab(i|j|a|b)) * 0.5;
    2.0 * (t1_ijab(i|j|a|b) + 2.0 * t2_ijab(i|j|a|b)) * 0.5;

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void labeled_anytensor_test::test_expr_complex_2()
    throw(libtest::test_exception) {

    static const char *testname =
        "labeled_anytensor_test::test_expr_complex_2()";

    try {

    anytensor2< 4, std::complex<double> > t1_ijab, t2_ijab;
    anytensor2< 4, std::complex<double> > t3_jiab, t4_jiab;
    anytensor2< 4, std::complex<double> > &t1i_ijab(t1_ijab);

    std::complex<double> ah(0.5), a2(2.0), a3(3.0);

    letter i, j, a, b;

    t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = -t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = -(t1_ijab(i|j|a|b) * a2);
    t3_jiab(j|i|a|b) = +t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = +(t1_ijab(i|j|a|b) * a2);
    t3_jiab(j|i|a|b) = t1i_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
    t2_ijab(i|j|a|b) = t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) - a2 * t3_jiab(j|i|a|b);
    a2 * t1_ijab(i|j|a|b) - t3_jiab(j|i|a|b);
    a2 * t1_ijab(i|j|a|b) - a3 * t3_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b);
    t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b));
    (t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)) +
        (t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b));

    t3_jiab(j|i|a|b) = ah * t1_ijab(i|j|a|b);
    t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b) * ah;
    t3_jiab(j|i|a|b) = ah * t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
    t1_ijab(i|j|a|b) * a2 + t2_ijab(i|j|a|b);
    t1_ijab(i|j|a|b) + ah * t2_ijab(i|j|a|b);
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) * a2;
    ah * t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) * a2;
    t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + ah * t3_jiab(j|i|a|b);
    ah * t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b));
    ah * t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + a2 * t3_jiab(j|i|a|b));
    ah * (t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b));
    (t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)) * ah;
    ah * (t1_ijab(i|j|a|b) + a2 * t2_ijab(i|j|a|b));
    t4_jiab(j|i|a|b) = (t1_ijab(i|j|a|b) + a2 * t2_ijab(i|j|a|b)) * ah;
    a2 * (t1_ijab(i|j|a|b) + a2 * t2_ijab(i|j|a|b)) * ah;

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


} // namespace libtensor
