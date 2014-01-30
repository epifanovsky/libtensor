#include <libtensor/exception.h>
#include <libtensor/iface/eval/eval_register.h>
#include "eval_register_test.h"

namespace libtensor {


void eval_register_test::perform() throw(libtest::test_exception) {

    test_1();
}


using namespace iface;


namespace {

class eval_1 : public eval_i {
    virtual bool can_evaluate(const expr::expr_tree &e) const {
        return false;
    }
    virtual void evaluate(const expr::expr_tree &e) const {

    }

};

class eval_selector : public eval_selector_i {
private:
    unsigned m_ntries;

public:
    eval_selector() : m_ntries(0) { }

    unsigned get_ntries() const { return m_ntries; }

    virtual void try_evaluator(const eval_i &e) {
        m_ntries++;
    }
};

} // unnamed namespace


void eval_register_test::test_1() {

    static const char testname[] = "eval_register_test::test_1()";

    try {

    eval_1 e1, e2;
    eval_selector es1, es2, es3;

    eval_register::get_instance().try_evaluators(es1);
    if(es1.get_ntries() != 0) {
        fail_test(testname, __FILE__, __LINE__, "es1.get_ntries() != 0");
    }

    eval_register::get_instance().add_evaluator(e1);
    eval_register::get_instance().add_evaluator(e2);
    eval_register::get_instance().try_evaluators(es2);
    if(es2.get_ntries() != 2) {
        fail_test(testname, __FILE__, __LINE__, "es2.get_ntries() != 0");
    }

    eval_register::get_instance().remove_evaluator(e1);
    eval_register::get_instance().remove_evaluator(e2);
    eval_register::get_instance().try_evaluators(es3);
    if(es3.get_ntries() != 0) {
        fail_test(testname, __FILE__, __LINE__, "es3.get_ntries() != 0");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

